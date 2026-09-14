"""化验单图像识别：图片 → 结构化条目 → 可回填文本。

定位（为什么不把图片直接塞进 `LabReportTool`）
---------------------------------------------
`LabReportTool` 的职责是**判定**：拿数值和参考范围比对，输出 H / L / N。
图像识别的职责是**把图上的字变成可靠的文本**。混在一起有两个后果：

1. 判定逻辑被迫感知图片形态（mime / base64 / 上传上限 / 解码），主链路的工具被污染；
2. 识别质量与判定质量耦合，线上出问题时说不清是哪一段的锅。

所以本模块**只做识别 + 规范化 + 质量分层报告**，输出一段文本指标清单
（`名称：数值 单位`，一行一个）。该文本回流到输入框，由**现有的文本链路**
（`lab_item_parser.parse_lab_items` → `LabReportTool`）消费。
对主流程的侵入面 = 0：图路与文路共用同一个下游，判定口径天然一致，
"识别得对不对"和"判定得对不对"可以分开单独验证。

三级质量门（每一级都可能丢条目，且每一级都**显式报告**丢在哪）
--------------------------------------------------------------
1. **模型侧**（`chat_completion_vision`）：strict `json_schema` 约束字段名与类型；
   空响应 / 截断 / 校验失败三类分别计数，不混为一谈。
2. **客户端**（本模块 `_clean_value`）：值的形态校验。`11.2 10^9/L` 要把单位剥出来；
   `<0.05` 这种**比较符值**不是精确值，一律不进回填文本。
3. **参考库**（本模块 + `LabReferenceService`）：名称归一化（别名 → 正式名）
   与**单位一致性**。图上写 `mg/dL`、库里是 `mmol/L` 时排除回填 ——
   否则 `血糖 117 mg/dL` 会被拿去和 `3.9-6.1 mmol/L` 比大小并给出错误判定。

安全原则与 `lab_item_parser` 一致：**宁可漏抽，不可错抽。**
漏抽退化成"这项没识别出来"，用户看得见；错抽会变成一句口气确定的错误医疗判断。

为什么回填文本必须用全角冒号 `：`
--------------------------------
回填文本要能被 `lab_item_parser._EXPLICIT_SEPARATORS`（`：:=＝`）识别。
实测（`多模态化验单识别-选型实验与集成方案.md`）：`名称：数值 单位` 形态 5/5 可回收；
改成空格或换行分隔后掉到 3/5（`血红蛋白`、`中性粒细胞百分比` 静默丢失，
因为白名单只有 12 个短名，靠泛化写法收不全）。**格式是实测出来的，不是审美选择。**
"""

from __future__ import annotations

import base64
import binascii
import re

from app.common.exceptions import LLMCallException, ParamException, PayloadTooLargeException
from app.common.logger import get_logger
from app.config.settings import settings
from app.core.llm.llm_service import LLMService
from app.core.rag.lab_reference_service import LabReferenceService
from app.schema.lab_schema import LabVisionReport

logger = get_logger(__name__)


#: `data:image/png;base64,xxxx` 形态的前缀。有前缀时 mime 以前缀为准 ——
#: 它由浏览器按文件内容填，比用户单独传的字段可信。
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[a-z0-9.+-]+/[a-z0-9.+-]+)?\s*;?\s*base64,", re.I)

#: base64 字符集：解码前先粗筛，避免把明显不是图片的东西送进解码器。
_B64_RE = re.compile(r"^[A-Za-z0-9+/\s]*={0,2}$")


#: 识别提示词。要点与坑（2026-09-13 实测，2026-09-14 增异常标记字段）：
#:
#: - `test_value` **只填数值**。图上 `11.2 10^9/L` 若整串填进来，回填后会被
#:   文本链路当数值参与判定，而它根本不是数字；
#: - 结果列的 `↑` / `↓` / `H` / `L` 不是数值的一部分，但**不能丢** ——
#:   它是这家医院按该患者给出的判定方向，口径比库内通用区间准。所以单独用
#:   `abnormal_flag` 承载：合在 `test_value` 里会让这一条的数值整条进不了判定；
#: - **空白项 = 正常**（医院只给异常项打标记），所以"该行没有标记"是有效信息。
#:   但它**必须如实填空串**，由下游按"空白即正常"的规范处理 —— 不能让模型自己
#:   去推断哪一项该判正常，那是把判定权交给识别环节；
#: - `reference_range` **照抄**，不要换算。库里有通用口径、图上有这家医院的口径，
#:   模型自作主张换算会破坏这个前提（判定侧不再拿它比大小，它用于展示与人工核对）；
#: - 表头、患者信息、页码都不是检验项目，不能进 `items`；
#: - 看不清留空，**不要猜**。猜错一位数字在医疗场景就是明确危害。
VISION_PROMPT = (
    "这是一张化验单（检验报告）图片。请逐行提取其中**全部检验项目**，输出结构化结果。\n"
    "要求：\n"
    "1) `item_name` 用中文检验项目名（如「白细胞计数」）；\n"
    "2) `test_value` 只填**数值本身**：不要带单位、不要带 `↑`/`↓`/`H`/`L` 等异常标记、"
    "不要带 `<`/`>` 比较符、不要写「阴性/阳性」这类定性描述；\n"
    "3) `abnormal_flag` 专装结果列的异常标记：该行有 `↑` 或有 `H` 就填 `H`，"
    "有 `↓` 或有 `L` 就填 `L`，**该行没有任何标记就填空字符串**"
    "（化验单只给异常项打标记，没有标记即正常）；\n"
    "4) `unit` 填单位列的内容，`reference_range` 填参考范围列的内容，"
    "**严格照抄图中文字**，不要换算单位、不要用你自己的知识改写；\n"
    "5) 只输出检验项目行：表头、医院名、患者姓名/性别/年龄、采样时间、页码、"
    "医生签名、备注说明这些**都不是**检验项目，不要放进 `items`；\n"
    "6) 任何字段看不清或图中没有，就填**空字符串**，**绝对不要猜测或用常识补全**；\n"
    "7) 如果图中没有任何检验项目，`items` 返回空数组。"
)


#: 单张化验单的条目上限。正常生化全项 40 项上下，60 给出余量；
#: 超出说明模型在编造，截断并告警，不静默采信。
MAX_ITEMS = 60

#: 化验单上的异常/备注标记字符，出现在结果列但不是数值的一部分。
_MARK_CHARS = "↑↓▲▼⇑⇓⇧⇩"

#: 纯数值形态。`120/80`（血压形态）一并放行：形态合法，判定侧能不能用另说。
_VALUE_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?$")

#: 从混合串里切出「开头的数值 + 剩余部分」，用于清理"值里带了单位"的脏数据。
_LEADING_VALUE_RE = re.compile(
    r"^\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)\s*(.*)$", re.S
)

#: 比较符前缀：`<0.05` / `≤1.7` / `>11.3`。**带比较符的不是精确值。**
#: 为什么必须挡住：文本链路的 `_VALUE_RE` 只用了「前一字符不是数字或小数点」的
#: 负向断言，`<0.05` 会被它抽成 `0.05`，然后**当作精确值**参与区间判定。
#: 这是典型的"看起来正常、结论错一位"的隐性错误，所以宁可不回填。
_COMPARATOR_RE = re.compile(r"^\s*(?:<=|>=|[<>≤≥＜＞])")

#: 名称里尾部的括号备注，如「白细胞计数（WBC）」。
_TRAILING_NOTE_RE = re.compile(r"[（(][^（()）]{0,24}[）)]\s*$")

#: 名称里用来切段的空白与分隔符。**故意不含 `/`**：
#: 参考库里有 `A/G`（白球比）这种把斜杠当内容一部分的英文名，切开就再也对不上了。
_NAME_SPLIT_RE = re.compile(r"[\s|,，、]+")

#: 上标数字 → 普通数字，用于单位归一化（`10⁹/L` 与 `10^9/L` 是同一个单位）。
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUPERSCRIPT_RUN_RE = re.compile(r"([0-9])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)")

#: 能被 Pillow 解出来的来源格式。**判型以解码结果为准**，不看客户端声明的 mime。
#: 送模型前一律重编码成 JPEG / PNG，所以来源可以放宽（连 TIFF 也能吃）。
#:
#: ⚠️ 这份名单要与 **Pillow 实际能力**对齐，否则会出现"白拒"：
#: 实测 `PIL.Image.registered_extensions()` 里有 `.avif`（Pillow 12 自带 libavif），
#: 但本名单最初漏了它 → AVIF 图会被"不支持的图片格式"挡在门外，而它其实完全能读。
#: **HEIC/HEIF 确实读不了**（无 `pillow-heif`，注册表里没有 `.heic`）—— 这是真限制，
#: 不是配置遗漏；iPhone 默认拍照格式就是 HEIC，要用得先转 JPG。
_ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF", "AVIF"})

#: 送入模型前的最长边上限（像素）。依据见 `prepare_image` 的说明。
MAX_IMAGE_SIDE = 2000


def _strip_marks(raw: str) -> str:
    """剥掉结果列上不属于数值的标记（`↑↓`、独立的 `H`/`L`、尾随 `*`/`#`）。"""
    out = (raw or "").strip()
    for ch in _MARK_CHARS:
        out = out.replace(ch, "")
    # 独立的 H / L：只剥**紧贴数值**的那一个，避免把内容本身吃掉
    out = re.sub(r"(?i)^(?:H|L|HIGH|LOW)\s*[::]?\s*(?=[-+]?\d)", "", out)
    out = re.sub(r"(?i)\s+(?:H|L|HIGH|LOW)$", "", out)
    out = re.sub(r"[\s*#]+$", "", out)
    return out.strip()


#: 方向标记字符 —— `_MARK_CHARS` 按方向拆开，供 `_extract_flag` 判断偏高/偏低。
#: 箭头是**无歧义**的，认到就算，不存在把单位误判成标记的可能。
_FLAG_UP_CHARS = "↑▲⇑⇧"
_FLAG_DOWN_CHARS = "↓▼⇓⇩"

#: `H` / `L`（含英文全写）的**保守**形态：要么在数值**前面**（`H 6.5`），
#: 要么在**空格之后**的末尾（`6.5 H`）。**不许紧贴匹配** ——
#: 单位里就有 `L`（`g/L`、`mmol/L`、`10^9/L`），宽松匹配会把 `1.2 g/L` 判成偏低。
#: 与 `_strip_marks` 用的是同一组形态，两处必须一起改。
_FLAG_UP_RE = re.compile(r"(?i)^(?:HIGH|H)\s*[::]?\s*(?=[-+]?\d)|\s(?:HIGH|H)$")
_FLAG_DOWN_RE = re.compile(r"(?i)^(?:LOW|L)\s*[::]?\s*(?=[-+]?\d)|\s(?:LOW|L)$")

#: 模型直接给出的标记词的**白名单**。不在表内的一律不做语义猜测，回到正则兜底。
_FLAG_ALIAS = {
    "h": "H", "high": "H", "↑": "H", "偏高": "H", "高": "H",
    "l": "L", "low": "L", "↓": "L", "偏低": "L", "低": "L",
}


def _extract_flag(raw: str) -> str:
    """从一段原始文本里判断异常方向，返回 `H` / `L` / 空串。"""
    s = (raw or "").strip()
    if not s:
        return ""
    if any(c in s for c in _FLAG_UP_CHARS):
        return "H"
    if any(c in s for c in _FLAG_DOWN_CHARS):
        return "L"
    if _FLAG_UP_RE.search(s):
        return "H"
    if _FLAG_DOWN_RE.search(s):
        return "L"
    return ""


def _normalize_flag(model_flag: str, *fallbacks: str) -> str:
    """把模型给的标记规整成 `H` / `L` / 空串。

    模型这一栏的值**不可直接采信**：可能大小写不一、可能写「偏高」，也可能
    箭头留在 `test_value` 里而把这一栏留空（改了 prompt 之后最可能出现的形态）。
    所以顺序是：① 按白名单转换它给的值；② 转不出来再从 `test_value` 兜底提取。

    这样做的实际意义是**向后兼容**：即使模型完全忽略新字段、仍按老习惯把
    `6.5↑` 整串填进 `test_value`，标记也不会丢 —— 而 `_clean_value` 本来就会把
    标记从 `test_value` 里剥掉，所以数值侧照样干净。
    """
    key = (model_flag or "").strip().casefold()
    if key in _FLAG_ALIAS:
        return _FLAG_ALIAS[key]
    for fb in fallbacks:
        got = _extract_flag(fb)
        if got:
            return got
    return ""


def _looks_like_unit(text: str) -> bool:
    """判断一段文本**看起来是不是单位**（而非另一个数字或一句解释）。

    用途：清理「值里带了单位」的脏数据时，必须确认剩余部分确实是单位才能接受。
    若剩余部分是数字（如 `11.2 78.5`），说明模型把两个字段串位了，一律拒绝。
    """
    s = (text or "").strip()
    if not s:
        return False
    if not re.fullmatch(r"[A-Za-zμµ%‰/^0-9.·×*\-×\s]+", s):
        return False
    return bool("/" in s or "%" in s or any(c.isalpha() for c in s))


def _normalize_unit(unit: str | None) -> str:
    """单位归一化，只用于**比较**，不改写回填内容。

    各厂家化验单对同一个单位的写法并不统一，实测见过的等价写法：

    - `10^9/L` / `10⁹/L` / `×10^9/L` / `x10^9/L` / `10*9/L`
    - `μmol/L` / `umol/L`（以及 `µ` 这个 MICRO SIGN 变体）

    归一化策略：小写 → 上标数字补 `^` → 去掉乘号/上标符/空白 → `μ`/`µ` 统一成 `u`。
    大小写差异一并抹平（`g/L` 与 `G/L`）；`mmol` 与 `mol` 不会被合并，
    这是有意的 —— 它们是不同的量级，合并会制造假一致。
    """
    s = (unit or "").strip().casefold()
    if not s:
        return ""
    s = _SUPERSCRIPT_RUN_RE.sub(
        lambda m: m.group(1) + "^" + m.group(2).translate(_SUPERSCRIPT_MAP), s
    )
    for ch in ("×", "x", "*", "·", "^", " "):
        s = s.replace(ch, "")
    for src in ("μ", "µ"):
        s = s.replace(src, "u")
    return s


def decode_upload(
    payload: str, mime_hint: str = "", *, max_mb: float | None = None
) -> tuple[bytes, str]:
    """base64（可带 data URL 前缀）→ `(原始字节, mime)`。

    这是**所有图片入口的唯一闸门**：`/lab/image-extract`（用户手工选图后单独识别）
    与聊天路径（图片随消息一起发）都走这里。刻意只此一份 —— 两条入口各写一版尺寸/
    字符集校验，迟早会在某一版上漂移，而其中一版会漏掉体积门。

    顺序是有讲究的：**体积闸门放在解码之前**。base64 膨胀约 4/3，用编码长度反推
    上界就能拦掉超大文件，不必先把它解成几十 MB 的 bytes 再报错（那时内存已经吃掉了）。

    抛 `PayloadTooLargeException`（→413，可压小重试）与 `ParamException`（→400，数据本身坏）。
    调用方负责把 `code` 映射成自己的响应形态（HTTP 状态码 / 对话内的说明文案）。
    """
    max_mb = float(settings.LAB_IMAGE_MAX_MB or 0) if max_mb is None else float(max_mb)
    max_bytes = int(max_mb * 1024 * 1024)

    payload = (payload or "").strip()
    mime = (mime_hint or "").split(";")[0].strip().lower()

    # --- 1) 允许 data URL 前缀；前缀里的 mime 优先于独立字段 ---
    m = _DATA_URL_RE.match(payload)
    if m:
        mime = (m.group("mime") or mime).lower()
        payload = payload[m.end():]

    # --- 2) 体积闸门（解码之前）---
    if max_bytes and len(payload) > (max_bytes * 4 // 3) + 1024:
        raise PayloadTooLargeException(f"图片过大，请压缩到 {max_mb:g}MB 以内再试")

    payload = re.sub(r"\s+", "", payload)
    if not _B64_RE.match(payload):
        raise ParamException("图片数据不是合法的 base64，请重新选择文件")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ParamException("图片数据不是合法的 base64，请重新选择文件") from e

    if not raw:
        raise ParamException("图片内容为空")
    # 字符集合法但体积超限（例如填充字符使编码长度恰好没触发上面的粗筛）→ 仍按 413
    if max_bytes and len(raw) > max_bytes:
        raise PayloadTooLargeException(f"图片过大，请压缩到 {max_mb:g}MB 以内再试")

    return raw, mime


def prepare_image(raw: bytes) -> tuple[bytes, str, dict]:
    """把上传的原始字节整理成**可直接送模型**的图片，返回 `(字节, mime, meta)`。

    做三件事，都对"手机翻拍"这个主要场景直接有用：

    1. **按真实格式判型**：客户端声明的 mime 不可信（改后缀、抓包重放都能伪造），
       以解码结果为准；不在白名单里就明确报错，不把未知格式硬塞给模型。
    2. **按 EXIF 摆正方向**：手机竖拍的照片通常是"像素横着存 + EXIF 标 90°"，
       而多数多模态模型不读 EXIF —— 直接送原图等于送一张躺倒的化验单，
       识别质量取决于拍摄方向，这种不确定性必须在入口消掉。
    3. **限制最长边**：4000px 直出照片的像素量是 1000px 的 16 倍，而化验单的
       信息量在 2000px 就饱和了。降采样是**控成本 / 防超限**，不是提精度的手段。

    输出**一律重编码为 JPEG 或 PNG**（来源是 JPEG 就保持 JPEG，其余转 PNG），
    所以模型侧只会见到两种格式，来源格式的多样性被收在入口。

    依赖 Pillow（`requirements.txt` 已显式固定）。**惰性导入 + 失败降级**：
    Pillow 缺失时原样返回字节并记警告 —— 宁可功能降级，也不让导入期炸掉整个应用。
    """
    if not raw:
        raise ParamException("图片内容为空")

    meta: dict = {"orig_bytes": len(raw)}
    try:
        import io

        from PIL import Image, ImageOps
    except Exception as e:  # noqa: BLE001
        logger.warning("[vision] Pillow 不可用，跳过图片预处理（EXIF 摆正/降采样均未执行）: %s", e)
        return raw, "", meta

    try:
        with Image.open(io.BytesIO(raw)) as im:
            meta["format"] = (im.format or "").upper()
            meta["size"] = list(im.size)
            if meta["format"] not in _ALLOWED_FORMATS:
                raise ParamException(
                    f"不支持的图片格式（{meta['format'] or '未知'}），请使用 JPG / PNG 格式的图片"
                )
            im.load()  # 强制完整解码：截断文件在这里报错，而不是等到模型返回空结果
            im = ImageOps.exif_transpose(im)

            longest = max(im.size)
            if longest > MAX_IMAGE_SIDE:
                ratio = MAX_IMAGE_SIDE / longest
                new_size = (max(1, round(im.width * ratio)), max(1, round(im.height * ratio)))
                im = im.resize(new_size, Image.LANCZOS)
                meta["resized_to"] = list(new_size)

            # 灰度扫描件保持 L（转 RGB 会让 PNG 体积约翻三倍），其余统一 RGB
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")

            buf = io.BytesIO()
            if meta["format"] == "JPEG":
                im.save(buf, format="JPEG", quality=88, optimize=True)
                mime = "image/jpeg"
            else:
                im.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            out = buf.getvalue()
    except ParamException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("[vision] 图片解码失败: %s", e)
        raise ParamException("无法解析该图片，请换一张图片或改用手工输入") from e

    meta["out_bytes"] = len(out)
    meta["out_mime"] = mime
    return out, mime, meta


def _clean_value(raw: str) -> tuple[str, str]:
    """清理并校验 `test_value`，返回 `(值, 原因码)`；原因码为空串表示通过。

    原因码（都会出现在接口返回的 `dropped` 里，**不静默丢弃**）：

    - `empty_value`      —— 空值／图中看不清
    - `comparator_value` —— 带 `<`/`>` 前缀，非精确值（见 `_COMPARATOR_RE` 注释）
    - `value_not_numeric`—— 不是数值（定性描述，或字段串位）
    """
    s = _strip_marks(raw)
    if not s:
        return "", "empty_value"
    if _COMPARATOR_RE.match(s):
        return s, "comparator_value"
    if _VALUE_RE.match(s):
        return re.sub(r"\s+", "", s), ""
    m = _LEADING_VALUE_RE.match(s)
    if m:
        head, rest = m.group(1), m.group(2).strip()
        # 只有剩余部分**确实像单位**才接受；是数字说明串位，是句子说明不是数值
        if rest and _looks_like_unit(rest):
            return re.sub(r"\s+", "", head), ""
    return s, "value_not_numeric"


def _name_candidates(raw: str) -> list[str]:
    """一个名称的多种写法，按可信度从高到低排列。

    模型给的名字不总是干净的：「白细胞计数（WBC）」「WBC 白细胞计数」「白细胞」都可能。
    这里只做**确定性的形态放宽**（去括号备注、按空白切段），不做任何语义猜测；
    真正"别名 → 正式名"的归一化交给参考库（它有别名表，是唯一权威）。
    第一个在库里命中的候选就是结果。
    """
    base = (raw or "").strip()
    out: list[str] = []

    def _add(v: str) -> None:
        v = (v or "").strip()
        if v and v not in out:
            out.append(v)

    _add(base)
    bare = _TRAILING_NOTE_RE.sub("", base)
    _add(bare)
    for part in _NAME_SPLIT_RE.split(bare):
        _add(part)
        _add(_TRAILING_NOTE_RE.sub("", part))
    return out


def build_fill_text(items: list[dict]) -> str:
    """把可回填的条目拼成文本清单。

    形态固定为 `名称：数值 单位`，一行一个：
    前两个 token 之间用**全角冒号**（`lab_item_parser` 的显式分隔符集合里只有
    `：:=＝`），数值与单位之间用**空格**（空格不是子句边界，不会截断窗口）。

    只输出 `fillable=True` 的条目 —— 单位冲突和比较符值在这里被挡掉，
    它们是"会产生错误判定的输入"，不能流到下游。
    """
    lines: list[str] = []
    for it in items:
        if not it.get("fillable"):
            continue
        name = (it.get("item_name") or "").strip()
        value = (it.get("test_value") or "").strip()
        if not name or not value:
            continue
        unit = (it.get("unit") or "").strip()
        lines.append(f"{name}：{value}{(' ' + unit) if unit else ''}")
    return "\n".join(lines)


class LabReportVisionTool:
    """化验单图像识别：图片 → 条目 → 回填文本。**不做异常判定。**"""

    def __init__(self) -> None:
        self._llm = LLMService()
        self._ref = LabReferenceService()

    async def extract_upload(self, raw: bytes) -> dict:
        """从**上传的原始字节**识别：先 `prepare_image` 整理，再 `extract`。

        路由层调这个入口，就能完全不碰图片格式、EXIF、降采样这些细节。
        `meta` 会并进返回结果，便于前端显示"已压缩到 xxx"这类实话。
        """
        prepared, mime, meta = prepare_image(raw)
        result = await self.extract(image_bytes=prepared, mime=mime)
        result["image_meta"] = meta
        return result

    async def extract(self, *, image_bytes: bytes, mime: str) -> dict:
        """识别一张化验单图。

        入参是**已通过校验的原始字节**（mime 白名单、大小上限、可解码性由
        调用方 `/lab/image-extract` 负责）；本方法只做识别与规范化。

        抛 `LLMCallException`：未配置视觉模型（`LLM_VISION_MODEL_NAME` 为空）、
        调用超时、或重试后仍拿不到合法结构。这些是**没有部分结果可给**的失败，
        不适合用返回值表达。
        """
        if not image_bytes:
            raise ParamException("图片内容为空")

        report = await self._llm.chat_completion_vision(
            prompt=VISION_PROMPT,
            image_bytes=image_bytes,
            mime=mime,
            schema=LabVisionReport,
            timeout_s=settings.LLM_VISION_TIMEOUT_S,
            schema_name="lab_vision_report",
        )
        if report is None:
            raise LLMCallException("图像识别失败：模型未返回可解析的结构化结果，请重试或改用手工输入")

        return await self._normalize(report)

    async def _normalize(self, report: LabVisionReport) -> dict:
        """识别结果 → 分层报告。**这一步是纯函数式的，不依赖模型。**"""
        raw_items = report.items
        truncated = False
        if len(raw_items) > MAX_ITEMS:
            truncated = True
            logger.warning("[vision] 模型返回 %d 条，超过上限 %d，已截断", len(raw_items), MAX_ITEMS)
            raw_items = raw_items[:MAX_ITEMS]

        # ---- 一次批量查库：把所有候选写法摊平成一维，避免逐项发查询 ----
        cand_per_item: list[list[str]] = [_name_candidates(it.item_name) for it in raw_items]
        flat: list[str] = []
        for cands in cand_per_item:
            flat.extend(cands)
        matches = await self._ref.match_items(flat) if flat else []
        match_map = {m["query"]: m["match"] for m in matches}

        items: list[dict] = []
        uncovered: list[str] = []
        unit_mismatch: list[dict] = []
        dropped: list[dict] = []

        for raw, cands in zip(raw_items, cand_per_item):
            raw_name = (raw.item_name or "").strip()
            if not raw_name:
                # 连名字都没有的条目没有任何回填价值，直接丢
                dropped.append({"raw_name": "", "reason": "empty_name"})
                continue

            value, reason = _clean_value(raw.test_value)
            #: 异常标记与数值清理**互相独立**：值不合法（比较符 / 定性描述）时，
            #: 这一条进不了回填文本，但标记本身仍然是有效的识别结果，
            #: 所以放在这里算、不放在 `reason` 的分支里。
            image_flag = _normalize_flag(getattr(raw, "abnormal_flag", ""), raw.test_value)
            unit = (raw.unit or "").strip()

            # 候选按可信度排序，第一个在库里命中的就是它
            ref = None
            for c in cands:
                if match_map.get(c):
                    ref = match_map[c]
                    break

            canonical = (ref or {}).get("item_name") or _TRAILING_NOTE_RE.sub("", raw_name).strip()
            lib_unit = (ref or {}).get("unit") or ""

            # 单位一致性：两边都有值才判断，缺一边视为"无从判断"
            unit_status = "unknown"
            if unit and lib_unit:
                unit_status = "ok" if _normalize_unit(unit) == _normalize_unit(lib_unit) else "mismatch"

            fillable = True
            note = ""
            if reason:
                fillable = False
                note = reason
                dropped.append({"raw_name": raw_name, "reason": reason, "test_value": raw.test_value})
            elif unit_status == "mismatch":
                # 单位不一致**不再排除回填**（2026-09-14 起）。
                #
                # 原先把"单位不同"当成硬门槛，是因为判定要拿数值去比库内区间 ——
                # mg/dL 的值对 mmol/L 的区间会给出错误结论。现在判定改为
                # **采信图上的异常标记**，不再比库内区间，这个风险不在路径上了；
                # 继续排除反而是净损失：本来能判的一项会被整条丢掉。
                #
                # 仍然记录在案，因为"库内参考范围与这张单子不可比"这件事
                # 影响**展示**：不能把库内区间和图上数值放在一起给用户看。
                note = "unit_mismatch"
                unit_mismatch.append(
                    {
                        "raw_name": raw_name,
                        "item_name": canonical,
                        "image_unit": unit,
                        "lib_unit": lib_unit,
                    }
                )

            if ref is None:
                uncovered.append(raw_name)

            items.append(
                {
                    "raw_name": raw_name,
                    "item_name": canonical,
                    "test_value": value,
                    "image_flag": image_flag,
                    "unit": unit,
                    "reference_range": (raw.reference_range or "").strip(),
                    "in_reference_base": ref is not None,
                    "lib_unit": lib_unit or None,
                    "unit_status": unit_status,
                    "fillable": fillable,
                    "note": note,
                }
            )

        fill_text = build_fill_text(items)
        warnings: list[str] = []
        if truncated:
            warnings.append(f"识别条目超过 {MAX_ITEMS} 条，已只保留前 {MAX_ITEMS} 条，请核对原图。")
        if uncovered:
            warnings.append(
                f"有 {len(uncovered)} 项未纳入参考库（{'、'.join(uncovered[:5])}"
                f"{'…' if len(uncovered) > 5 else ''}），这些项没有可参考的解释文本。"
            )
        if unit_mismatch:
            detail = "、".join(f"{u['item_name']}（图上 {u['image_unit']}，库内 {u['lib_unit']}）" for u in unit_mismatch[:3])
            warnings.append(
                f"有 {len(unit_mismatch)} 项单位与参考库不一致（{detail}）。"
                "判定依据是化验单上标注的异常标记，不受单位影响；"
                "但库内参考范围与这张单子的单位不可比，请以单子上印的参考范围为准。"
            )
        if dropped:
            warnings.append(
                f"有 {len(dropped)} 项未纳入回填（空值 / 带比较符 / 非数值），"
                "原始内容仍在上方条目中展示，请自行核对。"
            )
        if not items:
            warnings.append("未识别到检验项目：请确认图片是检验报告且清晰完整，或改用手工输入。")

        logger.info(
            "[vision] 识别完成 条目=%d 回填=%d 未收录=%d 单位冲突=%d 丢弃=%d",
            len(items),
            len(fill_text.splitlines()),
            len(uncovered),
            len(unit_mismatch),
            len(dropped),
        )

        return {
            "item_count": len(items),
            "items": items,
            "fill_text": fill_text,
            "uncovered": uncovered,
            "unit_mismatch": unit_mismatch,
            "dropped": dropped,
            "warnings": warnings,
        }
