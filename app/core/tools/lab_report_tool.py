from __future__ import annotations

import re

from app.common.exceptions import ParamException, UserAuthException
from app.core.rag.lab_reference_service import LabReferenceService
from app.db.crud.archive_crud import ArchiveCRUD


#: 参考范围的三种可判定形态。注意 `<=` / `>=` 必须排在 `[<≤]` / `[>≥]` 之前，
#: 否则 `<=1.7` 会先被 `[<≤]` 吃掉 `<`，剩下的 `=1.7` 无法匹配数值。
_RANGE_RE = re.compile(
    r"^\s*(?:"
    r"(?P<low>[-+]?\d+(?:\.\d+)?)\s*-\s*(?P<high>[-+]?\d+(?:\.\d+)?)"
    r"|(?:<=|[<≤])\s*(?P<only_high>[-+]?\d+(?:\.\d+)?)"
    r"|(?:>=|[>≥])\s*(?P<only_low>[-+]?\d+(?:\.\d+)?)"
    r")\s*$"
)


def _try_float(v: str) -> float | None:
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _parse_reference_range(text: str) -> tuple[float | None, float | None]:
    """把参考范围字符串解析成 `(low, high)`；无法识别时返回 `(None, None)`。

    支持形态：

    - 双侧闭区间 `3.9-6.1` → `(3.9, 6.1)`
    - 仅上限 `<1.7` / `≤1.7` / `<=1.7` → `(None, 1.7)`
    - 仅下限 `>1.0` / `≥1.0` / `>=1.0` → `(1.0, None)`

    为什么必须有单侧形态：血脂四项在化验单上就是这么印的（总胆固醇/甘油三酯/
    低密度脂蛋白给上限，高密度脂蛋白给下限）。若用 `0-上限` 硬凑，
    高密度脂蛋白会把**正常的高值反向误报成偏高** —— 这属于"错抽"，不是"漏抽"。

    其余一切无法识别的写法（如 `阴性`、`男：130-175 女：115-150`）一律返回
    `(None, None)`，由调用方按"无法解析、不做判定"处理，**绝不猜**。
    """
    m = _RANGE_RE.match(text or "")
    if not m:
        return None, None
    if m.group("only_high") is not None:
        return None, float(m.group("only_high"))
    if m.group("only_low") is not None:
        return float(m.group("only_low")), None
    return float(m.group("low")), float(m.group("high"))


#: 图上这一行**没有**异常标记时的含义。医院只给异常项打标记，所以"没标记"就是正常。
_MEANING_IMAGE_BLANK = (
    "通用科普信息：化验单上这一项未标注异常标记，按常见报告规范即为正常范围，仅供参考。"
)

#: 判定依据在图上、但这一项不在参考库里 —— **有结论、没有解释文本**。
#: 不能沿用"无法提供解读服务"那句话：我们明明已经知道它偏高/偏低，
#: 说"无法解读"是错的，只是说不出原因和建议而已。
_MEANING_NO_REFERENCE_IMAGE = (
    "该指标暂未纳入参考库，无法提供可能原因与建议；"
    "判断依据是化验单上标注的异常标记，请以化验单与医生的解释为准。"
)

_MEANING_NO_REFERENCE = "当前检验指标暂未纳入参考库，无法提供解读服务，请核对指标名称或咨询执业医师。"
_MEANING_UNPARSABLE = "数值或参考范围格式无法解析，建议核对化验单原始内容或咨询执业医师。"
_MEANING_NORMAL = "通用科普信息：该指标在参考范围内，仅供参考，具体以检验机构与医生解释为准。"
_MEANING_DEFAULT = "通用科普信息：请结合复查与医生意见综合评估。"

#: `final_desc` 里最多给几项展开「可能原因 / 建议 / 何时就医」。
#: 一张生化全项能异常十几项，全量展开会把最终回复撑成一篇文档，
#: 真正要看的那几项反而被淹没。
_MAX_ADVICE_ITEMS = 6


def _norm_name(v: str | None) -> str:
    """项名归一化，**只用于在图标注索引里找对应条目**。

    为什么不直接等值比较：同一项在图上的写法与库内正式名可能差一两个字符
    （图上「γ-谷氨酰转移酶」经文本解析可能掉成「谷氨酰转移酶」，γ 不在
    解析器的名称字符集里）。差这一点就查不到标记，于是白白退回通用口径 ——
    等于把图上已有的信息丢了。
    """
    return re.sub(r"[^0-9a-z\u4e00-\u9fff\u0370-\u03ff]+", "", (v or "").casefold())


def _lookup_image_entry(index: dict[str, dict], *candidates: str) -> tuple[bool, dict]:
    """在归一化索引里找条目，返回 `(是否来自图上, 条目；没有则空字典)`。

    先精确匹配；不中再退一步做**后缀包含**匹配，且**只在候选唯一时采纳** ——
    多个候选说明有歧义（「红细胞」对「红细胞计数」和「红细胞压积」都成立），
    这时宁可不认，也不能把别的项的标记安到这一项上。
    """
    keys = [k for k in (_norm_name(c) for c in candidates) if k]
    for k in keys:
        if k in index:
            return True, index[k]
    for k in keys:
        hits = [v for kk, v in index.items() if kk.endswith(k) or k.endswith(kk)]
        if len(hits) == 1:
            return True, hits[0]
    return False, {}


class LabReportTool:
    """化验单通用解读工具：**判定优先采信图上的异常标记**，没有标记才回落参考区间比对。"""

    async def interpret(
        self,
        *,
        user_id: str,
        lab_item_list: list[dict],
        sync_to_archive: bool,
        image_items: dict[str, dict] | None = None,
    ) -> dict:
        """判定与解释。

        判定来源有两条，**优先级固定**：

        1. **图上标注**（`image_items` 里有对应项的）：化验单上印的 `↑`/`↓`/`H`/`L`
           是这家医院按该患者给出的方向结论，口径比通用参考区间准，直接采信；
           **该行没有标记就是正常** —— 医院只给异常项打标记。
        2. **库内参考区间**（`image_items` 里没有的项）：用户用文字打来的指标没有图，
           自然没有标记，只能拿数值和参考范围比大小。

        `image_items` 形状：`{归一化项名: {"flag": "H"/"L"/"", "reference_range": "…"}}`。
        **"键存在"本身就是信息**：它表示这一项来自那张化验单，因此适用"空白即正常"。
        没有这个键的项才走区间比对 —— 少了这个区分，会把"用户手打的数值"也
        按"图上没标 = 正常"处理，那是两件完全不同的事。

        参考范围只用于**展示**，同样优先图上那份（这家医院这张单子的口径）。
        """
        if not user_id:
            raise UserAuthException("未授权")
        if not lab_item_list:
            raise ParamException("lab_item_list 不能为空")

        names = [i.get("item_name", "").strip() for i in lab_item_list if i.get("item_name")]
        svc = LabReferenceService()
        matches = await svc.match_items(names)
        m_map = {m["query"]: m["match"] for m in matches}
        image_index = {
            k: v for k, v in ((_norm_name(k), v) for k, v in (image_items or {}).items()) if k
        }

        item_list: list[dict] = []
        for it in lab_item_list:
            name = it.get("item_name", "").strip()
            value = str(it.get("test_value", "")).strip()

            ref = m_map.get(name)
            canonical = (ref or {}).get("item_name") or name

            from_image, img = _lookup_image_entry(image_index, name, canonical)
            image_flag = (img.get("flag") or "").strip().upper() if from_image else ""
            image_range = (img.get("reference_range") or "").strip() if from_image else ""

            # 参考范围：图上的优先（该院按该患者印的），没有图才用库内通用口径
            ref_range = image_range or ((ref or {}).get("reference_range") or "")

            if from_image:
                abnormal_flag: str | None = image_flag or "N"
                judge_source: str | None = "image_flag" if image_flag else "image_blank"
            else:
                low, high = _parse_reference_range(ref_range)
                v = _try_float(value)
                if v is None or (low is None and high is None):
                    abnormal_flag = None
                elif low is not None and v < low:
                    abnormal_flag = "L"
                elif high is not None and v > high:
                    abnormal_flag = "H"
                else:
                    abnormal_flag = "N"
                judge_source = "general_range" if abnormal_flag else None

            if ref is None:
                # 库里没有这一项：图上给了方向就照给（只是没有解释文本），
                # 图上也没给才回到"无法提供解读服务"。
                meaning = (
                    _MEANING_NO_REFERENCE_IMAGE
                    if judge_source == "image_flag"
                    else _MEANING_NO_REFERENCE
                )
            elif abnormal_flag == "H":
                meaning = ref.get("high_meaning") or _MEANING_DEFAULT
            elif abnormal_flag == "L":
                meaning = ref.get("low_meaning") or _MEANING_DEFAULT
            elif abnormal_flag == "N":
                meaning = _MEANING_IMAGE_BLANK if judge_source == "image_blank" else _MEANING_NORMAL
            else:
                meaning = _MEANING_UNPARSABLE

            item_list.append(
                {
                    "item_name": canonical,
                    "test_value": value,
                    "reference_range": ref_range or None,
                    "abnormal_flag": abnormal_flag,
                    #: 这一条的状态是谁给的：「图上标的」还是「系统按通用口径算的」。
                    #: 下游文案要如实区分，用户有权知道哪句话是医院给的结论。
                    "judge_source": judge_source,
                    "meaning": meaning,
                }
            )

        # 「偏高/偏低之后怎么办」的文本：按 (项目名, 方向) 批量取一次，不逐项发查询
        advice_pairs = list(
            {
                (i["item_name"], i["abnormal_flag"])
                for i in item_list
                if i.get("abnormal_flag") in ("H", "L")
            }
        )
        advices = await svc.get_advices(advice_pairs) if advice_pairs else {}
        for i in item_list:
            if i.get("abnormal_flag") not in ("H", "L"):
                continue
            got = advices.get((i["item_name"], i["abnormal_flag"]))
            if got:
                i["advice"] = got

        if sync_to_archive:
            await ArchiveCRUD().sync_lab_items(user_id=user_id, items=item_list)

        final_lines = ["化验指标通用解读（科普信息，来源：化验单标注 + 结构化检验参考库）："]
        advice_shown = 0
        for o in item_list:
            final_lines.append(
                f"- {o['item_name']}：{o['test_value']}（参考：{o.get('reference_range') or '未知'}）"
                f" 状态：{o.get('abnormal_flag') or '未知'}；{o['meaning']}"
            )
            adv = o.get("advice")
            if not adv or advice_shown >= _MAX_ADVICE_ITEMS:
                continue
            advice_shown += 1
            for label, key in (
                ("可能原因", "causes"),
                ("建议", "advice"),
                ("何时就医", "when_to_see_doctor"),
            ):
                text = (adv.get(key) or "").strip()
                if text:
                    final_lines.append(f"  · {label}：{text}")
            note = (adv.get("disclaimer") or "").strip()
            if note:
                final_lines.append(f"  · 说明：{note}")

        return {"item_list": item_list, "final_desc": "\n".join(final_lines)}
