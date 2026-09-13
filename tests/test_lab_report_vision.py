"""化验单**图像识别**链路的契约测试。

与 `test_lab_report.py`（判定链路）分开：这里测的是"图 → 回填文本"这一段，
判定本身不在这里重复测。

覆盖四类**悄悄坏掉也不会报错**的地方：

1. **值清理** —— 模型把 `11.2 10^9/L` 整串填进 `test_value`、或填 `<0.05`。
   后者若被当成精确值参与区间比较，会给出一个**看起来正常但错一位**的判定；
2. **单位归一化** —— `10⁹/L` 与 `10^9/L` 是同一个单位（写法不同），
   而 `mmol/L` 与 `μmol/L` **不是**（量级不同）。两种情况误判方向相反，都要卡住；
3. **回填文本的可回收性** —— 这是整条链路的核心契约：生成的回填文本必须能被
   既有 `parse_lab_items` 解析回**同一批指标**（含白名单外的指标）。
   实测过：换成空格/换行分隔就掉到 3/5，静默丢项；
4. **接口闸门** —— 体积上限、base64 合法性、未配置视觉模型时的报错。
   这些一旦失效，表现为"上传后没反应"或"打满内存"，都很难从现象反推。
"""

import base64
import io
from types import SimpleNamespace

import pytest

from app.api.lab_router import image_extract
from app.core.tools.lab_report_vision import (
    LabReportVisionTool,
    _clean_value,
    _name_candidates,
    _normalize_unit,
    build_fill_text,
    prepare_image,
)
from app.db.database import get_engine
from app.db.init_db import ensure_min_csv, import_min_kb, init_schema
from app.schema.lab_schema import LabImageExtractResponse, LabVisionItem, LabVisionReport


async def _prepare() -> None:
    ensure_min_csv()
    await init_schema(get_engine())
    await import_min_kb(get_engine())


def _req(user_id: str = "u1") -> SimpleNamespace:
    """路由函数只读 `request.state`，直接调用时给个最小替身即可。"""
    return SimpleNamespace(state=SimpleNamespace(user_id=user_id, request_id="test"))


def _png(size: tuple[int, int] = (300, 200)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# 1. 值清理
# --------------------------------------------------------------------------


def test_clean_value_forms():
    """纯数值通过；单位/箭头被剥掉；比较符值与定性描述被**拒绝**（给原因码）。"""
    assert _clean_value("11.2") == ("11.2", "")
    assert _clean_value("  4.51  ") == ("4.51", "")
    assert _clean_value("120/80") == ("120/80", "")

    # 单位混进值里 → 剥出来，只留数值
    assert _clean_value("11.2 10^9/L") == ("11.2", "")
    assert _clean_value("78.5 %") == ("78.5", "")
    # 异常标记不是数值的一部分
    assert _clean_value("11.2↑") == ("11.2", "")
    assert _clean_value("H 11.2") == ("11.2", "")
    assert _clean_value("11.2 H") == ("11.2", "")

    # 比较符值：不是精确值，必须拒绝而不是剥掉比较符
    for raw in ("<0.05", ">11.3", "≤1.7", ">=1.0"):
        value, reason = _clean_value(raw)
        assert reason == "comparator_value", raw
        assert raw in value

    assert _clean_value("") == ("", "empty_value")
    assert _clean_value("阴性")[1] == "value_not_numeric"
    # 两个数字 → 字段串位，宁可不收
    assert _clean_value("11.2 78.5")[1] == "value_not_numeric"


# --------------------------------------------------------------------------
# 2. 单位归一化
# --------------------------------------------------------------------------


def test_normalize_unit_equivalents_and_collisions():
    """等价写法必须合并；量级不同的单位**必须不合并**。"""
    same = ["10^9/L", "10⁹/L", "×10^9/L", "x10^9/L", "10*9/L", "10^9/l"]
    assert len({_normalize_unit(u) for u in same}) == 1, "同一单位的等价写法没有合并"

    assert _normalize_unit("μmol/L") == _normalize_unit("umol/L")
    assert _normalize_unit("µmol/L") == _normalize_unit("umol/L")  # MICRO SIGN
    assert _normalize_unit("g/L") == _normalize_unit("G/L")
    assert _normalize_unit("") == ""

    # 不同量级不能合并 —— 合并了会造出假的"单位一致"，进而拿错区间做判定
    assert _normalize_unit("mmol/L") != _normalize_unit("μmol/L")
    assert _normalize_unit("10^9/L") != _normalize_unit("10^12/L")


def test_name_candidates_forms():
    """名称的形态放宽：去括号备注、按空白切段；`A/G` 不能被斜杠切坏。"""
    assert _name_candidates("白细胞计数") == ["白细胞计数"]
    assert "白细胞计数" in _name_candidates("白细胞计数（WBC）")
    assert "白细胞计数" in _name_candidates("WBC 白细胞计数")
    assert _name_candidates("A/G") == ["A/G"]
    assert _name_candidates("   ") == []


# --------------------------------------------------------------------------
# 3. 回填文本 → 文本链路（核心契约）
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_text_round_trip_through_parser():
    """回填文本必须被 `parse_lab_items` 回收成**同一批指标**。

    含两类难缠的条目：

    - 白名单内的正式名（白细胞计数）—— 解析器会归一成短名「白细胞」，
      靠参考库的别名表再回到正式名；
    - 白名单外的指标（糖化血红蛋白 / 维生素 D）—— 只能靠显式 `名称：数值` 收，
      这正是"分隔符必须是全角冒号"的理由（换成空格时这两项会静默丢失）。
    """
    from app.core.rag.lab_reference_service import LabReferenceService
    from app.core.tools.lab_item_parser import parse_lab_items

    await _prepare()
    items = [
        {"item_name": "白细胞计数", "test_value": "11.2", "unit": "10^9/L", "fillable": True},
        {"item_name": "血红蛋白", "test_value": "128", "unit": "g/L", "fillable": True},
        {"item_name": "糖化血红蛋白", "test_value": "6.5", "unit": "%", "fillable": True},
        {"item_name": "维生素D", "test_value": "32", "unit": "ng/mL", "fillable": True},
    ]
    fill_text = build_fill_text(items)
    assert fill_text.count("\n") == 3, "回填文本应为一行一个条目"

    parsed = parse_lab_items(fill_text)
    assert parsed, "回填文本没有被解析器回收"

    # 取消回填的条目不得出现在文本里
    excluded = build_fill_text(
        [{"item_name": "血糖", "test_value": "117", "unit": "mg/dL", "fillable": False}]
    )
    assert excluded == ""

    # 名字可能被归一成短名，所以统一经参考库映射回正式名再比
    got = {}
    for r in await LabReferenceService().match_items([p["item_name"] for p in parsed]):
        name = (r["match"] or {}).get("item_name") or r["query"]
        got[name] = parsed[[p["item_name"] for p in parsed].index(r["query"])]["test_value"]

    assert got.get("白细胞计数") == "11.2"
    assert got.get("血红蛋白") == "128"
    assert got.get("糖化血红蛋白") == "6.5", "白名单外的指标没能回收"
    assert got.get("维生素D") == "32", "白名单外且不在参考库的指标没能回收"


# --------------------------------------------------------------------------
# 4. 分层报告（单位冲突 / 丢弃 / 未收录）
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_reports_unit_mismatch_and_excludes_from_fill_text():
    """单位与参考库不一致的条目：**进不了回填文本**，但要在 `items` 里如实展示。

    为什么必须排除：`LabReportTool` 只比数值、不看单位。`血糖 117 mg/dL`
    放进去就会和库里的 `3.9-6.1 mmol/L` 比大小并判出 H —— 数值方向或许凑巧对，
    但那是巧合，不是结论。
    """
    await _prepare()
    report = LabVisionReport(
        items=[
            LabVisionItem(item_name="血糖", test_value="117", unit="mg/dL", reference_range="70-110"),
            LabVisionItem(item_name="血红蛋白", test_value="128", unit="g/L", reference_range="130-175"),
        ]
    )
    out = await LabReportVisionTool()._normalize(report)

    by_name = {i["item_name"]: i for i in out["items"]}
    assert by_name["血糖"]["unit_status"] == "mismatch"
    assert by_name["血糖"]["fillable"] is False
    assert by_name["血红蛋白"]["unit_status"] == "ok"
    assert by_name["血红蛋白"]["fillable"] is True

    assert [u["item_name"] for u in out["unit_mismatch"]] == ["血糖"]
    assert "血糖" not in out["fill_text"]
    assert "血红蛋白" in out["fill_text"]
    assert any("单位" in w for w in out["warnings"]), "单位冲突必须在 warnings 里说出来"


@pytest.mark.asyncio
async def test_normalize_reports_dropped_values():
    """空值 / 比较符值 / 非数值：不进回填，但**不静默**——条目仍在 `items` 且给出原因码。"""
    await _prepare()
    report = LabVisionReport(
        items=[
            LabVisionItem(item_name="C反应蛋白", test_value="<0.05", unit="mg/L"),
            LabVisionItem(item_name="血小板计数", test_value="", unit="10^9/L"),
            LabVisionItem(item_name="尿蛋白", test_value="阴性", unit=""),
            LabVisionItem(item_name="白细胞计数", test_value="11.2", unit="10^9/L"),
        ]
    )
    out = await LabReportVisionTool()._normalize(report)

    reasons = {d["raw_name"]: d["reason"] for d in out["dropped"]}
    assert reasons["C反应蛋白"] == "comparator_value"
    assert reasons["血小板计数"] == "empty_value"
    assert reasons["尿蛋白"] == "value_not_numeric"

    assert len(out["items"]) == 4, "被丢弃的条目仍要在 items 里展示原始内容"
    assert out["fill_text"] == "白细胞计数：11.2 10^9/L"
    assert any("未纳入回填" in w for w in out["warnings"])


@pytest.mark.asyncio
async def test_normalize_maps_alias_to_canonical_and_keeps_uncovered():
    """别名/缩写归一成正式名；参考库外的指标如实标出但仍可回填。"""
    await _prepare()
    report = LabVisionReport(
        items=[
            LabVisionItem(item_name="WBC 白细胞计数", test_value="11.2", unit="10^9/L"),
            LabVisionItem(item_name="维生素D", test_value="32", unit="ng/mL"),
        ]
    )
    out = await LabReportVisionTool()._normalize(report)

    by_raw = {i["raw_name"]: i for i in out["items"]}
    assert by_raw["WBC 白细胞计数"]["item_name"] == "白细胞计数"
    assert by_raw["WBC 白细胞计数"]["in_reference_base"] is True
    assert by_raw["维生素D"]["in_reference_base"] is False
    assert out["uncovered"] == ["维生素D"]
    # 未收录的指标仍回填：让下游如实回答"未纳入参考库"，比悄悄丢掉更诚实
    assert "维生素D" in out["fill_text"]


# --------------------------------------------------------------------------
# 5. 图片预处理
# --------------------------------------------------------------------------


def test_prepare_image_normalizes_format_and_resizes():
    """输出格式收成 PNG/JPEG；超长边降采样；EXIF/格式差异在入口消掉。"""
    from PIL import Image

    # 非 JPEG 一律转 PNG
    out, mime, meta = prepare_image(_png((300, 200)))
    assert mime == "image/png"
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    assert meta["format"] == "PNG"
    assert meta["size"] == [300, 200]

    # JPEG 保持 JPEG（避免无损体量膨胀）
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, format="JPEG")
    out_j, mime_j, meta_j = prepare_image(buf.getvalue())
    assert mime_j == "image/jpeg"
    assert meta_j["format"] == "JPEG"

    # 超长边降采样到上限
    out_big, mime_big, meta_big = prepare_image(_png((3000, 600)))
    assert "resized_to" in meta_big
    assert max(meta_big["resized_to"]) == 2000
    from PIL import Image as _I

    with _I.open(io.BytesIO(out_big)) as im:
        assert im.size == (2000, 400), "降采样后的实际尺寸与 meta 不一致"


def test_prepare_image_accepts_avif_but_not_heic():
    """白名单必须与 Pillow 的**实际能力**对齐：能读的要放行，读不了的才拒。

    这是"白拒"类缺陷的守门测试 —— 名单里少一个格式，功能看起来正常，
    只是某些用户的图**永远传不进来**，而且报错文案是"不支持的图片格式"，
    会被读成"你传错格式了"，而不是"我们漏配了"。

    实测：Pillow 12 自带 libavif（`.avif` 在 `registered_extensions()` 里），
    所以 AVIF 必须放行；`pillow-heif` 未安装（`.heic`/`.heif` 不在表里），
    HEIC 只能拒 —— 这是真限制，不是配置遗漏。
    """
    from PIL import Image

    from app.core.tools.lab_report_vision import _ALLOWED_FORMATS

    # AVIF：Pillow 能读写 → 必须放行，且重编码成 PNG
    buf = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buf, format="AVIF")
    out, mime, meta = prepare_image(buf.getvalue())
    assert meta["format"] == "AVIF"
    assert mime == "image/png", "非 JPEG 来源应统一转 PNG 送模型"
    assert out[:8] == b"\x89PNG\r\n\x1a\n"

    # 白名单与 Pillow 能力的一致性：注册表里有的扩展名，格式应被接受
    registered = {e.lower() for e in Image.registered_extensions()}
    assert ".avif" in registered, "Pillow 不再支持 AVIF？那要把 AVIF 从白名单移除"
    assert "AVIF" in _ALLOWED_FORMATS

    # HEIC 读不了 → 必须在白名单之外（否则会走到解码失败的分支）
    assert not {".heic", ".heif"} & registered, "环境装上了 pillow-heif，应把 HEIC 加入白名单"
    assert "HEIC" not in _ALLOWED_FORMATS


def test_prepare_image_rejects_non_image():
    """不是图片要给明确错误，不能把未知字节塞给模型（那会得到一段编造的结果）。"""
    from app.common.exceptions import ParamException

    with pytest.raises(ParamException):
        prepare_image(b"this is definitely not an image")
    with pytest.raises(ParamException):
        prepare_image(b"")


# --------------------------------------------------------------------------
# 6. 接口闸门
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_extract_endpoint_gates():
    """未授权 / 未开启 / 非法 base64 / 超体积 —— 四种都要**明确报错**。"""
    from fastapi import HTTPException

    from app.config.settings import settings
    from app.schema.lab_schema import LabImageExtractRequest

    good = base64.b64encode(_png((60, 40))).decode()

    # 未授权
    with pytest.raises(HTTPException) as e1:
        await image_extract(LabImageExtractRequest(image_base64=good), _req(user_id=""))
    assert e1.value.status_code == 401

    # 未配置视觉模型 → 明确说"未开启"，而不是等到调用失败再报 500
    old = settings.LLM_VISION_MODEL_NAME
    settings.LLM_VISION_MODEL_NAME = ""
    try:
        with pytest.raises(HTTPException) as e2:
            await image_extract(LabImageExtractRequest(image_base64=good), _req())
        assert e2.value.status_code == 503
    finally:
        settings.LLM_VISION_MODEL_NAME = old

    # 非法 base64
    with pytest.raises(HTTPException) as e3:
        await image_extract(LabImageExtractRequest(image_base64="!" * 64), _req())
    assert e3.value.status_code == 400

    # 超过体积上限 —— 闸门必须在**解码前**生效，否则先吃满内存再报错
    old_mb = settings.LAB_IMAGE_MAX_MB
    settings.LAB_IMAGE_MAX_MB = 0.001
    try:
        with pytest.raises(HTTPException) as e4:
            await image_extract(LabImageExtractRequest(image_base64="A" * 4000), _req())
        assert e4.value.status_code == 413
    finally:
        settings.LAB_IMAGE_MAX_MB = old_mb


def test_unauthenticated_request_returns_401_not_500():
    """未鉴权必须是 **401**，不能是 500。

    这是一条**全站级**的回归守门：`AuthMiddleware` 继承 `BaseHTTPMiddleware`，
    它挂在 `ExceptionMiddleware` 的**外层**，所以在 `dispatch` 里 `raise HTTPException`
    不会被翻译成响应，而是冒泡成 500（且 body 为空）。

    现象不起眼，后果很实：前端靠 `res.status === 401` 判断 token 过期并登出，
    拿到 500 就会走进"系统暂时无法响应"的兜底话术 → **用户卡在一个登不出去的状态里**。

    ⚠️ 这条只有**真实 HTTP 调用**能发现 —— 直接调路由函数时中间件根本不参与
    （本文件其余接口测试就是这样，所以它们全部"绿"，却漏掉了这个洞）。
    """
    from fastapi.testclient import TestClient

    from app.main import app

    # 不用 with：跳过 lifespan（会预加载词典、探测模型档位，都不该在单测里发生）
    client = TestClient(app)
    resp = client.post(
        "/api/v1/lab/image-extract",
        json={"image_base64": "A" * 64},
    )
    assert resp.status_code == 401, f"未鉴权应返回 401，实际 {resp.status_code}"
    assert resp.json().get("msg") == "未授权"


@pytest.mark.asyncio
async def test_image_extract_endpoint_response_shape(monkeypatch):
    """成功路径的**响应契约**：结果字典必须能装配成 `LabImageExtractResponse`。

    这条不测模型（模型侧由 `scripts/probe_lab_vision.py` 的真实链路覆盖），
    测的是"识别结果 → 接口响应"的字段有没有对齐 —— 少一个键就是 500，
    而这类错在只跑识别模块的单测里永远看不见。
    """
    from app.config.settings import settings

    canned = {
        "item_count": 1,
        "items": [
            {
                "raw_name": "白细胞计数",
                "item_name": "白细胞计数",
                "test_value": "11.2",
                "unit": "10^9/L",
                "reference_range": "3.5-9.5",
                "in_reference_base": True,
                "lib_unit": "10^9/L",
                "unit_status": "ok",
                "fillable": True,
                "note": "",
            }
        ],
        "fill_text": "白细胞计数：11.2 10^9/L",
        "uncovered": [],
        "unit_mismatch": [],
        "dropped": [],
        "warnings": [],
        "image_meta": {"format": "PNG", "out_mime": "image/png"},
    }

    async def _fake(self, raw: bytes) -> dict:  # noqa: ANN001
        return dict(canned)

    monkeypatch.setattr(LabReportVisionTool, "extract_upload", _fake)
    old = settings.LLM_VISION_MODEL_NAME
    settings.LLM_VISION_MODEL_NAME = "qwen3-vl-flash"
    try:
        from app.schema.lab_schema import LabImageExtractRequest

        resp = await image_extract(
            LabImageExtractRequest(image_base64=base64.b64encode(_png((60, 40))).decode()), _req()
        )
    finally:
        settings.LLM_VISION_MODEL_NAME = old

    assert isinstance(resp.data, LabImageExtractResponse)
    assert resp.data.item_count == 1
    assert resp.data.fill_text == "白细胞计数：11.2 10^9/L"
    assert resp.data.image_meta["out_mime"] == "image/png"
