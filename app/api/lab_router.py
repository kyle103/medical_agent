from __future__ import annotations

import base64
import binascii
import re

from fastapi import APIRouter, HTTPException, Request

from app.common.exceptions import AppException
from app.common.logger import get_logger
from app.config.settings import settings
from app.core.tools.lab_report_tool import LabReportTool
from app.core.tools.lab_report_vision import LabReportVisionTool
from app.schema.base import APIResponse
from app.schema.lab_schema import (
    LabImageExtractRequest,
    LabImageExtractResponse,
    LabReportInterpretRequest,
    LabReportInterpretResponse,
)

router = APIRouter()
logger = get_logger(__name__)

#: `data:image/png;base64,xxxx` 形态的前缀。有前缀时 mime 以前缀为准 ——
#: 它由浏览器按文件内容填，比用户单独传的字段可信。
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[a-z0-9.+-]+/[a-z0-9.+-]+)?\s*;?\s*base64,", re.I)

#: base64 字符集：解码前先粗筛，避免把明显不是图片的东西送进解码器。
_B64_RE = re.compile(r"^[A-Za-z0-9+/\s]*={0,2}$")


@router.post("/report-interpret", response_model=APIResponse[LabReportInterpretResponse])
async def report_interpret(req: LabReportInterpretRequest, request: Request):
    user_id = getattr(request.state, "user_id", None)

    tool = LabReportTool()
    # 入参是 `LabItemInput` 模型，工具侧按 dict 取值（`i.get(...)`）——
    # 必须 `model_dump()` 转换。此前直接把模型对象传下去，接口必然 500
    # （`AttributeError: 'LabItemInput' object has no attribute 'get'`）。
    # 图执行链路（`tool_executor._execute_lab_report`）传的本来就是 dict，
    # 所以只有这个 REST 入口受影响。
    result = await tool.interpret(
        user_id=user_id,
        lab_item_list=[i.model_dump() for i in req.lab_item_list],
        sync_to_archive=req.sync_to_archive,
    )

    # 合规检查已禁用，不再添加免责声明
    final_desc = result["final_desc"]

    return APIResponse(
        data=LabReportInterpretResponse(**{**result, "final_desc": final_desc}),
        request_id=getattr(request.state, "request_id", ""),
    )


@router.post("/image-extract", response_model=APIResponse[LabImageExtractResponse])
async def image_extract(req: LabImageExtractRequest, request: Request):
    """化验单图片 → 检验项目清单（**只识别，不解读**）。

    职责刻意收窄：本接口只把图变成"可回填的文本"，不产出任何异常判定。
    用户在前端确认／修改后，文本走原有的对话或 `/report-interpret` 完成解读。
    这样"图路"与"文路"最终经过**同一个**判定实现（`LabReportTool`），
    不会出现两套口径；主流程的图、端点、状态机一行未改。
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未授权")

    if not (settings.LLM_VISION_MODEL_NAME or "").strip():
        raise HTTPException(status_code=503, detail="图像识别未开启：未配置视觉模型（LLM_VISION_MODEL_NAME）")

    max_mb = float(settings.LAB_IMAGE_MAX_MB or 0)
    max_bytes = int(max_mb * 1024 * 1024)

    # --- 1) 取 payload 与 mime；允许 data URL 前缀 ---
    payload = (req.image_base64 or "").strip()
    mime_hint = (req.mime or "").split(";")[0].strip().lower()
    m = _DATA_URL_RE.match(payload)
    if m:
        mime_hint = (m.group("mime") or mime_hint).lower()
        payload = payload[m.end():]

    # --- 2) 体积闸门放在**解码之前** ---
    # base64 膨胀约 4/3，用编码长度反推上界即可拦掉超大文件，
    # 不必先把它解成几十 MB 的 bytes 再报错。
    if max_bytes and len(payload) > (max_bytes * 4 // 3) + 1024:
        raise HTTPException(status_code=413, detail=f"图片过大，请压缩到 {max_mb:g}MB 以内再试")

    payload = re.sub(r"\s+", "", payload)
    if not _B64_RE.match(payload):
        raise HTTPException(status_code=400, detail="图片数据不是合法的 base64，请重新选择文件")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail="图片数据不是合法的 base64，请重新选择文件") from e

    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if max_bytes and len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"图片过大，请压缩到 {max_mb:g}MB 以内再试")

    # --- 3) 识别 ---
    # 真正的格式判定在 `prepare_image` 里按解码结果做，`mime_hint` 只进日志。
    try:
        result = await LabReportVisionTool().extract_upload(raw)
    except AppException as e:
        # ParamException（图不合法）→ 400，LLMCallException（模型侧失败）→ 503
        raise HTTPException(status_code=getattr(e, "code", 500), detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.error("[vision] 图片识别失败: %s", e)
        raise HTTPException(status_code=500, detail="图像识别失败，请重试或改用手工输入") from e

    logger.info(
        "[vision] 图片识别 user=%s mime_hint=%s 条目=%s bytes=%s→%s",
        user_id,
        mime_hint or "-",
        result.get("item_count"),
        len(raw),
        (result.get("image_meta") or {}).get("out_bytes"),
    )

    return APIResponse(
        data=LabImageExtractResponse(**result),
        request_id=getattr(request.state, "request_id", ""),
    )
