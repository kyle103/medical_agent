from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.common.exceptions import AppException
from app.common.logger import get_logger
from app.config.settings import settings
from app.core.tools.lab_report_tool import LabReportTool
from app.core.tools.lab_report_vision import LabReportVisionTool, decode_upload
from app.schema.base import APIResponse
from app.schema.lab_schema import (
    LabImageExtractRequest,
    LabImageExtractResponse,
    LabReportInterpretRequest,
    LabReportInterpretResponse,
)

router = APIRouter()
logger = get_logger(__name__)


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

    # --- 1) 共享闸门：尺寸（解码前）→ data URL 前缀 → base64 字符集 → 解码 ---
    # 与聊天路径（图片随消息一起发）**共用同一份实现**，否则两条入口的校验迟早漂移。
    # 异常带 `code`（413/400），下面统一按 code 映射成 HTTP 状态码。
    try:
        raw, mime_hint = decode_upload(req.image_base64 or "", req.mime or "")
    except AppException as e:
        raise HTTPException(status_code=getattr(e, "code", 400), detail=str(e)) from e

    # --- 2) 识别 ---
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
