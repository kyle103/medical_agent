from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime
import json
import uuid
import asyncio

from app.common.exceptions import AppException, UserAuthException
from app.common.logger import get_logger, log_session_start, log_session_end
from app.config.settings import settings
from app.core.agent.stream_events import progress_payload
from app.core.agent.workflow import MedicalAgent
from app.core.compliance.compliance_service import ComplianceService
from app.core.memory.memory_service import MemoryService
from app.core.memory.long_memory_service import LongMemoryService
from app.core.tools.lab_report_vision import LabReportVisionTool, decode_upload
from app.schema.base import APIResponse
from app.schema.chat_schema import ChatCompletionRequest, ChatCompletionResponse

import time

router = APIRouter()
logger = get_logger(__name__)


#: 只发图、不打字、图又没读出来时回给用户的话。
#: 不能返回空回复：用户看到一片空白只会以为系统卡了，多半会再传一次同一张图。
_IMAGE_ONLY_FAILED_REPLY = (
    "我这边没能从这张图片里读出可以解读的检验指标。"
    "可以换一张更清晰、拍全一些的图片再试，或者直接把指标按「名称 + 数值」用文字发来，"
    "例如：血糖 6.5；血压 120/80。"
)


async def _extract_lab_image(image_base64: str, image_mime: str) -> tuple[str, list, str]:
    """随消息上传的图片 → `(识别文本, 结构化条目, 说明)`。

    **识别失败一律非致命。** 未配置视觉模型、尺寸超限、格式非法、模型超时、合规拦截
    —— 每一种都只是"这张图没读成"，把它记进说明后照常把用户这一轮走完：
    用户可能同时打了字，绝不能因为图片读不出来就 503 掉整轮对话
    （`/lab/image-extract` 那个独立端点可以硬报错，因为那条路径的唯一目的就是读这张图）。

    **注入防护**：图上的文字是**不可信输入**，识别结果会被拼进生成 prompt。
    一张写着"忽略以上指令"的化验单就是一条注入路径，所以先过一遍
    `input_compliance_check`。命中就整个丢弃识别结果并如实告知 ——
    注意是**丢弃**，不是拦下这一轮：用户打的字仍然要正常回答。
    """
    if not (image_base64 or "").strip():
        return "", [], ""

    try:
        raw, mime_hint = decode_upload(image_base64, image_mime)
    except AppException as e:
        logger.info("chat image rejected: %s", e)
        return "", [], str(e)

    if not (settings.LLM_VISION_MODEL_NAME or "").strip():
        logger.info("chat image skipped: LLM_VISION_MODEL_NAME 未配置")
        return "", [], "当前未开启图像识别（未配置视觉模型）。"

    try:
        result = await LabReportVisionTool().extract_upload(raw)
    except AppException as e:
        logger.warning("chat image extract failed: %s", e)
        return "", [], str(e)
    except Exception as e:  # noqa: BLE001
        # 视觉调用是外部依赖，任何未预期异常都不该把用户的整轮对话带走
        logger.error("chat image extract crashed: %s", e)
        return "", [], "图片识别时出现异常，请重试。"

    items = result.get("items") or []
    text = (result.get("fill_text") or "").strip()
    logger.info(
        "[vision] chat 图片识别完成 mime=%s 条目=%s 可判定=%s",
        mime_hint or "-",
        result.get("item_count"),
        len(text.splitlines()),
    )

    if not text:
        # 读到了条目但没有一条能参与判定（都因单位不一致/比较符/看不清被排除）。
        # 条目本身仍然要带上：生成阶段那段话要能说清"N 项为什么没纳入"。
        # 另外把识别自带的告警（含未收录项）一并交代，否则用户以为整张单子被判过了。
        warnings = "；".join(result.get("warnings") or [])
        return "", items, warnings or "没能从这张图里读出可参与判定的指标。"

    ok, msg = ComplianceService().input_compliance_check(text)
    if not ok:
        logger.warning("chat image text blocked by compliance: %s", msg)
        return "", [], f"图片里识别出的文字未通过合规检查，已忽略其内容（{msg}）"

    return text, items, ""


@router.post("/completion", response_model=APIResponse[ChatCompletionResponse])
async def completion(req: ChatCompletionRequest, request: Request):
    user_id = getattr(request.state, "user_id", None)

    session_id = req.session_id
    if not session_id or session_id.strip() == "":
        session_id = str(uuid.uuid4())
        logger.info("Generated new session for user %s: %s", user_id, session_id)

    if req.stream:
        return StreamingResponse(
            _stream_generator(user_id=user_id, session_id=session_id, req=req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        t0 = time.perf_counter()

        # 识别放在建图之前：图片字节到此为止，进图的只有识别出的小文本。
        image_lab_text, image_lab_items, image_note = await _extract_lab_image(
            req.image_base64, req.image_mime
        )
        # 只发图、不打字、图又没读出来 —— 这一轮没有任何可以据以回答的东西。
        # 直接给一句明确的说明，**不入图**：空输入进去只会白跑一整条 LLM 流水线
        # 再加一份合规检查，最后产出一段没有依据的话。
        if not req.user_input.strip() and not image_lab_text and not image_lab_items:
            return APIResponse(
                data=ChatCompletionResponse(
                    session_id=session_id,
                    user_input="",
                    assistant_output=image_note or _IMAGE_ONLY_FAILED_REPLY,
                    intent="lab",
                    create_time=datetime.now().isoformat(timespec="seconds"),
                ),
                request_id=getattr(request.state, "request_id", ""),
            )

        agent = MedicalAgent()
        t1 = time.perf_counter()

        log_session_start(
            session_id=session_id,
            user_id=user_id or "",
            user_input=req.user_input,
            stream=False,
        )

        result = await agent.run(
            user_id=user_id,
            session_id=session_id,
            user_input=req.user_input,
            stream=False,
            enable_archive_link=req.enable_archive_link,
            image_lab_text=image_lab_text,
            image_lab_items=image_lab_items,
            image_note=image_note,
        )
        t2 = time.perf_counter()

        total_ms = int((t2 - t0) * 1000)
        logger.info(
            "chat_completion perf: build_agent_ms=%s run_ms=%s total_ms=%s intent=%s",
            int((t1 - t0) * 1000),
            int((t2 - t1) * 1000),
            total_ms,
            result.get("intent"),
        )

        log_session_end(
            session_id=session_id,
            total_ms=total_ms,
            intent=result.get("intent", ""),
        )

        result["session_id"] = session_id

        return APIResponse(
            data=ChatCompletionResponse(**result),
            request_id=getattr(request.state, "request_id", ""),
        )
    except UserAuthException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        log_session_end(session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="服务内部错误")


async def _stream_generator(user_id: str, session_id: str, req: ChatCompletionRequest):
    t0 = time.perf_counter()
    user_input = req.user_input
    try:
        # 识别在开流之前就得算完，但**界面不能干等着**：视觉调用动辄数秒，
        # 这段时间里前端只挂着一个还没发过任何事件的流。先把 progress 推出去，
        # 让用户看到"正在识别化验单"而不是一个假死的界面。
        # 事件契约见 stream_events.py；前端标签注册在 app.js 的 names 表。
        if (req.image_base64 or "").strip():
            yield f"data: {json.dumps(progress_payload('lab_vision'), ensure_ascii=False)}\n\n"

        image_lab_text, image_lab_items, image_note = await _extract_lab_image(
            req.image_base64, req.image_mime
        )

        # 只发图、不打字、图又没读出来：见非流式分支的同类守卫说明。
        # 这里必须**照常收尾**（progress + chunk + done），否则前端停在"正在识别化验单"。
        if not user_input.strip() and not image_lab_text and not image_lab_items:
            reply = image_note or _IMAGE_ONLY_FAILED_REPLY
            yield f"data: {json.dumps({'type': 'chunk', 'content': reply}, ensure_ascii=False)}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "session_id": session_id,
                        "intent": "lab",
                        "needs_confirmation": False,
                        "conversation_turns": 0,
                        "cache": None,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            total_ms = int((time.perf_counter() - t0) * 1000)
            log_session_end(session_id=session_id, total_ms=total_ms)
            return

        agent = MedicalAgent()
        log_session_start(
            session_id=session_id,
            user_id=user_id or "",
            user_input=user_input,
            stream=True,
        )
        async for line in agent.run_stream(
            user_id=user_id,
            session_id=session_id,
            user_input=user_input,
            enable_archive_link=req.enable_archive_link,
            image_lab_text=image_lab_text,
            image_lab_items=image_lab_items,
            image_note=image_note,
        ):
            yield f"data: {line}\n\n"
        total_ms = int((time.perf_counter() - t0) * 1000)
        log_session_end(session_id=session_id, total_ms=total_ms)
    except UserAuthException:
        total_ms = int((time.perf_counter() - t0) * 1000)
        log_session_end(session_id=session_id, total_ms=total_ms, error="unauthorized")
        yield f"data: {json.dumps({'type': 'error', 'content': '未授权'}, ensure_ascii=False)}\n\n"
    except Exception as e:
        total_ms = int((time.perf_counter() - t0) * 1000)
        logger.error("stream completion failed: %s", e)
        log_session_end(session_id=session_id, total_ms=total_ms, error=str(e))
        yield f"data: {json.dumps({'type': 'error', 'content': '服务内部错误'}, ensure_ascii=False)}\n\n"


class SessionEndRequest(BaseModel):
    session_id: str = Field(..., description="要结束的会话ID")


@router.post("/session/end", response_model=APIResponse[dict])
async def end_session(req: SessionEndRequest, request: Request):
    """结束会话并触发长期记忆批量写入。"""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未授权")

    session_id = req.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    try:
        mem = MemoryService()
        history = await mem.get_user_memory(user_id=user_id, session_id=session_id, limit=100)

        svc = LongMemoryService()
        if svc.is_enabled() and history:
            result = await svc.batch_write_session(
                user_id=user_id, session_id=session_id, history=history
            )
            logger.info("session_end long_memory flush: user=%s session=%s result=%s", user_id, session_id, result)
        else:
            result = {"written": 0, "skipped": 0, "replaced": 0}

        # Step 3.5：此处原有一段「get_state() 读整行 → 写 long_memory_flushed=True → upsert_state() 整行回写」，
        # 已整段删除，原因有三：
        #   1. 该标记全仓库**零处读取**（已 grep 确认；agent_session_state 也没有对应列，它只活在 state_json 里）。
        #   2. 它与 `nodes.py::memory_update` 构成同一行的**两个无协调写者**：后者每轮写
        #      pending_confirmation / private_scratchpads / last_decision，一旦与本段的「读—改—写回」交错，
        #      就会把刚写进去的新一轮 last_decision 覆盖成旧值（后写者胜）。
        #   3. 长期记忆的幂等去重由 `batch_write_session` 的游标（source_chat_id）负责，不依赖该标记；
        #      本次刷写结果也已通过下方 APIResponse.long_memory_result 直接返回给调用方。
        # 删除后 `agent_session_state` 的唯一写者收敛为 memory_update 节点，为 Step 4 引入 checkpointer 扫清
        # 「同一份业务态多个 owner」的不一致面。

        return APIResponse(
            data={"session_id": session_id, "long_memory_result": result},
            request_id=getattr(request.state, "request_id", ""),
        )
    except Exception as e:
        logger.error("end_session failed: %s", e)
        raise HTTPException(status_code=500, detail="服务内部错误")
