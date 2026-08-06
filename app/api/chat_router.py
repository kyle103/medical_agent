from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uuid
import asyncio

from app.common.exceptions import UserAuthException
from app.common.logger import get_logger, log_session_start, log_session_end
from app.core.agent.workflow import MedicalAgent
from app.core.memory.memory_service import MemoryService
from app.core.memory.long_memory_service import LongMemoryService
from app.schema.base import APIResponse
from app.schema.chat_schema import ChatCompletionRequest, ChatCompletionResponse

import time

router = APIRouter()
logger = get_logger(__name__)


@router.post("/completion", response_model=APIResponse[ChatCompletionResponse])
async def completion(req: ChatCompletionRequest, request: Request):
    user_id = getattr(request.state, "user_id", None)

    session_id = req.session_id
    if not session_id or session_id.strip() == "":
        session_id = str(uuid.uuid4())
        logger.info("Generated new session for user %s: %s", user_id, session_id)

    if req.stream:
        return StreamingResponse(
            _stream_generator(user_id=user_id, session_id=session_id, user_input=req.user_input, enable_archive_link=req.enable_archive_link),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        t0 = time.perf_counter()
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


async def _stream_generator(user_id: str, session_id: str, user_input: str, enable_archive_link: bool):
    t0 = time.perf_counter()
    try:
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
            enable_archive_link=enable_archive_link,
        ):
            yield f"data: {line}\n\n"
        total_ms = int((time.perf_counter() - t0) * 1000)
        log_session_end(session_id=session_id, total_ms=total_ms)
    except UserAuthException:
        import json
        total_ms = int((time.perf_counter() - t0) * 1000)
        log_session_end(session_id=session_id, total_ms=total_ms, error="unauthorized")
        yield f"data: {json.dumps({'type': 'error', 'content': '未授权'}, ensure_ascii=False)}\n\n"
    except Exception as e:
        import json
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

        try:
            from app.core.session.agent_state_store import AgentStateStore
            rt_state = await AgentStateStore().get_state(user_id=user_id, session_id=session_id)
            if isinstance(rt_state, dict):
                rt_state["long_memory_flushed"] = True
                await AgentStateStore().upsert_state(user_id=user_id, session_id=session_id, state=rt_state)
        except Exception:
            pass

        return APIResponse(
            data={"session_id": session_id, "long_memory_result": result},
            request_id=getattr(request.state, "request_id", ""),
        )
    except Exception as e:
        logger.error("end_session failed: %s", e)
        raise HTTPException(status_code=500, detail="服务内部错误")
