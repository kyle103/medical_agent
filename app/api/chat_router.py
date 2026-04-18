from fastapi import APIRouter, Request

from app.common.logger import get_logger
from app.core.agent.workflow import MedicalAgent
from app.core.session.session_manager import SessionManager
from app.schema.base import APIResponse
from app.schema.chat_schema import ChatCompletionRequest, ChatCompletionResponse

import time

router = APIRouter()
logger = get_logger(__name__)
session_manager = SessionManager()


@router.post("/completion", response_model=APIResponse[ChatCompletionResponse])
async def completion(req: ChatCompletionRequest, request: Request):
    user_id = getattr(request.state, "user_id", None)
    
    # 如果session_id为空，生成新的会话ID
    session_id = req.session_id
    if not session_id or session_id.strip() == "":
        session_id = session_manager.create_session(user_id=user_id)
        logger.info(f"Created new session for user {user_id}: {session_id}")

    t0 = time.perf_counter()
    agent = MedicalAgent()
    t1 = time.perf_counter()

    result = await agent.run(
        user_id=user_id,
        session_id=session_id,
        user_input=req.user_input,
        stream=req.stream,
        enable_archive_link=req.enable_archive_link,
    )
    t2 = time.perf_counter()

    logger.info(
        "chat_completion perf: build_agent_ms=%s run_ms=%s total_ms=%s intent=%s",
        int((t1 - t0) * 1000),
        int((t2 - t1) * 1000),
        int((t2 - t0) * 1000),
        result.get("intent"),
    )

    # 确保返回的结果中包含正确的session_id
    result["session_id"] = session_id
    
    return APIResponse(
        data=ChatCompletionResponse(**result),
        request_id=getattr(request.state, "request_id", ""),
    )
