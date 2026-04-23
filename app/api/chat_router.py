from fastapi import APIRouter, Request, HTTPException
import uuid

from app.common.exceptions import UserAuthException
from app.common.logger import get_logger
from app.core.agent.workflow import MedicalAgent
from app.schema.base import APIResponse
from app.schema.chat_schema import ChatCompletionRequest, ChatCompletionResponse

import time

router = APIRouter()
logger = get_logger(__name__)


@router.post("/completion", response_model=APIResponse[ChatCompletionResponse])
async def completion(req: ChatCompletionRequest, request: Request):
    user_id = getattr(request.state, "user_id", None)
    
    # 如果session_id为空，优先复用当前用户最近活跃会话，减少上下文丢失
    session_id = req.session_id
    if not session_id or session_id.strip() == "":
        # 生产链路以DB会话为准，避免内存SessionManager造成双轨上下文
        session_id = str(uuid.uuid4())
        logger.info("Generated new session for user %s: %s", user_id, session_id)

    try:
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
    except UserAuthException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误")
