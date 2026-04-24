import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.common.log_context import clear_request_context, set_request_context
from app.common.logger import get_logger


logger = get_logger(__name__)


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        trace_id = request.headers.get("X-Trace-Id") or None
        request.state.request_id = request_id

        ctx = set_request_context(trace_id=trace_id, request_id=request_id)
        start = time.time()
        logger.info("request start method=%s path=%s", request.method, request.url.path)

        response: Response | None = None
        try:
            response = await call_next(request)
        finally:
            cost_ms = int((time.time() - start) * 1000)
            logger.info("request end status=%s cost_ms=%s", getattr(response, "status_code", "-"), cost_ms)
            clear_request_context()

        if response is None:
            return Response(status_code=500)

        response.headers["X-Request-Id"] = ctx.request_id
        response.headers["X-Trace-Id"] = ctx.trace_id
        response.headers["X-Response-Time-Ms"] = str(cost_ms)
        return response
