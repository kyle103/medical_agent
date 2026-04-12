import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.time()
        response: Response = await call_next(request)
        cost_ms = int((time.time() - start) * 1000)

        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time-Ms"] = str(cost_ms)
        return response
