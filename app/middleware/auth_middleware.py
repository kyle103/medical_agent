from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.common.auth import parse_bearer_token


class AuthMiddleware(BaseHTTPMiddleware):
    """将 user_id 注入 request.state。

    规则：
    - /api/v1/user/register 不需要鉴权
    - /api/v1/user/login 不需要鉴权
    - 其他 /api/v1/** 需要 Authorization: Bearer <token>
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.endswith("/api/v1/user/register") or path.endswith("/api/v1/user/login"):
            return await call_next(request)

        if path.startswith("/api/v1/"):
            auth = request.headers.get("Authorization", "")
            user_id = parse_bearer_token(auth)
            request.state.user_id = user_id

        return await call_next(request)
