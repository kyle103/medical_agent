from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
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
        # 处理 OPTIONS 请求，跳过鉴权
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path.endswith("/api/v1/user/register") or path.endswith("/api/v1/user/login"):
            return await call_next(request)

        if path.startswith("/api/v1/"):
            auth = request.headers.get("Authorization", "")
            try:
                user_id = parse_bearer_token(auth)
                request.state.user_id = user_id
            except Exception as e:  # noqa: BLE001
                # ⚠️ 这里**必须 return 一个响应，不能 `raise HTTPException`**。
                #
                # `BaseHTTPMiddleware` 挂在 `ExceptionMiddleware` 的**外层**，
                # 而把 `HTTPException` 转成响应的正是后者。所以在 dispatch 里抛
                # `HTTPException(401)` 不会被翻译成 401，而是冒泡到
                # `ServerErrorMiddleware` → 客户端拿到 **500 且 body 为空**
                # （2026-09-13 实测：curl 未带 token → HTTP 500，响应体 0 字节）。
                #
                # 后果不只是"状态码不好看"：前端靠 `res.status === 401` 判断 token 过期
                # 并登出（`app.js` 的 `sendMessage` / `handleLabImageFile`），
                # 拿到 500 就会走进"系统暂时无法响应"的兜底话术 —— 用户被卡在
                # 一个永远登不出的状态里。
                #
                # 响应体同时给 `detail` 与 `APIResponse` 信封两种形状：
                # 前者是 FastAPI 默认错误的约定（既有前端在读），后者与本项目其余接口一致。
                return JSONResponse(
                    status_code=401,
                    content={
                        "code": 401,
                        "msg": "未授权",
                        "detail": "未授权",
                        "data": {},
                        "request_id": getattr(request.state, "request_id", ""),
                    },
                )

        return await call_next(request)
