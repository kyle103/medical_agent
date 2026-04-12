from __future__ import annotations

from datetime import datetime, timedelta

from jose import JWTError, jwt

from app.common.exceptions import UserAuthException
from app.config.settings import settings


def create_access_token(*, user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": user_id, "exp": expire}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")


def parse_bearer_token(authorization_header: str) -> str:
    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise UserAuthException("未授权")
    token = authorization_header.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise UserAuthException("用户身份非法，请重新获取用户标识")
        return str(user_id)
    except JWTError as e:
        raise UserAuthException("用户身份非法，请重新获取用户标识") from e
