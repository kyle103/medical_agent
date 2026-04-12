from __future__ import annotations

from pydantic import BaseModel, Field


class UserRegisterRequest(BaseModel):
    phone: str = Field(..., min_length=6, max_length=32, description="手机号/登录名")
    password: str = Field(..., min_length=6, max_length=128, description="密码")
    user_nickname: str | None = Field(default=None, max_length=32)


class UserLoginRequest(BaseModel):
    phone: str = Field(..., min_length=6, max_length=32, description="手机号/登录名")
    password: str = Field(..., min_length=6, max_length=128, description="密码")


class UserAuthResponse(BaseModel):
    user_id: str
    access_token: str
    expires_in: int
