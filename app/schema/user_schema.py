from __future__ import annotations

from pydantic import BaseModel, Field


class UserRegisterRequest(BaseModel):
    user_nickname: str | None = Field(default=None, max_length=32)


class UserRegisterResponse(BaseModel):
    user_id: str
    access_token: str
    expires_in: int
