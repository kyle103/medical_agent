from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    code: int = 200
    msg: str = "success"
    data: T | dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""
