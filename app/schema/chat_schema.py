from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    session_id: str = Field(default="", max_length=64)
    user_input: str = Field(min_length=1, max_length=4000)
    stream: bool = False
    enable_archive_link: bool = True


class ChatCompletionResponse(BaseModel):
    session_id: str
    user_input: str
    assistant_output: str
    intent: str
    create_time: str
