from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    session_id: str = Field(default="", max_length=64)
    user_input: str = Field(min_length=1, max_length=4000)
    stream: bool = False
    enable_archive_link: bool = True


class IntentAnalysisInfo(BaseModel):
    intent_type: str = ""
    confidence: float = 0.0
    reason: str = ""
    target_name: str = ""


class ChatCompletionResponse(BaseModel):
    session_id: str
    user_input: str
    assistant_output: str
    intent: str
    create_time: str
    intent_analysis: Optional[IntentAnalysisInfo] = None
    target_agent: str = ""
    needs_confirmation: bool = False
    conversation_turns: int = 0
