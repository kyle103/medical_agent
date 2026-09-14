from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class ChatCompletionRequest(BaseModel):
    session_id: str = Field(default="", max_length=64)
    # 允许空串：用户可能**只发一张化验单图片**、不打任何字（见下面的 validators）。
    # 非空内容长度仍受 max_length 约束。
    user_input: str = Field(default="", max_length=4000)
    stream: bool = False
    enable_archive_link: bool = True

    # 随消息一起上传的图片。用 base64-JSON 而不是 multipart：`python-multipart`
    # 未装在依赖里，引入它会连累整个应用启动。
    #
    # ⚠️ 这里的字符串只活到路由层。识别在路由层完成，进图的是识别出的文本
    # （`image_lab_text`），图片字节**绝不**进 state —— base64 有几 MB，
    # 进 tracked 字段就会被写进每一份 checkpointer 快照。
    image_base64: str = Field(default="", max_length=30_000_000)
    image_mime: str = Field(default="", max_length=64)

    @model_validator(mode="after")
    def _require_text_or_image(self) -> "ChatCompletionRequest":
        """文本非空 **或** 带图片，二者至少有一个。

        放宽 `user_input` 的 min_length 就必须在这里补上这条约束，否则一个空请求体
        也能走进图、跑完整轮 LLM 流水线，最后产出一段针对"空气"的回答。
        """
        if not self.user_input.strip() and not self.image_base64.strip():
            raise ValueError("user_input 与 image_base64 至少需要提供一个")
        return self


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
    # 写操作二次确认的选项（needs_confirmation 为真时非空）。SSE 路径由
    # stream_events.options_payload 单独发 options 事件，这里保证 stream=false 的
    # 返回体语义完整——否则调用方只看到"需要确认"却拿不到可选项。
    options: list[dict] = []
    conversation_turns: int = 0
