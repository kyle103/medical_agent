"""流式事件契约的唯一构造源。

背景（Step 3 / astream 合并）：
    改造前事件是在 `workflow.py::run_stream` 里手写 `json.dumps(...)` 拼出来的，
    而 `nodes.py::llm_generate` 完全不知道事件的存在。合并后 token 要从图节点内部
    推给前端，于是同一个 `intent` 事件会出现两个产出点，字段极易漂移。

    本模块把「前端能看到什么」收成一个地方：任何字段变更只改这里。
    消费方是 `frontend/app.js::handleSSEEvent`，那边识别 5 类事件。

契约（**改字段必须同步 app.js**）：

    progress  {"type":"progress","node":<节点名>}
    intent    {"type":"intent","intent":...,"intent_analysis":{...},"target_agent":...}
              —— 注意有两处产出点，字段**刻意不同**（见 intent_payload 的 full 参数）
    chunk     {"type":"chunk","content":<增量文本>}
    error     {"type":"error","content":<错误文本>}
    done      {"type":"done","session_id":...,"intent":...,"needs_confirmation":...,
               "conversation_turns":...,"cache":...}

设计说明：
    - 产出 payload dict 与序列化分离：节点内用 `writer(xxx_payload(...))` 推流，
      传输层用 `serialize()` 统一转成 SSE 行，两边共用同一份字段定义。
    - 保留改造前的**非对称 intent**：`intent_node` 处发完整版（带 intent_analysis /
      target_agent），`llm_generate` 内那个发精简版（只有 intent）。这是改造前的既有
      行为，本次不顺手改，以免前端契约在同一批次里产生非必要变更。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

__all__ = [
    "progress_payload",
    "intent_payload",
    "chunk_payload",
    "error_payload",
    "done_payload",
    "serialize",
    "iter_sentence_chunks",
    "PROGRESS_EXCLUDED_NODES",
]

# 改造前按句切分模拟流式用的分隔符（保留捕获组，切出的分隔符本身也要发出去）
_SENTENCE_SPLIT_RE = re.compile(r"([。！？\n])")

#: 这些节点完成时**不发** progress 事件。
#: - `err`：错误路径单独发 error 事件，再发 progress 会让前端状态错乱
#: - `mem`：记忆落库是收尾副作用，此时回答已经吐完，发 progress 会把前端气泡
#:   从"回答正文"覆盖成"正在…"（app.js 的 progress 分支是直接 innerHTML 覆写）
PROGRESS_EXCLUDED_NODES = frozenset({"err", "mem"})


def serialize(payload: dict[str, Any]) -> str:
    """统一序列化为 SSE 数据行（chat_router 会再加 `data: ` 前缀）。"""
    return json.dumps(payload, ensure_ascii=False) + "\n"


def progress_payload(node_name: str) -> dict[str, Any]:
    return {"type": "progress", "node": node_name}


def intent_payload(state: dict[str, Any], *, full: bool) -> dict[str, Any]:
    """构造 intent 事件。

    full=True  —— 完整版，带 intent_analysis / target_agent（在 `intent_node` 完成后发）
    full=False —— 精简版，只有 intent（在 `llm_generate` 内、开始吐 chunk 前发）

    full=False 是刻意保留的历史行为：改造前 `_intent_event()` 只发 intent 字段。
    """
    payload: dict[str, Any] = {
        "type": "intent",
        "intent": state.get("intent", "general"),
    }
    if not full:
        return payload

    ia = state.get("intent_analysis") or {}
    payload["intent_analysis"] = {
        "intent_type": ia.get("intent_type", state.get("intent_type", "")),
        "confidence": ia.get("confidence", state.get("intent_confidence", 0.0)),
        "reason": ia.get("reason", state.get("intent_reason", "")),
        "target_name": ia.get("target_name", state.get("target_agent", "")),
    }
    payload["target_agent"] = state.get("target_agent", "")
    return payload


def chunk_payload(content: str) -> dict[str, Any]:
    return {"type": "chunk", "content": content}


def error_payload(content: str) -> dict[str, Any]:
    return {"type": "error", "content": content}


def done_payload(
    *,
    session_id: str,
    state: dict[str, Any],
    cache_stats: Any,
) -> dict[str, Any]:
    history = state.get("history") or []
    return {
        "type": "done",
        "session_id": session_id,
        "intent": state.get("intent", "general"),
        "needs_confirmation": bool(state.get("needs_confirmation")),
        "conversation_turns": len(history) // 2 if history else 0,
        "cache": cache_stats,
    }


def iter_sentence_chunks(text: str) -> Iterator[str]:
    """按句切分文本，用于上游已产出完整文本时模拟流式。

    `final_response` 分支用：该分支的文本由前置节点整段产出，直接一次性吐出会
    让前端"啪"地出现一大段。逐句吐出视觉上更接近真流式。

    切分规则（保持改造前 workflow.py 的行为）：累积到 12 字或遇到句末标点就吐一次。
    """
    sentence_buf = ""
    for part in _SENTENCE_SPLIT_RE.split(text):
        sentence_buf += part
        if len(sentence_buf) >= 12 or part in ("。", "！", "？", "\n"):
            if sentence_buf.strip():
                yield sentence_buf
            sentence_buf = ""
    if sentence_buf.strip():
        yield sentence_buf
