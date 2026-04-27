from __future__ import annotations

import json
from typing import Any


class StateAccessor:
    SHARED_READONLY_KEYS = frozenset({
        "user_id", "session_id", "user_input", "stream", "enable_archive_link",
        "shared_facts", "memory_summary", "history_text", "long_memory_text",
        "long_memory_items", "retrieved_knowledge", "execution_plan",
    })

    def __init__(self, state: dict, agent_name: str):
        self._state = state
        self._agent_name = agent_name

    @property
    def agent_name(self) -> str:
        return self._agent_name

    def read_shared(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def read_private(self, key: str, default: Any = None) -> Any:
        scratchpad = self._state.get("private_scratchpads", {})
        agent_data = scratchpad.get(self._agent_name, {})
        return agent_data.get(key, default)

    def write_private(self, key: str, value: Any) -> None:
        scratchpad = self._state.setdefault("private_scratchpads", {})
        scratchpad.setdefault(self._agent_name, {})[key] = value

    def clear_private(self) -> None:
        scratchpad = self._state.setdefault("private_scratchpads", {})
        scratchpad[self._agent_name] = {}

    def propose_shared_update(self, key: str, value: Any, priority: int = 0) -> None:
        updates = self._state.setdefault("proposed_updates", [])
        updates.append({
            "scope": "shared",
            "key": key,
            "value": value,
            "source": self._agent_name,
            "priority": priority,
        })

    def read_visible_state(self) -> dict:
        from app.core.agent.registry import build_agent_registry
        agents = build_agent_registry()
        agent = agents.get(self._agent_name)
        if not agent:
            return {}
        card = agent.get_agent_card()
        visible_keys = set(card.visible_state_keys or [])
        result = {}
        for k in visible_keys:
            if k in self._state:
                val = self._state[k]
                if isinstance(val, (dict, list)):
                    result[k] = val
                else:
                    result[k] = val
        return result

    def build_memory_context(self) -> str:
        visible = self.read_visible_state()
        parts = []

        memory_summary = visible.get("memory_summary")
        if isinstance(memory_summary, str) and memory_summary.strip():
            parts.append(f"会话摘要：\n{memory_summary}")

        shared_facts = visible.get("shared_facts")
        if isinstance(shared_facts, dict) and shared_facts:
            parts.append(f"共享事实：\n{json.dumps(shared_facts, ensure_ascii=False)}")

        retrieved_knowledge = visible.get("retrieved_knowledge")
        if isinstance(retrieved_knowledge, dict) and retrieved_knowledge:
            parts.append(f"医疗知识检索结果：\n{json.dumps(retrieved_knowledge, ensure_ascii=False)}")

        long_memory_items = visible.get("long_memory_items")
        if isinstance(long_memory_items, list) and long_memory_items:
            safe_items = [f"- ({it.get('memory_type', 'fact')}) {it.get('text', '')}" for it in long_memory_items[:4]]
            parts.append("召回长期记忆（可能不准确，仅供参考）：\n" + "\n".join(safe_items))

        history_text = visible.get("history_text")
        if isinstance(history_text, str) and history_text.strip():
            parts.append(f"近期对话历史：\n{history_text}")

        return "\n\n".join(parts)
