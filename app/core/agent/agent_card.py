from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentCard:
    """描述 Agent 能力，供调度层发现和路由。"""

    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    visible_state_keys: list[str] = field(default_factory=list)
    priority: int = 1

