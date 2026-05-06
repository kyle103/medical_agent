from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.common.logger import get_logger
from app.core.agent.agent_card import AgentCard
from app.core.agent.state_accessor import StateAccessor
from app.core.compliance.compliance_service import ComplianceService
from app.core.llm.llm_service import LLMService

logger = get_logger(__name__)


class BaseAgent(ABC):

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm_service = LLMService()
        self._compliance = ComplianceService()

    @abstractmethod
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        pass

    def _create_accessor(self, state: dict[str, Any]) -> StateAccessor:
        return StateAccessor(state, self.agent_name)

    async def _call_llm(self, prompt: str, system_prompt: str, state: dict[str, Any] = None) -> str:
        try:
            final_prompt = prompt
            if state:
                accessor = self._create_accessor(state)
                memory_context = accessor.build_memory_context()
                user_input = state.get("user_input", "")

                if memory_context:
                    final_prompt = f"{memory_context}\n\n用户当前查询：{user_input}\n\n{prompt}"

            response = await self.llm_service.chat_completion(
                prompt=final_prompt,
                system_prompt=system_prompt,
            )
            return response
        except Exception as e:
            return f"抱歉，处理请求时出现错误：{str(e)}"

    def _log_agent_call(self, user_id: str, agent_name: str, input_text: str, output_text: str):
        logger.info("Agent调用 - 用户:%s Agent:%s 输入长度:%s 输出长度:%s", user_id, agent_name, len(input_text), len(output_text))

    def _check_compliance(self, content: str) -> tuple[bool, str]:
        ok, msg = self._compliance.output_compliance_check(content)
        if not ok:
            logger.warning("合规拦截 agent=%s msg=%s", self.agent_name, msg)
        return ok, msg

    def _add_disclaimer(self, content: str) -> str:
        return self._compliance.add_disclaimer(content)
