"""
专业Agent基类模块
提供所有专业Agent的公共基类和标准化接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import json

from app.core.llm.llm_service import LLMService


class BaseAgent(ABC):
    """专业Agent基类"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm_service = LLMService()
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent核心处理逻辑"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """获取Agent专用系统提示词"""
        pass
    
    async def _call_llm(self, prompt: str, system_prompt: str, state: Dict[str, Any] = None) -> str:
        """统一LLM调用方法，支持记忆注入"""
        try:
            # 如果有state参数，检查是否需要注入记忆
            final_prompt = prompt
            if state:
                # 检查是否需要上下文记忆
                user_input = state.get("user_input", "")
                history_text = state.get("history_text", "")
                long_memory_text = state.get("long_memory_text", "")
                
                # 判断是否需要记忆
                need_memory = self._need_memory_injection(user_input, history_text, long_memory_text)
                
                if need_memory:
                    # 构建包含记忆的prompt
                    memory_context = self._build_memory_context(history_text, long_memory_text)
                    final_prompt = f"""{memory_context}

用户当前查询：{user_input}

{prompt}"""
            
            response = await self.llm_service.chat_completion(
                prompt=final_prompt,
                system_prompt=system_prompt
            )
            return response
        except Exception as e:
            return f"抱歉，处理请求时出现错误：{str(e)}"
    
    def _need_memory_injection(self, user_input: str, history_text: str, long_memory_text: str) -> bool:
        """判断是否需要注入记忆"""
        # 如果有长期记忆，总是注入
        if long_memory_text and len(long_memory_text.strip()) > 0:
            return True
        
        # 检查用户输入是否需要上下文
        if not user_input:
            return False
            
        # 短追问/承接
        if len(user_input) <= 12:
            # 排除简单的陈述性语句
            simple_statements = ["我有", "我是", "我在", "我要", "我想"]
            if not any(statement in user_input for statement in simple_statements):
                return True

        # 指代/省略
        pronouns = ["那个", "它", "这", "这样", "上面", "刚才", "之前", "继续", "然后", "还要", "还用", "还需要"]
        if any(p in user_input for p in pronouns):
            return True

        # 回忆/核对类
        recall = ["总结", "回顾", "复盘", "你还记得", "你记得", "还记得", "之前", "刚才", "上次", "昨天", "今天", "最近", "回顾", "总结"]
        if any(k in user_input for k in recall):
            return True

        return False
    
    def _build_memory_context(self, history_text: str, long_memory_text: str) -> str:
        """构建记忆上下文"""
        memory_parts = []
        
        if history_text and len(history_text.strip()) > 0:
            memory_parts.append(f"近期对话历史：\n{history_text}")
        
        if long_memory_text and len(long_memory_text.strip()) > 0:
            memory_parts.append(f"相关长期记忆：\n{long_memory_text}")
        
        if memory_parts:
            return "\n\n".join(memory_parts)
        
        return ""
    
    def _check_compliance(self, content: str) -> tuple[bool, str]:
        """合规校验（已禁用）"""
        return True, ""
    
    def _add_disclaimer(self, content: str) -> str:
        """添加免责声明（已禁用）"""
        return content
    
    def _log_agent_call(self, user_id: str, agent_name: str, input_text: str, output_text: str):
        """记录Agent调用日志"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Agent调用 - 用户:{user_id} Agent:{agent_name} 输入长度:{len(input_text)} 输出长度:{len(output_text)}")