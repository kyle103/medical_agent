"""会话管理器 - 自动管理多轮对话历史"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json


class SessionManager:
    """会话管理器，用于自动管理多轮对话历史"""
    
    def __init__(self):
        # 内存中的会话存储（生产环境应该使用数据库）
        self.sessions: Dict[str, Dict[str, Any]] = {}
        # 会话过期时间（小时）
        self.session_timeout_hours = 24
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "conversation_history": [],
            "long_term_memory": {}
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        session = self.sessions.get(session_id)
        
        if session:
            # 检查会话是否过期
            if self._is_session_expired(session):
                self.delete_session(session_id)
                return None
            
            # 更新最后访问时间
            session["last_accessed"] = datetime.now()
            
        return session
    
    def add_message_to_session(self, session_id: str, role: str, content: str) -> bool:
        """添加消息到会话历史"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        session["conversation_history"].append(message)
        
        # 限制历史记录长度（保留最近50条消息）
        if len(session["conversation_history"]) > 50:
            session["conversation_history"] = session["conversation_history"][-50:]
        
        return True
    
    def get_conversation_history(self, session_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
        """获取会话历史（简化格式）"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        # 返回简化格式的历史记录
        history = []
        for msg in session["conversation_history"][-max_messages:]:
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return history
    
    def update_long_term_memory(self, session_id: str, key: str, value: Any) -> bool:
        """更新长期记忆"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session["long_term_memory"][key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
        
        return True
    
    def get_long_term_memory(self, session_id: str, key: str) -> Optional[Any]:
        """获取长期记忆"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        memory = session["long_term_memory"].get(key)
        return memory["value"] if memory else None
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
    
    def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """检查会话是否过期"""
        last_accessed = session["last_accessed"]
        expiration_time = last_accessed + timedelta(hours=self.session_timeout_hours)
        
        return datetime.now() > expiration_time


# 全局会话管理器实例
session_manager = SessionManager()