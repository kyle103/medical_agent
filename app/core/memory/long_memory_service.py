from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass

from app.common.exceptions import ServiceUnavailableException
from app.core.llm.llm_service import LLMService
from app.db.vector_store import init_chroma_collection


DEFAULT_COLLECTION = "user_long_memory"
MIN_CONFIDENCE = 0.7  # 最低置信度阈值


@dataclass
class LongMemoryItem:
    """长期记忆条目。

    为后续“进阶版”（结构化 facts 表、可更新/可删除、多类型召回）预留字段。
    """

    memory_id: str
    text: str
    memory_type: str = "fact"  # fact/preference/profile/summary
    source: str = "chat"
    session_id: str | None = None
    created_at: int = 0
    confidence: float = 1.0  # 记忆置信度，0-1之间

    def to_metadata(self, *, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "source": self.source,
            "session_id": self.session_id or "",
            "created_at": int(self.created_at or 0),
            "confidence": float(self.confidence or 0),
        }


class LongMemoryService:
    """基于向量库（Chroma file persist）的长期记忆服务。

    设计目标：
    - 与 DB 全量对话分离：这里只存“可复用事实/偏好/画像摘要”，用于语义召回。
    - 隔离方式：统一使用 user_id（来自 JWT sub），避免使用手机号。
    - 易扩展：后续可替换为 facts 表 + 向量库仅存 embedding/doc_id。
    """

    def __init__(self, *, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        self.llm_service = LLMService()

    def _collection(self):
        return init_chroma_collection(collection_name=self.collection_name)

    def is_enabled(self) -> bool:
        try:
            # init_chroma_collection 内部会校验 VECTOR_STORE_TYPE
            self._collection()
            return True
        except ServiceUnavailableException:
            return False
        except Exception:
            # 初始化失败也视为不可用（不阻塞主流程）
            return False

    async def extract_candidates(self, *, user_input: str) -> list[LongMemoryItem]:
        """从用户输入中抽取可写入长期记忆的候选条目（LLM版）。

        策略：
        - 使用LLM提取记忆，提高准确性和覆盖面
        - 增加置信度评估，过滤低质量记忆
        - 控制提取数量，避免过多噪声
        """

        text = (user_input or "").strip()
        if not text:
            return []

        now = int(time.time())
        
        # 尝试使用LLM提取记忆
        try:
            llm_items = await self._extract_with_llm(user_input=text)
            # 过滤低置信度记忆
            filtered_items = [item for item in llm_items if item.confidence >= MIN_CONFIDENCE]
            # 控制数量
            return filtered_items[:3]
        except Exception:
            # LLM提取失败时，回退到规则提取
            return self._extract_with_rules(user_input=text)

    async def _extract_with_llm(self, *, user_input: str) -> list[LongMemoryItem]:
        """使用LLM提取记忆。"""
        
        system_prompt = """
你是一个医疗记忆提取助手，负责从用户的对话中提取有价值的医疗相关信息，用于长期记忆存储。

请从以下用户输入中提取可能需要长期记忆的信息，包括但不限于：
1. 过敏史
2. 用药情况
3. 健康偏好
4. 病史/慢病信息
5. 其他重要的健康相关事实

对于每条提取的信息，请提供：
- 记忆内容（简洁明了的陈述句）
- 记忆类型（fact/preference/profile）
- 置信度（0-1之间，反映该信息的可靠性）

输出格式为JSON数组，每个元素包含text、memory_type和confidence字段。
如果没有需要提取的信息，请返回空数组。
"""

        prompt = f"用户输入：{user_input}"
        
        response = await self.llm_service.chat_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            timeout_s=10.0,
            max_tokens=500
        )
        
        # 解析LLM输出
        import json
        try:
            items_data = json.loads(response)
            items = []
            # 确保items_data是列表
            if isinstance(items_data, list):
                for data in items_data:
                    if isinstance(data, dict) and 'text' in data:
                        items.append(
                            LongMemoryItem(
                                memory_id=uuid.uuid4().hex,
                                text=data['text'],
                                memory_type=data.get('memory_type', 'fact'),
                                confidence=data.get('confidence', 1.0),
                                created_at=int(time.time())
                            )
                        )
            return items
        except Exception:
            # 解析失败时返回空列表
            return []

    def _extract_with_rules(self, *, user_input: str) -> list[LongMemoryItem]:
        """使用规则提取记忆（回退方案）。"""
        
        text = (user_input or "").strip()
        if not text:
            return []

        lowered = text
        items: list[LongMemoryItem] = []
        now = int(time.time())

        patterns: list[tuple[str, str]] = [
            (r"(我|本人).{0,4}(对|存在).{0,4}(过敏)", "fact"),
            (r"(我|本人).{0,6}(不吃|不喝|不喜欢|讨厌|不能吃|不能喝)", "preference"),
            (r"(我|本人).{0,6}(既往史|病史|慢病|高血压|糖尿病|冠心病|哮喘)", "profile"),
            (r"(我|本人).{0,8}(正在|目前|长期).{0,6}(用药|吃|服用)", "fact"),
        ]

        for pat, mtype in patterns:
            if re.search(pat, lowered):
                items.append(
                    LongMemoryItem(
                        memory_id=uuid.uuid4().hex,
                        text=text,
                        memory_type=mtype,
                        confidence=0.8,  # 规则提取的默认置信度
                        created_at=now,
                    )
                )
                break

        # 控制数量
        return items[:2]

    def add_items(self, *, user_id: str, session_id: str, items: list[LongMemoryItem]) -> int:
        if not user_id:
            return 0
        if not items:
            return 0

        # 去重处理
        try:
            unique_items = self._deduplicate_items(user_id=user_id, items=items)
            if not unique_items:
                return 0

            col = self._collection()

            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict] = []

            for it in unique_items:
                it.session_id = session_id
                ids.append(it.memory_id)
                docs.append(it.text)
                metas.append(it.to_metadata(user_id=user_id))

            col.add(ids=ids, documents=docs, metadatas=metas)
            return len(ids) if isinstance(ids, list) else 0
        except Exception:
            # 出错时默认返回0，避免阻塞主流程
            return 0

    def _deduplicate_items(self, *, user_id: str, items: list[LongMemoryItem]) -> list[LongMemoryItem]:
        """去重处理，避免存储重复记忆。"""
        
        try:
            if not items:
                return []

            # 首先对输入的items进行去重
            seen_texts = set()
            unique_input_items = []
            for item in items:
                text = item.text.strip()
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_input_items.append(item)

            # 然后与已有记忆进行去重
            final_unique_items = []
            for item in unique_input_items:
                # 检查是否与已有记忆重复
                if not self._is_duplicate(user_id=user_id, text=item.text):
                    final_unique_items.append(item)

            return final_unique_items
        except Exception:
            # 出错时默认返回空列表，避免阻塞主流程
            return []

    def _is_duplicate(self, *, user_id: str, text: str) -> bool:
        """检查文本是否与已有记忆重复。"""
        
        try:
            col = self._collection()
            # 搜索相似记忆
            res = col.query(
                query_texts=[text],
                n_results=3,
                where={"user_id": user_id}
            )
            
            # 确保docs是列表
            docs = res.get("documents")
            if not isinstance(docs, list) or not docs:
                return False
            
            # 获取第一个结果集
            first_docs = docs[0] if isinstance(docs[0], list) else []
            if not isinstance(first_docs, list):
                return False
            
            # 简单文本相似度判断，实际项目中可使用更复杂的相似度算法
            for doc in first_docs:
                if isinstance(doc, str):
                    if text == doc:
                        return True
                    # 检查文本是否高度相似（例如，一个是另一个的子集）
                    if text in doc or doc in text:
                        return True
        except Exception:
            # 出错时默认返回False，避免阻塞主流程
            pass
        
        return False

    def recall(self, *, user_id: str, query: str, top_k: int = 3) -> list[LongMemoryItem]:
        if not user_id:
            return []
        q = (query or "").strip()
        if not q:
            return []

        col = self._collection()
        res = col.query(query_texts=[q], n_results=max(1, int(top_k)), where={"user_id": user_id})

        # 处理ids
        ids = res.get("ids")
        if not isinstance(ids, list) or not ids:
            return []
        first_ids = ids[0] if isinstance(ids[0], list) else []
        if not isinstance(first_ids, list):
            return []
        
        # 处理docs
        docs = res.get("documents")
        if not isinstance(docs, list) or not docs:
            return []
        first_docs = docs[0] if isinstance(docs[0], list) else []
        if not isinstance(first_docs, list):
            return []
        
        # 处理metas
        metas = res.get("metadatas")
        if not isinstance(metas, list) or not metas:
            metas = [[]]
        first_metas = metas[0] if isinstance(metas[0], list) else []
        if not isinstance(first_metas, list):
            first_metas = []

        out: list[LongMemoryItem] = []
        for i in range(min(len(first_ids), len(first_docs))):
            md = first_metas[i] if i < len(first_metas) and isinstance(first_metas[i], dict) else {}
            out.append(
                LongMemoryItem(
                    memory_id=str(first_ids[i]),
                    text=str(first_docs[i] or ""),
                    memory_type=str(md.get("memory_type") or "fact"),
                    source=str(md.get("source") or "chat"),
                    session_id=str(md.get("session_id") or "") or None,
                    created_at=int(md.get("created_at") or 0),
                    confidence=float(md.get("confidence") or 1.0),
                )
            )

        return out
