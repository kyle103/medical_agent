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
        - 必要时进行结构化处理，提高记忆质量
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
1. 过敏史（如：对青霉素过敏）
2. 用药情况（如：今天吃了布洛芬）
3. 健康偏好（如：不喜欢吃辣）
4. 病史/慢病信息（如：有高血压病史）
5. 其他重要的健康相关事实

提取规则：
- 每条提取的信息必须是一个完整的事实陈述
- 信息要简洁明了，去除冗余内容
- 优先提取具体的医疗相关信息，如药物名称、症状、病史等
- 对于药物相关信息，尽量包含药物名称和使用情况

对于每条提取的信息，请提供：
- 记忆内容（简洁明了的陈述句，如：用户对青霉素过敏）
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
                        # 结构化处理：确保文本格式统一，以"用户"开头
                        text = data['text']
                        if not text.startswith('用户'):
                            text = f"用户{text}"
                        items.append(
                            LongMemoryItem(
                                memory_id=uuid.uuid4().hex,
                                text=text,
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

        # 增强的规则模式，覆盖更多医疗相关场景
        patterns: list[tuple[str, str]] = [
            # 过敏史
            (r"(我|本人).{0,4}(对|存在).{0,10}(过敏|过敏原)", "fact"),
            # 用药情况
            (r"(我|本人).{0,8}(吃|服用|用了|用).{0,10}(药|药物|胶囊|片|丸)", "fact"),
            (r"(我|本人).{0,8}(今天|昨天|最近).{0,10}(吃|服用|用了|用).{0,10}(药|药物)", "fact"),
            # 健康偏好
            (r"(我|本人).{0,6}(不吃|不喝|不喜欢|讨厌|不能吃|不能喝|避免)", "preference"),
            # 病史/慢病信息
            (r"(我|本人).{0,6}(有|患|得了).{0,10}(病|症|高血压|糖尿病|冠心病|哮喘|胃炎|肝炎)", "profile"),
            (r"(我|本人).{0,6}(既往史|病史|慢病|长期病)", "profile"),
            # 症状信息
            (r"(我|本人).{0,6}(感到|感觉|出现|有).{0,10}(头痛|头晕|发烧|咳嗽|腹痛|恶心|呕吐)", "fact"),
        ]

        for pat, mtype in patterns:
            if re.search(pat, lowered):
                # 结构化处理：提取关键信息，生成更规范的记忆文本
                structured_text = self._structure_memory_text(text, mtype)
                items.append(
                    LongMemoryItem(
                        memory_id=uuid.uuid4().hex,
                        text=structured_text,
                        memory_type=mtype,
                        confidence=0.8,  # 规则提取的默认置信度
                        created_at=now,
                    )
                )
                # 不break，允许提取多个记忆项

        # 控制数量
        return items[:3]

    def _structure_memory_text(self, text: str, memory_type: str) -> str:
        """对提取的记忆文本进行结构化处理，生成更规范的记忆内容。"""
        
        # 替换第一人称
        text = text.replace("我", "用户").replace("本人", "用户")
        
        # 根据记忆类型进行不同的结构化处理
        if memory_type == "fact":
            # 确保事实类记忆是完整的陈述句
            if not text.endswith('。'):
                text = f"{text}。"
        elif memory_type == "preference":
            # 确保偏好类记忆清晰表达
            if "不喜欢" in text or "讨厌" in text:
                text = text.replace("不喜欢", "不喜欢")
                text = text.replace("讨厌", "不喜欢")
        elif memory_type == "profile":
            # 确保病史类记忆准确表达
            if "有" in text and "病" in text:
                pass  # 保持原样
        
        return text

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
                n_results=5,
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
            
            # 改进的文本相似度判断
            for doc in first_docs:
                if isinstance(doc, str):
                    # 完全相同
                    if text == doc:
                        return True
                    # 检查文本是否高度相似（例如，一个是另一个的子集）
                    if text in doc or doc in text:
                        return True
                    # 检查是否包含相同的关键信息（如药物名称）
                    if self._has_same_key_information(text, doc):
                        return True
        except Exception:
            # 出错时默认返回False，避免阻塞主流程
            pass
        
        return False
    
    def _has_same_key_information(self, text1: str, text2: str) -> bool:
        """检查两个文本是否包含相同的关键信息（如药物名称）。"""
        
        # 药物关键词列表
        drug_names = ["布洛芬", "阿司匹林", "青霉素", "抗生素", "降压药", "降糖药", "感冒药", "止痛药"]
        
        # 检查两个文本是否包含相同的药物名称
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for drug in drug_names:
            if drug in text1_lower and drug in text2_lower:
                return True
        
        # 检查是否包含相同的健康状况关键词
        health_keywords = ["高血压", "糖尿病", "冠心病", "哮喘", "过敏", "头痛", "头晕", "发烧"]
        for keyword in health_keywords:
            if keyword in text1 and keyword in text2:
                return True
        
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
