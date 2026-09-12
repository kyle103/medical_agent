from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
import json

from app.common.logger import get_logger
from app.core.utils.text_splitter import TextSplitter
from app.common.exceptions import ServiceUnavailableException
from app.core.llm.llm_service import LLMService
from app.core.prompts import Prompts
from app.core.llm.embedding_service import EmbeddingService
from app.db.database import get_sessionmaker
from app.db.chroma_store import (
    delete_long_memory,
    ensure_long_memory_collection,
    get_chroma_client,
    has_collection,
    insert_long_memory,
    parse_metadata,
    vector_search,
)
from app.core.session.agent_state_store import AgentStateStore
from app.core.tools.drug_record_tool import DrugRecordTool

logger = get_logger(__name__)


DEFAULT_COLLECTION = "user_long_memory"
MIN_CONFIDENCE = 0.7  # 最低置信度阈值


@dataclass
class LongMemoryItem:
    """长期记忆条目。

    source_chat_id: 来源对话消息 id，用于"每条消息只提取一次"的游标去重。
    """

    memory_id: str
    text: str
    memory_type: str = "fact"  # fact/preference/profile/summary/drug_event
    source: str = "chat"
    session_id: str | None = None
    source_chat_id: int | None = None
    created_at: int = 0
    confidence: float = 1.0  # 记忆置信度，0-1之间

    def to_metadata(self, *, user_id: str) -> dict:
        meta = {
            "user_id": user_id,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "source": self.source,
            "session_id": self.session_id or "",
            "created_at": int(self.created_at or 0),
            "confidence": float(self.confidence or 0),
        }
        if self.source_chat_id is not None:
            meta["source_chat_id"] = int(self.source_chat_id)
        return meta


class LongMemoryService:
    """基于云端向量库（Milvus）的长期记忆服务。

    设计目标：
    - 与 DB 全量对话分离：这里只存“可复用事实/偏好/画像摘要”，用于语义召回。
    - 隔离方式：统一使用 user_id（来自 JWT sub），避免使用手机号。
    - 易扩展：后续可替换为 facts 表 + 向量库仅存 embedding/doc_id。
    """

    def __init__(self, *, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        self.llm_service = LLMService()
        self.embedder = EmbeddingService()

    def _coll(self, user_id: str) -> str:
        """按用户物理隔离的长期记忆集合名（每用户一个集合，杜绝跨用户泄露）。"""
        return f"user_long_memory_{user_id}"

    def _escape_expr_value(self, text: str) -> str:
        return (text or "").replace("\\", "\\\\").replace('"', '\\"')

    def is_enabled(self) -> bool:
        try:
            # 使用本地 Chroma 作为长期记忆向量库
            _ = get_chroma_client()
            return True
        except ServiceUnavailableException:
            return False
        except Exception:
            # 初始化失败也视为不可用（不阻塞主流程）
            return False

    async def extract_candidates(self, *, user_input: str, source_chat_id: int | None = None) -> list[LongMemoryItem]:
        """从用户输入中抽取可写入长期记忆的候选条目（LLM版），包括用药事件。

        source_chat_id: 来源消息 id，供游标去重使用。
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
            items = filtered_items[:3]
        except Exception:
            # LLM提取失败时，回退到规则提取
            items = self._extract_with_rules(user_input=text)

        for it in items:
            it.source_chat_id = source_chat_id
        return items

    async def _extract_with_llm(self, *, user_input: str) -> list[LongMemoryItem]:
        """使用LLM提取记忆。"""
        
        system_prompt = Prompts.get_prompt("MEMORY_EXTRACTION")

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
        # 用药事件
        (r"(今天|昨天|刚才|现在|早上|中午|晚上|下午).{0,10}(吃|服用|用了|用).{0,10}([^，。！？\s]{1,30})", "drug_event"),
        (r"([^，。！？\s]{1,30}).{0,10}(片|粒|胶囊|支|瓶|袋|贴).{0,10}(今天|昨天|刚才|现在|早上|中午|晚上|下午)?", "drug_event"),
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
        
    def _split_text(self, text: str) -> list[str]:
        """文本切分
        使用基于标点符号的语义分割，结合固定长度限制和20%的重叠策略
        """
        return TextSplitter.split_text(
            text=text,
            chunk_size=500,  # 固定长度限制
            chunk_overlap=100,  # 20%的重叠策略 (500*0.2=100)
            separators=["\n\n", "\n", "。", "！", "？", "，", "、"]  # 中文标点符号作为分隔符
        )

    async def add_items(self, *, user_id: str, session_id: str, items: list[LongMemoryItem]) -> int:
        if not user_id:
            return 0
        if not items:
            return 0

        # 去重处理
        try:
            unique_items = await self._deduplicate_items(user_id=user_id, items=items)
            if not unique_items:
                return 0

            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict] = []

            for it in unique_items:
                it.session_id = session_id
                
                # 对长文本进行切分
                if len(it.text) > 500:
                    chunks = self._split_text(it.text)
                    for i, chunk in enumerate(chunks):
                        # 为每个chunk生成唯一ID
                        chunk_id = f"{it.memory_id}_{i}"
                        ids.append(chunk_id)
                        docs.append(chunk)
                        # 保留原始记忆项的元数据
                        meta = it.to_metadata(user_id=user_id)
                        # 添加chunk信息
                        meta["parent_id"] = it.memory_id
                        meta["chunk_index"] = i
                        metas.append(meta)
                else:
                    # 短文本直接存储
                    ids.append(it.memory_id)
                    docs.append(it.text)
                    metas.append(it.to_metadata(user_id=user_id))

            if not docs:
                return 0

            embeddings = await self.embedder.embed_documents(docs)
            if not embeddings:
                return 0

            coll = self._coll(user_id)
            ensure_long_memory_collection(
                collection_name=coll,
                dim=len(embeddings[0]),
            )

            json_metas = [json.dumps(m, ensure_ascii=False) for m in metas]
            user_ids = [user_id for _ in ids]
            insert_long_memory(
                collection_name=coll,
                ids=ids,
                user_ids=user_ids,
                documents=docs,
                metadatas=json_metas,
                embeddings=embeddings,
            )
            return len(ids) if isinstance(ids, list) else 0
        except Exception as e:
            # 出错时默认返回0，避免阻塞主流程
            import logging
            logging.error(f"添加记忆项失败: {e}")
            return 0

    async def _deduplicate_items(self, *, user_id: str, items: list[LongMemoryItem]) -> list[LongMemoryItem]:
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
                if not await self._is_duplicate(user_id=user_id, text=item.text):
                    final_unique_items.append(item)

            return final_unique_items
        except Exception as e:
            # 出错时默认返回空列表，避免阻塞主流程
            logger.error(f"记忆去重失败: {e}")
            return []

    async def _is_duplicate(self, *, user_id: str, text: str) -> bool:
        """检查文本是否与已有记忆重复。"""

        try:
            vectors = await self.embedder.embed_documents([text])
            if not vectors:
                return False

            coll = self._coll(user_id)
            if not has_collection(coll):
                return False
            expr_user = self._escape_expr_value(user_id)
            try:
                hits = vector_search(
                    collection_name=coll,
                    query_vectors=vectors,
                    limit=5,
                    output_fields=["document", "metadata", "user_id"],
                    where={"user_id": expr_user},
                )
            except ServiceUnavailableException:
                return False

            if not hits:
                return False
            
            # 改进的文本相似度判断
            for hit in hits:
                entity = hit.get("entity") or {}
                doc = entity.get("document")
                if isinstance(doc, str):
                    if text == doc:
                        return True
                    if text in doc or doc in text:
                        return True
                    if await self._has_same_key_information(text, doc):
                        return True
        except Exception as e:
            # 出错时默认返回False，避免阻塞主流程
            logger.error(f"检查重复记忆失败: {e}")
            pass
        
        return False
    
    async def _has_same_key_information(self, text1: str, text2: str) -> bool:
        """检查两个文本是否包含相同的关键信息（基于药品知识库识别）。"""
        
        from app.core.rag.drug_knowledge_service import DrugKnowledgeService
        
        # 从文本中提取可能的药品名称
        def extract_drug_names(text):
            import re
            patterns = [
                r'(?:吃了|服用了|用了|吃|服用|使用|用)([^，。！？\s]{1,30})',
                r'([^，。！？\s]{1,30})(?:片|粒|胶囊|支|瓶|袋|贴)',
                r'(?:药名|药品|药物)\s*[:：]\s*([^，。！？\s]{1,30})'
            ]
            
            candidate_names = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                candidate_names.extend(matches)
            
            return list(set(candidate_names))
        
        # 提取两个文本中的药品名称
        drugs1 = extract_drug_names(text1)
        drugs2 = extract_drug_names(text2)
        
        # 如果两个文本都包含药品名称，使用药品知识库进行匹配
        if drugs1 and drugs2:
            try:
                # 异步调用药品知识库服务
                svc = DrugKnowledgeService()
                matched_drugs1 = await svc.match_drugs(drugs1)
                matched_drugs2 = await svc.match_drugs(drugs2)
                
                # 检查是否有相同的匹配药品
                matched_names1 = set()
                matched_names2 = set()
                
                for result in matched_drugs1:
                    if result.get("match"):
                        matched_names1.add(result["match"]["drug_name"])
                
                for result in matched_drugs2:
                    if result.get("match"):
                        matched_names2.add(result["match"]["drug_name"])
                
                # 如果有相同的匹配药品名，返回True
                if matched_names1 & matched_names2:
                    return True
                    
            except Exception as e:
                # 如果药品知识库调用失败，回退到简单的关键词匹配
                logger.error(f"药品知识库调用失败: {e}")
                pass
        
        # 回退方案：检查是否包含相同的健康状况关键词
        health_keywords = ["高血压", "糖尿病", "冠心病", "哮喘", "过敏", "头痛", "头晕", "发烧"]
        for keyword in health_keywords:
            if keyword in text1 and keyword in text2:
                return True
        
        return False

    async def recall(self, *, user_id: str, query: str, top_k: int = 3) -> list[LongMemoryItem]:
        if not user_id:
            return []
        q = (query or "").strip()
        if not q:
            return []

        vectors = await self.embedder.embed_documents([q])
        if not vectors:
            return []

        coll = self._coll(user_id)
        if not has_collection(coll):
            return []
        expr_user = self._escape_expr_value(user_id)
        try:
            hits = vector_search(
                collection_name=coll,
                query_vectors=vectors,
                limit=max(1, int(top_k)),
                output_fields=["document", "metadata", "user_id"],
                where={"user_id": expr_user},
            )
        except ServiceUnavailableException:
            return []
        if not hits:
            return []

        out: list[LongMemoryItem] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            md = parse_metadata(entity.get("metadata"))
            out.append(
                LongMemoryItem(
                    memory_id=str(hit.get("id") or ""),
                    text=str(entity.get("document") or ""),
                    memory_type=str(md.get("memory_type") or "fact"),
                    source=str(md.get("source") or "chat"),
                    session_id=str(md.get("session_id") or "") or None,
                    source_chat_id=int(md["source_chat_id"]) if md.get("source_chat_id") is not None else None,
                    created_at=int(md.get("created_at") or 0),
                    confidence=float(md.get("confidence") or 1.0),
                )
            )

        return out

    async def detect_conflicts(self, *, user_id: str, items: list[LongMemoryItem]) -> list[dict]:
        """检测候选记忆与已有记忆之间的冲突。

        返回冲突列表，每个元素格式：
        {
            "candidate": LongMemoryItem,
            "conflict_with": LongMemoryItem,
            "conflict_type": "contradict" | "supersede" | "duplicate",
            "resolution": "skip" | "replace" | "merge"
        }
        """
        if not items or not user_id:
            return []

        conflicts = []
        coll = self._coll(user_id)
        if not has_collection(coll):
            return []
        for item in items:
            try:
                vectors = await self.embedder.embed_documents([item.text])
                if not vectors:
                    continue

                expr_user = self._escape_expr_value(user_id)
                try:
                    hits = vector_search(
                        collection_name=coll,
                        query_vectors=vectors,
                        limit=5,
                        output_fields=["document", "metadata", "user_id"],
                        where={"user_id": expr_user},
                    )
                except ServiceUnavailableException:
                    continue

                if not hits:
                    continue

                for hit in hits:
                    entity = hit.get("entity") or {}
                    existing_text = str(entity.get("document") or "")
                    md = parse_metadata(entity.get("metadata"))
                    existing_item = LongMemoryItem(
                        memory_id=str(hit.get("id") or ""),
                        text=existing_text,
                        memory_type=str(md.get("memory_type") or "fact"),
                        source=str(md.get("source") or "chat"),
                        session_id=str(md.get("session_id") or "") or None,
                        created_at=int(md.get("created_at") or 0),
                        confidence=float(md.get("confidence") or 1.0),
                    )

                    conflict_type = self._classify_conflict(item.text, existing_text)
                    if conflict_type:
                        resolution = self._resolve_conflict(conflict_type, item, existing_item)
                        conflicts.append({
                            "candidate": item,
                            "conflict_with": existing_item,
                            "conflict_type": conflict_type,
                            "resolution": resolution,
                        })
                        logger.info(
                            "long_memory conflict detected: type=%s resolution=%s candidate=%s existing=%s",
                            conflict_type, resolution, item.text[:50], existing_text[:50],
                        )
            except Exception as e:
                logger.error("detect_conflicts item failed: %s", e)

        return conflicts

    def _classify_conflict(self, new_text: str, existing_text: str) -> str | None:
        """判断新记忆与已有记忆的冲突类型。"""
        if new_text.strip() == existing_text.strip():
            return "duplicate"

        if new_text in existing_text or existing_text in new_text:
            return "supersede"

        contradict_patterns = [
            (r"对(.+?)过敏", r"对(.+?)不过敏|对(.+?)已脱敏"),
            (r"有(.+?)病史", r"没有(.+?)病史|(.+?)已治愈"),
            (r"正在服用(.+?)", r"已停用(.+?)|不再服用(.+?)"),
            (r"患有(.+?)", r"未患有(.+?)|排除(.+?)"),
        ]
        for pat_new, pat_existing in contradict_patterns:
            new_match = re.search(pat_new, new_text)
            if new_match:
                existing_match = re.search(pat_existing, existing_text)
                if existing_match:
                    return "contradict"

        return None

    def _resolve_conflict(self, conflict_type: str, new_item: LongMemoryItem, existing_item: LongMemoryItem) -> str:
        """根据冲突类型决定解决策略。"""
        if conflict_type == "duplicate":
            return "skip"
        if conflict_type == "supersede":
            if new_item.created_at >= existing_item.created_at:
                return "replace"
            return "skip"
        if conflict_type == "contradict":
            if new_item.created_at >= existing_item.created_at and new_item.confidence >= existing_item.confidence:
                return "replace"
            return "skip"
        return "skip"

    async def write_with_conflict_check(self, *, user_id: str, session_id: str, items: list[LongMemoryItem], source: str = "chat") -> dict:
        """带冲突检测的写入，返回写入结果摘要。

        分流：
        - drug_event 类型 → SQL（UserDrugRecord，结构化档案）
        - 其余（fact/profile/preference/summary）→ 向量库（带冲突检测）
        """
        if not items:
            return {"written": 0, "skipped": 0, "replaced": 0}

        # 1) drug_event 走 SQL，不进向量库
        drug_events = [it for it in items if it.memory_type == "drug_event"]
        others = [it for it in items if it.memory_type != "drug_event"]

        result = {"written": 0, "skipped": 0, "replaced": 0}
        if drug_events:
            sql_written = await self._write_drug_events_to_sql(user_id=user_id, items=drug_events)
            result["written"] += sql_written
            result["skipped"] += len(drug_events) - sql_written

        if not others:
            return result

        # 2) 其余类型走向量库（冲突检测）
        conflicts = await self.detect_conflicts(user_id=user_id, items=others)
        conflict_map = {}
        for c in conflicts:
            conflict_map[c["candidate"].memory_id] = c

        to_write = []
        skipped = 0
        replaced = 0

        for item in others:
            conflict = conflict_map.get(item.memory_id)
            if not conflict:
                to_write.append(item)
                continue

            resolution = conflict["resolution"]
            if resolution == "skip":
                skipped += 1
                logger.info("long_memory write skipped (conflict): %s", item.text[:50])
            elif resolution == "replace":
                replaced += 1
                to_write.append(item)
                logger.info("long_memory write replace (conflict): new=%s old=%s", item.text[:50], conflict["conflict_with"].text[:50])
            elif resolution == "merge":
                skipped += 1
                logger.info("long_memory write merge (conflict): %s", item.text[:50])

        for item in to_write:
            item.source = source

        written = await self.add_items(user_id=user_id, session_id=session_id, items=to_write)
        result["written"] += written
        result["skipped"] += skipped
        result["replaced"] += replaced

        logger.info(
            "long_memory write_with_conflict_check: source=%s total=%d written=%d skipped=%d replaced=%d",
            source, len(items), result["written"], result["skipped"], result["replaced"],
        )
        return result

    async def _write_drug_events_to_sql(self, *, user_id: str, items: list[LongMemoryItem]) -> int:
        """drug_event 写入 SQL（UserDrugRecord），用 add_record 的幂等去重防重复。"""
        written = 0
        tool = DrugRecordTool()
        for it in items:
            original = (it.text or "").replace("用户", "我")
            drug_name = await self._extract_drug_name(original)
            if not drug_name:
                continue
            time_text = self._extract_time_text(original)
            try:
                res = await tool.add_record(user_id=user_id, drug_name=drug_name, time_text=time_text)
                if res.get("created"):
                    written += 1
                    logger.info("drug_event -> SQL: user=%s drug=%s", user_id, drug_name)
                elif not res.get("ok"):
                    logger.warning("drug_event SQL write failed: %s", res.get("message"))
            except Exception as e:
                logger.error("drug_event SQL write error: %s", e)
        return written

    @staticmethod
    def _extract_time_text(text: str) -> str:
        """从用药事件文本粗提取时间描述（"今天"/"昨天下午"等），无则空串。"""
        for kw in ["今天晚上", "昨天晚上", "今天中午", "昨天下午", "昨天早上", "前天晚上",
                   "前天", "昨天", "今天", "晚上", "下午", "上午", "早上", "中午", "凌晨", "半夜"]:
            if kw in (text or ""):
                return kw
        return ""

    async def _extract_drug_name(self, text: str) -> str | None:
        """从用药事件文本提取药名。

        正则优先（确定性，避开"用户"里的"用"误匹配），LLM 兜底（处理非"吃了X"句式）。
        """
        m = re.search(r"(?:吃了|服用了|服用过|使用了|使用过|服用)([^，。！？\s：:，]{1,12})", text)
        if m:
            candidate = m.group(1).strip().strip("了")
            if 2 <= len(candidate) <= 24:
                return candidate
        try:
            from app.core.agent.llm_decision_service import LLMDecisionService
            name = await LLMDecisionService().extract_drug_name_from_event(text)
            if name:
                return name
        except Exception:
            pass
        return None

    async def get_cursor(self, *, user_id: str, session_id: str) -> int:
        from sqlalchemy import select
        from app.db.models import UserLongMemoryCursor
        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(UserLongMemoryCursor.chat_id)
                .where(
                    UserLongMemoryCursor.user_id == user_id,
                    UserLongMemoryCursor.session_id == session_id,
                )
                .order_by(UserLongMemoryCursor.id.desc())
                .limit(1)
            )
            return res.scalar() or 0

    async def update_cursor(self, *, user_id: str, session_id: str, chat_id: int) -> None:
        from sqlalchemy import select
        from app.db.models import UserLongMemoryCursor
        if not user_id or not session_id or not chat_id:
            return
        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(UserLongMemoryCursor)
                .where(
                    UserLongMemoryCursor.user_id == user_id,
                    UserLongMemoryCursor.session_id == session_id,
                )
                .order_by(UserLongMemoryCursor.id.desc())
                .limit(1)
            )
            rec = res.scalar_one_or_none()
            if rec:
                rec.chat_id = max(rec.chat_id, int(chat_id))
            else:
                session.add(
                    UserLongMemoryCursor(user_id=user_id, session_id=session_id, chat_id=int(chat_id))
                )
            await session.commit()

    async def batch_write_session(self, *, user_id: str, session_id: str, history: list[dict]) -> dict:
        """批量写入一个 session 的对话历史到长期记忆。

        游标去重：只处理 cursor 之后的新消息（flush/compress/session_end 多时机幂等）。
        """
        if not history or not user_id:
            return {"written": 0, "skipped": 0, "replaced": 0}

        cursor = await self.get_cursor(user_id=user_id, session_id=session_id)

        all_items: list[LongMemoryItem] = []
        max_chat_id = cursor
        for msg in history:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            chat_id = msg.get("chat_id")
            if chat_id is not None and int(chat_id) <= cursor:
                continue  # 该消息已提取过
            try:
                items = await self.extract_candidates(
                    user_input=content,
                    source_chat_id=int(chat_id) if chat_id is not None else None,
                )
                all_items.extend(items)
            except Exception as e:
                logger.error("batch_write_session extract failed for msg: %s", e)
            if chat_id is not None:
                max_chat_id = max(max_chat_id, int(chat_id))

        if not all_items:
            return {"written": 0, "skipped": 0, "replaced": 0}

        result = await self.write_with_conflict_check(
            user_id=user_id, session_id=session_id, items=all_items, source="session_end"
        )

        if max_chat_id > cursor:
            await self.update_cursor(user_id=user_id, session_id=session_id, chat_id=max_chat_id)
        return result
