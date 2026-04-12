from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass

from app.common.exceptions import ServiceUnavailableException
from app.db.vector_store import init_chroma_collection


DEFAULT_COLLECTION = "user_long_memory"


@dataclass
class LongMemoryItem:
    """极简长期记忆条目。

    为后续“进阶版”（结构化 facts 表、可更新/可删除、多类型召回）预留字段。
    """

    memory_id: str
    text: str
    memory_type: str = "fact"  # fact/preference/profile/summary
    source: str = "chat"
    session_id: str | None = None
    created_at: int = 0

    def to_metadata(self, *, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "source": self.source,
            "session_id": self.session_id or "",
            "created_at": int(self.created_at or 0),
        }


class LongMemoryService:
    """基于向量库（Chroma file persist）的极简长期记忆服务。

    设计目标：
    - 与 DB 全量对话分离：这里只存“可复用事实/偏好/画像摘要”，用于语义召回。
    - 隔离方式：统一使用 user_id（来自 JWT sub），避免使用手机号。
    - 易扩展：后续可替换为 facts 表 + 向量库仅存 embedding/doc_id。
    """

    def __init__(self, *, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name

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

    def extract_candidates(self, *, user_input: str) -> list[LongMemoryItem]:
        """从用户输入中抽取可写入长期记忆的候选条目（规则版）。

        极简策略：
        - 只抽取很少量（<=2条），降低噪声。
        - 后续可以替换为 LLM 抽取 + 去重 + 置信度。
        """

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

        col = self._collection()

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for it in items:
            it.session_id = session_id
            ids.append(it.memory_id)
            docs.append(it.text)
            metas.append(it.to_metadata(user_id=user_id))

        col.add(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def recall(self, *, user_id: str, query: str, top_k: int = 3) -> list[LongMemoryItem]:
        if not user_id:
            return []
        q = (query or "").strip()
        if not q:
            return []

        col = self._collection()
        res = col.query(query_texts=[q], n_results=max(1, int(top_k)), where={"user_id": user_id})

        ids = (res.get("ids") or [[]])[0] or []
        docs = (res.get("documents") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []

        out: list[LongMemoryItem] = []
        for i in range(min(len(ids), len(docs))):
            md = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            out.append(
                LongMemoryItem(
                    memory_id=str(ids[i]),
                    text=str(docs[i] or ""),
                    memory_type=str(md.get("memory_type") or "fact"),
                    source=str(md.get("source") or "chat"),
                    session_id=str(md.get("session_id") or "") or None,
                    created_at=int(md.get("created_at") or 0),
                )
            )

        return out
