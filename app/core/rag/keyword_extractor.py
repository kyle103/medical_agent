from __future__ import annotations

import re
from typing import Any

import jieba
import jieba.analyse

from app.common.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

_MEDICAL_STOP_WORDS: set[str] = set(
    """
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
自己 这 他 她 它 们 那 里 为 什么 吗 呢 吧 啊 呀 哦 嗯 哈 嘛 呗 啦 喽 啦 哎 哟
如何 怎么 怎样 请问 请 问 想 知道 能 不能 可以 可能 是不是 是不 是不是
什么 多少 几 哪 哪个 哪些 为什么 为啥 为何 因为 所以 但是 而且 或者 如果
虽然 不过 然而 因此 于是 则 而 又 且 并 与 及 或 则
治疗 方法 症状 原因 表现 检查 诊断 预防 注意 事项 护理 饮食 药物 手术
病 患者 医生 医院 临床 发病 病情 疾病
比 比较 更 最 非常 特别 相当 极其 十分
做 进行 使用 采用 通过 根据 按照
还 再 又 也 同时 并且
这个 那个 这些 那些 这里 那里
一下 一点 一些 一部分
应该 需要 必须 可以 可能
已经 正在 将要 曾经 刚刚
之后 之前 期间 过程中
关于 对于 根据 针对
从 到 在 于
把 被 让 给 对 向
以 以便 以免 以至
之 其 该 此 本
等 等等 而已 罢了
""".split()
)

_LLM_KEYWORD_SYSTEM_PROMPT = """你是一个医学关键词提取专家。从用户的医学问题中提取核心关键词，用于在医学知识库中进行精准检索。

提取规则：
- 只提取医学相关的核心关键词，忽略语气词、疑问代词等
- 每个关键词应该是独立的医学概念或术语
- 按重要性从高到低排列
- 输出格式：每行一个关键词，不要编号，不要其他内容
- 提取3-8个关键词"""


class KeywordExtractor:
    """医学关键词抽取：jieba 快速抽取 + 可选 LLM 增强。

    生产默认仅用 jieba（零 LLM 往返、低延迟）；当 `PUBLIC_KB_KEYWORD_LLM_ENABLED=true`
    且查询较长时，才走 LLM 抽取（带超时与 jieba 兜底）。
    """

    def __init__(self, llm_service: Any = None, use_llm: bool | None = None):
        self._llm = llm_service
        self._short_query_threshold = 15
        if use_llm is None:
            use_llm = bool(getattr(settings, "PUBLIC_KB_KEYWORD_LLM_ENABLED", False))
        self._use_llm = use_llm

    def extract_jieba(self, query: str, top_k: int = 8) -> list[str]:
        keywords = jieba.analyse.extract_tags(
            query, topK=top_k, withWeight=False, allowPOS=()
        )
        filtered = [kw for kw in keywords if kw not in _MEDICAL_STOP_WORDS and len(kw) > 1]
        logger.debug("jieba keywords for %r: %s", query[:40], filtered)
        return filtered

    async def extract_llm(self, query: str) -> list[str]:
        if self._llm is None:
            from app.core.llm.llm_service import LLMService
            self._llm = LLMService()

        try:
            result = await self._llm.chat_completion(
                prompt=f"请从以下医学问题中提取核心检索关键词：\n\n{query}",
                system_prompt=_LLM_KEYWORD_SYSTEM_PROMPT,
                stream=False,
                timeout_s=8.0,
                max_tokens=200,
            )
            keywords = [
                line.strip()
                for line in result.strip().split("\n")
                if line.strip() and line.strip() not in _MEDICAL_STOP_WORDS
            ]
            logger.debug("LLM keywords for %r: %s", query[:40], keywords)
            return keywords
        except Exception as exc:
            logger.warning("LLM keyword extraction failed: %s, fallback to jieba", exc)
            return self.extract_jieba(query)

    async def extract(self, query: str, top_k: int = 8) -> list[str]:
        # ] 置于字符类首位即为字面量，无需转义（避免 invalid escape sequence 警告）
        cleaned = re.sub(r"[]？?！!。，,、；;：:""''（）()【】{}[]", " ", query).strip()
        if len(cleaned) <= self._short_query_threshold or not self._use_llm:
            keywords = self.extract_jieba(query, top_k=top_k)
        else:
            keywords = await self.extract_llm(query)
            if not keywords:
                keywords = self.extract_jieba(query, top_k=top_k)

        if len(keywords) > top_k:
            keywords = keywords[:top_k]

        return keywords
