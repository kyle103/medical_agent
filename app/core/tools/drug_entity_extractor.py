from __future__ import annotations

import re
from typing import List


class DrugEntityExtractor:
    """共享药物实体抽取能力（规则型，可复用）。"""

    _STOP_WORDS = {
        "我", "你", "他", "她", "它", "请问", "一下", "这个", "那个", "还有", "是否", "能", "不能",
        "帮我", "请帮我", "记录一下", "删除", "查询", "查看", "病史"
    }
    _NOISE_TERMS = [
        "一起吃", "同服", "相互作用", "有冲突", "冲突", "禁忌", "配伍", "能不能", "可不可以", "可以", "吗", "么", "呢",
        "帮我记录", "帮我", "记录一下", "删除一下", "查一下"
    ]

    @classmethod
    def extract_drug_candidates(cls, text: str, *, max_items: int = 10) -> List[str]:
        raw = (text or "").strip()
        if not raw:
            return []

        normalized = (
            raw.replace("？", " ")
            .replace("?", " ")
            .replace("。", " ")
            .replace("，", ",")
            .replace("、", ",")
            .replace("；", ",")
            .replace(";", ",")
        )
        for noise in cls._NOISE_TERMS:
            normalized = normalized.replace(noise, " ")

        parts = re.split(r"[,\s]|和|与|及|加上|配合", normalized)
        out: List[str] = []
        for p in parts:
            cand = (p or "").strip()
            if len(cand) < 2 or len(cand) > 24:
                continue
            if cand in cls._STOP_WORDS:
                continue
            if re.search(r"[0-9]", cand):
                continue
            out.append(cand)

        # 保序去重
        dedup: List[str] = []
        seen = set()
        for x in out:
            if x in seen:
                continue
            seen.add(x)
            dedup.append(x)
        return dedup[:max_items]

