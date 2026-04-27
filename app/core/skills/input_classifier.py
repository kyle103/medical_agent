from __future__ import annotations

import re


class InputClassifier:
    AFFIRMATIVE = {"是", "是的", "确认", "确定", "好", "好的", "y", "yes", "同意", "添加", "保存", "嗯", "行", "可以"}
    NEGATIVE = {"不", "取消", "不要", "不用", "否", "no", "n", "算了", "放弃", "不用了"}

    DRUG_DOSAGE_RE = re.compile(r"\d+\.?\d*\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)", re.I)
    DRUG_FREQUENCY_RE = re.compile(r"(一天|每日|每天|早晚|早中晚)\s*\d*\s*次", re.I)
    DATE_RE = re.compile(r"(今天|昨天|前天|\d+月\d+日|\d{4}-\d{2}-\d{2})", re.I)
    PURPOSE_RE = re.compile(r"(头痛|发烧|感冒|疼痛|炎症|高血压|糖尿病|降压|降糖|消炎|止咳|退烧|镇痛)")

    @classmethod
    def classify(cls, text: str, expected_field: str | None = None) -> str:
        t = (text or "").strip().lower()
        if not t:
            return "irrelevant"

        if t in cls.AFFIRMATIVE:
            return "affirmative"
        if t in cls.NEGATIVE:
            return "negative"

        if expected_field == "dosage" and cls.DRUG_DOSAGE_RE.search(t):
            return "field_answer"
        if expected_field == "frequency" and cls.DRUG_FREQUENCY_RE.search(t):
            return "field_answer"
        if expected_field == "start_date_text" and cls.DATE_RE.search(t):
            return "field_answer"
        if expected_field == "purpose" and cls.PURPOSE_RE.search(t):
            return "field_answer"

        if len(t) <= 30 and not any(kw in t for kw in ["帮我", "请问", "我想", "查询", "检查", "相互作用", "档案"]):
            return "likely_field_answer"

        return "irrelevant"
