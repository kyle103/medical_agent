from __future__ import annotations

import re

from app.common.utils import contains_sensitive_personal_info, detect_prompt_injection
from app.config.compliance_rules import (
    BANNED_INTENT_HINTS,
    FORBIDDEN_OUTPUT_PATTERNS,
    STANDARD_DISCLAIMER,
)
from app.config.settings import settings


class ComplianceService:
    def __init__(self):
        self.forbidden_patterns = FORBIDDEN_OUTPUT_PATTERNS
        self.banned_intents = BANNED_INTENT_HINTS
        self.standard_disclaimer = STANDARD_DISCLAIMER

    def input_compliance_check(self, user_input: str) -> tuple[bool, str]:
        if not settings.ENABLE_INPUT_CHECK:
            return True, ""

        if contains_sensitive_personal_info(user_input):
            return False, "检测到敏感个人信息采集内容，请勿输入身份证号、医保卡号、详细住址等信息。"

        if detect_prompt_injection(user_input):
            return False, "检测到疑似提示词注入/越权指令，已拦截。"

        for w in self.banned_intents:
            if w in user_input:
                return False, "该请求可能涉及诊疗或用药决策，已按合规要求拦截。"

        return True, ""

    def output_compliance_check(self, output_content: str) -> tuple[bool, str]:
        if not settings.ENABLE_OUTPUT_CHECK:
            return True, ""

        for pattern, label in self.forbidden_patterns:
            if re.search(pattern, output_content):
                return False, f"检测到违规医疗内容（{label}），已拦截。"

        return True, ""

    def add_disclaimer(self, content: str) -> str:
        if not content:
            return self.standard_disclaimer
        if self.standard_disclaimer in content or "免责声明" in content:
            return content
        return f"{content}\n\n{self.standard_disclaimer}"
