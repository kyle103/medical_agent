import re

SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{17}[0-9Xx]\b"),  # 身份证号
    re.compile(r"\b\d{15}\b"),
    re.compile(r"医保卡|社保卡|银行卡|信用卡"),
    re.compile(r"详细住址|家庭住址|身份证|身份证号"),
]


def contains_sensitive_personal_info(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in SENSITIVE_PATTERNS)


def detect_prompt_injection(text: str) -> bool:
    if not text:
        return False
    keywords = [
        "忽略以上", "ignore previous", "system prompt", "developer message",
        "越狱", "jailbreak", "你现在是", "扮演", "泄露", "reveal",
    ]
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)
