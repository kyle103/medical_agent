import pytest

from app.core.agent.intent_classifier import IntentClassifier


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,expected",
    [
        ("阿莫西林和布洛芬能一起吃吗", "drug"),
        ("血糖 6.1 mmol/L 正常吗", "lab"),
        ("我昨天吃的什么药", "archive"),
        ("如何缓解失眠", "general"),
    ],
)
async def test_intent_rule_predict(text: str, expected: str):
    clf = IntentClassifier()
    res = await clf.predict(text=text, stream=False)
    assert res.intent == expected
    assert 0.0 <= res.confidence <= 1.0
