"""意图识别测试。

⚠️ 2026-09-13 修正：本文件原先名为 `test_intent_rule_predict`，却调用 `clf.predict()` ——
而 `predict()` 是 **LLM 优先**的：

    predict() → classify_intent_and_route()（结构化调用）
              → 失败则退回 _rule_predict() 拿基线
              → 再试一次 chat_completion(INTENT_CLASSIFIER 提示词)
              → 只要它的自评置信度 ≥ 规则置信度，就用它的结论

两个后果：
1. **名实不符**：名为"测规则"，实际在测 LLM，规则层在 LLM 可用时完全没被覆盖；
2. **不稳定**：末段那个弱提示词回退路径偶尔给出与规则不一致的标签，
   于是跑全套时随机挂（实测同一句"删除我的关于阿司匹林的用药记录"连挂 3 次，
   单独复跑又通过）。

现拆成两条，各测各的契约：
- `test_intent_rule_predict`：**纯规则、无 LLM、决定性** —— 这才是它的名字承诺的东西；
- `test_intent_predict_returns_legal_intent`：`predict()` 的真实契约是"返回合法意图 + 合法置信度"，
  **不是**"对某句话必须给出某个标签"（那是 LLM 行为，不是接口契约）。
"""

import pytest

from app.core.agent.intent_classifier import IntentClassifier

#: 规则层应稳定给出的判定（无 LLM 依赖）
RULE_CASES = [
    ("阿莫西林和布洛芬能一起吃吗", "drug"),
    ("血糖 6.1 mmol/L 正常吗", "lab"),
    ("我昨天吃的什么药", "archive"),
    ("如何缓解失眠", "general"),
    ("删除我的关于阿司匹林的用药记录", "drug"),
]


@pytest.mark.parametrize("text,expected", RULE_CASES)
def test_intent_rule_predict(text: str, expected: str):
    """规则层：不碰网络，决定性。"""
    res = IntentClassifier._rule_predict(text)
    assert res.intent == expected, f"{text!r} → {res.intent}（reason={res.reason}）"
    assert 0.0 <= res.confidence <= 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("text,_expected", RULE_CASES)
async def test_intent_predict_returns_legal_intent(text: str, _expected: str):
    """`predict()` 的接口契约：返回合法意图与合法置信度。

    不断言具体标签 —— `predict()` 是 LLM 优先的，对某句具体话给出什么标签
    取决于模型，属部署属性而非代码契约（同 附一之三 对结构化输出的结论）。
    """
    res = await IntentClassifier().predict(text=text, stream=False)
    assert res.intent in IntentClassifier.INTENTS, f"{text!r} → 非法意图 {res.intent!r}"
    assert 0.0 <= res.confidence <= 1.0
