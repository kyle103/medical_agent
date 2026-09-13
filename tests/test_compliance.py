"""合规输出拦截的契约测试。

⚠️ 口径说明（2026-09-13 修正）：
`app/config/compliance_rules.py` 的 `FORBIDDEN_OUTPUT_PATTERNS` 在 commit `62a153c` 由
「裸子串列表 `["确诊", "诊断", ...]`」收窄为**正则**。收窄方向是**对的**：裸子串会把
"如果确诊了高血压，日常需要注意…"这类正常科普一并拦掉（误伤），正则只拦
"**AI 自称替用户确诊/开方改药**"这一类越界表述。

但收窄时写成 `您?`，**只认「您」不认「你」**，而提示词里并没有强制模型使用「您」——
`"根据你的描述，我为你确诊为高血压"` 会被静默放行。本次已把称谓改为 `[您你]`，
本测试同时固化两个称谓的行为，并把「正常科普不得误伤」作为边界用例一并断言。
"""

import pytest

from app.core.compliance.compliance_service import ComplianceService


def test_input_sensitive_info_block():
    ok, msg = ComplianceService().input_compliance_check("身份证号 11010119900307876X")
    assert ok is False
    assert "敏感" in msg


@pytest.mark.parametrize(
    "text,expect_hit",
    [
        # —— 越界表述：AI 自称替用户确诊 / 开方 / 改药 —— 必须拦截 ——
        # 口语「你」与敬语「您」都要覆盖（这正是 2026-09-13 修掉的洞）
        ("根据你的描述，我为你确诊为高血压", "确诊"),
        ("我为您确诊为高血压", "确诊"),
        ("我为你诊断出这是感冒", "诊断"),
        ("为您诊断出这是感冒", "诊断"),
        ("为你开具处方：氨氯地平 5mg", "处方"),
        ("为你开药：布洛芬", "开药"),
        ("建议你停药三天观察", "停药"),
        ("建议您停药三天观察", "停药"),
        ("建议你加量到每天两次", "加量"),
        ("建议你换药", "换药"),
        ("为你制定治疗方案如下", "治疗方案"),
        ("鉴别诊断：原发性高血压", "鉴别诊断"),
        # —— 边界：正常科普 / 引导就医，不得误伤 ——
        ("建议你确诊后再治疗", None),
        ("如果确诊为高血压，日常需要注意低盐饮食", None),
        ("请及时前往正规医疗机构就诊，遵医嘱用药", None),
        ("高血压患者应遵医嘱服药，不要自行停药", None),
    ],
)
def test_output_forbidden_pattern_contract(text, expect_hit):
    ok, msg = ComplianceService().output_compliance_check(text)

    if expect_hit is None:
        assert ok is True, f"正常科普应放行，实际被拦：{msg!r} ← {text!r}"
    else:
        assert ok is False, f"越界表述应被拦截，实际放行 ← {text!r}"
        assert expect_hit in msg, f"拦截提示应点名命中项 {expect_hit!r}，实际：{msg!r}"


@pytest.mark.parametrize("pronoun", ["你", "您"])
def test_forbidden_patterns_cover_both_pronouns(pronoun):
    """固化不变量：**每条** `为[您你]?X` / `建议[您你]?X` 规则对两种称谓行为一致。

    防复发：以后往规则表里加条目时漏掉称谓类，这条会直接指出是哪一条。
    """
    import re

    from app.config.compliance_rules import FORBIDDEN_OUTPUT_PATTERNS

    #: 规则 → 用该称谓填出的样本（样本必须能被对应规则命中）
    _PROBES = {
        "确诊": "我为{谁}确诊为高血压",
        "诊断": "我为{谁}诊断出这是感冒",
        "处方": "为{谁}开处方：氨氯地平 5mg",
        "开药": "为{谁}开药：布洛芬",
        "加量": "建议{谁}加量到每天两次",
        "停药": "建议{谁}停药三天观察",
        "换药": "建议{谁}换药",
        "治疗方案": "为{谁}制定治疗方案如下",
    }

    # 前置检查：规则表本身必须还在（防止被整段删掉后本测试空转通过）
    declared = {pattern for pattern, _ in FORBIDDEN_OUTPUT_PATTERNS}
    for label in _PROBES:
        assert any(label == lab for _, lab in FORBIDDEN_OUTPUT_PATTERNS), (
            f"规则表已无 {label!r} 条目，请同步更新本测试"
        )

    for pattern, label in FORBIDDEN_OUTPUT_PATTERNS:
        if label not in _PROBES:
            continue
        sample = _PROBES[label].format(谁=pronoun)
        assert re.search(pattern, sample), (
            f"规则 {pattern!r}（{label}）应能命中称谓「{pronoun}」的样本 {sample!r}"
        )
