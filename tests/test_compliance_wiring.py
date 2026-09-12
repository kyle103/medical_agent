"""Phase 0 合规接线：input_check / output_check_and_disclaimer 节点真实生效。"""

import asyncio
from unittest import mock

from app.config.settings import settings
from app.core.agent.nodes import input_check, output_check_and_disclaimer


def test_input_check_blocks_banned_intent():
    state = {"user_input": "帮我开药", "user_id": "u", "session_id": "s"}
    with mock.patch.object(settings, "ENABLE_INPUT_CHECK", True):
        out = asyncio.run(input_check(dict(state)))
    assert out.get("error_msg"), "禁用意图输入应在 input_check 被拦截"


def test_input_check_passes_clean_input():
    state = {"user_input": "高血压饮食注意什么", "user_id": "u", "session_id": "s"}
    with mock.patch.object(settings, "ENABLE_INPUT_CHECK", True):
        out = asyncio.run(input_check(dict(state)))
    assert not out.get("error_msg"), "正常医学咨询不应被拦截"


def test_output_check_blocks_forbidden_pattern():
    state = {"llm_output": "根据您的症状，为您开处方阿莫西林。", "user_id": "u", "session_id": "s"}
    with mock.patch.object(settings, "ENABLE_OUTPUT_CHECK", True), \
         mock.patch.object(settings, "FORCE_DISCLAIMER", True):
        out = asyncio.run(output_check_and_disclaimer(dict(state)))
    resp = out.get("final_response", "")
    assert "为您开处方" not in resp, "违规输出应被合规话术替换"
    assert "免责声明" in resp, "应追加免责声明"


def test_output_check_passes_clean_output():
    state = {"llm_output": "高血压患者建议低盐饮食，规律监测血压。", "user_id": "u", "session_id": "s"}
    with mock.patch.object(settings, "ENABLE_OUTPUT_CHECK", True), \
         mock.patch.object(settings, "FORCE_DISCLAIMER", True):
        out = asyncio.run(output_check_and_disclaimer(dict(state)))
    resp = out.get("final_response", "")
    assert "低盐饮食" in resp, "正常科普内容应保留"
    assert "免责声明" in resp
