"""用药记录写操作二次确认（选项式）测试。

分四层，各测各的契约：

1. **纯函数**（无 DB / 无 LLM）——`build_options` / `build_card_message` /
   `resolve_answer`。确认流程的全部判断都收在这三个纯函数里，所以这里能穷举。
2. **agent 门控**——歧义时**一次都不许落库**、用户确认后**恰好落一条**。
   断言方式是给 agent 塞一个记账用的假工具，直接看它有没有被调用，而不是看返回值文案。
3. **零 LLM 往返**——回答确认卡（"A" / "不记录"）时不得走 LLM 路由。
   做法是把 LLM 调用替换成"一被调用就抛错"，以此证明它没被调用。
4. **回归护栏**——`pending_confirmation` 必须**留在** turn_reset 之外（加进去等于每轮
   自毁，跨轮确认直接失效）；确认字段必须在 execute 的步骤结果白名单里（漏了就是
   静默失效：卡片不出、或者选了没反应，且不报任何错）。
"""

import asyncio
import json

import pytest

from app.core.agent import nodes
from app.core.agent.drug_record_agent import DrugRecordAgent
from app.core.agent.llm_decision_service import LLMDecisionService
from app.core.agent.stream_events import options_payload
from app.core.rag.entity_dictionary import EntityDictionary
from app.core.skills import drug_write_confirmation as dwc


# --------------------------------------------------------------------------
# 测试脚手架
# --------------------------------------------------------------------------

def _payload(name: str) -> dict:
    return {
        "drug_name": name,
        "drug_alias": "",
        "interaction_drugs": "[]",
        "interaction_desc": "{}",
    }


def _seed_dict() -> EntityDictionary:
    """与 tests/test_entity_dictionary.py 同构的词典（含别名与未收录两类样本）。"""
    d = EntityDictionary()
    d.add_entry(kind="drug", canonical_name="对乙酰氨基酚",
                terms=["对乙酰氨基酚", "扑热息痛"], payload=_payload("对乙酰氨基酚"))
    d.add_entry(kind="drug", canonical_name="复方氨酚烷胺片",
                terms=["复方氨酚烷胺片", "感康"], payload=_payload("复方氨酚烷胺片"))
    d.add_entry(kind="drug", canonical_name="布洛芬",
                terms=["布洛芬"], payload=_payload("布洛芬"))
    d.add_entry(kind="drug", canonical_name="阿司匹林",
                terms=["阿司匹林"], payload=_payload("阿司匹林"))
    d.mark_ready()
    return d


@pytest.fixture
def seeded(monkeypatch) -> EntityDictionary:
    """把模块级单例换成种好的内存词典（纯函数测试与 agent 测试共用）。"""
    d = _seed_dict()
    monkeypatch.setattr(dwc, "get_dictionary", lambda: d)
    return d


@pytest.fixture
def empty_dict(monkeypatch) -> EntityDictionary:
    """词典未加载（DB 异常时的静默降级态）——此时**不许**弹确认卡。"""
    d = EntityDictionary()
    monkeypatch.setattr(dwc, "get_dictionary", lambda: d)
    return d


class _FakeTool:
    """记账用的假工具：不落库，只记录被怎么调用过。"""

    def __init__(self, *, created: bool = True):
        self.calls: list[tuple[str, dict]] = []
        self._created = created

    async def add_record(self, **kw) -> dict:
        self.calls.append(("add", kw))
        return {"ok": True, "created": self._created, "message": "用药记录已存在，无需重复添加。"}

    async def update_by_name(self, **kw) -> dict:
        self.calls.append(("update", kw))
        return {"ok": True, "message": "已更新"}

    async def soft_delete_latest_by_name(self, **kw) -> dict:
        self.calls.append(("delete", kw))
        return {"ok": True, "message": "已删除最近一条记录"}


def _stub_parse(info: dict):
    """替换 `_parse_drug_info`，把 LLM 抽取结果钉死（确认流程本身不依赖抽取质量）。"""
    async def _f(user_input: str, state: dict) -> dict:
        return dict(info)
    return _f


def _agent(seeded) -> DrugRecordAgent:
    agent = DrugRecordAgent()
    agent.drug_record_tool = _FakeTool()
    return agent


# --------------------------------------------------------------------------
# 1. 纯函数：build_options
# --------------------------------------------------------------------------

def test_multi_drug_sentence_offers_candidate_names(seeded):
    """A 类歧义：一轮只会落成一条记录，两个药必须让用户选。"""
    options = dwc.build_options(
        draft={"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次"},
        raw_input="我吃了布洛芬和阿司匹林",
    )
    labels = [o["label"] for o in options]
    assert labels == ["布洛芬", "阿司匹林", dwc._cancel_label("add")]
    assert [o["id"] for o in options] == ["A", "B", "C"]
    assert options[0]["kind"] == "drug"


def test_alias_offers_canonical_and_verbatim(seeded):
    """B 类歧义：用户写"感康"、档案里存规范名，不能让代码替用户决定。"""
    options = dwc.build_options(draft={"drug_name": "感康"}, raw_input="我吃了感康")
    labels = [o["label"] for o in options]
    assert labels[0] == "记录为复方氨酚烷胺片"
    assert labels[1] == "按原文记录（感康）"
    assert labels[2] == dwc._cancel_label("add")
    assert [o["value"] for o in options[:2]] == ["复方氨酚烷胺片", "感康"]


def test_unknown_drug_offers_verbatim_and_cancel(seeded):
    """C 类：词典未收录——未经知识库校验的药名进档案前先确认一次。"""
    options = dwc.build_options(draft={"drug_name": "头孢克肟"}, raw_input="我吃了头孢克肟")
    assert [o["label"] for o in options] == ["按原文记录（头孢克肟）", dwc._cancel_label("add")]


def test_missing_fields_offers_action_options(seeded):
    """D 类：药名无歧义但缺剂量/频次，缺项会被落成"未指定"，用户有权先看到。"""
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬")
    assert [o["label"] for o in options] == ["确认记录", dwc._cancel_label("add")]
    assert [o["value"] for o in options] == [dwc.CONFIRM_VALUE, dwc.CANCEL_VALUE]


def test_unambiguous_and_complete_returns_empty(seeded):
    """E 类：无歧义且字段齐全 → 空列表 = 直接落库，不进确认流程。"""
    assert dwc.build_options(
        draft={"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次"},
        raw_input="我吃了布洛芬，一片，每天两次",
    ) == []


def test_dosage_only_still_asks(seeded):
    """只补了剂量、没补频次，仍算字段不全（不能只看第一个字段就放行）。"""
    assert dwc.build_options(
        draft={"drug_name": "布洛芬", "dosage": "一片"}, raw_input="我吃了布洛芬一片"
    )


def test_update_and_delete_do_not_ask_for_missing_fields(seeded):
    """update 只改用户点到的字段、delete 只按药名定位，缺字段不构成确认理由。"""
    assert dwc.build_options(
        draft={"drug_name": "布洛芬"}, raw_input="布洛芬的剂量改成100mg", operation="update"
    ) == []
    assert dwc.build_options(
        draft={"drug_name": "布洛芬"}, raw_input="删除布洛芬的记录", operation="delete"
    ) == []


def test_delete_wording_never_says_record(seeded):
    """删除路径不能出现"记录为"字样——用户会以为自己点的是新增。"""
    options = dwc.build_options(
        draft={"drug_name": "感康"}, raw_input="删除我昨天吃的感康", operation="delete"
    )
    labels = [o["label"] for o in options]
    assert labels[0] == "删除复方氨酚烷胺片的记录"
    assert labels[1] == "按原文删除（感康）"
    assert labels[2] == "不删除"
    assert not any("记录为" in lb for lb in labels)


def test_empty_dictionary_never_asks(empty_dict):
    """词典静默降级为空时无法区分"歧义"与"未收录"，此时宁可直落库也不弹卡。"""
    assert dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="布洛芬") == []


def test_missing_drug_name_never_asks(seeded):
    """没有药名是另一类问题（提示补全），不该走确认卡。"""
    assert dwc.build_options(draft={"drug_name": ""}, raw_input="我吃药了") == []


# --------------------------------------------------------------------------
# 2. 纯函数：build_card_message
# --------------------------------------------------------------------------

def test_card_lists_options_as_text_and_lettered(seeded):
    """选项必须同时是文本：没有按钮的客户端（stream=false）照着打字也能走通。"""
    options = dwc.build_options(
        draft={"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次"},
        raw_input="我吃了布洛芬和阿司匹林",
    )
    msg = dwc.build_card_message(draft={"drug_name": "布洛芬"}, options=options)
    assert "• 药品：布洛芬" in msg
    for opt in options:
        assert f"{opt['id']}. {opt['label']}" in msg
    # 隐式"其他"跟在已列选项之后
    assert f"{dwc.other_option_id(options)}. 其他" in msg


def test_card_shows_unspecified_for_add_only(seeded):
    """只有新增会把缺项真的落成"未指定"；update/delete 显示它反而误导。"""
    options = [{"id": "A", "label": "确认记录", "value": dwc.CONFIRM_VALUE, "kind": "action"}]
    add_msg = dwc.build_card_message(draft={"drug_name": "布洛芬"}, options=options)
    assert "• 剂量：未指定" in add_msg and "• 频次：未指定" in add_msg

    upd_msg = dwc.build_card_message(
        draft={"drug_name": "布洛芬"}, options=options, operation="update"
    )
    assert "未指定" not in upd_msg


def test_card_for_delete_omits_field_list_and_says_delete(seeded):
    options = dwc.build_options(
        draft={"drug_name": "布洛芬"}, raw_input="删除布洛芬和阿司匹林", operation="delete"
    )
    msg = dwc.build_card_message(
        draft={"drug_name": "布洛芬"}, options=options, operation="delete"
    )
    assert "删除" in msg
    assert "未指定" not in msg


# --------------------------------------------------------------------------
# 3. 纯函数：resolve_answer
# --------------------------------------------------------------------------

def _drug_choice_card() -> list[dict]:
    """多药同句那张卡：只有候选药名 + 取消，**没有**"确认"动作选项。"""
    return dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬和阿司匹林")


def _confirm_card() -> list[dict]:
    """药名无歧义、字段缺失那张卡：有"确认记录"动作选项。"""
    return dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬")


def _pending_for(options: list[dict], *, attempts: int = 0) -> dict:
    return {
        "id": "abc123",
        "type": dwc.PENDING_TYPE,
        "operation": "add",
        "draft": {"drug_name": "布洛芬"},
        "options": options,
        "attempts": attempts,
        "raw_user_input": "我吃了布洛芬和阿司匹林",
    }


@pytest.mark.parametrize("answer", ["A", "a", "1", "A.", "A、", "a)", "布洛芬", " 布洛芬 "])
def test_resolve_select_by_index_label_and_free_text(seeded, answer):
    """序号 / label 精确匹配 / 自由文本写药名，三条路都要能选中同一个选项。"""
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬和阿司匹林")
    action, payload = dwc.resolve_answer(answer, _pending_for(options))
    assert action == "select", answer
    assert payload["value"] == "布洛芬"


def test_resolve_second_option(seeded):
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬和阿司匹林")
    action, payload = dwc.resolve_answer("B", _pending_for(options))
    assert (action, payload["value"]) == ("select", "阿司匹林")


@pytest.mark.parametrize("answer", ["确认", "嗯", "好", "可以", "yes", "OK"])
def test_resolve_affirmative(seeded, answer):
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬")
    action, payload = dwc.resolve_answer(answer, _pending_for(options))
    assert action == "affirm"
    assert payload["value"] == dwc.CONFIRM_VALUE


@pytest.mark.parametrize("answer", ["确认", "嗯", "好", "yes"])
def test_bare_affirmative_is_not_an_answer_on_a_drug_choice_card(seeded, answer):
    """"选哪个药"那张卡上没有"确认"选项——一句"嗯"没指定任何药名。

    若按 affirm 处理，代码就会替用户在两个药里挑一个（删除场景 = 删错药）。
    这种回答必须落到答非所问，由 attempts 计数决定重问还是放弃。
    """
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬和阿司匹林")
    assert not any(o.get("value") == dwc.CONFIRM_VALUE for o in options)
    assert dwc.resolve_answer(answer, _pending_for(options)) == ("unrelated", None)


def test_affirmative_still_works_after_the_drug_choice_is_settled(seeded):
    """用户先用自由文本指定了药名（第 4 条规则），此后的"嗯"就该被认下来。"""
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬和阿司匹林")
    assert dwc.resolve_answer("阿司匹林", _pending_for(options))[0] == "select"


@pytest.mark.parametrize("answer", ["不记录", "取消", "算了", "不用了", "不", "no"])
def test_resolve_negative(seeded, answer):
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬")
    assert dwc.resolve_answer(answer, _pending_for(options)) == ("deny", None)


@pytest.mark.parametrize("answer", ["今天天气不错", "", "帮我查一下血糖"])
def test_resolve_unrelated(seeded, answer):
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬")
    assert dwc.resolve_answer(answer, _pending_for(options)) == ("unrelated", None)


def test_resolve_free_text_with_two_drugs_is_unrelated(seeded):
    """自由文本里解析出两个药名时不算"唯一解析"，不能猜——交给 attempts 逻辑。"""
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬和阿司匹林")
    assert dwc.resolve_answer("布洛芬还有阿司匹林", _pending_for(options)) == ("unrelated", None)


def test_resolve_cancel_option_maps_to_deny(seeded):
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬")
    cancel_idx = len(options) - 1  # 列表末项固定是取消
    assert options[cancel_idx]["value"] == dwc.CANCEL_VALUE
    assert dwc.resolve_answer(dwc.OPTION_LETTERS[cancel_idx], _pending_for(options)) == ("deny", None)


# --------------------------------------------------------------------------
# 4. agent 门控：歧义时不落库，确认后恰好落一条
# --------------------------------------------------------------------------

def test_add_with_ambiguity_writes_nothing(seeded):
    agent = _agent(seeded)
    agent._parse_drug_info = _stub_parse(
        {"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次", "time": "今天"}
    )
    state: dict = {}
    out = asyncio.run(agent._handle_add_operation("u1", "我吃了布洛芬和阿司匹林", state))

    assert out is None, "确认卡路径不得产出 final_response，否则会吃掉 needs_confirmation 分支"
    assert agent.drug_record_tool.calls == [], "出确认卡就必须一次都不落库"
    assert state["needs_confirmation"] is True
    assert state["pending_confirmation"]["type"] == dwc.PENDING_TYPE
    assert state["pending_confirmation"]["operation"] == "add"
    assert state["pending_confirmation"]["attempts"] == 0
    assert state["confirmation_message"]


def test_gate_reads_user_original_words_not_planner_query(seeded):
    """歧义检测必须看用户原话，而不是 planner 改写后的步骤 query。

    `_build_sub_state` 会把 sub_state["user_input"] 换成步骤 query（可能带
    "[背景信息：…]" 前缀或换了措辞）。若拿它去检测多药同句，第二个药名就没了，
    歧义被吞掉——用户说"布洛芬和阿司匹林"，系统静默只记布洛芬。
    """
    agent = _agent(seeded)
    agent._parse_drug_info = _stub_parse(
        {"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次", "time": "今天"}
    )
    state = {"original_user_input": "我吃了布洛芬和阿司匹林"}
    out = asyncio.run(agent._handle_add_operation("u1", "记录用户的用药情况：布洛芬", state))

    assert out is None
    labels = [o["label"] for o in state["pending_confirmation"]["options"]]
    assert "阿司匹林" in labels, "改写后的步骤 query 里没有第二个药名，必须回退到原话"
    assert state["pending_confirmation"]["raw_user_input"] == "我吃了布洛芬和阿司匹林"


def test_add_unambiguous_writes_directly(seeded):
    agent = _agent(seeded)
    agent._parse_drug_info = _stub_parse(
        {"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次", "time": "今天"}
    )
    state: dict = {}
    out = asyncio.run(agent._handle_add_operation("u1", "我吃了布洛芬一片每天两次", state))

    assert isinstance(out, str) and out
    assert [c[0] for c in agent.drug_record_tool.calls] == ["add"]
    assert agent.drug_record_tool.calls[0][1]["drug_name"] == "布洛芬"
    assert "pending_confirmation" not in state


def test_delete_with_two_candidates_asks_instead_of_dropping_one(seeded):
    """旧实现直接取 drugs[0]，用户说删两个时其中一个被静默忽略——现在必须先问。"""
    agent = _agent(seeded)
    state: dict = {}
    out = asyncio.run(agent._handle_delete_operation("u1", "删除布洛芬和阿司匹林", state))

    assert out is None
    assert agent.drug_record_tool.calls == []
    labels = [o["label"] for o in state["pending_confirmation"]["options"]]
    assert "布洛芬" in labels and "阿司匹林" in labels
    assert state["pending_confirmation"]["operation"] == "delete"


def test_delete_single_candidate_writes_directly(seeded):
    agent = _agent(seeded)
    state: dict = {}
    out = asyncio.run(agent._handle_delete_operation("u1", "删除布洛芬的记录", state))

    assert out == "已删除最近一条记录"
    assert [c[0] for c in agent.drug_record_tool.calls] == ["delete"]
    # 规则抽取器对这句会整句兜底（候选名 = "删除布洛芬的记录"），必须被词典纠正回药名
    assert agent.drug_record_tool.calls[0][1]["drug_name"] == "布洛芬"


def test_preferred_drug_name_prefers_dictionary_over_sentence_fallback(seeded):
    """规则抽取器的整句兜底不能进档案表：抽不到药名时才允许用 fallback。"""
    assert dwc.preferred_drug_name(raw_input="删除布洛芬的记录", fallback="删除布洛芬的记录") == "布洛芬"
    assert dwc.preferred_drug_name(raw_input="我吃了个新药", fallback="新药") == "新药"
    assert dwc.preferred_drug_name(
        raw_input="删除布洛芬和阿司匹林", fallback="整句"
    ) == "布洛芬"


def test_process_uses_confirmed_drug_name(seeded):
    """用户选了 B（阿司匹林）→ 落库的必须是阿司匹林，不是 draft 里的布洛芬。"""
    agent = _agent(seeded)
    agent._parse_drug_info = _stub_parse({"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次"})
    state: dict = {}
    asyncio.run(agent._handle_add_operation("u1", "我吃了布洛芬和阿司匹林", state))
    pending = state["pending_confirmation"]

    async def _must_not_be_called(*a, **k):
        raise AssertionError("确认卡的回答不该重新判操作类型")

    agent._parse_operation_type = _must_not_be_called
    state["confirmation_resolution"] = {
        "action": "select",
        "payload": {"id": "B", "label": "阿司匹林", "value": "阿司匹林", "kind": "drug"},
        "pending": pending,
    }
    state.update({"user_id": "u1", "user_input": "B"})
    out_state = asyncio.run(agent.process(state))

    assert [c[0] for c in agent.drug_record_tool.calls] == ["add"]
    assert agent.drug_record_tool.calls[0][1]["drug_name"] == "阿司匹林"
    assert out_state["pending_confirmation"] == {}, "确认后必须清 pending，否则下一轮会被重复拦截"
    assert out_state["needs_confirmation"] is False
    assert "阿司匹林" in out_state["final_response"]


def test_process_deny_writes_nothing_and_clears_pending(seeded):
    agent = _agent(seeded)
    agent._parse_drug_info = _stub_parse({"drug_name": "布洛芬", "dosage": "一片", "frequency": "每天两次"})
    state: dict = {}
    asyncio.run(agent._handle_add_operation("u1", "我吃了布洛芬和阿司匹林", state))
    pending = state["pending_confirmation"]

    state["confirmation_resolution"] = {"action": "deny", "payload": None, "pending": pending}
    state.update({"user_id": "u1", "user_input": "不记录"})
    out_state = asyncio.run(agent.process(state))

    assert agent.drug_record_tool.calls == []
    assert out_state["pending_confirmation"] == {}
    assert "不写" in out_state["final_response"]


def test_confirmed_delete_targets_chosen_drug(seeded):
    agent = _agent(seeded)
    state: dict = {}
    asyncio.run(agent._handle_delete_operation("u1", "删除布洛芬和阿司匹林", state))
    pending = state["pending_confirmation"]

    state["confirmation_resolution"] = {
        "action": "select",
        "payload": {"id": "A", "label": "布洛芬", "value": "布洛芬", "kind": "drug"},
        "pending": pending,
    }
    state.update({"user_id": "u1", "user_input": "A"})
    asyncio.run(agent.process(state))

    assert [c[0] for c in agent.drug_record_tool.calls] == ["delete"]
    assert agent.drug_record_tool.calls[0][1]["drug_name"] == "布洛芬"


# --------------------------------------------------------------------------
# 5. 节点级：确认回答零 LLM 往返
# --------------------------------------------------------------------------

def _llm_must_not_run(monkeypatch) -> None:
    """把 LLM 路由类调用换成"一被调用就抛错"，以此证明它没被调用。"""
    monkeypatch.setattr(nodes, "_llm_enabled_for_nodes", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("回答确认卡时不得发起 LLM 调用")

    monkeypatch.setattr(LLMDecisionService, "classify_route_and_extract", _boom)
    monkeypatch.setattr(LLMDecisionService, "classify_operation_type", _boom)


@pytest.mark.parametrize(
    "card,answer,expected,expected_value",
    [
        (_drug_choice_card, "A", "select", "布洛芬"),
        (_drug_choice_card, "B", "select", "阿司匹林"),
        (_confirm_card, "确认", "affirm", dwc.CONFIRM_VALUE),
        (_confirm_card, "不记录", "deny", None),
    ],
)
def test_intent_node_resolves_confirmation_without_llm(
    seeded, monkeypatch, card, answer, expected, expected_value
):
    """用户答"B""确认"这类短回答绝不能被交给分类器——落到 general 就等于确认丢失。"""
    _llm_must_not_run(monkeypatch)
    state = {"user_input": answer, "pending_confirmation": _pending_for(card())}

    out = asyncio.run(nodes.intent_recognition(state))

    assert out["intent"] == "drug"
    assert out["target_agent"] == "drug_record_agent"
    assert out["is_multi_intent"] is False
    resolution = out["confirmation_resolution"]
    assert resolution["action"] == expected
    assert (resolution["payload"] or {}).get("value") == expected_value
    assert resolution["pending"] == state["pending_confirmation"]


def test_unrelated_answer_keeps_pending_until_attempts_exhausted(seeded):
    """答非所问不是"取消"：先记一次数继续等，连续超限才自动放弃（避免永久悬挂）。"""
    options = dwc.build_options(draft={"drug_name": "布洛芬"}, raw_input="我吃了布洛芬")
    pending = _pending_for(options, attempts=0)
    state = {"user_input": "今天天气不错", "pending_confirmation": pending}

    assert nodes._intercept_pending_confirmation(state, "今天天气不错") is False
    assert state["pending_confirmation"]["attempts"] == 1
    assert "confirmation_resolution" not in state, "没听懂不等于用户回答，不得写入 resolution"
    assert state["pending_confirmation"]["options"] == pending["options"], "重问要保留原选项"

    state2 = {"user_input": "今天天气不错", "pending_confirmation": state["pending_confirmation"]}
    assert nodes._intercept_pending_confirmation(state2, "今天天气不错") is True
    assert state2["confirmation_resolution"]["action"] == "deny"


def test_intercept_ignores_foreign_pending_type(seeded):
    """pending 是别的确认类型时不接管，免得把无关流程的挂起态当用药确认吃掉。"""
    state = {"pending_confirmation": {"type": "something_else", "options": [{"id": "A"}]}}
    assert nodes._intercept_pending_confirmation(state, "A") is False
    assert "confirmation_resolution" not in state


# --------------------------------------------------------------------------
# 6. 回归护栏
# --------------------------------------------------------------------------

def test_pending_confirmation_survives_turn_reset():
    """跨轮载体必须在 turn_reset 之外。

    这是个容易顺手改坏的地方：`pending_confirmation` 和 `needs_confirmation` 名字相近、
    都在"确认"语义下，很容易被一起加进 _TURN_LOCAL_FIELDS——加进去就等于每轮开头自毁，
    用户答"B"时 pending 已经没了，症状是"确认卡出了但选了没反应"。
    """
    assert "pending_confirmation" not in nodes._TURN_LOCAL_FIELDS
    assert "pending_confirmation" not in nodes._TURN_LOCAL_DEFAULTS
    # 反过来：单轮有效的回答字段**必须**被重置，残留会让下一轮误触发落库
    assert "confirmation_resolution" in nodes._TURN_LOCAL_FIELDS
    assert nodes._TURN_LOCAL_DEFAULTS["confirmation_resolution"] == {}


def test_confirmation_keys_are_in_step_result_whitelist():
    """execute 步骤结果白名单是硬闸门，漏一个键就是静默失效（不报错、只是没反应）。"""
    for key in ("needs_confirmation", "confirmation_message", "pending_confirmation"):
        assert key in nodes._STEP_RESULT_KEYS, f"{key} 必须在 _STEP_RESULT_KEYS 里"


def test_project_step_result_keeps_confirmation_fields():
    projected = nodes._project_step_result({
        "final_response": "", "needs_confirmation": True,
        "confirmation_message": "卡片", "pending_confirmation": {"id": "x"},
        "unrelated_key": "丢弃",
    })
    assert set(projected) == {
        "final_response", "needs_confirmation", "confirmation_message", "pending_confirmation"
    }


def test_decision_context_keeps_pending_block_beyond_budget():
    """待确认块必须活过 max_chars 整体截断：它常是本轮路由的唯一依据（用户只回一个"A"）。"""
    state = {
        "history": [
            {"role": "user", "content": "很长的一段近期对话" * 60},
            {"role": "assistant", "content": "同样很长的一段回答" * 60},
        ],
        "pending_confirmation": _pending_for(
            [{"id": "A", "label": "布洛芬", "value": "布洛芬", "kind": "drug"}]
        ),
    }
    ctx = nodes._build_decision_context(state)
    assert "待用户确认的用药记录" in ctx
    assert '"options"' in ctx, "options 被截断的话路由 LLM 看不到可选项"
    assert json.loads(ctx.split("待用户确认的用药记录：\n", 1)[1])
    assert len(ctx) > 600, "该块算在预算之外，整体长度可以超预算"


def test_confirmation_card_outranks_final_response():
    """确认卡必须压过同轮其他步骤产出的 final_response。

    真实触发路径（线上复现过）：用户说"我吃了布洛芬和阿司匹林" →
      s1 drug_record_agent 抛出确认卡（不设 final_response）→
      execute_node 判定"单句多药却没做相互作用检查" → replan 追加 s_conflict_check_1 →
      该步产出 final_response。
    如果 final_response 的短路分支排在前面，卡片就被整个吞掉：`done` 事件里
    needs_confirmation 仍是 true、pending 也已落库，但用户**只看到相互作用提示**，
    永远不知道有条记录在等确认；下一句无关的话还会被拦截器当成对这张隐身卡片的回答。
    """
    state = {
        "final_response": "布洛芬和阿司匹林同属 NSAIDs，不建议自行同时服用。",
        "needs_confirmation": True,
        "confirmation_message": "我理解您想记录这条用药信息：\n\nA. 布洛芬\nB. 阿司匹林",
    }
    plan = nodes.build_generation_prompt(state)
    assert plan["branch"] == "confirmation", "final_response 抢先命中会把确认卡吞掉"
    # 同轮的安全提示不能因为写操作被挂起就丢，由 content 带到卡片前面
    assert "NSAIDs" in plan["content"]


def test_confirmation_card_outranks_multi_intent():
    """确认卡还必须压过 multi_intent 分支——这是实际踩到的那一个。

    multi_intent 的判定在函数最开头，且它自己会发起一次 LLM 整合调用把多个
    reconciled_sections 揉成一段回答。多步路径下（确认卡 + 相互作用检查）
    sections 数 >1，于是整合回答把卡片顶掉了：用户看到一段"## 本轮对话中提到的药物…
    ## 跨任务药物冲突提醒"的漂亮回答，而那张卡片一次都没出现。
    """
    state = {
        "needs_confirmation": True,
        "confirmation_message": "请确认：\nA. 布洛芬\nB. 阿司匹林",
        "reconciled_sections": [
            "**布洛芬和阿司匹林**\n同属 NSAIDs，合用增加出血风险。",
        ],
    }
    plan = nodes.build_generation_prompt(state)
    assert plan["branch"] == "confirmation", "multi_intent 抢先命中会把确认卡吞掉"
    assert "NSAIDs" in plan["content"], "相互作用告警要保留在卡片前面，不能被丢掉"


def test_confirmation_card_still_wins_without_other_step_output():
    """没有其他步骤产出时（单步路径）行为不变，content 为空串而非 None。"""
    state = {"needs_confirmation": True, "confirmation_message": "请确认："}
    plan = nodes.build_generation_prompt(state)
    assert plan["branch"] == "confirmation"
    assert plan["content"] == ""


def test_final_response_branch_still_reachable_when_no_confirmation():
    """反向断言：确认字段为假值时，final_response 短路分支必须照常生效。"""
    assert nodes.build_generation_prompt({"final_response": "普通回答"})["branch"] == "final_response"


def test_options_payload_contract():
    """前端 app.js 的 options 分支依赖这四个字段，改字段必须两边一起改。"""
    payload = options_payload(
        confirmation_id="abc", options=[{"id": "A", "label": "布洛芬", "value": "布洛芬", "kind": "drug"}]
    )
    assert payload == {
        "type": "options",
        "confirmation_id": "abc",
        "options": [{"id": "A", "label": "布洛芬", "value": "布洛芬", "kind": "drug"}],
        "allow_other": True,
    }
