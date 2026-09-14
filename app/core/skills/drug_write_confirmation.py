"""用药记录写操作的选项式二次确认。

为什么需要：
    `user_drug_records` 是医疗档案。LLM 只负责抽取与表达，**落库必须经用户确认**——
    药名有歧义或关键字段不齐时不落库，改为出一张选项卡；用户选中即确认落库，
    选 other 表达不记录即取消。

本模块是确认流程的**唯一决策源**，三个纯函数（零 LLM、零 I/O、零状态），
因此可被单测穷举：

    build_options      → 按歧义类型生成选项；**返回空列表表示"无歧义，直接落库"**
    build_card_message → 渲染确认卡文本（确定性模板，不走 LLM）
    resolve_answer     → 解析用户对上一轮确认卡的回答

跨轮载体是 `state["pending_confirmation"]`（由 AgentStateStore 持久化），
**不是** `candidate_drug_events` —— 后者是 UntrackedValue，`turn_reset` 每轮开头清零，
靠它做跨轮确认必失败（这正是改造前确认分支不可达的根因）。
"""

from __future__ import annotations

import re

from app.core.rag.entity_dictionary import get_dictionary, normalize_text

__all__ = [
    "build_options",
    "build_card_message",
    "preferred_drug_name",
    "resolve_answer",
    "other_option_id",
    "AFFIRMATIVE",
    "NEGATIVE",
    "CONFIRM_VALUE",
    "CANCEL_VALUE",
    "MAX_ATTEMPTS",
    "PENDING_TYPE",
]

#: pending_confirmation["type"] 取值，用于和未来的其他确认类型区分
PENDING_TYPE = "drug_write"

#: 选项字母表。列表最多 3 项，第 4 个字母留给隐式的"其他"
OPTION_LETTERS = "ABCD"
MAX_LISTED_OPTIONS = 3
#: 多药同句时最多列几个候选药名（留一位给"不记录"）
MAX_DRUG_OPTIONS = 2

#: 连续多少轮答非所问后自动放弃确认（沿用被删 InputClassifier 的 MAX_IRRELEVANT_COUNT 语义）。
#: 没有这个上限的话，用户换话题后 pending 会永久悬挂，之后每轮都被拦截。
MAX_ATTEMPTS = 2

#: 动作型选项的机器值（label 是给人看的，value 给代码分支用）
CONFIRM_VALUE = "__confirm__"
CANCEL_VALUE = "__cancel__"

# 肯定/否定词表。移植自 commit 217b03e 删除的 `InputClassifier`
# （其 AFFIRMATIVE 比仍留在 medication_confirmation_skill.py 里的那份多"嗯/行/可以"）。
AFFIRMATIVE = frozenset(
    {
        "是", "是的", "对", "对的", "好", "好的", "行", "行的", "可以", "确认",
        "确定", "嗯", "同意", "添加", "保存", "记录", "y", "yes", "ok",
    }
)
NEGATIVE = frozenset(
    {
        "不", "否", "不是", "不要", "不用", "不用了", "不记录", "不保存",
        "不更新", "不删除", "别记", "取消", "算了", "放弃", "n", "no",
    }
)

_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("drug_name", "药品"),
    ("dosage", "剂量"),
    ("frequency", "频次"),
    ("time", "时间"),
)

#: 写操作类型 → 卡片开场白 / 取消选项文案。
#: 删除不可逆，文案必须说"删除"，不能复用"不记录"——用户会以为自己只是不新增。
_OPERATION_LEAD: dict[str, str] = {
    "add": "我理解您想记录这条用药信息：",
    "update": "我理解您想更新这条用药信息：",
    "delete": "我理解您想删除这条用药信息：",
}
_OPERATION_CANCEL: dict[str, str] = {
    "add": "不记录",
    "update": "不更新",
    "delete": "不删除",
}
#: 规范名选项 / 按原文选项的文案也要分操作——删除路径上写"记录为布洛芬"会误导用户，
#: 他会以为自己点的是新增。
_OPERATION_CANONICAL_LABEL: dict[str, str] = {
    "add": "记录为{name}",
    "update": "更新{name}的记录",
    "delete": "删除{name}的记录",
}
_OPERATION_RAW_LABEL: dict[str, str] = {
    "add": "按原文记录（{name}）",
    "update": "按原文更新（{name}）",
    "delete": "按原文删除（{name}）",
}


def _cancel_label(operation: str) -> str:
    return _OPERATION_CANCEL.get(operation, _OPERATION_CANCEL["add"])


def _canonical_label(operation: str, name: str) -> str:
    tpl = _OPERATION_CANONICAL_LABEL.get(operation, _OPERATION_CANONICAL_LABEL["add"])
    return tpl.format(name=name)


def _raw_label(operation: str, name: str) -> str:
    tpl = _OPERATION_RAW_LABEL.get(operation, _OPERATION_RAW_LABEL["add"])
    return tpl.format(name=name)

#: 选项序号：字母或数字，允许尾随一个分隔符（"A" / "a." / "2、" / "B）"）
_INDEX_RE = re.compile(r"^([a-dA-D1-4])\s*[.、,，)）:：]?$")
_PUNCT_RE = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’()（）\[\]【】]")


def _clean(value: object) -> str:
    """归一化可能来自 LLM 的空值。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in ("未指定", "None", "null", "nan") else text


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", (text or "").strip().lower())


def _renumber(options: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, opt in enumerate(options[:MAX_LISTED_OPTIONS]):
        item = dict(opt)
        item["id"] = OPTION_LETTERS[i]
        out.append(item)
    return out


def other_option_id(options: list[dict]) -> str:
    """隐式"其他"选项的序号（跟在已列选项之后）。"""
    idx = min(len(options), len(OPTION_LETTERS) - 1)
    return OPTION_LETTERS[idx]


def build_options(*, draft: dict, raw_input: str, operation: str = "add") -> list[dict]:
    """生成确认选项。返回空列表 = 无歧义且字段齐全，调用方应直接落库。

    四类触发条件（互斥，按优先级）：
        A. 多药同句   —— 原文提到 ≥2 个规范药名，但一轮只会落成一条记录
        B. 别名/商品名 —— 抽出的名字能解析到**不同**的规范名（感康 → 复方氨酚烷胺片）
        C. 词典未收录 —— 药名不在知识库中
        D. 字段缺失   —— **仅 add**：药名无歧义，但缺剂量/频次（缺项会被写成"未指定"）
    """
    cancel = _cancel_label(operation)
    drug_name = _clean(draft.get("drug_name"))
    if not drug_name:
        # 没有药名是另一类问题（提示用户补全），不该走确认卡
        return []

    dictionary = get_dictionary()
    if not dictionary.is_ready():
        # 词典未加载（DB 异常时会静默降级为空词典）时无法区分"歧义"与"未收录"。
        # 此时**不做确认**：给每一次记录都弹卡是比现状更差的回归。
        return []

    mentioned = [
        c for c in dictionary.candidates_for(raw_input, kinds=("drug",)) if not c["negated"]
    ]

    # A. 多药同句：DrugRecordInfo.drug_name 是标量，LLM 要么只取一个、要么把两个名字
    #    拼成一个怪字符串（"布洛芬和阿司匹林"），两种都会写坏档案。
    if len(mentioned) >= 2:
        names = [c["canonical_name"] for c in mentioned[:MAX_DRUG_OPTIONS]]
        return _renumber(
            [{"label": n, "value": n, "kind": "drug"} for n in names]
            + [{"label": cancel, "value": CANCEL_VALUE, "kind": "action"}]
        )

    canonical = dictionary.canonical_name_for(drug_name, kinds=("drug",))

    # B. 别名/商品名命中：用户写"感康"，档案里存"复方氨酚烷胺片"——这是医疗语义上的
    #    替换，不能让代码替用户决定。
    if canonical is not None and normalize_text(canonical) != normalize_text(drug_name):
        return _renumber(
            [
                {"label": _canonical_label(operation, canonical), "value": canonical, "kind": "drug"},
                {"label": _raw_label(operation, drug_name), "value": drug_name, "kind": "drug"},
                {"label": cancel, "value": CANCEL_VALUE, "kind": "action"},
            ]
        )

    # C. 词典未收录：把未经知识库校验的药名写进医疗档案前先确认一次
    if canonical is None:
        return _renumber(
            [
                {"label": _raw_label(operation, drug_name), "value": drug_name, "kind": "drug"},
                {"label": cancel, "value": CANCEL_VALUE, "kind": "action"},
            ]
        )

    # D. 药名无歧义，但关键字段缺失。**仅新增**：add_record 会把缺项落成"未指定"，
    #    用户有权在写库前看到这一点；而 update/delete 只作用在用户明确点到的字段上
    #    （update_by_name 跳过空值与"未指定"），缺字段不代表任何风险，不需要确认。
    missing = [f for f in ("dosage", "frequency") if not _clean(draft.get(f))]
    if missing and operation == "add":
        return _renumber(
            [
                {"label": "确认记录", "value": CONFIRM_VALUE, "kind": "action"},
                {"label": cancel, "value": CANCEL_VALUE, "kind": "action"},
            ]
        )

    # E. 无歧义且齐全 → 直接落库
    return []


def preferred_drug_name(*, raw_input: str, fallback: str) -> str:
    """定出这次写操作真正作用在哪个药名上。

    `DrugEntityExtractor.extract_drug_candidates` 是**整句兜底**的：抽不到已知药名时
    会把整段输入原样当候选返回（"删除布洛芬的记录" → 候选就是"删除布洛芬的记录"）。
    拿这种名字去删/更新必然落空，更糟的是它会被当成"词典未收录的药名"去出确认卡，
    卡片上写着"按原文删除（删除布洛芬的记录）"——比不确认还糟。

    所以词典能解析出药名时一律以词典为准，抽不出任何药名才退回 fallback。
    """
    names = [
        c["canonical_name"]
        for c in get_dictionary().candidates_for(raw_input, kinds=("drug",))
        if not c["negated"]
    ]
    return names[0] if names else fallback


def build_card_message(*, draft: dict, options: list[dict], operation: str = "add") -> str:
    """渲染确认卡。

    刻意把选项**也列成文本**：`stream=false` 的客户端没有按钮，照着打字仍然可用；
    SSE 客户端会额外收到 options 事件渲染按钮，两者不冲突。
    """
    cancel = _cancel_label(operation)
    lines = [_OPERATION_LEAD.get(operation, _OPERATION_LEAD["add"]), ""]
    for key, label in _FIELD_LABELS:
        value = _clean(draft.get(key))
        if value:
            lines.append(f"• {label}：{value}")
        elif operation == "add":
            # 只有新增才会把缺项真的落成"未指定"，所以只有新增需要显示它。
            # update 只改用户点到的字段、delete 只按药名定位，显示"未指定"会让人
            # 以为这次操作会把该字段清空。
            lines.append(f"• {label}：未指定")

    lines.append("")
    has_drug_option = any(o.get("kind") == "drug" for o in options)
    lines.append("药名需要您确认一下：" if has_drug_option else "请确认：")
    for opt in options:
        lines.append(f"{opt['id']}. {opt['label']}")
    lines.append(f"{other_option_id(options)}. 其他（请直接输入药品名）")

    lines.append("")
    lines.append(f"回复对应字母即可；也可以直接输入药品名，或回复“{cancel}”取消。")
    return "\n".join(lines)


def resolve_answer(user_input: str, pending: dict) -> tuple[str, dict | None]:
    """解析用户对确认卡的回答。

    返回 `(action, payload)`，action ∈ {"select", "affirm", "deny", "unrelated"}：
        select   → 选中了某个具体药名（payload 为该选项 dict）
        affirm   → 确认按 draft 落库（payload 是卡片上的"确认"选项 dict）
        deny     → 取消，不落库（payload 为 None）
        unrelated→ 答非所问，调用方决定重问还是放弃（payload 为 None）
    """
    options = pending.get("options") or []
    text = (user_input or "").strip()
    norm = _normalize(text)

    # 1) 序号选择（前端按钮发的是 label，这条主要服务手输 "A" / "1" / "B."）
    m = _INDEX_RE.match(text)
    if m:
        idx = _index_of(m.group(1))
        if 0 <= idx < len(options):
            return _option_action(options[idx])

    # 2) 与某个选项 label 精确相等（前端按钮走这条）
    for opt in options:
        if norm and norm == _normalize(str(opt.get("label") or "")):
            return _option_action(opt)

    # 3) 纯肯定/否定词。肯定词**只在卡片上确有"确认"动作选项时**才成立：
    #    多药同句那张卡问的是"选哪个药"，一句"嗯"没指定任何药名，按 affirm 处理等于
    #    替用户挑了一个——删除场景下就是删错药。这种回答按答非所问走 attempts 计数。
    if norm in AFFIRMATIVE:
        confirm_opt = next(
            (o for o in options if o.get("kind") == "action" and o.get("value") == CONFIRM_VALUE),
            None,
        )
        if confirm_opt is not None:
            return ("affirm", confirm_opt)
        return ("unrelated", None)
    if norm in NEGATIVE:
        return ("deny", None)

    # 4) other：自由文本里能唯一解析出药名 → 视为用户在更正药名
    names = get_dictionary().resolve(text, kinds=("drug",), drop_negated=True)
    if len(names) == 1:
        return ("select", {"id": "", "label": names[0], "value": names[0], "kind": "drug"})

    return ("unrelated", None)


def _index_of(token: str) -> int:
    t = token.lower()
    return int(t) - 1 if t.isdigit() else "abcd".index(t)


def _option_action(opt: dict) -> tuple[str, dict | None]:
    if opt.get("kind") != "action":
        return ("select", opt)
    if opt.get("value") == CANCEL_VALUE:
        return ("deny", None)
    return ("affirm", opt)
