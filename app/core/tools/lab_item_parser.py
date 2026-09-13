"""化验指标文本解析：从自然语言里抽出「指标名 + 数值」对。

为什么单独成模块
----------------
抽取结果同时被两处消费，放在中立模块里可避免 `planner_agent <-> tool_executor` 互相 import：

1. `PlannerAgent._route_by_intent_and_text` —— 判断这条消息**是否包含可解读的指标数据**，
   据此决定要不要路由到 `lab_report` 工具；
2. `ToolExecutor._execute_lab_report` —— 真正把指标交给 `LabReportTool`。

设计前提（**安全相关，改这里务必先读**）
----------------------------------------
抽错一个数值，下游 `LabReportTool` 会拿它去和参考范围比对，并输出明确的
**H/L 异常判定**。也就是说，如果把「我今年 35 岁」里的 35 当成血糖值，系统会
一本正经地告诉用户"血糖偏高"。这不是体验问题，是**错误的医疗判断**。

因此本模块遵循一条不可让步的原则：**宁可漏抽，不可错抽。**
任何拿不准的数值一律不产出条目 —— 漏抽会退化成通用问答（可接受），
错抽会输出错误结论（不可接受）。

历史坑（2026-09-13 实测，原文见 `重构方案-Agent范式四步补强.md` 附一之九）
--------------------------------------------------------------------
旧实现有两条产错路径，都会静默给出错误解读：

- 把 `是` / `为` 当键值分隔符 → `"血糖偏高是怎么回事"` 被拆成
  `血糖偏高 = 怎么回事`，非数值也当成了检验值；
- 兜底逻辑「匹配到指标名后，抓整句里第一个数字挂上去」→
  `"血糖不高，我今年35岁"` → `血糖 = 35`；`"我血糖正常，但体重70公斤"` → `血糖 = 70`；
  `"血常规里白细胞高，我发烧38度5"` → `白细胞 = 38`。且多个指标会共用同一个数字。
"""

from __future__ import annotations

import re

#: 能直接对上参考库的高频指标。命中后在其**紧邻子句窗口**内找数值。
COMMON_LAB_ITEMS: tuple[str, ...] = (
    "血糖", "血压", "血脂", "胆固醇", "肝功能", "肾功能",
    "白细胞", "红细胞", "血小板", "尿酸", "转氨酶",
)

#: 子句边界：到这里就不再往后找数值，避免跨句把无关数字套到指标上。
_CLAUSE_BOUNDARY = "，,。；;！!？?\n\r\t"

#: 数值：整数 / 小数，或「数字/数字」（如血压 120/80）。
_VALUE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)")

#: 数字后面紧跟这些量词（或单位）时，说明它**不是检验值**：
#: 年龄、体重、体温、天数、饮酒量、次数…… 出现在此处即拒绝该数字。
_NON_LAB_UNITS: tuple[str, ...] = (
    "岁", "公斤", "kg", "KG", "Kg", "斤", "度", "天", "年", "个月", "周",
    "次", "片", "粒", "杯", "瓶", "包", "两", "米", "cm", "CM", "个", "只",
    "位", "名", "小时", "分钟", "段", "层", "级", "期", "条", "号",
)

#: 显式键值分隔符。**故意不含「是」「为」**：它们在自然句里太常见
#: （"我是不是""为什么""胆固醇偏高怎么办"），当分隔符会造出大量假数值。
_EXPLICIT_SEPARATORS = "：:=＝"

#: 显式写法里合法的指标名：2~10 个中英文字符（末尾不再含括号内容，见 `_trim_tail`）。
_NAME_TAIL_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]{1,12}$")

#: 名称后紧跟的括号备注，如「白细胞计数（WBC）」→ 剔除「（WBC）」
_TRAILING_PAREN_RE = re.compile(r"[（(][^（()）]{0,20}[）)]\s*$")

#: 指标名至少 2 字：1 个汉字的"名"几乎都是误切（如"是"→"值"）
_MIN_NAME_LEN = 2
_MAX_NAME_LEN = 10

#: 「名称」位置上的词其实不是检验指标：报告名 / 泛称。命中的显式键值对直接丢弃，
#: 否则会出现「血常规：白细胞11.2」被解析成 `血常规 = 11.2` 这种噪音条目。
_NON_ITEM_NAMES: tuple[str, ...] = (
    "血常规", "尿常规", "大便常规", "化验单", "化验", "检验", "检查单",
    "报告单", "报告", "体检", "指标", "结果", "数值", "参考范围", "正常值",
)

#: 以这些词结尾的「名称」同样是泛称而非指标（如「化验结果」「参考数值」）。
_NON_ITEM_NAME_SUFFIXES: tuple[str, ...] = (
    "结果", "数值", "指标", "报告", "参考范围", "正常值",
)

#: 单个指标的搜索窗口长度（字符）。够覆盖「血压（坐位）120/80」这类写法，
#: 又不至于跨到下一句。
_WINDOW_LIMIT = 16


def _window(text: str, start: int, limit: int = _WINDOW_LIMIT) -> str:
    """取 `text[start:]` 中到最近子句边界为止的片段（长度上限 `limit`）。"""
    rest = text[start:start + limit]
    for i, ch in enumerate(rest):
        if ch in _CLAUSE_BOUNDARY:
            return rest[:i]
    return rest


def _first_lab_value(window: str, *, exclude_items: tuple[str, ...] = ()) -> str | None:
    """在窗口里取第一个**看起来是检验数值**的数字；没有则返回 `None`。

    `exclude_items` 用于排除跨指标的串值：若该数字之前又出现了别的指标名，
    说明这句话的数字归属不明（如「血糖和血压都是6.5」），一律放弃。
    """
    for m in _VALUE_RE.finditer(window):
        if exclude_items:
            prefix = window[: m.start()]
            if any(item in prefix for item in exclude_items):
                continue
        tail = window[m.end():].lstrip()
        if any(tail.startswith(u) for u in _NON_LAB_UNITS):
            continue
        return m.group(1).replace(" ", "")
    return None


def _trailing_name(left: str) -> str:
    """取分隔符左边**紧邻**的名称片段（剔除末尾括号备注）。"""
    cleaned = _TRAILING_PAREN_RE.sub("", left)
    m = _NAME_TAIL_RE.search(cleaned)
    name = m.group(0).strip() if m else ""
    if not (_MIN_NAME_LEN <= len(name) <= _MAX_NAME_LEN):
        return ""
    return name


def _is_non_item_name(name: str) -> bool:
    """判断「名称」位置上的词是泛称/报告名，而不是检验指标。"""
    if name in _NON_ITEM_NAMES:
        return True
    return any(name.endswith(s) for s in _NON_ITEM_NAME_SUFFIXES)


def _explicit_pairs(text: str, *, taken: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """抽显式键值对：`名称：数值` / `名称=数值`（任意指标名，不限白名单）。

    `taken` 是已经被白名单收走的指标名；显式名称与它互相包含时跳过，
    避免「白细胞计数（WBC）：11.2」同时产出 `白细胞` 和 `白细胞计数` 两条重复项。
    """
    out: list[tuple[str, str]] = []
    for i, ch in enumerate(text):
        if ch not in _EXPLICIT_SEPARATORS:
            continue
        name = _trailing_name(text[:i])
        if not name or _is_non_item_name(name):
            continue
        if any(name in t or t in name for t in taken):
            continue
        value = _first_lab_value(_window(text, i + 1))
        if value is None:
            continue
        out.append((name, value))
    return out


def _whitelist_items(text: str) -> list[tuple[str, str]]:
    """抽白名单指标：指标名后**同一子句内**紧跟的数值。"""
    out: list[tuple[str, str]] = []
    for name in COMMON_LAB_ITEMS:
        idx = text.find(name)
        if idx < 0:
            continue
        window = _window(text, idx + len(name))
        others = tuple(n for n in COMMON_LAB_ITEMS if n != name)
        value = _first_lab_value(window, exclude_items=others)
        if value is None:
            continue
        out.append((name, value))
    return out


def infer_unit(test_value: str) -> str:
    """按数值形态粗推单位。缺失比猜错好，所以拿不准就返回空串。"""
    if "/" in test_value:
        return "mmHg"
    try:
        v = float(test_value)
        if "." in test_value and v < 20:
            return "mmol/L"
    except (ValueError, TypeError):
        pass
    return ""


def parse_lab_items(user_input: str) -> list[dict]:
    """从文本里解析化验指标。**不产出任何非数值条目。**

    返回 `[{"item_name", "test_value", "unit"}, ...]`；无可用数据时返回 `[]`。
    """
    text = (user_input or "").strip()
    if not text:
        return []

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    # 白名单优先：它带"指标名紧邻数值"的窗口约束，比通用显式写法更可信
    for name, value in _whitelist_items(text):
        if name in seen:
            continue
        seen.add(name)
        pairs.append((name, value))
    for name, value in _explicit_pairs(text, taken=tuple(seen)):
        if name in seen:
            continue
        seen.add(name)
        pairs.append((name, value))

    return [
        {"item_name": name, "test_value": value, "unit": infer_unit(value)}
        for name, value in pairs
    ]


def has_lab_values(user_input: str) -> bool:
    """文本里是否存在**可解读**的「指标名 + 数值」。

    路由层用它区分两种都含「化验/血常规」的消息：
    - 有数值 → 用户要解读 → 路由到 `lab_report` 工具；
    - 无数值 → 用户在问流程/怎么看/能不能发图片 → 应走通用问答，
      否则工具会零抽取并回一句「请提供指标名称和数值」，答非所问且会反复复读。
    """
    return bool(parse_lab_items(user_input))
