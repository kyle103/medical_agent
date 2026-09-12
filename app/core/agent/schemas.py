"""决策类 LLM 调用的输出 schema（Pydantic）。

替代 `llm_decision_service.py` 里"正则捞 JSON + 手写 valid_xxx 集合"的做法：
- 枚举交给 schema（`intent` / `target_type` 是静态 `Literal`，能进 JSON Schema 的 enum）
- `target_name` 由 `CAPABILITY_REGISTRY` **代码生成**校验，不手抄（registry 加工具时自动同步）
- 校验失败会带**字段路径**抛错，可作为反馈回灌给模型重试（见 `LLMService.chat_completion_json`）

## 两条设计约束（避免踩坑）

1. **schema 里不出现自由形态对象**（如 `dict[str, Any]`）。
   strict json_schema 要求对象声明 `properties` 并把它们全部列入 `required`；
   自由形态对象既无法满足严格模式，也让"模型该输出什么"变得不可断言。
   → 实体字段一律**拍平**为具名字段，由 service 层再组装成原有的 `entities` 字典。

2. **数值范围不在 schema 里硬约束**。`confidence` 用裸 `float` 接收，由 service 层
   `max(min(x, 1.0), 0.0)` 收敛——这是改造前的既有行为，schema 只负责结构和枚举，
   不顺手改语义（否则一个 1.2 的 confidence 会让整次决策从"可用"变成"丢弃"）。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from app.core.agent.capabilities import valid_target_names

# 静态枚举：直接进 JSON Schema 的 enum，模型生成不出非法值
Intent = Literal["archive", "drug", "lab", "general"]
TargetType = Literal["agent", "tool"]

# 统一配置：extra="forbid" 让 Pydantic 生成 additionalProperties: false，
# 配合 structured_output.strictify 满足 strict 模式的两条硬要求。
_STRICT = ConfigDict(extra="forbid")

# --- 容错类型 ---------------------------------------------------------------
#
# 改造前的手写解析对"字段值为 null"是宽容的（大量 `data.get(k, "") or ""`）。
# 换成 schema 后若用裸 `str` / `float`，一个 null 就会让**整次**校验失败——
# 那是比改造前更差的行为，所以这里用 before-validator 把 null 收敛掉。
#
# 只处理 None 与非列表容器，**不做类型强转**（"123" 不会被当成数字）。


def _none_to_empty_str(v: Any) -> Any:
    return "" if v is None else v


def _none_to_zero(v: Any) -> Any:
    return 0.0 if v is None else v


def _to_str_list(v: Any) -> Any:
    """非 list（含 None、裸字符串）一律收敛成 []。

    对齐改造前 `isinstance(x, list)` 不成立即跳过的判断；元素统一 `str()`，
    对齐旧代码的 `str(n).strip()`（数字被写成字符串时不至于让整次校验失败）。
    """
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


def _to_obj_list(v: Any) -> Any:
    """list 里只保留 dict 元素；非 list 收敛成 []。

    改造前 `_extract_lab_entities` 是对 lab_items 原样透传，非 dict 元素要到下游
    消费时才炸。这里提前滤掉，把失败点前移到校验层。
    """
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    return []


Text = Annotated[str, BeforeValidator(_none_to_empty_str)]
Number = Annotated[float, BeforeValidator(_none_to_zero)]
TextList = Annotated[list[str], BeforeValidator(_to_str_list)]


def _ensure_registered_target(v: str) -> str:
    """target_name 必须是注册表里的名字。

    为何不用 `Literal[...]` 动态生成：合法取值来自运行期注册表，用 `field_validator`
    更直接，且错误信息可以带上候选列表——这对"回灌给模型重试"很关键。
    """
    names = valid_target_names()
    if v not in names:
        raise ValueError(f"target_name 必须是 {list(names)} 之一，收到 {v!r}")
    return v


class LabItem(BaseModel):
    """化验单条目。字段名保持与既有 `lab_items` 结构一致（勿改，下游按此取值）。"""

    model_config = _STRICT

    item_name: Text = ""
    test_value: Text = ""
    unit: Text = ""


# lab_items 的列表类型：先滤掉非 dict 元素，再逐个校验成 LabItem
LabItems = Annotated[list[LabItem], BeforeValidator(_to_obj_list)]


class RouteDecision(BaseModel):
    """`classify_intent_and_route` 的输出。"""

    model_config = _STRICT

    intent: Intent
    intent_type: Text = ""
    target_type: TargetType
    target_name: Text
    confidence: Number = 0.0
    reason: Text = ""

    @field_validator("target_name")
    @classmethod
    def _v_target_name(cls, v: str) -> str:
        return _ensure_registered_target(v)


class RouteAndEntities(BaseModel):
    """`classify_route_and_extract` 的输出：路由 + 实体，一次调用。

    实体字段**拍平**在此模型上（而非嵌套一个自由形态 `entities` 对象），
    由 service 层按 intent 重新组装成既有契约里的 `entities` 字典。
    """

    model_config = _STRICT

    intent: Intent
    intent_type: Text = ""
    target_type: TargetType
    target_name: Text
    confidence: Number = 0.0
    reason: Text = ""
    is_multi_intent: bool = False

    # 药物类实体
    drug_name_list: TextList = Field(default_factory=list)
    dosage: Text = ""
    frequency: Text = ""
    start_date_text: Text = ""
    purpose: Text = ""

    # 化验类实体
    lab_items: LabItems = Field(default_factory=list)

    @field_validator("target_name")
    @classmethod
    def _v_target_name(cls, v: str) -> str:
        return _ensure_registered_target(v)


class BatchRouteItem(BaseModel):
    """批量路由的单个元素：**故意放宽枚举约束**。

    原因：批量路由在改造前的语义是**逐项容错**——某一条不合法，只把该条置 `None`，
    其余照常返回。若这里直接用 `RouteDecision`（`Literal` 硬约束）做元素类型，
    一条非法就会让**整批**校验失败、"逐项降级"退化成"整批降级"——比改造前更差。

    因此这里只约束**结构**（字段齐全、无多余字段），
    枚举合法性与注册表校验交由 service 层逐条 `RouteDecision.model_validate()` 执行。
    """

    model_config = _STRICT

    intent: str
    intent_type: Text = ""
    target_type: str
    target_name: Text
    confidence: Number = 0.0
    reason: Text = ""


class BatchRouteDecision(BaseModel):
    """`batch_route_queries` 的输出。

    包一层 `decisions` 对象而不是直接用数组根：strict json_schema 要求根类型是 object，
    数组根会被拒。因此提示词也相应要求输出 `{"decisions": [...]}`。
    """

    model_config = _STRICT

    decisions: list[BatchRouteItem] = Field(default_factory=list)


# ===========================================================================
# Step 2.5：P1/P2 调用点的契约
#
# 这一批改造前都是「正则捞 JSON → 手写 valid_xxx 集合 → 逐条过滤」。
# 换成 schema 时踩的是同一类坑，两条规则先说在前面：
#
# 1. **字典键要提升为字段。** `{"s1": {...}, "s2": {...}}` 这种以步骤 id 为键的
#    动态字典，strict json_schema 表达不了（对象必须声明 properties 并全部 required）。
#    → 统一改成数组 + 显式 `step_id`（本模块顶部约束 1 的具体应用）。
#
# 2. **批量元素里的枚举一律放宽成 `str`。** 改造前这些位置是**逐条容错**，
#    用 `Literal` 会让一条非法导致整批校验失败。枚举合法性与注册表校验
#    交由 service 层逐条执行——与 `BatchRouteItem` 同一套理由。
# ===========================================================================


class StepRouteItem(BaseModel):
    """带显式 `step_id` 的路由元素（替代改造前的 `"s1"` 字典键）。

    只约束结构，`intent` / `target_type` / `target_name` 都是裸 `str`——
    理由见上方第 2 条与 `BatchRouteItem` 的文档。
    """

    model_config = _STRICT

    step_id: Text
    intent: str
    intent_type: Text = ""
    target_type: str
    target_name: Text
    confidence: Number = 0.0
    reason: Text = ""


class DepsItem(BaseModel):
    """依赖声明：`step_id` 这一步依赖 `depends_on` 里的步骤。"""

    model_config = _STRICT

    step_id: Text
    depends_on: TextList = Field(default_factory=list)


class BatchRouteWithDeps(BaseModel):
    """`batch_route_with_deps` 的输出（改造前是 `{"routes": {...}, "deps": {...}}`）。"""

    model_config = _STRICT

    routes: list[StepRouteItem] = Field(default_factory=list)
    deps: list[DepsItem] = Field(default_factory=list)


class SplitRouteDeps(BaseModel):
    """`split_route_deps` 的输出：拆分 + 逐条路由 + 依赖，一次调用完成。"""

    model_config = _STRICT

    sub_queries: TextList = Field(default_factory=list)
    routes: list[StepRouteItem] = Field(default_factory=list)
    deps: list[DepsItem] = Field(default_factory=list)


class ReplanAction(BaseModel):
    """重规划的单个动作。

    `action` **故意用裸 `str`**（而非 `Literal["retry", ...]`）：改造前是逐条过滤
    非法动作，用 `Literal` 会让一条非法导致整批校验失败——同上第 2 条。
    """

    model_config = _STRICT

    step_id: Text
    action: str
    query: Text = ""
    target_name: Text = ""
    reason: Text = ""


class ReplanResult(BaseModel):
    """`replan_failed_steps` 的输出。"""

    model_config = _STRICT

    actions: list[ReplanAction] = Field(default_factory=list)


class QuerySplit(BaseModel):
    """`split_queries` 的输出。

    改造前是**数组根** `["q1", "q2"]`；strict json_schema 要求根类型是 object，
    数组根会被拒 → 包一层 `queries`，提示词同步改为输出对象。
    """

    model_config = _STRICT

    queries: TextList = Field(default_factory=list)


class DrugEntities(BaseModel):
    """`_extract_drug_entities` 的输出。"""

    model_config = _STRICT

    drug_name_list: TextList = Field(default_factory=list)
    dosage: Text = ""
    frequency: Text = ""
    start_date_text: Text = ""
    purpose: Text = ""


class LabEntities(BaseModel):
    """`_extract_lab_entities` 的输出。"""

    model_config = _STRICT

    lab_items: LabItems = Field(default_factory=list)


class DrugRecordInfo(BaseModel):
    """`extract_drug_info` 的输出。"""

    model_config = _STRICT

    drug_name: Text = ""
    dosage: Text = ""
    frequency: Text = ""
    start_date_text: Text = ""
    purpose: Text = ""
