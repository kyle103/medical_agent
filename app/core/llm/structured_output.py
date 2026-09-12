"""结构化输出的能力层：决定向提供方下发哪种 response_format。

## 为什么需要这一层

实测（2026-09-12，百炼 compatible-mode）：**`response_format` 是否被真正履行，取决于具体模型快照**。

- 支持：`deepseek-v4-flash` / `deepseek-v4-pro` / `deepseek-v3.2` / `qwen3.8-flash` / `qwen3.7-flash` 等
- **静默不生效**：`deepseek-v4-flash-0731`（带日期快照）/ `qwen3.6-flash`
  —— 接口不报错、返回的 JSON 也合法，但字段名 / 枚举 / 类型**全不受 schema 约束**。
  这是最危险的一类失败：不报错、无信号，你以为约束上了其实没有。

所以「模型支持某能力」是**部署属性**，不是能写进代码的契约。本模块把它变成一个
**运行时确定的档位**，并强制在启动日志里打印，使"静默降级"变成"看得见的降级"。

## 档位

| 档位 | response_format | 含义 |
|---|---|---|
| `json_schema` | strict schema | 服务端约束解码，非法结构生成不出来（首选） |
| `json_object` | `{"type":"json_object"}` | 只保证顶层是合法 JSON；字段对齐靠 prompt + 客户端校验 |

`STRUCTURED_OUTPUT_MODE`：`auto`（默认，探针判定并缓存）/ `json_schema` / `json_object`。

> **无论哪个档位，客户端 Pydantic 校验都是不变量**（见 `LLMService.chat_completion_json`）。
> 本模块只决定"能不能额外拿到一层服务端约束"，不决定正确性。
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, Literal

from app.common.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

StructuredMode = Literal["json_schema", "json_object"]

# 能力探针的判据字段名：在 schema 里声明，且 **绝不出现于任何 prompt**。
# 模型若返回它，唯一的解释是 schema 被真正下发到了解码层（猜不出来）。
_PROBE_FIELD = "zz_schema_capability_probe"

# (base_url, model) -> 档位。进程内缓存，避免每次决策都探针。
_MODE_CACHE: dict[tuple[str, str], StructuredMode] = {}
_PROBE_LOCK = asyncio.Lock()

# 最近一次确定出的档位（便于运行期自查 / 测试断言）
_last_mode: StructuredMode | None = None
_last_mode_source: str = ""


def _llm_configured() -> bool:
    def _ok(v: str) -> bool:
        v = (v or "").strip()
        return bool(v) and not (v.startswith("{{") and v.endswith("}}"))

    return _ok(settings.LLM_API_BASE) and _ok(settings.LLM_API_KEY) and _ok(settings.LLM_MODEL_NAME)


def configured_mode() -> str:
    """读取配置档位，非法值回退 auto。"""
    raw = (getattr(settings, "STRUCTURED_OUTPUT_MODE", "auto") or "auto").strip().lower()
    return raw if raw in ("auto", "json_schema", "json_object") else "auto"


def current_mode() -> StructuredMode | None:
    return _last_mode


def current_mode_source() -> str:
    return _last_mode_source


def reset_capability_cache() -> None:
    """清空能力缓存与最近状态（测试用）。"""
    global _last_mode, _last_mode_source
    _MODE_CACHE.clear()
    _last_mode = None
    _last_mode_source = ""


def _cache_key(model: str) -> tuple[str, str]:
    return ((settings.LLM_API_BASE or "").strip().rstrip("/"), (model or "").strip())


async def probe_schema_support(
    client: Any, model: str, *, timeout_s: float = 10.0, attempts: int = 2
) -> bool | None:
    """极小请求探针：判定该模型是否真的把 json_schema 下发到了解码层。

    返回三态，**不要当成 bool 用**：
      - `True`  → 确认支持
      - `False` → 确认不支持（内容完整但判据字段缺席）
      - `None`  → **无结论**（截断 / 空返回 / 调用异常用尽重试）
    区分 `False` 与 `None` 的意义：`False` 可以缓存，`None` **不能**——
    否则一次网络抖动就会被当成"该模型不支持"永久缓存，又是一次静默降级。

    超时与重试保持较小（10s × 2）：探针会阻塞调用方，不能让它拖慢决策链路。


    判据是**无提示的独有字段名**——它只出现在 schema 中，prompt 里没有。
    因此"返回里出现它"只有一种解释：schema 确实被下发了。

    判定顺序与两条**实测得来**的防假阴性规则（都很反直觉，改之前先读这里）：

    1. **判据落在原始文本的子串匹配上，不要求能解析成合法 JSON。**
       输出被截断时 JSON 必然不合法；若用"能否解析"做判据，就会把支持 schema 的
       模型误判为不支持。子串匹配能救回"字段名之后才被截断"这一类（如
       `{"zz_schema_capability_probe": "阿司匹`）。

    2. **必须区分「无结论」与「否」—— 截断与空返回都属无结论。**
       实测：同一模型连打 6 次有 1 次 `content=''` 且 `finish_reason='length'`；
       另有 `max_tokens` 偏小导致字段名**本身**被截断（`{"zz_schema_capability_`）的情况，
       此时文本非空但不含判据字段——若按"否"处理就会误判。
       因此 `finish_reason == "length"` 一律重试，绝不当作否定结论。

    判定优先级：
      ① 命中判据字段            → True（决定性）
      ② 截断 / 空返回 / 调用异常 → 无结论，重试
      ③ 内容完整且判据字段缺席   → False（决定性）
    全部无结论时返回 None —— 调用方本次退到 json_object，但**不缓存**结论。
    """
    schema = {
        "type": "object",
        "properties": {_PROBE_FIELD: {"type": "string"}},
        "required": [_PROBE_FIELD],
        "additionalProperties": False,
    }
    last_note = "未执行"

    for i in range(max(1, attempts)):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "只输出一个 JSON 对象，不要任何解释文字。"},
                        {"role": "user", "content": "用两个字回应：你好"},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "capability_probe", "strict": True, "schema": schema},
                    },
                    # 与生产路径（chat_completion_json）保持一致：探针要测的就是实际跑的那条路。
                    # 顺带避免推理吃满 max_tokens 造成截断，进而污染三态判定。
                    **(
                        {"extra_body": {"enable_thinking": False}}
                        if getattr(settings, "LLM_DISABLE_THINKING", True)
                        else {}
                    ),
                ),
                timeout=timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            last_note = f"调用异常 {type(e).__name__}: {str(e)[:160]}"
            logger.warning("[structured] 能力探针第 %d/%d 次失败: %s", i + 1, attempts, last_note)
            continue

        raw = ""
        finish_reason = None
        try:
            choice = resp.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
        except Exception as e:  # noqa: BLE001
            last_note = f"响应结构异常 {type(e).__name__}"
            continue

        # ① 命中判据字段 → 决定性为真（即便后续被截断也算数）
        if _PROBE_FIELD in raw:
            return True

        # ② 截断 → 无结论。此时字段名可能被从中间切断，不足以判定"不支持"
        if finish_reason == "length":
            last_note = f"输出被截断(finish_reason=length, len={len(raw)})"
            logger.info("[structured] 能力探针第 %d/%d 次输出被截断（无结论），重试", i + 1, attempts)
            continue

        # ② 空返回 → 无结论（实测偶发）
        if not raw.strip():
            last_note = "空返回"
            logger.info("[structured] 能力探针第 %d/%d 次返回空内容（无结论），重试", i + 1, attempts)
            continue

        # ③ 内容完整但判据字段缺席 → 决定性结论：schema 未下发
        logger.info(
            "[structured] 能力探针判定：内容完整但判据字段缺席 → schema 未下发（content=%r）", raw[:120]
        )
        return False

    logger.warning(
        "[structured] 能力探针 %d 次均无有效结论（最后：%s）→ 本次退到 json_object，但不缓存该结论",
        attempts,
        last_note,
    )
    return None


async def resolve_mode(client: Any, model: str) -> StructuredMode:
    """确定当前应使用的档位。

    auto 模式下用探针判定；**只缓存明确结论**（True/False），
    "无结论"（探测被截断 / 网络抖动）只在本次退到 json_object，
    不写缓存，下次调用会重新探。这样既不会因一次抖动永久降级，
    也不会把不确定的结论伪装成确定结论。
    """
    global _last_mode, _last_mode_source

    cfg = configured_mode()
    if cfg in ("json_schema", "json_object"):
        _last_mode, _last_mode_source = cfg, "config"
        return cfg

    key = _cache_key(model)
    cached = _MODE_CACHE.get(key)
    if cached is not None:
        _last_mode, _last_mode_source = cached, "auto-probe(cached)"
        return cached

    async with _PROBE_LOCK:
        cached = _MODE_CACHE.get(key)
        if cached is not None:
            _last_mode, _last_mode_source = cached, "auto-probe(cached)"
            return cached

        verdict = await probe_schema_support(client, model)

        if verdict is None:
            # 无结论：本次保守使用 json_object，但绝不缓存。
            # 若把 None 当成 False 缓存，一次网络抖动就会让整个进程永久停在低档位——
            # 那正是本项目一直在消除的"静默降级"。
            _last_mode, _last_mode_source = "json_object", "auto-probe(inconclusive)"
            logger.warning(
                "[structured] 能力探针无结论，本次使用 json_object（不缓存，后续会重试探针）model=%s", model
            )
            return "json_object"

        mode: StructuredMode = "json_schema" if verdict else "json_object"
        _MODE_CACHE[key] = mode
        _last_mode, _last_mode_source = mode, "auto-probe"
        logger.info(
            "[structured] 能力探针结果: mode=%s model=%s base_url=%s"
            + ("" if verdict else "（该模型快照未下发 schema，退化为 json_object；正确性由客户端校验保障）"),
            mode,
            model,
            key[0],
        )
        return mode


async def warmup(client: Any = None, *, budget_s: float = 25.0) -> StructuredMode | None:
    """启动预热：提前确定档位并打日志，避免首个用户请求承担探针延迟。

    整体套 `budget_s` 上限：提供方不可达时不能把启动流程卡住
    （探针自身最坏 10s × 2 次，25s 足够覆盖）。
    失败不抛异常（启动流程不应因它中断）；后续调用会惰性重试。
    """
    if not _llm_configured():
        logger.info("[structured] 未配置 LLM，跳过结构化输出预热")
        return None
    try:
        if client is None:
            from app.core.llm.llm_service import get_shared_client

            client = await get_shared_client()
        mode = await asyncio.wait_for(resolve_mode(client, settings.LLM_MODEL_NAME), timeout=budget_s)
        logger.info(
            "[structured] 生效档位=%s（来源=%s）model=%s",
            mode,
            current_mode_source(),
            settings.LLM_MODEL_NAME,
        )
        return mode
    except Exception as e:  # noqa: BLE001
        logger.warning("[structured] 预热失败，首个请求将惰性确定档位: %s", e)
        return None


def strictify(schema: dict) -> dict:
    """把 Pydantic 生成的 JSON Schema 调整为"严格模式友好"的形式。

    OpenAI strict 的两条硬要求，Pydantic 默认都不满足：
      1. 每个 object 都要 `additionalProperties: false`（`extra="forbid"` 可满足，但不保证）
      2. **所有**属性都必须出现在 `required` 里（Pydantic 会漏掉带默认值的字段）

    因此这里统一补齐。注意：只对**声明了 properties 的 object** 关掉额外属性，
    自由形态对象（没有 properties）保持原样，避免把 `dict[str, Any]` 之类的字段锁死。
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and props:
                node["additionalProperties"] = False
                node["required"] = list(props.keys())
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    out = copy.deepcopy(schema)
    _walk(out)
    return out


def build_response_format(schema_cls: type, mode: StructuredMode, *, name: str | None = None) -> dict:
    """按档位构造 response_format 参数。"""
    if mode == "json_schema":
        raw = schema_cls.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name or getattr(schema_cls, "__name__", "structured_output"),
                "strict": True,
                "schema": strictify(raw),
            },
        }
    return {"type": "json_object"}


def extract_json_candidate(raw: str) -> tuple[str, bool]:
    """从模型原始输出中取出最可能的 JSON 文本。

    返回 `(文本, 是否走了兜底提取)`。兜底命中会被计数并打日志——
    这是"response_format 未被履行"的可观测信号：若该计数长期为 0，说明约束真的生效了。

    存在的意义：档位退化到 `json_object` 时，模型仍可能加代码围栏或前置解释；
    相比直接判失败，先尝试提取能避免把可救的响应丢掉（客户端校验仍是不变量）。
    """
    s = (raw or "").strip()
    if not s:
        return "", False

    # 1) 本身就是合法 JSON
    try:
        json.loads(s)
        return s, False
    except Exception:  # noqa: BLE001
        pass

    # 2) 剥 ``` 代码围栏
    if s.startswith("```"):
        body = s
        first_nl = body.find("\n")
        if first_nl != -1:
            body = body[first_nl + 1 :]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        body = body.strip()
        if body:
            try:
                json.loads(body)
                return body, True
            except Exception:  # noqa: BLE001
                pass
            s = body

    # 3) 取最外层 {} 或 []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        i, j = s.find(open_ch), s.rfind(close_ch)
        if i != -1 and j > i:
            cand = s[i : j + 1]
            try:
                json.loads(cand)
                return cand, True
            except Exception:  # noqa: BLE001
                continue

    return s, True
