"""Step 2.5 真实链路冒烟：验证 7 处迁移点在**真实模型**下的契约与提示词遵从度。

为什么必须单独做这一步（而不是只信 mock 测试）：
    Step 2.5 与 P0 三处不同——它**改了提示词的输出形态**：
      - batch_route_with_deps / split_route_deps：`{"s1": {...}}` 动态键字典
        → `[{"step_id": "s1", ...}]` 数组
      - split_queries：数组根 `["q1","q2"]` → 包一层 `{"queries": [...]}`
    mock 测试只能证明"**收到新形态时**解析正确"，证明不了"**模型会按新形态输出**"。
    若模型坚持旧形态，schema 会拒 → 重试 → 最终静默返回 None，
    功能退化但不报错——正是本项目一直在消除的那类失败。所以必须实打实跑一次。

用法（在 medical_agent 目录下执行）：
    ./.venv/Scripts/python.exe scripts/probe_step25_smoke.py

会真实调用 LLM（约 8 次请求）。不打印、不落盘任何密钥。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.settings import settings  # noqa: E402
from app.core.agent import llm_decision_service as lds  # noqa: E402
from app.core.agent.llm_decision_service import LLMDecisionService  # noqa: E402
from app.core.llm import structured_output as so  # noqa: E402
from app.core.llm.llm_service import structured_output_stats  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


def dump(x) -> str:
    return json.dumps(x, ensure_ascii=False)


async def main() -> int:
    print(f"model={settings.LLM_MODEL_NAME}  mode_cfg={so.configured_mode()}")
    mode = await so.warmup()
    print(f"生效档位={mode}（来源={so.current_mode_source()}）\n")

    svc = LLMDecisionService()

    # --- 1. batch_route_with_deps：两个子查询，第二个依赖第一个 ---
    qs = ["我的血糖正常吗", "那布洛芬能和它一起吃吗"]
    routes, deps = await svc.batch_route_with_deps(qs)
    # 断言的边界：本脚本验证的是**迁移是否生效**（新数组形态被接受、step_id 被解析、
    # 依赖被正确挂到下标上），不替模型的语义质量背书。
    # 因此"路由条数对齐 + 至少一条成功 + 依赖解析出来"即通过；
    # 被丢弃的条数单独报出来，属于已知的 intent 混淆问题（见 diag_route_enum.py）。
    dropped = sum(1 for r in routes if r is None)
    ok = len(routes) == 2 and any(r is not None for r in routes) and any(d for d in deps)
    detail = f"routes={dump(routes)}\n        deps={dump(deps)}"
    if dropped:
        detail += f"\n        ⚠ {dropped}/{len(routes)} 条路由被丢弃（模型把 intent_type 的取值写进了 intent）"
    record("batch_route_with_deps（数组形态 + 依赖）", ok, detail)

    # --- 2. split_route_deps：一次完成拆分 + 路由 + 依赖 ---
    text = "我的血糖正常吗？布洛芬能和它一起吃吗？另外帮我记录一下今天吃了阿莫西林"
    sub_queries, s_routes, s_deps = await svc.split_route_deps(text)
    # 同上：只看"拆分成功且 routes/deps 与子查询等长"（形态与对齐），
    # 条目的语义合法性由逐条校验负责，不在此断言。
    ok = bool(sub_queries) and len(s_routes or []) == len(sub_queries)
    detail = f"sub_queries={dump(sub_queries)}\n        routes={dump(s_routes)}\n        deps={dump(s_deps)}"
    if ok and sub_queries:
        dropped_n = sum(1 for r in s_routes if r is None)
        if dropped_n:
            detail += f"\n        ⚠ {dropped_n}/{len(sub_queries)} 条路由被丢弃（同上的 intent 混淆）"
    record("split_route_deps（数组形态 + 拆分对齐）", ok, detail)

    # --- 3. split_queries：数组根 → {"queries": [...]} ---
    split = await svc.split_queries("我的血糖正常吗？布洛芬能和它一起吃吗？")
    record("split_queries（对象包裹）", bool(split), f"-> {dump(split)}")

    # --- 4. replan_failed_steps：逐条动作过滤 ---
    failed = [
        {
            "step_id": "s2",
            "query": "查一下它的数值",
            "target_name": "lab_report",
            "error_msg": "缺少检验指标名称与数值，无法解读",
        }
    ]
    done = [{"step_id": "s1", "query": "我的血糖正常吗", "target_name": "lab_report", "summary": "血糖 6.5"}]
    actions = await svc.replan_failed_steps("我的血糖正常吗？那它正常吗", failed, done)
    ok = bool(actions) and all(
        a["action"] in {"retry", "rewrite", "reroute", "drop"} for a in actions
    )
    record("replan_failed_steps（动作枚举逐条过滤）", ok, f"-> {dump(actions)}")

    # --- 5. 实体抽取三件套 ---
    drug = await svc.extract_entities("我今天早上吃了两片布洛芬，为了退烧", "drug")
    ok = bool(drug) and bool(drug.get("drug_name_list"))
    record("extract_entities(drug)", ok, f"-> {dump(drug)}")

    lab = await svc.extract_entities("血糖 6.5 mmol/L，白细胞 11.2", "lab")
    ok = bool(lab) and bool(lab.get("lab_items")) and lab.get("raw") is not None
    record("extract_entities(lab，含 raw 字段)", ok, f"-> {dump(lab)}")

    info = await svc.extract_drug_info("昨晚八点吃了一片阿司匹林，为了抗凝", [])
    ok = bool(info) and bool(info.get("drug_name"))
    record("extract_drug_info", ok, f"-> {dump(info)}")

    # --- 6. 可观测：兜底提取计数应保持低位，说明约束真的在起作用 ---
    stats = structured_output_stats()
    print(f"\n结构化输出计数: {dump(stats)}")
    print(
        "（extract_fallback 长期为 0 或极小 = response_format 被真正履行；"
        "validation_failure 高 = 模型不遵从新形态，需要回看提示词）"
    )

    # --- 7. step_id 解析的旁证 ---
    print(f"\n_step_id_to_index 自检: {[lds._step_id_to_index(s, 3) for s in ('s1', 's2', 's9', 'x')]}")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n==== {passed}/{len(RESULTS)} 通过 ====")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
