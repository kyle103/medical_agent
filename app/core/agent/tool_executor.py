from __future__ import annotations

import time

from app.common.logger import get_logger
from app.core.rag.drug_knowledge_service import DrugKnowledgeService
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
from app.core.tools.drug_interaction_tool import DrugInteractionTool
from app.core.tools.lab_item_parser import lab_route_text, parse_lab_items
from app.core.tools.lab_report_tool import LabReportTool

logger = get_logger(__name__)


#: 化验工具「零抽取」时回给上游的说明文本。
#:
#: 两点必须写清，否则用户会反复问同一件事：
#: 1) **这一轮到底怎么了** —— 是"没给数值"还是"给了但读不出来"。
#:    曾经这里写的是「本系统…不支持上传或拍照化验单（没有图片识别能力）」：
#:    图片识别接入后这句话就成了明确的假话，而且用户照着它去改也没用
#:    （他可能已经发了图，问题在于图没读出来）；
#: 2) **可用的输入格式**：给出可直接照抄的示例。
#: 这段文本只在 `no_data=True` 时出现，不会再被当作"化验结论"输出（见 `_decide_response_mode`）。
_LAB_NO_DATA_HINT = (
    "这一轮我没拿到可以解读的检验指标。"
    "你可以直接上传化验单照片，也可以把指标按「名称 + 数值」用文字发来，"
    "一行一项即可，例如：血糖 6.5；血压 120/80。"
)


def _image_flag_map(state: dict) -> dict[str, dict]:
    """`state["image_lab_items"]` → `{项名: {"flag": …, "reference_range": …}}`。

    为什么必须走这条独立通道：异常标记**故意不进回填文本** ——
    `6.5↑` 整串填进 `test_value` 会让数值侧的形态校验失败，这一条的数值就没了。
    所以标记只能从结构化条目里搬。

    这里只做搬运，不判定。**"键存在即来自图上"**这个语义由
    `LabReportTool.interpret` 使用：它决定该走"图标注优先"还是"区间比对"。
    """
    out: dict[str, dict] = {}
    for it in state.get("image_lab_items") or []:
        if not isinstance(it, dict):
            continue
        name = (it.get("item_name") or it.get("raw_name") or "").strip()
        if not name:
            continue
        out[name] = {
            "flag": (it.get("image_flag") or "").strip().upper(),
            "reference_range": (it.get("reference_range") or "").strip(),
        }
    return out


class ToolExecutor:
    """统一工具执行器：根据工具名称和状态调用对应工具。

    将原 DrugConflictAgent / LabReportAgent 的核心逻辑下沉为 Tool 直接调用，
    LLM 格式化由 Workflow 的 llm_generate 节点统一处理。
    """

    async def execute(self, tool_name: str, state: dict) -> dict:
        dispatch = {
            "drug_interaction": self._execute_drug_interaction,
            "lab_report": self._execute_lab_report,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            logger.error("ToolExecutor unknown tool=%s", tool_name)
            return {"error_msg": f"未知工具: {tool_name}", "final_desc": ""}

        start_time = time.perf_counter()
        try:
            result = await handler(state)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "[TOOL] %s | latency=%dms | OK | has_error=%s",
                tool_name,
                latency_ms,
                bool(result.get("error_msg")),
            )
            return result
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "[TOOL] %s | latency=%dms | FAIL | error=%s",
                tool_name,
                latency_ms,
                str(e)[:100],
            )
            return {"error_msg": str(e), "final_desc": ""}

    async def _execute_drug_interaction(self, state: dict) -> dict:
        tool = DrugInteractionTool()
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")

        entities = state.get("extract_entities") or {}
        drug_names = entities.get("drug_name_list") if isinstance(entities, dict) else []
        if not drug_names:
            drug_names = DrugEntityExtractor.extract_drug_candidates(user_input, max_items=10)

        if not drug_names:
            step_context = state.get("step_context") or {}
            dep_summaries = step_context.get("dep_summaries") or []
            if dep_summaries:
                combined_text = " ".join(dep_summaries)
                dep_names = DrugEntityExtractor.extract_drug_candidates(combined_text, max_items=10)
                if dep_names:
                    drug_names = dep_names

        # 词典归一 + 前向最大匹配补充：先把已有候选归一到标准名（避免“泰诺”与
        # “对乙酰氨基酚”这类同药异名重复进查询），再补回规则漏掉的无分隔符药名
        try:
            svc = DrugKnowledgeService()
            merged: list[str] = await svc.canonicalize_names(drug_names)
            seen = set(merged)
            source_texts = [user_input]
            step_context = state.get("step_context") or {}
            source_texts.extend(step_context.get("dep_summaries") or [])
            for src in source_texts:
                if not src or not src.strip():
                    continue
                for canon in await svc.resolve_text(src):
                    if canon not in seen:
                        seen.add(canon)
                        merged.append(canon)
            drug_names = merged[:12]
        except Exception:  # noqa: BLE001 - 词典为增强层，失败沿用既有候选
            pass

        if not drug_names:
            return {
                "tool_result": {
                    "drug_list": [],
                    "interaction_result": [],
                    "final_desc": "未识别到有效的药品名称。请提供药品的通用名，例如：布洛芬、阿司匹林等。",
                },
                "intent_type": "drug_conflict",
            }

        tool_result = await tool.check_interactions(
            user_id=user_id,
            drug_name_list=drug_names,
            sync_to_archive=False,
        )
        return {
            "tool_result": tool_result,
            "extract_entities": {"drug_name_list": drug_names},
            "intent_type": "drug_conflict",
        }

    async def _execute_lab_report(self, state: dict) -> dict:
        tool = LabReportTool()
        user_id = state.get("user_id", "")

        # 图路与文路在此汇入**同一个** `parse_lab_items` → `LabReportTool`：
        # 图片识别产出的文本就是按 `parse_lab_items` 的显式分隔符定制的
        # （见 lab_report_vision 模块说明），所以图路不新开第二条判定路径。
        #
        # ⚠️ 但异常标记**不走文本**：`6.5↑` 整串填进来会让数值侧校验失败，
        # 所以标记走 `image_items` 这条独立通道（`_image_flag_map`）。
        # 判定优先级（图标注优先 / 无图才比区间）统一在 `interpret` 里实现。
        lab_items = parse_lab_items(lab_route_text(state))
        if not lab_items:
            return {
                "tool_result": {
                    "item_list": [],
                    # 「工具拿到了零数据」必须与「工具给出了结论」区分开。
                    # `_decide_response_mode` 见这个标记会回到对话模式，
                    # 否则会把下面这段说明当成"解读结论"输出 —— 用户问
                    # 「我直接发单子给你吗」时会收到「请提供指标名称和数值」，
                    # 再问一次仍收到同一句（实测死循环）。
                    "no_data": True,
                    "final_desc": _LAB_NO_DATA_HINT,
                },
                "intent_type": "lab_report",
            }

        tool_result = await tool.interpret(
            user_id=user_id,
            lab_item_list=lab_items,
            sync_to_archive=False,
            image_items=_image_flag_map(state),
        )
        return {
            "tool_result": tool_result,
            "intent_type": "lab_report",
        }
