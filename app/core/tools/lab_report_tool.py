from __future__ import annotations

import re

from app.common.exceptions import ParamException, UserAuthException
from app.core.rag.lab_reference_service import LabReferenceService
from app.db.crud.archive_crud import ArchiveCRUD


#: 参考范围的三种可判定形态。注意 `<=` / `>=` 必须排在 `[<≤]` / `[>≥]` 之前，
#: 否则 `<=1.7` 会先被 `[<≤]` 吃掉 `<`，剩下的 `=1.7` 无法匹配数值。
_RANGE_RE = re.compile(
    r"^\s*(?:"
    r"(?P<low>[-+]?\d+(?:\.\d+)?)\s*-\s*(?P<high>[-+]?\d+(?:\.\d+)?)"
    r"|(?:<=|[<≤])\s*(?P<only_high>[-+]?\d+(?:\.\d+)?)"
    r"|(?:>=|[>≥])\s*(?P<only_low>[-+]?\d+(?:\.\d+)?)"
    r")\s*$"
)


def _try_float(v: str) -> float | None:
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _parse_reference_range(text: str) -> tuple[float | None, float | None]:
    """把参考范围字符串解析成 `(low, high)`；无法识别时返回 `(None, None)`。

    支持形态：

    - 双侧闭区间 `3.9-6.1` → `(3.9, 6.1)`
    - 仅上限 `<1.7` / `≤1.7` / `<=1.7` → `(None, 1.7)`
    - 仅下限 `>1.0` / `≥1.0` / `>=1.0` → `(1.0, None)`

    为什么必须有单侧形态：血脂四项在化验单上就是这么印的（总胆固醇/甘油三酯/
    低密度脂蛋白给上限，高密度脂蛋白给下限）。若用 `0-上限` 硬凑，
    高密度脂蛋白会把**正常的高值反向误报成偏高** —— 这属于"错抽"，不是"漏抽"。

    其余一切无法识别的写法（如 `阴性`、`男：130-175 女：115-150`）一律返回
    `(None, None)`，由调用方按"无法解析、不做判定"处理，**绝不猜**。
    """
    m = _RANGE_RE.match(text or "")
    if not m:
        return None, None
    if m.group("only_high") is not None:
        return None, float(m.group("only_high"))
    if m.group("only_low") is not None:
        return float(m.group("only_low")), None
    return float(m.group("low")), float(m.group("high"))


class LabReportTool:
    """化验单通用解读工具：异常判断 100% 来自 reference_range 对比。"""

    async def interpret(self, *, user_id: str, lab_item_list: list[dict], sync_to_archive: bool) -> dict:
        if not user_id:
            raise UserAuthException("未授权")
        if not lab_item_list:
            raise ParamException("lab_item_list 不能为空")

        names = [i.get("item_name", "").strip() for i in lab_item_list if i.get("item_name")]
        svc = LabReferenceService()
        matches = await svc.match_items(names)
        m_map = {m["query"]: m["match"] for m in matches}

        item_list = []
        for it in lab_item_list:
            name = it.get("item_name", "").strip()
            value = str(it.get("test_value", "")).strip()
            unit = (it.get("unit") or "").strip() or None

            ref = m_map.get(name)
            if not ref:
                item_list.append(
                    {
                        "item_name": name,
                        "test_value": value,
                        "reference_range": None,
                        "abnormal_flag": None,
                        "meaning": "当前检验指标暂未纳入参考库，无法提供解读服务，请核对指标名称或咨询执业医师。",
                    }
                )
                continue

            ref_range = ref.get("reference_range")
            low, high = _parse_reference_range(ref_range)

            v = _try_float(value)
            abnormal_flag = None
            meaning = "通用科普信息：请结合复查与医生意见综合评估。"

            if v is None or (low is None and high is None):
                abnormal_flag = None
                meaning = "数值或参考范围格式无法解析，建议核对化验单原始内容或咨询执业医师。"
            elif low is not None and v < low:
                abnormal_flag = "L"
                meaning = ref.get("low_meaning") or meaning
            elif high is not None and v > high:
                abnormal_flag = "H"
                meaning = ref.get("high_meaning") or meaning
            else:
                abnormal_flag = "N"
                meaning = "通用科普信息：该指标在参考范围内，仅供参考，具体以检验机构与医生解释为准。"

            item_list.append(
                {
                    "item_name": ref.get("item_name") or name,
                    "test_value": value,
                    "reference_range": ref_range,
                    "abnormal_flag": abnormal_flag,
                    "meaning": meaning,
                }
            )

        if sync_to_archive:
            await ArchiveCRUD().sync_lab_items(user_id=user_id, items=item_list)

        final_lines = ["化验指标通用解读（科普信息，来源：结构化检验参考库）："]
        for o in item_list:
            final_lines.append(
                f"- {o['item_name']}：{o['test_value']}（参考：{o.get('reference_range') or '未知'}）"
                f" 状态：{o.get('abnormal_flag') or '未知'}；{o['meaning']}"
            )

        return {"item_list": item_list, "final_desc": "\n".join(final_lines)}
