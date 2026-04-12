from __future__ import annotations

import re

from app.common.exceptions import ParamException, UserAuthException
from app.core.rag.lab_reference_service import LabReferenceService
from app.db.crud.archive_crud import ArchiveCRUD


_RANGE_RE = re.compile(r"^\s*(?P<low>[-+]?\d+(?:\.\d+)?)\s*-\s*(?P<high>[-+]?\d+(?:\.\d+)?)\s*$")


def _try_float(v: str) -> float | None:
    try:
        return float(str(v).strip())
    except Exception:
        return None


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
            m = _RANGE_RE.match(ref_range or "")
            low = float(m.group("low")) if m else None
            high = float(m.group("high")) if m else None

            v = _try_float(value)
            abnormal_flag = None
            meaning = "通用科普信息：请结合复查与医生意见综合评估。"

            if v is None or low is None or high is None:
                abnormal_flag = None
                meaning = "数值或参考范围格式无法解析，建议核对化验单原始内容或咨询执业医师。"
            else:
                if v < low:
                    abnormal_flag = "L"
                    meaning = ref.get("low_meaning") or meaning
                elif v > high:
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
