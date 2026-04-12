from __future__ import annotations

from itertools import combinations

from app.common.exceptions import ParamException, ServiceUnavailableException, UserAuthException
from app.core.rag.drug_knowledge_service import DrugKnowledgeService
from app.db.crud.archive_crud import ArchiveCRUD


class DrugInteractionTool:
    """药物冲突检测工具：结论 100% 来自 drug_knowledge_base。"""

    async def check_interactions(self, *, user_id: str, drug_name_list: list[str], sync_to_archive: bool) -> dict:
        if not user_id:
            raise UserAuthException("未授权")
        if not drug_name_list:
            raise ParamException("drug_name_list 不能为空")

        svc = DrugKnowledgeService()
        matches = await svc.match_drugs(drug_name_list)

        drug_list = []
        matched_drugs = []
        match_map = {}
        for m in matches:
            if not m["match"]:
                drug_list.append({"drug_name": m["query"], "match_status": "匹配失败"})
            else:
                dn = m["match"]["drug_name"]
                drug_list.append({"drug_name": dn, "match_status": "匹配成功"})
                matched_drugs.append(dn)
                match_map[dn] = m["match"]

        interaction_result = []
        for a, b in combinations(sorted(set(matched_drugs)), 2):
            a_drugs, a_desc = svc.parse_interactions(match_map[a])
            b_drugs, b_desc = svc.parse_interactions(match_map[b])

            desc = None
            if b in a_desc:
                desc = a_desc.get(b)
            elif a in b_desc:
                desc = b_desc.get(a)
            elif b in a_drugs or a in b_drugs:
                desc = "提示：知识库记录存在相互作用标记，但缺少详细说明。请以说明书/执业药师建议为准。"

            if desc:
                interaction_result.append(
                    {
                        "drug_a": a,
                        "drug_b": b,
                        "risk_level": "需注意",
                        "interaction_desc": str(desc),
                    }
                )

        if matched_drugs and sync_to_archive:
            await ArchiveCRUD().sync_drugs(user_id=user_id, drug_names=sorted(set(matched_drugs)))

        if not matched_drugs:
            final_desc = "未在知识库中匹配到有效药品名称，请核对药品通用名/商品名后再试。"
        elif not interaction_result:
            final_desc = "已匹配到药品，但未查询到两两相互作用记录；如需更精确信息，请查阅说明书或咨询执业药师。"
        else:
            lines = ["药物相互作用（科普信息，来源：结构化药品知识库）："]
            for it in interaction_result:
                lines.append(f"- {it['drug_a']} + {it['drug_b']}：{it['interaction_desc']}")
            final_desc = "\n".join(lines)

        return {
            "drug_list": drug_list,
            "interaction_result": interaction_result,
            "final_desc": final_desc,
        }
