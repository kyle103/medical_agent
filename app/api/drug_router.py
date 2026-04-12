from fastapi import APIRouter, Request

from app.core.compliance.compliance_service import ComplianceService
from app.core.tools.drug_interaction_tool import DrugInteractionTool
from app.schema.base import APIResponse
from app.schema.drug_schema import DrugInteractionCheckRequest, DrugInteractionCheckResponse

router = APIRouter()


@router.post("/interaction-check", response_model=APIResponse[DrugInteractionCheckResponse])
async def interaction_check(req: DrugInteractionCheckRequest, request: Request):
    user_id = getattr(request.state, "user_id", None)

    tool = DrugInteractionTool()
    result = await tool.check_interactions(
        user_id=user_id,
        drug_name_list=req.drug_name_list,
        sync_to_archive=req.sync_to_archive,
    )

    compliance = ComplianceService()
    final_desc = compliance.add_disclaimer(result["final_desc"])

    return APIResponse(
        data=DrugInteractionCheckResponse(**{**result, "final_desc": final_desc}),
        request_id=getattr(request.state, "request_id", ""),
    )
