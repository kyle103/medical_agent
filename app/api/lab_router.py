from fastapi import APIRouter, Request

from app.core.compliance.compliance_service import ComplianceService
from app.core.tools.lab_report_tool import LabReportTool
from app.schema.base import APIResponse
from app.schema.lab_schema import LabReportInterpretRequest, LabReportInterpretResponse

router = APIRouter()


@router.post("/report-interpret", response_model=APIResponse[LabReportInterpretResponse])
async def report_interpret(req: LabReportInterpretRequest, request: Request):
    user_id = getattr(request.state, "user_id", None)

    tool = LabReportTool()
    result = await tool.interpret(user_id=user_id, lab_item_list=req.lab_item_list, sync_to_archive=req.sync_to_archive)

    compliance = ComplianceService()
    final_desc = compliance.add_disclaimer(result["final_desc"])

    return APIResponse(
        data=LabReportInterpretResponse(**{**result, "final_desc": final_desc}),
        request_id=getattr(request.state, "request_id", ""),
    )
