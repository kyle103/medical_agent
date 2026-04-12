from fastapi import APIRouter, Request

from app.schema.base import APIResponse

router = APIRouter()


@router.get("/{archive_type}/list")
async def list_archive(archive_type: str, request: Request):
    # MVP：仅占位，后续补齐 CRUD
    return APIResponse(data={"archive_type": archive_type, "items": []}, request_id=getattr(request.state, "request_id", ""))


@router.post("/search")
async def search_archive(request: Request):
    # MVP：仅占位（需向量检索与 user_id filter）
    return APIResponse(data={"items": []}, request_id=getattr(request.state, "request_id", ""))
