from fastapi import APIRouter

from app.api.user_router import router as user_router
from app.api.chat_router import router as chat_router
from app.api.drug_router import router as drug_router
from app.api.lab_router import router as lab_router
from app.api.archive_router import router as archive_router

api_router = APIRouter()

api_router.include_router(user_router, prefix="/user", tags=["user"])
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(drug_router, prefix="/drug", tags=["drug"])
api_router.include_router(lab_router, prefix="/lab", tags=["lab"])
api_router.include_router(archive_router, prefix="/archive", tags=["archive"])
