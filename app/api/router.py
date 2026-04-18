from fastapi import APIRouter

from app.api.user_router import router as user_router
from app.api.chat_router import router as chat_router
from app.api.drug_router import router as drug_router
from app.api.lab_router import router as lab_router
from app.api.archive_router import router as archive_router
from app.api.smart_router import router as smart_router

api_router = APIRouter()

# 核心API接口
api_router.include_router(user_router, prefix="/user", tags=["用户认证"])
api_router.include_router(chat_router, prefix="/chat", tags=["智能对话"])

# 专业功能API
api_router.include_router(drug_router, prefix="/drug", tags=["药物管理"])
api_router.include_router(lab_router, prefix="/lab", tags=["化验单解读"])
api_router.include_router(archive_router, prefix="/archive", tags=["档案管理"])

# 测试和调试API
api_router.include_router(smart_router, prefix="/smart", tags=["智能路由测试"])
