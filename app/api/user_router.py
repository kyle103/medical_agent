import uuid

from fastapi import APIRouter, Request

from app.common.auth import create_access_token
from app.config.settings import settings
from app.db.crud.user_crud import UserCRUD
from app.schema.base import APIResponse
from app.schema.user_schema import UserRegisterRequest, UserRegisterResponse

router = APIRouter()


@router.post("/register", response_model=APIResponse[UserRegisterResponse])
async def register(req: UserRegisterRequest, request: Request):
    user_id = uuid.uuid4().hex
    nickname = req.user_nickname or "匿名用户"

    await UserCRUD().create_user(user_id=user_id, user_nickname=nickname)

    token = create_access_token(user_id=user_id)
    return APIResponse(
        data=UserRegisterResponse(
            user_id=user_id,
            access_token=token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        ),
        request_id=getattr(request.state, "request_id", ""),
    )
