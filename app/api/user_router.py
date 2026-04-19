import uuid

from fastapi import APIRouter, HTTPException, Request

from app.common.auth import create_access_token
from app.common.passwords import hash_password, verify_password
from app.config.settings import settings
from app.db.crud.user_crud import UserCRUD
from app.schema.base import APIResponse
from app.schema.user_schema import UserAuthResponse, UserLoginRequest, UserRegisterRequest

router = APIRouter()


@router.post("/register", response_model=APIResponse[UserAuthResponse])
async def register(req: UserRegisterRequest, request: Request):
    phone = (req.phone or "").strip()
    if not phone:
        raise HTTPException(status_code=400, detail="phone required")

    crud = UserCRUD()
    existing = await crud.get_user_by_phone(phone=phone)
    if existing is not None:
        raise HTTPException(status_code=400, detail="phone already registered")

    user_id = uuid.uuid4().hex
    nickname = req.user_nickname or "匿名用户"
    pwd_hash = hash_password(req.password)

    await crud.create_user(
        user_id=user_id,
        user_nickname=nickname,
        phone=phone,
        password_hash=pwd_hash,
    )

    token = create_access_token(user_id=user_id)
    return APIResponse(
        data=UserAuthResponse(
            user_id=user_id,
            access_token=token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        ),
        request_id=getattr(request.state, "request_id", ""),
    )


@router.post("/login", response_model=APIResponse[UserAuthResponse])
async def login(req: UserLoginRequest, request: Request):
    phone = (req.phone or "").strip()
    crud = UserCRUD()
    user = await crud.get_user_by_phone(phone=phone)
    if user is None or not user.password_hash:
        raise HTTPException(status_code=401, detail="invalid phone or password")

    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid phone or password")

    token = create_access_token(user_id=user.user_id)
    return APIResponse(
        data=UserAuthResponse(
            user_id=user.user_id,
            access_token=token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        ),
        request_id=getattr(request.state, "request_id", ""),
    )


@router.get("/me", response_model=APIResponse[dict])
async def get_current_user(request: Request):
    """获取当前登录用户信息"""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未授权")

    crud = UserCRUD()
    user = await crud.get_user(user_id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    return APIResponse(
        data={
            "user_id": user.user_id,
            "user_nickname": user.user_nickname,
            "phone": user.phone,
        },
        request_id=getattr(request.state, "request_id", ""),
    )
