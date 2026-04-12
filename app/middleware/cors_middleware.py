from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings


def setup_cors(app: FastAPI) -> None:
    allowed_hosts = settings.ALLOWED_HOSTS
    allow_origins = [o.strip() for o in allowed_hosts.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
        if "*" in allow_origins
        else ["Authorization", "Content-Type", "X-Request-Id"],
    )
