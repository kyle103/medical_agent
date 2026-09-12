from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.router import api_router
from app.common.logger import get_logger, setup_logging
from app.middleware.auth_middleware import AuthMiddleware
from app.middleware.cors_middleware import setup_cors
from app.middleware.log_middleware import RequestLogMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # 启动预加载 药品/化验 实体词典；失败不阻塞（首次调用时会惰性重试）
    try:
        from app.core.rag.entity_dictionary import reload_entity_dictionary

        await reload_entity_dictionary()
    except Exception as e:  # noqa: BLE001
        logger.warning("entity_dictionary preload failed (will lazy-load): %s", e)

    # 预热结构化输出档位：提前确定模型是否真的下发 schema，并把生效档位打进启动日志。
    # 目的是让"模型换了导致约束静默失效"这件事在启动时就可见，而不是等到线上解析失败。
    try:
        from app.core.llm.structured_output import warmup as warmup_structured_output

        await warmup_structured_output()
    except Exception as e:  # noqa: BLE001
        logger.warning("structured_output warmup failed (will lazy-resolve): %s", e)

    yield


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="medical_agent", version="1.0.0", lifespan=_lifespan)

    setup_cors(app)

    app.add_middleware(RequestLogMiddleware)
    app.add_middleware(AuthMiddleware)

    app.include_router(api_router, prefix="/api/v1")

    # Swagger/OpenAPI: 增加 Bearer JWT 鉴权按钮（Authorize）
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
        schema.setdefault("security", []).append({"BearerAuth": []})

        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[assignment]

    return app


app = create_app()
