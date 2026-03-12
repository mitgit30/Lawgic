from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers.auth import router as auth_router
from backend.api.routers.health import router as health_router
from backend.api.routers.legal import router as legal_router
from src.core.config import get_settings
from src.core.logging import setup_logging


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(legal_router)
    return app


app = create_app()
