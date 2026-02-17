"""
Main FastAPI application with logging, monitoring, and middleware setup.
"""
from contextlib import asynccontextmanager

from dishka import make_async_container
from dishka.integrations.fastapi import DishkaRoute
from dishka.integrations import fastapi as fastapi_integration
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middlewares.response_middleware import StandardResponseMiddleware
from src.api.exceptions.exception_handlers import register_exception_handlers
from src.api.v1.controllers.detection import router as detection_router
from src.core.config import config, Config
from src.core.logging import get_logger, setup_logging
from src.ioc import AppProvider
from src.services.detection_service import DetectionService

setup_logging(
    level="DEBUG" if config.debug else "INFO",
    json_logs=not config.debug,
)

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application with Dishka DI container.
    """
    # Create Dishka container with AppProvider and inject config context
    container = make_async_container(AppProvider(), context={Config: config})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("application_startup", app_name=config.app_name)
        detection_service = await container.get(DetectionService)
        await detection_service.load()
        logger.info("model_loaded")
        yield
        logger.info("application_shutdown", app_name=config.app_name)
        await container.close()

    app = FastAPI(
        title=config.app_name,
        description="FastAPI application with structured logging and monitoring",
        version="0.1.0",
        lifespan=lifespan,
    )

    fastapi_integration.setup_dishka(container, app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(StandardResponseMiddleware)

    register_exception_handlers(app)

    return app


app = create_app()


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Basic liveness check endpoint.
    """
    return {"status": "healthy", "service": config.app_name}

health_router = APIRouter(route_class=DishkaRoute, tags=["Health"])


@health_router.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    """
    return {
        "status": "ready",
        "service": config.app_name
    }


app.include_router(health_router)
app.include_router(detection_router)