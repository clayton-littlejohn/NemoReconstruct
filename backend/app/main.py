from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.reconstructions import router as reconstructions_router
from app.api.workflows import router as workflows_router
from app.core.config import settings
from app.core.database import Base, engine, ensure_runtime_schema
from app.schemas import HealthResponse
from app.workers.runner import runner


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    runner.mark_inflight_jobs_failed()
    runner.start()
    yield


app = FastAPI(
    title="NemoReconstruct API",
    version="0.1.0",
    description="Minimal MOV-to-fVDB reconstruction workflow for Isaac Sim and agent orchestration.",
    docs_url=settings.docs_url,
    openapi_url=settings.openapi_url,
    redoc_url=settings.redoc_url,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse()


app.include_router(reconstructions_router)
app.include_router(workflows_router)