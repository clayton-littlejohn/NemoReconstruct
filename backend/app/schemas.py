from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ReconstructionParams(BaseModel):
    frame_rate: float | None = Field(default=None, ge=0.25, le=12.0)
    sequential_matcher_overlap: int | None = Field(default=None, ge=2, le=50)
    fvdb_max_epochs: int | None = Field(default=None, ge=5, le=500)
    fvdb_sh_degree: int | None = Field(default=None, ge=0, le=4)
    fvdb_image_downsample_factor: int | None = Field(default=None, ge=1, le=12)
    splat_only_mode: bool | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "NemoReconstruct"


class PipelineInfo(BaseModel):
    slug: str
    name: str
    description: str
    source_type: str = "video"
    output_types: list[str]
    steps: list[str]
    requirements: list[str]
    tunable_params: dict[str, str] = Field(default_factory=dict)


class ReconstructionBase(BaseModel):
    id: str
    name: str
    description: str | None = None
    status: str
    pipeline_slug: str
    processing_step: str | None = None
    processing_pct: int = 0
    error_message: str | None = None
    source_video_filename: str
    frame_count: int | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_params: ReconstructionParams = Field(default_factory=ReconstructionParams)


class ReconstructionDetail(ReconstructionBase):
    artifact_ply_url: str | None = None
    artifact_usdz_url: str | None = None
    artifact_bundle_url: str | None = None
    artifact_log_url: str | None = None
    artifact_metadata_url: str | None = None


class ReconstructionStatusResponse(BaseModel):
    id: str
    status: str
    processing_step: str | None = None
    processing_pct: int = 0
    error_message: str | None = None
    updated_at: datetime


class ReconstructionArtifacts(BaseModel):
    source_video_url: str
    splat_ply_url: str | None = None
    scene_usdz_url: str | None = None
    sim_bundle_url: str | None = None
    run_log_url: str | None = None
    metadata_url: str | None = None


class MetricsEntry(BaseModel):
    epoch: int
    metric: str
    value: float


class MetricsResponse(BaseModel):
    id: str
    summary: dict[str, float] = Field(default_factory=dict)
    entries: list[MetricsEntry] = Field(default_factory=list)


class UploadResponse(ReconstructionDetail):
    pass


class RetryResponse(ReconstructionDetail):
    pass


class DeleteResponse(BaseModel):
    id: str
    deleted: bool = True


class UploadFormData(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class RetryRequest(BaseModel):
    params: ReconstructionParams | None = None


class NotesUpdate(BaseModel):
    notes: str = Field(min_length=1, max_length=5000)


class IterationSummary(BaseModel):
    iteration: int
    params: ReconstructionParams = Field(default_factory=ReconstructionParams)
    loss: float | None = None
    ssim: float | None = None
    num_gaussians: int | None = None
    verdict: str | None = None
    reason: str | None = None
    ply_url: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class IterationHistoryResponse(BaseModel):
    reconstruction_id: str
    iterations: list[IterationSummary] = Field(default_factory=list)


class WorkflowDetail(BaseModel):
    id: str
    scene_name: str
    video_filename: str
    status: str
    current_agent: str | None = None
    current_step: str | None = None
    iteration: int = 0
    max_iterations: int = 3
    last_verdict: str | None = None
    last_reason: str | None = None
    reconstruction_id: str | None = None
    error_message: str | None = None
    pid: int | None = None
    created_at: datetime
    updated_at: datetime


class WorkflowStateUpdate(BaseModel):
    status: str | None = None
    current_agent: str | None = None
    current_step: str | None = None
    iteration: int | None = None
    last_verdict: str | None = None
    last_reason: str | None = None
    reconstruction_id: str | None = None
    error_message: str | None = None
