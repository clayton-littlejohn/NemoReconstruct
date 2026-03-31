from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReconstructionStatus(str, enum.Enum):
    uploading = "uploading"
    queued = "queued"
    extracting_frames = "extracting_frames"
    feature_extraction = "feature_extraction"
    feature_matching = "feature_matching"
    sparse_reconstruction = "sparse_reconstruction"
    fvdb_reconstruction = "fvdb_reconstruction"
    grut_reconstruction = "grut_reconstruction"
    exporting = "exporting"
    generating_collision_mesh = "generating_collision_mesh"
    completed = "completed"
    failed = "failed"


class Reconstruction(Base):
    __tablename__ = "reconstructions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(64), default=ReconstructionStatus.uploading.value, nullable=False)
    pipeline_slug: Mapped[str] = mapped_column(String(64), default="nemo-reconstruct-mvp", nullable=False)

    processing_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    processing_pct: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_params_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    source_video_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_video_path: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_dir: Mapped[str] = mapped_column(Text, nullable=False)

    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_ply_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_usdz_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_bundle_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_collision_mesh_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_metadata_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Workflow(Base):
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scene_name: Mapped[str] = mapped_column(String(255), nullable=False)
    video_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    video_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="pending", nullable=False)
    current_agent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    iteration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    accept_psnr_threshold: Mapped[float] = mapped_column(Float, default=25.0, nullable=False)
    accept_ssim_threshold: Mapped[float] = mapped_column(Float, default=0.85, nullable=False)
    last_verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reconstruction_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class IterationRecord(Base):
    __tablename__ = "iteration_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reconstruction_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ply_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    psnr: Mapped[float | None] = mapped_column(Float, nullable=True)
    ssim: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_gaussians: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
