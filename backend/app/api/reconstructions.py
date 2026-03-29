from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models import IterationRecord, Reconstruction, ReconstructionStatus
from app.schemas import DatasetInfo, DeleteResponse, IterationHistoryResponse, IterationSummary, MetricsResponse, MetricsEntry, NotesUpdate, PipelineInfo, ReconstructionArtifacts, ReconstructionDetail, ReconstructionParams, ReconstructionStatusResponse, RetryRequest, RetryResponse, UploadResponse
from app.services.pipeline import PIPELINE_INFO
from app.services.storage import ensure_parent, remove_workspace
from app.workers.runner import runner


router = APIRouter(prefix=settings.api_prefix, tags=["reconstructions"])

ALLOWED_EXTENSIONS = {".mov", ".mp4", ".m4v"}
ARTIFACT_FIELDS = {
    "source_video": "source_video_path",
    "splat_ply": "artifact_ply_path",
    "scene_usdz": "artifact_usdz_path",
    "collision_mesh": "artifact_collision_mesh_path",
    "sim_bundle": "artifact_bundle_path",
    "run_log": "artifact_log_path",
    "metadata": "artifact_metadata_path",
}


def build_download_url(reconstruction_id: str, artifact: str) -> str:
    return f"{settings.api_prefix}/reconstructions/{reconstruction_id}/download/{artifact}"


def serialize_processing_params(reconstruction: Reconstruction) -> ReconstructionParams:
    if not reconstruction.processing_params_json:
        return ReconstructionParams()
    try:
        payload = json.loads(reconstruction.processing_params_json)
    except json.JSONDecodeError:
        return ReconstructionParams()
    if not isinstance(payload, dict):
        return ReconstructionParams()
    try:
        return ReconstructionParams(**payload)
    except Exception:
        return ReconstructionParams()


def build_processing_params(
    frame_rate: float | None,
    sequential_matcher_overlap: int | None,
    colmap_mapper_type: str | None,
    colmap_max_num_features: int | None,
    reconstruction_backend: str | None,
    fvdb_max_epochs: int | None,
    fvdb_sh_degree: int | None,
    fvdb_image_downsample_factor: int | None,
    grut_n_iterations: int | None,
    grut_render_method: str | None,
    grut_strategy: str | None,
    grut_downsample_factor: int | None,
    splat_only_mode: bool | None,
    collision_mesh_enabled: bool | None = None,
    collision_mesh_method: str | None = None,
    collision_mesh_target_faces: int | None = None,
    collision_mesh_alpha: float | None = None,
    collision_mesh_downsample: int | None = None,
) -> ReconstructionParams:
    try:
        return ReconstructionParams(
            frame_rate=frame_rate,
            sequential_matcher_overlap=sequential_matcher_overlap,
            colmap_mapper_type=colmap_mapper_type,
            colmap_max_num_features=colmap_max_num_features,
            reconstruction_backend=reconstruction_backend,
            fvdb_max_epochs=fvdb_max_epochs,
            fvdb_sh_degree=fvdb_sh_degree,
            fvdb_image_downsample_factor=fvdb_image_downsample_factor,
            grut_n_iterations=grut_n_iterations,
            grut_render_method=grut_render_method,
            grut_strategy=grut_strategy,
            grut_downsample_factor=grut_downsample_factor,
            splat_only_mode=splat_only_mode,
            collision_mesh_enabled=collision_mesh_enabled,
            collision_mesh_method=collision_mesh_method,
            collision_mesh_target_faces=collision_mesh_target_faces,
            collision_mesh_alpha=collision_mesh_alpha,
            collision_mesh_downsample=collision_mesh_downsample,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())


def serialize_reconstruction(reconstruction: Reconstruction) -> ReconstructionDetail:
    return ReconstructionDetail(
        id=reconstruction.id,
        name=reconstruction.name,
        description=reconstruction.description,
        status=reconstruction.status,
        pipeline_slug=reconstruction.pipeline_slug,
        processing_step=reconstruction.processing_step,
        processing_pct=reconstruction.processing_pct,
        error_message=reconstruction.error_message,
        source_video_filename=reconstruction.source_video_filename,
        frame_count=reconstruction.frame_count,
        created_at=reconstruction.created_at,
        updated_at=reconstruction.updated_at,
        started_at=reconstruction.started_at,
        completed_at=reconstruction.completed_at,
        processing_params=serialize_processing_params(reconstruction),
        artifact_ply_url=build_download_url(reconstruction.id, "splat_ply") if reconstruction.artifact_ply_path else None,
        artifact_usdz_url=build_download_url(reconstruction.id, "scene_usdz") if reconstruction.artifact_usdz_path else None,
        artifact_collision_mesh_url=build_download_url(reconstruction.id, "collision_mesh") if reconstruction.artifact_collision_mesh_path else None,
        artifact_bundle_url=build_download_url(reconstruction.id, "sim_bundle") if reconstruction.artifact_bundle_path else None,
        artifact_log_url=build_download_url(reconstruction.id, "run_log") if reconstruction.artifact_log_path else None,
        artifact_metadata_url=build_download_url(reconstruction.id, "metadata") if reconstruction.artifact_metadata_path else None,
    )


def get_reconstruction_or_404(db: Session, reconstruction_id: str) -> Reconstruction:
    reconstruction = db.get(Reconstruction, reconstruction_id)
    if reconstruction is None:
        raise HTTPException(status_code=404, detail="Reconstruction not found")
    return reconstruction


@router.get("/pipelines", response_model=list[PipelineInfo], tags=["pipelines"])
def list_pipelines() -> list[PipelineInfo]:
    return [PipelineInfo(**PIPELINE_INFO)]


@router.get("/datasets", response_model=list[DatasetInfo], tags=["datasets"])
def list_datasets() -> list[DatasetInfo]:
    data_dir = settings.data_dir
    if not data_dir.is_dir():
        return []
    datasets: list[DatasetInfo] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images_dir = child / "images"
        image_count = len(list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))) if images_dir.is_dir() else 0
        has_sparse = (child / "sparse" / "0" / "cameras.bin").exists()
        factors = []
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and sub.name.startswith("images_"):
                try:
                    factors.append(int(sub.name.split("_")[1]))
                except ValueError:
                    pass
        datasets.append(DatasetInfo(
            name=child.name,
            image_count=image_count,
            has_sparse=has_sparse,
            downsampled_factors=factors,
            description=f"{image_count} images, COLMAP sparse model {'available' if has_sparse else 'missing'}",
        ))
    return datasets


@router.post("/reconstructions/from-dataset", response_model=UploadResponse, status_code=201)
def create_from_dataset(
    dataset_name: str = Form(...),
    name: str = Form(...),
    description: str | None = Form(None),
    frame_rate: float | None = Form(None),
    sequential_matcher_overlap: int | None = Form(None),
    colmap_mapper_type: str | None = Form(None),
    colmap_max_num_features: int | None = Form(None),
    reconstruction_backend: str | None = Form(None),
    fvdb_max_epochs: int | None = Form(None),
    fvdb_sh_degree: int | None = Form(None),
    fvdb_image_downsample_factor: int | None = Form(None),
    grut_n_iterations: int | None = Form(None),
    grut_render_method: str | None = Form(None),
    grut_strategy: str | None = Form(None),
    grut_downsample_factor: int | None = Form(None),
    splat_only_mode: bool | None = Form(None),
    collision_mesh_enabled: bool | None = Form(None),
    collision_mesh_method: str | None = Form(None),
    collision_mesh_target_faces: int | None = Form(None),
    collision_mesh_alpha: float | None = Form(None),
    collision_mesh_downsample: int | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    dataset_path = (settings.data_dir / dataset_name).resolve()
    if not dataset_path.is_dir() or not dataset_path.is_relative_to(settings.data_dir.resolve()):
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_images = dataset_path / "images"
    dataset_sparse = dataset_path / "sparse" / "0"
    if not dataset_images.is_dir() or not dataset_sparse.is_dir():
        raise HTTPException(status_code=400, detail="Dataset missing images/ or sparse/0/ directory")

    processing_params = build_processing_params(
        frame_rate, sequential_matcher_overlap, colmap_mapper_type,
        colmap_max_num_features, reconstruction_backend, fvdb_max_epochs,
        fvdb_sh_degree, fvdb_image_downsample_factor, grut_n_iterations,
        grut_render_method, grut_strategy, grut_downsample_factor,
        splat_only_mode, collision_mesh_enabled, collision_mesh_method,
        collision_mesh_target_faces, collision_mesh_alpha, collision_mesh_downsample,
    )

    reconstruction = Reconstruction(
        name=name,
        description=description,
        status=ReconstructionStatus.uploading.value,
        source_video_filename=f"{dataset_name} (dataset)",
        source_video_path=str(dataset_path),
        workspace_dir="",
        processing_params_json=processing_params.model_dump_json(exclude_none=True),
    )
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    workspace_dir = settings.storage_dir / reconstruction.id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy images and sparse model from dataset into workspace
    workspace_images = workspace_dir / "images"
    workspace_sparse = workspace_dir / "sparse" / "0"
    shutil.copytree(str(dataset_images), str(workspace_images))
    workspace_sparse.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(dataset_sparse), str(workspace_sparse))

    # Also copy any downsampled image directories (images_2, images_4, etc.)
    for sub in sorted(dataset_path.iterdir()):
        if sub.is_dir() and sub.name.startswith("images_"):
            shutil.copytree(str(sub), str(workspace_dir / sub.name))

    reconstruction.workspace_dir = str(workspace_dir)
    reconstruction.status = ReconstructionStatus.queued.value
    reconstruction.processing_step = "queued"
    reconstruction.processing_pct = 1
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    runner.enqueue(reconstruction.id)
    return UploadResponse(**serialize_reconstruction(reconstruction).model_dump())


@router.post("/reconstructions/upload", response_model=UploadResponse, status_code=201)
def upload_reconstruction(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
    frame_rate: float | None = Form(None),
    sequential_matcher_overlap: int | None = Form(None),
    colmap_mapper_type: str | None = Form(None),
    colmap_max_num_features: int | None = Form(None),
    reconstruction_backend: str | None = Form(None),
    fvdb_max_epochs: int | None = Form(None),
    fvdb_sh_degree: int | None = Form(None),
    fvdb_image_downsample_factor: int | None = Form(None),
    grut_n_iterations: int | None = Form(None),
    grut_render_method: str | None = Form(None),
    grut_strategy: str | None = Form(None),
    grut_downsample_factor: int | None = Form(None),
    splat_only_mode: bool | None = Form(None),
    collision_mesh_enabled: bool | None = Form(None),
    collision_mesh_method: str | None = Form(None),
    collision_mesh_target_faces: int | None = Form(None),
    collision_mesh_alpha: float | None = Form(None),
    collision_mesh_downsample: int | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    suffix = Path(file.filename or "upload.mov").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only MOV, MP4, and M4V uploads are supported")

    processing_params = build_processing_params(
        frame_rate,
        sequential_matcher_overlap,
        colmap_mapper_type,
        colmap_max_num_features,
        reconstruction_backend,
        fvdb_max_epochs,
        fvdb_sh_degree,
        fvdb_image_downsample_factor,
        grut_n_iterations,
        grut_render_method,
        grut_strategy,
        grut_downsample_factor,
        splat_only_mode,
        collision_mesh_enabled,
        collision_mesh_method,
        collision_mesh_target_faces,
        collision_mesh_alpha,
        collision_mesh_downsample,
    )

    reconstruction = Reconstruction(
        name=name,
        description=description,
        status=ReconstructionStatus.uploading.value,
        source_video_filename=file.filename or f"upload{suffix}",
        source_video_path="",
        workspace_dir="",
        processing_params_json=processing_params.model_dump_json(exclude_none=True),
    )
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    workspace_dir = settings.storage_dir / reconstruction.id
    source_video_path = workspace_dir / f"source{suffix}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(source_video_path)

    with source_video_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    reconstruction.workspace_dir = str(workspace_dir)
    reconstruction.source_video_path = str(source_video_path)
    reconstruction.status = ReconstructionStatus.queued.value
    reconstruction.processing_step = "queued"
    reconstruction.processing_pct = 1
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    runner.enqueue(reconstruction.id)
    return UploadResponse(**serialize_reconstruction(reconstruction).model_dump())


@router.get("/reconstructions", response_model=list[ReconstructionDetail])
def list_reconstructions(db: Session = Depends(get_db)) -> list[ReconstructionDetail]:
    reconstructions = db.query(Reconstruction).order_by(Reconstruction.created_at.desc()).all()
    return [serialize_reconstruction(item) for item in reconstructions]


@router.get("/reconstructions/{reconstruction_id}", response_model=ReconstructionDetail)
def get_reconstruction(reconstruction_id: str, db: Session = Depends(get_db)) -> ReconstructionDetail:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    return serialize_reconstruction(reconstruction)


@router.get("/reconstructions/{reconstruction_id}/status", response_model=ReconstructionStatusResponse)
def get_reconstruction_status(reconstruction_id: str, db: Session = Depends(get_db)) -> ReconstructionStatusResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    return ReconstructionStatusResponse(
        id=reconstruction.id,
        status=reconstruction.status,
        processing_step=reconstruction.processing_step,
        processing_pct=reconstruction.processing_pct,
        error_message=reconstruction.error_message,
        updated_at=reconstruction.updated_at,
    )


@router.get("/reconstructions/{reconstruction_id}/artifacts", response_model=ReconstructionArtifacts)
def get_reconstruction_artifacts(reconstruction_id: str, db: Session = Depends(get_db)) -> ReconstructionArtifacts:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    return ReconstructionArtifacts(
        source_video_url=build_download_url(reconstruction.id, "source_video"),
        splat_ply_url=build_download_url(reconstruction.id, "splat_ply") if reconstruction.artifact_ply_path else None,
        scene_usdz_url=build_download_url(reconstruction.id, "scene_usdz") if reconstruction.artifact_usdz_path else None,
        collision_mesh_url=build_download_url(reconstruction.id, "collision_mesh") if reconstruction.artifact_collision_mesh_path else None,
        sim_bundle_url=build_download_url(reconstruction.id, "sim_bundle") if reconstruction.artifact_bundle_path else None,
        run_log_url=build_download_url(reconstruction.id, "run_log") if reconstruction.artifact_log_path else None,
        metadata_url=build_download_url(reconstruction.id, "metadata") if reconstruction.artifact_metadata_path else None,
    )


@router.get("/reconstructions/{reconstruction_id}/download/{artifact}")
def download_artifact(reconstruction_id: str, artifact: str, db: Session = Depends(get_db)) -> FileResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    field_name = ARTIFACT_FIELDS.get(artifact)
    if field_name is None:
        raise HTTPException(status_code=404, detail="Unknown artifact")

    artifact_path = getattr(reconstruction, field_name)
    if not artifact_path:
        raise HTTPException(status_code=404, detail="Artifact not available")

    path = Path(artifact_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing on disk")

    media_type = None
    if artifact == "scene_usdz":
        media_type = "model/vnd.usdz+zip"
    elif artifact == "collision_mesh":
        media_type = "model/obj"
    elif artifact == "sim_bundle":
        media_type = "application/zip"
    elif artifact == "run_log":
        media_type = "text/plain"
    elif artifact == "metadata":
        media_type = "application/json"

    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/reconstructions/{reconstruction_id}/metrics", response_model=MetricsResponse)
def get_reconstruction_metrics(reconstruction_id: str, db: Session = Depends(get_db)) -> MetricsResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    workspace = Path(reconstruction.workspace_dir)
    # Use only the latest run's CSV (highest-dated folder)
    csv_files = sorted(workspace.rglob("metrics_log.csv"))
    if not csv_files:
        return MetricsResponse(id=reconstruction.id)
    latest_csv = csv_files[-1]
    entries: list[MetricsEntry] = []
    for line in latest_csv.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",", 2)
        if len(parts) != 3:
            continue
        try:
            entries.append(MetricsEntry(epoch=int(parts[0]), metric=parts[1], value=float(parts[2])))
        except (ValueError, TypeError):
            continue
    summary: dict[str, float] = {}
    if entries:
        last_epoch = max(e.epoch for e in entries)
        for e in entries:
            if e.epoch == last_epoch:
                summary[e.metric] = e.value
    # Return only the last few entries to keep the response small for LLM agents
    last_entries = [e for e in entries if e.epoch >= last_epoch - 10] if entries else []
    return MetricsResponse(id=reconstruction.id, summary=summary, entries=last_entries)


@router.post("/reconstructions/{reconstruction_id}/retry", response_model=RetryResponse)
def retry_reconstruction(
    reconstruction_id: str,
    request: RetryRequest | None = Body(default=None),
    db: Session = Depends(get_db),
) -> RetryResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    if reconstruction.status not in {ReconstructionStatus.failed.value, ReconstructionStatus.completed.value}:
        raise HTTPException(status_code=409, detail="Only completed or failed reconstructions can be retried")

    reconstruction.status = ReconstructionStatus.queued.value
    reconstruction.processing_step = "queued"
    reconstruction.processing_pct = 1
    reconstruction.error_message = None
    reconstruction.started_at = None
    reconstruction.completed_at = None
    reconstruction.artifact_ply_path = None
    reconstruction.artifact_usdz_path = None
    reconstruction.artifact_collision_mesh_path = None
    reconstruction.artifact_bundle_path = None
    reconstruction.artifact_log_path = None
    reconstruction.artifact_metadata_path = None
    if request and request.params is not None:
        # Merge new params with existing ones (new values override)
        existing_params: dict = {}
        if reconstruction.processing_params_json:
            existing_params = json.loads(reconstruction.processing_params_json)
        new_params = request.params.model_dump(exclude_none=True)
        existing_params.update(new_params)
        reconstruction.processing_params_json = json.dumps(existing_params)
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    runner.enqueue(reconstruction.id)
    return RetryResponse(**serialize_reconstruction(reconstruction).model_dump())


@router.patch("/reconstructions/{reconstruction_id}/notes", response_model=ReconstructionDetail)
def update_notes(
    reconstruction_id: str,
    body: NotesUpdate,
    db: Session = Depends(get_db),
) -> ReconstructionDetail:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    existing = reconstruction.description or ""
    if existing:
        reconstruction.description = existing + "\n" + body.notes
    else:
        reconstruction.description = body.notes
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)
    return serialize_reconstruction(reconstruction)


@router.delete("/reconstructions/{reconstruction_id}", response_model=DeleteResponse)
def delete_reconstruction(reconstruction_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    workspace = Path(reconstruction.workspace_dir)
    db.query(IterationRecord).filter(IterationRecord.reconstruction_id == reconstruction_id).delete()
    db.delete(reconstruction)
    db.commit()
    remove_workspace(workspace)
    return DeleteResponse(id=reconstruction_id)


def _build_iteration_ply_url(reconstruction_id: str, iteration: int) -> str:
    return f"{settings.api_prefix}/reconstructions/{reconstruction_id}/iterations/{iteration}/download/splat_ply"


@router.get("/reconstructions/{reconstruction_id}/iterations", response_model=IterationHistoryResponse)
def get_iteration_history(reconstruction_id: str, db: Session = Depends(get_db)) -> IterationHistoryResponse:
    get_reconstruction_or_404(db, reconstruction_id)
    records = (
        db.query(IterationRecord)
        .filter(IterationRecord.reconstruction_id == reconstruction_id)
        .order_by(IterationRecord.iteration)
        .all()
    )
    iterations = []
    for rec in records:
        params = ReconstructionParams()
        if rec.params_json:
            try:
                params = ReconstructionParams(**json.loads(rec.params_json))
            except Exception:
                pass
        iterations.append(IterationSummary(
            iteration=rec.iteration,
            params=params,
            loss=rec.loss,
            ssim=rec.ssim,
            num_gaussians=rec.num_gaussians,
            verdict=rec.verdict,
            reason=rec.reason,
            ply_url=_build_iteration_ply_url(reconstruction_id, rec.iteration) if rec.ply_path else None,
            started_at=rec.started_at,
            completed_at=rec.completed_at,
        ))
    return IterationHistoryResponse(reconstruction_id=reconstruction_id, iterations=iterations)


@router.get("/reconstructions/{reconstruction_id}/iterations/{iteration}/download/splat_ply")
def download_iteration_ply(reconstruction_id: str, iteration: int, db: Session = Depends(get_db)) -> FileResponse:
    get_reconstruction_or_404(db, reconstruction_id)
    record = (
        db.query(IterationRecord)
        .filter(IterationRecord.reconstruction_id == reconstruction_id, IterationRecord.iteration == iteration)
        .first()
    )
    if record is None or not record.ply_path:
        raise HTTPException(status_code=404, detail="Iteration PLY not available")
    path = Path(record.ply_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Iteration PLY file missing on disk")
    return FileResponse(path, filename=f"iter_{iteration}_fvdb_output.ply")


@router.patch("/reconstructions/{reconstruction_id}/iterations/{iteration}/verdict")
def update_iteration_verdict(
    reconstruction_id: str,
    iteration: int,
    body: dict = Body(...),
    db: Session = Depends(get_db),
) -> dict:
    record = (
        db.query(IterationRecord)
        .filter(IterationRecord.reconstruction_id == reconstruction_id, IterationRecord.iteration == iteration)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Iteration record not found")
    if "verdict" in body:
        record.verdict = body["verdict"]
    if "reason" in body:
        record.reason = body["reason"]
    db.add(record)
    db.commit()
    return {"iteration": record.iteration, "verdict": record.verdict, "reason": record.reason}
