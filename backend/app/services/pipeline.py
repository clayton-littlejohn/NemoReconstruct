from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import IterationRecord, Reconstruction, ReconstructionStatus


PIPELINE_INFO = {
    "slug": "fvdb-isaac-sim-mvp",
    "name": "fVDB Splat Demo",
    "description": "Video upload -> ffmpeg frame extraction -> COLMAP -> fVDB splat (PLY)",
    "source_type": "video",
    "output_types": ["ply"],
    "steps": [
        "extract_frames",
        "feature_extraction",
        "feature_matching",
        "sparse_reconstruction",
        "fvdb_reconstruction",
    ],
    "requirements": ["ffmpeg", "colmap", "frgs"],
    "tunable_params": {
        "frame_rate": "Frames per second extracted by ffmpeg (0.25-12.0)",
        "sequential_matcher_overlap": "COLMAP sequential matcher overlap window (2-50)",
        "colmap_mapper_type": "COLMAP mapper algorithm: 'incremental' (default, robust) or 'global' (faster on large scenes, uses GLOMAP)",
        "colmap_max_num_features": "Max SIFT features per image (1000-32768, default 8192). More features = better matching but slower",
        "fvdb_max_epochs": "fVDB training epochs (5-500)",
        "fvdb_sh_degree": "Spherical harmonics degree for splats (0-4)",
        "fvdb_image_downsample_factor": "Input image downsampling for fVDB (1-12)",
        "splat_only_mode": "If true, skip USDZ conversion and ZIP bundle generation",
    },
}


class PipelineError(RuntimeError):
    pass


def load_processing_params(reconstruction: Reconstruction) -> dict[str, object]:
    if not reconstruction.processing_params_json:
        return {}
    try:
        parsed = json.loads(reconstruction.processing_params_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_fvdb_conda_prefix() -> Path:
    return settings.fvdb_conda_root.expanduser() / settings.fvdb_conda_env


def build_fvdb_env() -> dict[str, str]:
    env = dict(os.environ)
    conda_prefix = get_fvdb_conda_prefix()

    env["CUDA_HOME"] = str(conda_prefix)
    env["PATH"] = f"{conda_prefix / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_prefix / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CONDA_PREFIX"] = str(conda_prefix)
    env["CONDA_DEFAULT_ENV"] = settings.fvdb_conda_env
    env["PYTHONUNBUFFERED"] = "1"
    env["FVDB_HEADLESS"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NVIDIA_TF32_OVERRIDE"] = "1"
    env["CUDNN_BENCHMARK"] = "1"
    env["CUDNN_V9_ALLOW_TENSOR_OP_MATH_FP32"] = "1"
    env["TORCH_FLOAT32_MATMUL_PRECISION"] = "high"
    return env


@dataclass
class JobPaths:
    root: Path
    source_video: Path
    images_dir: Path
    colmap_database: Path
    sparse_dir: Path
    fvdb_dir: Path
    log_path: Path
    metadata_path: Path
    bundle_path: Path


def build_job_paths(reconstruction: Reconstruction) -> JobPaths:
    root = Path(reconstruction.workspace_dir)
    return JobPaths(
        root=root,
        source_video=Path(reconstruction.source_video_path),
        images_dir=root / "images",
        colmap_database=root / "database.db",
        sparse_dir=root / "sparse",
        fvdb_dir=root / "fvdb_output",
        log_path=root / "run.log",
        metadata_path=root / "metadata.json",
        bundle_path=root / "isaac_sim_bundle.zip",
    )


def update_reconstruction(
    db: Session,
    reconstruction: Reconstruction,
    *,
    status: ReconstructionStatus | None = None,
    step: str | None = None,
    pct: int | None = None,
    error_message: str | None = None,
) -> None:
    if status is not None:
        reconstruction.status = status.value
    if step is not None:
        reconstruction.processing_step = step
    if pct is not None:
        reconstruction.processing_pct = pct
    if error_message is not None:
        reconstruction.error_message = error_message
    reconstruction.updated_at = datetime.now(timezone.utc)
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)


def require_binary(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise PipelineError(f"Required binary '{binary}' is not available on PATH")
    return resolved


def resolve_frgs_binary() -> str:
    configured = Path(settings.frgs_bin).expanduser()
    if configured.is_file():
        return str(configured)

    resolved = shutil.which(settings.frgs_bin)
    if resolved is not None:
        return resolved

    conda_frgs = get_fvdb_conda_prefix() / "bin" / "frgs"
    if conda_frgs.is_file():
        return str(conda_frgs)

    raise PipelineError(
        f"Required binary '{settings.frgs_bin}' is not available on PATH and no fVDB frgs executable was found at {conda_frgs}"
    )


def run_command(
    command: list[str],
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n")
        log_file.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise PipelineError(f"Command failed ({process.returncode}): {' '.join(command)}")


def count_frames(frames_dir: Path) -> int:
    return len(sorted(frames_dir.glob("*.png")))


def reset_workspace(paths: JobPaths) -> None:
    for path in [paths.images_dir, paths.sparse_dir, paths.fvdb_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    for file_path in [paths.colmap_database, paths.log_path, paths.metadata_path, paths.bundle_path]:
        if file_path.exists():
            file_path.unlink()

    paths.log_path.touch()


def locate_sparse_model(sparse_dir: Path) -> Path:
    candidate = sparse_dir / "0"
    if candidate.exists():
        return candidate
    for child in sorted(sparse_dir.iterdir()):
        if child.is_dir():
            return child
    raise PipelineError("COLMAP mapper did not produce a sparse model")


def locate_first_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.rglob(pattern))
    return matches[0] if matches else None


def write_metadata(reconstruction: Reconstruction, paths: JobPaths) -> None:
    processing_params = load_processing_params(reconstruction)
    payload = {
        "id": reconstruction.id,
        "name": reconstruction.name,
        "description": reconstruction.description,
        "status": reconstruction.status,
        "pipeline_slug": reconstruction.pipeline_slug,
        "source_video_filename": reconstruction.source_video_filename,
        "frame_count": reconstruction.frame_count,
        "created_at": reconstruction.created_at.isoformat(),
        "updated_at": reconstruction.updated_at.isoformat(),
        "artifact_ply_path": reconstruction.artifact_ply_path,
        "artifact_usdz_path": reconstruction.artifact_usdz_path,
        "artifact_log_path": reconstruction.artifact_log_path,
        "processing_params": processing_params,
    }
    paths.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def package_bundle(reconstruction: Reconstruction, paths: JobPaths) -> None:
    with zipfile.ZipFile(paths.bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(paths.metadata_path, arcname="metadata.json")
        archive.write(paths.log_path, arcname="run.log")
        if reconstruction.artifact_ply_path:
            archive.write(reconstruction.artifact_ply_path, arcname="fvdb_output.ply")
        if reconstruction.artifact_usdz_path:
            archive.write(reconstruction.artifact_usdz_path, arcname="fvdb_output.usdz")


def save_iteration_snapshot(db: Session, reconstruction: Reconstruction, paths: JobPaths) -> None:
    """Copy PLY to iterations/<N>/ and record an IterationRecord in the DB."""
    # Determine iteration number from existing records
    existing = (
        db.query(IterationRecord)
        .filter(IterationRecord.reconstruction_id == reconstruction.id)
        .count()
    )
    iteration = existing + 1

    # Copy PLY to a preserved location
    ply_dest: str | None = None
    if reconstruction.artifact_ply_path and Path(reconstruction.artifact_ply_path).exists():
        iter_dir = paths.root / "iterations" / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        dest = iter_dir / "fvdb_output.ply"
        shutil.copy2(reconstruction.artifact_ply_path, dest)
        ply_dest = str(dest)

    # Read latest metrics summary
    csv_files = sorted(paths.root.rglob("metrics_log.csv"))
    metrics_summary: dict[str, float] = {}
    if csv_files:
        latest_csv = csv_files[-1]
        entries: list[tuple[int, str, float]] = []
        for line in latest_csv.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(",", 2)
            if len(parts) != 3:
                continue
            try:
                entries.append((int(parts[0]), parts[1], float(parts[2])))
            except (ValueError, TypeError):
                continue
        if entries:
            last_epoch = max(e[0] for e in entries)
            for epoch, metric, value in entries:
                if epoch == last_epoch:
                    metrics_summary[metric] = value

    record = IterationRecord(
        reconstruction_id=reconstruction.id,
        iteration=iteration,
        params_json=reconstruction.processing_params_json,
        metrics_json=json.dumps(metrics_summary) if metrics_summary else None,
        ply_path=ply_dest,
        loss=metrics_summary.get("reconstruct/loss"),
        ssim=metrics_summary.get("reconstruct/ssimloss"),
        num_gaussians=int(metrics_summary["reconstruct/num_gaussians"]) if "reconstruct/num_gaussians" in metrics_summary else None,
        started_at=reconstruction.started_at,
        completed_at=reconstruction.completed_at,
    )
    db.add(record)
    db.commit()


def process_reconstruction_job(db: Session, reconstruction_id: str) -> None:
    reconstruction = db.get(Reconstruction, reconstruction_id)
    if reconstruction is None:
        return

    paths = build_job_paths(reconstruction)
    processing_params = load_processing_params(reconstruction)

    frame_rate = float(processing_params.get("frame_rate", settings.frame_rate))
    sequential_matcher_overlap = int(processing_params.get("sequential_matcher_overlap", settings.sequential_matcher_overlap))
    colmap_mapper_type = str(processing_params.get("colmap_mapper_type", settings.colmap_mapper_type))
    colmap_max_num_features = int(processing_params.get("colmap_max_num_features", settings.colmap_max_num_features))
    fvdb_max_epochs = int(processing_params.get("fvdb_max_epochs", settings.fvdb_max_epochs))
    fvdb_sh_degree = int(processing_params.get("fvdb_sh_degree", settings.fvdb_sh_degree))
    fvdb_image_downsample_factor = int(
        processing_params.get("fvdb_image_downsample_factor", settings.fvdb_image_downsample_factor)
    )
    splat_only_mode = bool(processing_params.get("splat_only_mode", settings.splat_only_mode))

    paths.root.mkdir(parents=True, exist_ok=True)
    reset_workspace(paths)

    reconstruction.started_at = datetime.now(timezone.utc)
    update_reconstruction(db, reconstruction, status=ReconstructionStatus.extracting_frames, step="ffmpeg", pct=5)

    try:
        ffmpeg_bin = require_binary(settings.ffmpeg_bin)
        colmap_bin = require_binary(settings.colmap_bin)
        frgs_bin = resolve_frgs_binary()
        frgs_env = build_fvdb_env() if Path(frgs_bin).is_file() and "envs" in Path(frgs_bin).parts else None

        run_command(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(paths.source_video),
                "-vf",
                f"fps={frame_rate}",
                str(paths.images_dir / "frame_%06d.png"),
            ],
            paths.log_path,
        )
        reconstruction.frame_count = count_frames(paths.images_dir)
        db.add(reconstruction)
        db.commit()

        if not reconstruction.frame_count:
            raise PipelineError("Frame extraction produced zero frames")

        update_reconstruction(db, reconstruction, status=ReconstructionStatus.feature_extraction, step="colmap_feature_extractor", pct=20)
        run_command(
            [
                colmap_bin,
                "feature_extractor",
                "--database_path",
                str(paths.colmap_database),
                "--image_path",
                str(paths.images_dir),
                "--ImageReader.single_camera",
                "1",
                "--SiftExtraction.max_num_features",
                str(colmap_max_num_features),
            ],
            paths.log_path,
        )

        update_reconstruction(db, reconstruction, status=ReconstructionStatus.feature_matching, step="colmap_sequential_matcher", pct=35)
        run_command(
            [
                colmap_bin,
                "sequential_matcher",
                "--database_path",
                str(paths.colmap_database),
                "--SequentialMatching.overlap",
                str(sequential_matcher_overlap),
            ],
            paths.log_path,
        )

        mapper_cmd = "global_mapper" if colmap_mapper_type == "global" else "mapper"
        update_reconstruction(db, reconstruction, status=ReconstructionStatus.sparse_reconstruction, step=f"colmap_{mapper_cmd}", pct=55)
        run_command(
            [
                colmap_bin,
                mapper_cmd,
                "--database_path",
                str(paths.colmap_database),
                "--image_path",
                str(paths.images_dir),
                "--output_path",
                str(paths.sparse_dir),
            ],
            paths.log_path,
        )

        locate_sparse_model(paths.sparse_dir)

        update_reconstruction(db, reconstruction, status=ReconstructionStatus.fvdb_reconstruction, step="frgs_reconstruct", pct=75)
        output_ply = paths.fvdb_dir / "fvdb_output.ply"
        run_command(
            [
                frgs_bin,
                "reconstruct",
                str(paths.root),
                "--out-path",
                str(output_ply),
                "--dataset-type",
                "colmap",
                "--device",
                "cuda",
                "--cfg.max-epochs",
                str(fvdb_max_epochs),
                "--cfg.sh-degree",
                str(fvdb_sh_degree),
                "--tx.image-downsample-factor",
                str(fvdb_image_downsample_factor),
                "--update-viz-every",
                "-1",
                "--io.no-save-images",
            ],
            paths.log_path,
            cwd=paths.root,
            env=frgs_env,
        )

        if not output_ply.exists():
            alt_ply = locate_first_file(paths.fvdb_dir, "*.ply")
            if alt_ply is None:
                raise PipelineError("fVDB completed without producing a PLY output")
            output_ply = alt_ply

        reconstruction.artifact_ply_path = str(output_ply)
        db.add(reconstruction)
        db.commit()

        reconstruction.artifact_log_path = str(paths.log_path)
        write_metadata(reconstruction, paths)
        reconstruction.artifact_metadata_path = str(paths.metadata_path)
        if not splat_only_mode:
            update_reconstruction(db, reconstruction, status=ReconstructionStatus.exporting, step="frgs_convert", pct=90)
            output_usdz = paths.fvdb_dir / "fvdb_output.usdz"
            try:
                run_command(
                    [frgs_bin, "convert", str(output_ply), str(output_usdz)],
                    paths.log_path,
                    cwd=paths.root,
                    env=frgs_env,
                )
                if output_usdz.exists():
                    reconstruction.artifact_usdz_path = str(output_usdz)
            except PipelineError:
                with paths.log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("USDZ conversion failed; continuing with PLY-only output.\n")

            package_bundle(reconstruction, paths)
            reconstruction.artifact_bundle_path = str(paths.bundle_path)

        reconstruction.completed_at = datetime.now(timezone.utc)

        update_reconstruction(db, reconstruction, status=ReconstructionStatus.completed, step="done", pct=100, error_message=None)
        db.add(reconstruction)
        db.commit()

        # Preserve iteration snapshot (copy PLY + record metrics/params)
        save_iteration_snapshot(db, reconstruction, paths)
    except Exception as exc:
        update_reconstruction(
            db,
            reconstruction,
            status=ReconstructionStatus.failed,
            step="failed",
            pct=reconstruction.processing_pct,
            error_message=str(exc),
        )
