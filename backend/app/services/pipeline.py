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
    "slug": "nemo-reconstruct-mvp",
    "name": "NemoReconstruct Pipeline",
    "description": "Video upload -> ffmpeg frame extraction -> COLMAP -> 3DGRUT (or fVDB) reconstruction -> NuRec USDZ + PLY",
    "source_type": "video",
    "output_types": ["ply", "usdz", "collision_mesh_obj"],
    "steps": [
        "extract_frames",
        "feature_extraction",
        "feature_matching",
        "sparse_reconstruction",
        "grut_or_fvdb_reconstruction",
        "generate_collision_mesh",
    ],
    "requirements": ["ffmpeg", "colmap", "3dgrut", "frgs"],
    "tunable_params": {
        "frame_rate": "Frames per second extracted by ffmpeg (0.25-12.0, default 2.0)",
        "sequential_matcher_overlap": "COLMAP sequential matcher overlap window (2-50, default 12)",
        "colmap_mapper_type": "COLMAP mapper algorithm: 'incremental' (default, robust) or 'global' (faster on large scenes, uses GLOMAP)",
        "colmap_max_num_features": "Max SIFT features per image (1000-32768, default 8192). More features = better matching but slower",
        "reconstruction_backend": "Reconstruction backend: 'fvdb' (default, fVDB Reality Capture / frgs, fast) or '3dgrut' (NVIDIA 3D Gaussian Ray Tracing, higher quality)",
        "fvdb_max_epochs": "fVDB training epochs (5-500, default 40). Only used when backend=fvdb",
        "fvdb_sh_degree": "Spherical harmonics degree for splats (0-4, default 3). Only used when backend=fvdb",
        "fvdb_image_downsample_factor": "Input image downsampling for fVDB (1-12, default 6). Only used when backend=fvdb",
        "grut_n_iterations": "3DGRUT training iterations (1000-100000, default 30000). Only used when backend=3dgrut",
        "grut_render_method": "3DGRUT render method: '3dgrt' (ray tracing, default) or '3dgut' (unscented transform splatting). Only used when backend=3dgrut",
        "grut_strategy": "3DGRUT densification strategy: 'gs' (standard, default) or 'mcmc' (MCMC, better for tricky scenes). Only used when backend=3dgrut",
        "grut_downsample_factor": "Input image downsampling for 3DGRUT (1-12, default 2). Only used when backend=3dgrut",
        "splat_only_mode": "If true, skip USDZ conversion and ZIP bundle generation (default false)",
        "collision_mesh_enabled": "Generate a collision mesh from Gaussian centroids for Isaac Sim physics (default true)",
        "collision_mesh_method": "Collision mesh algorithm: 'alpha' (concave alpha shape, default) or 'convex_hull' (fast convex wrapper)",
        "collision_mesh_target_faces": "Target face count after decimation (500-500000, default 50000). Lower = faster physics, higher = more accurate collisions",
        "collision_mesh_alpha": "Alpha parameter for alpha-shape method (0.01-100.0, default auto). Higher = tighter fit, lower = smoother. 0 = auto-compute from point density",
        "collision_mesh_downsample": "Point cloud downsampling factor before meshing (1-64, default 4). Higher = faster but less detailed mesh",
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


def get_grut_conda_prefix() -> Path:
    return settings.fvdb_conda_root.expanduser() / settings.grut_conda_env


def build_3dgrut_env() -> dict[str, str]:
    env = dict(os.environ)
    conda_prefix = get_grut_conda_prefix()

    # CUDA headers/libs are in /usr/local/cuda, not the conda env
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{conda_prefix / 'bin'}:{cuda_home}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_prefix / 'lib'}:{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    # Ensure g++ can find cuda_runtime.h for JIT C++ extension builds
    env["CPATH"] = f"{cuda_home}/include:{env.get('CPATH', '')}"
    env["CONDA_PREFIX"] = str(conda_prefix)
    env["CONDA_DEFAULT_ENV"] = settings.grut_conda_env
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NVIDIA_TF32_OVERRIDE"] = "1"
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
    grut_dir: Path
    collision_mesh_dir: Path
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
        grut_dir=root / "grut_output",
        collision_mesh_dir=root / "collision_mesh",
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
    return len([f for f in frames_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])


def create_downsampled_images(images_dir: Path, factor: int, log_path: Path) -> None:
    """Create images_{factor}/ with resized images for 3DGRUT COLMAP convention."""
    if factor <= 1:
        return
    dest = images_dir.parent / f"images_{factor}"
    if dest.exists() and any(dest.iterdir()):
        return  # already created
    dest.mkdir(parents=True, exist_ok=True)
    for src_img in sorted(images_dir.iterdir()):
        if src_img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        run_command(
            ["ffmpeg", "-y", "-i", str(src_img),
             "-vf", f"scale=iw/{factor}:ih/{factor}",
             str(dest / src_img.name)],
            log_path,
        )


def has_valid_preprocessing(paths: JobPaths) -> bool:
    """Check if ffmpeg + COLMAP output already exists and is usable."""
    if not paths.images_dir.exists():
        return False
    has_images = any(
        f.suffix.lower() in (".png", ".jpg", ".jpeg")
        for f in paths.images_dir.iterdir()
    )
    if not has_images:
        return False
    sparse_model = paths.sparse_dir / "0"
    if not sparse_model.exists():
        return False
    # Must have at least cameras.bin and one of images.bin/frames.bin
    has_cameras = (sparse_model / "cameras.bin").exists()
    has_frames = (sparse_model / "images.bin").exists() or (sparse_model / "frames.bin").exists()
    return has_cameras and has_frames


def reset_workspace(paths: JobPaths, reconstruction_only: bool = False) -> None:
    if reconstruction_only:
        # Only reset reconstruction output; keep images, sparse, database,
        # and any pre-existing downsampled image dirs (images_2, images_4, etc.)
        for path in [paths.fvdb_dir, paths.grut_dir, paths.collision_mesh_dir]:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        for file_path in [paths.log_path, paths.metadata_path, paths.bundle_path]:
            if file_path.exists():
                file_path.unlink()
        paths.log_path.touch()
        return

    for path in [paths.images_dir, paths.sparse_dir, paths.fvdb_dir, paths.grut_dir, paths.collision_mesh_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    # Clean up undistorted and downsampled image dirs from prior runs
    for pattern in ["undistorted", "images_*"]:
        for old_dir in paths.root.glob(pattern):
            if old_dir.is_dir():
                shutil.rmtree(old_dir)

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


def resolve_grut_config_name(render_method: str, strategy: str) -> str:
    """Map render_method + strategy to a 3DGRUT Hydra config name."""
    suffix = "_mcmc" if strategy == "mcmc" else ""
    return f"apps/colmap_{render_method}{suffix}.yaml"


def resolve_grut_python() -> str:
    conda_prefix = get_grut_conda_prefix()
    python = conda_prefix / "bin" / "python"
    if not python.is_file():
        raise PipelineError(f"3DGRUT conda python not found at {python}")
    return str(python)


def resolve_grut_train_script() -> str:
    train_script = settings.grut_install_dir / "train.py"
    if not train_script.is_file():
        raise PipelineError(f"3DGRUT train.py not found at {train_script}")
    return str(train_script)


def write_grut_metrics_csv(grut_output_dir: Path, metrics_csv_path: Path) -> None:
    """Parse TensorBoard events from the 3DGRUT output dir and write metrics_log.csv."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        # Fallback: write a minimal CSV if tensorboard isn't available in the backend env
        metrics_csv_path.write_text("", encoding="utf-8")
        return

    # Find the tensorboard events directory (deepest run dir)
    event_files = sorted(grut_output_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        metrics_csv_path.write_text("", encoding="utf-8")
        return

    events_dir = str(event_files[-1].parent)
    ea = EventAccumulator(events_dir)
    ea.Reload()

    # Map 3DGRUT metric names to NemoReconstruct metric names
    # Training metrics are logged per step; validation/test metrics (psnr, ssim)
    # only appear when val_frequency triggers a validation run or test_last=true.
    # 3DGRUT writes test metrics as psnr/test, ssim/test (not psnr/val).
    metric_map = {
        "loss/total/train": "reconstruct/loss",
        "loss/l1/train": "reconstruct/l1loss",
        "loss/ssim/train": "reconstruct/ssimloss",
        "psnr/train": "reconstruct/psnr",
        "ssim/train": "reconstruct/ssim",
        "psnr/val": "reconstruct/psnr",
        "ssim/val": "reconstruct/ssim",
        "psnr/test": "reconstruct/psnr",
        "ssim/test": "reconstruct/ssim",
        "num_particles/train": "reconstruct/num_gaussians",
    }

    lines: list[str] = []
    for tb_name, csv_name in metric_map.items():
        if tb_name in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tb_name):
                lines.append(f"{event.step},{csv_name},{event.value}")

    # Sort by step for consistent output
    lines.sort(key=lambda l: int(l.split(",", 1)[0]))
    metrics_csv_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def locate_grut_ply(grut_dir: Path) -> Path | None:
    """Find the PLY exported by 3DGRUT."""
    # 3DGRUT writes to <out_dir>/<experiment>/<run_name>/export_last.ply
    matches = sorted(grut_dir.rglob("export_last.ply"))
    return matches[-1] if matches else None


def locate_grut_usdz(grut_dir: Path) -> Path | None:
    """Find the USDZ exported by 3DGRUT."""
    matches = sorted(grut_dir.rglob("export_last.usdz"))
    return matches[-1] if matches else None


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
        ssim=metrics_summary.get("reconstruct/ssim"),
        num_gaussians=int(metrics_summary["reconstruct/num_gaussians"]) if "reconstruct/num_gaussians" in metrics_summary else None,
        started_at=reconstruction.started_at,
        completed_at=reconstruction.completed_at,
    )
    db.add(record)
    db.commit()


def run_collision_mesh_generation(
    paths: JobPaths,
    ply_path: str,
    *,
    method: str = "alpha",
    target_faces: int = 50000,
    alpha: float = 0.0,
    downsample: int = 4,
) -> tuple[Path | None, dict]:
    """Run the collision mesh generation script as a subprocess in the 3dgrut conda env.

    Returns (output_obj_path, metrics_dict).
    """
    grut_python = resolve_grut_python()
    grut_env = build_3dgrut_env()

    script = Path(__file__).parent / "generate_collision_mesh.py"
    output_obj = paths.collision_mesh_dir / "collision_mesh.obj"
    paths.collision_mesh_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        grut_python,
        str(script),
        ply_path,
        str(output_obj),
        "--method", method,
        "--target-faces", str(target_faces),
        "--downsample", str(downsample),
    ]
    if alpha > 0:
        cmd.extend(["--alpha", str(alpha)])

    result = subprocess.run(
        cmd,
        env=grut_env,
        capture_output=True,
        text=True,
        check=False,
    )

    # Append stdout/stderr to the run log
    with paths.log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n")
        if result.stdout:
            log_file.write(result.stdout)
        if result.stderr:
            log_file.write(result.stderr)

    if result.returncode != 0:
        raise PipelineError(f"Collision mesh generation failed ({result.returncode})")

    # Parse metrics from stdout (JSON)
    metrics: dict = {}
    try:
        metrics = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        pass

    if output_obj.exists():
        return output_obj, metrics
    return None, metrics


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
    reconstruction_backend = str(processing_params.get("reconstruction_backend", settings.reconstruction_backend))
    fvdb_max_epochs = int(processing_params.get("fvdb_max_epochs", settings.fvdb_max_epochs))
    fvdb_sh_degree = int(processing_params.get("fvdb_sh_degree", settings.fvdb_sh_degree))
    fvdb_image_downsample_factor = int(
        processing_params.get("fvdb_image_downsample_factor", settings.fvdb_image_downsample_factor)
    )
    grut_n_iterations = int(processing_params.get("grut_n_iterations", settings.grut_n_iterations))
    grut_render_method = str(processing_params.get("grut_render_method", settings.grut_render_method))
    grut_strategy = str(processing_params.get("grut_strategy", settings.grut_strategy))
    grut_downsample_factor = int(processing_params.get("grut_downsample_factor", settings.grut_downsample_factor))
    splat_only_mode = bool(processing_params.get("splat_only_mode", settings.splat_only_mode))
    collision_mesh_enabled = bool(processing_params.get("collision_mesh_enabled", settings.collision_mesh_enabled))
    collision_mesh_method = str(processing_params.get("collision_mesh_method", settings.collision_mesh_method))
    collision_mesh_target_faces = int(processing_params.get("collision_mesh_target_faces", settings.collision_mesh_target_faces))
    collision_mesh_alpha = float(processing_params.get("collision_mesh_alpha", settings.collision_mesh_alpha))
    collision_mesh_downsample = int(processing_params.get("collision_mesh_downsample", settings.collision_mesh_downsample))

    paths.root.mkdir(parents=True, exist_ok=True)

    # If valid preprocessing data exists (images + sparse model), skip ffmpeg + COLMAP
    skip_preprocessing = has_valid_preprocessing(paths)
    reset_workspace(paths, reconstruction_only=skip_preprocessing)

    reconstruction.started_at = datetime.now(timezone.utc)

    try:
        ffmpeg_bin = require_binary(settings.ffmpeg_bin)
        colmap_bin = require_binary(settings.colmap_bin)

        # Resolve backend-specific dependencies upfront
        if reconstruction_backend == "3dgrut":
            grut_python = resolve_grut_python()
            grut_train = resolve_grut_train_script()
            grut_env = build_3dgrut_env()
        else:
            frgs_bin = resolve_frgs_binary()
            frgs_env = build_fvdb_env() if Path(frgs_bin).is_file() and "envs" in Path(frgs_bin).parts else None

        if skip_preprocessing:
            reconstruction.frame_count = count_frames(paths.images_dir)
            db.add(reconstruction)
            db.commit()
            update_reconstruction(db, reconstruction, status=ReconstructionStatus.sparse_reconstruction, step="skipped_preprocessing", pct=55)
        else:
            update_reconstruction(db, reconstruction, status=ReconstructionStatus.extracting_frames, step="ffmpeg", pct=5)
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

            # ── Undistort if using 3DGRUT (requires PINHOLE cameras) ──
            if reconstruction_backend == "3dgrut":
                undistorted_dir = paths.root / "undistorted"
                run_command(
                    [
                        colmap_bin,
                        "image_undistorter",
                        "--image_path",
                        str(paths.images_dir),
                        "--input_path",
                        str(paths.sparse_dir / "0"),
                        "--output_path",
                        str(undistorted_dir),
                        "--output_type",
                        "COLMAP",
                    ],
                    paths.log_path,
                )
                # Replace images and sparse with undistorted versions
                undist_images = undistorted_dir / "images"
                undist_sparse = undistorted_dir / "sparse"
                if undist_images.exists():
                    shutil.rmtree(paths.images_dir)
                    undist_images.rename(paths.images_dir)
                if undist_sparse.exists():
                    shutil.rmtree(paths.sparse_dir)
                    paths.sparse_dir.mkdir(parents=True)
                    # image_undistorter writes model files directly; 3DGRUT expects sparse/0/
                    undist_sparse.rename(paths.sparse_dir / "0")

        # ── Branch: 3DGRUT or fVDB ──────────────────────────────────
        if reconstruction_backend == "3dgrut":
            update_reconstruction(db, reconstruction, status=ReconstructionStatus.grut_reconstruction, step="3dgrut_train", pct=75)

            config_name = resolve_grut_config_name(grut_render_method, grut_strategy)
            export_usdz = not splat_only_mode

            if grut_downsample_factor > 1:
                create_downsampled_images(paths.images_dir, grut_downsample_factor, paths.log_path)

            run_command(
                [
                    grut_python,
                    grut_train,
                    "--config-name",
                    config_name,
                    f"path={paths.root}",
                    f"out_dir={paths.grut_dir}",
                    "experiment_name=reconstruction",
                    f"n_iterations={grut_n_iterations}",
                    f"dataset.downsample_factor={grut_downsample_factor}",
                    "export_ply.enabled=true",
                    f"export_usdz.enabled={str(export_usdz).lower()}",
                    "with_gui=false",
                    f"val_frequency={grut_n_iterations}",
                    "test_last=true",
                    "compute_extra_metrics=true",
                ],
                paths.log_path,
                cwd=settings.grut_install_dir,
                env=grut_env,
            )

            output_ply_path = locate_grut_ply(paths.grut_dir)
            if output_ply_path is None:
                raise PipelineError("3DGRUT completed without producing a PLY output")

            reconstruction.artifact_ply_path = str(output_ply_path)

            # Convert TensorBoard events to metrics_log.csv for compatibility
            metrics_csv = paths.grut_dir / "metrics_log.csv"
            write_grut_metrics_csv(paths.grut_dir, metrics_csv)

            if not splat_only_mode:
                output_usdz_path = locate_grut_usdz(paths.grut_dir)
                if output_usdz_path is not None:
                    reconstruction.artifact_usdz_path = str(output_usdz_path)

            db.add(reconstruction)
            db.commit()

        else:
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
        if not splat_only_mode and reconstruction_backend != "3dgrut":
            update_reconstruction(db, reconstruction, status=ReconstructionStatus.exporting, step="frgs_convert", pct=90)
            output_usdz = paths.fvdb_dir / "fvdb_output.usdz"
            try:
                run_command(
                    [frgs_bin, "convert", str(reconstruction.artifact_ply_path), str(output_usdz)],
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

        # ── Collision mesh generation ────────────────────────────────
        if collision_mesh_enabled and reconstruction.artifact_ply_path:
            update_reconstruction(
                db, reconstruction,
                status=ReconstructionStatus.generating_collision_mesh,
                step="collision_mesh", pct=92,
            )
            try:
                collision_obj, collision_metrics = run_collision_mesh_generation(
                    paths,
                    reconstruction.artifact_ply_path,
                    method=collision_mesh_method,
                    target_faces=collision_mesh_target_faces,
                    alpha=collision_mesh_alpha,
                    downsample=collision_mesh_downsample,
                )
                if collision_obj is not None:
                    reconstruction.artifact_collision_mesh_path = str(collision_obj)
                    # Write collision metrics to metadata
                    collision_meta_path = paths.collision_mesh_dir / "collision_metrics.json"
                    collision_meta_path.write_text(
                        json.dumps(collision_metrics, indent=2), encoding="utf-8"
                    )
            except PipelineError:
                with paths.log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("Collision mesh generation failed; continuing without collision mesh.\n")

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
