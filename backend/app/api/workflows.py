from __future__ import annotations

import os
import signal
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models import Workflow
from app.schemas import WorkflowDetail, WorkflowStateUpdate

router = APIRouter(prefix=settings.api_prefix, tags=["workflows"])

ALLOWED_EXTENSIONS = {".mov", ".mp4", ".m4v"}


def serialize_workflow(w: Workflow) -> WorkflowDetail:
    return WorkflowDetail(
        id=w.id,
        scene_name=w.scene_name,
        video_filename=w.video_filename,
        status=w.status,
        current_agent=w.current_agent,
        current_step=w.current_step,
        iteration=w.iteration,
        max_iterations=w.max_iterations,
        last_verdict=w.last_verdict,
        last_reason=w.last_reason,
        reconstruction_id=w.reconstruction_id,
        error_message=w.error_message,
        pid=w.pid,
        created_at=w.created_at,
        updated_at=w.updated_at,
    )


@router.get("/workflows", response_model=list[WorkflowDetail])
def list_workflows(db: Session = Depends(get_db)) -> list[WorkflowDetail]:
    workflows = db.query(Workflow).order_by(Workflow.created_at.desc()).all()
    return [serialize_workflow(w) for w in workflows]


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetail)
def get_workflow(workflow_id: str, db: Session = Depends(get_db)) -> WorkflowDetail:
    w = db.get(Workflow, workflow_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return serialize_workflow(w)


@router.post("/workflows/start", response_model=WorkflowDetail, status_code=201)
def start_workflow(
    file: UploadFile = File(...),
    scene_name: str = Form(...),
    max_iterations: int = Form(3),
    db: Session = Depends(get_db),
) -> WorkflowDetail:
    suffix = Path(file.filename or "upload.mov").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only MOV, MP4, and M4V uploads are supported")

    w = Workflow(
        scene_name=scene_name,
        video_filename=file.filename or f"upload{suffix}",
        video_path="",
        status="pending",
        max_iterations=max_iterations,
    )
    db.add(w)
    db.commit()
    db.refresh(w)

    # Save video to a known location
    video_dir = settings.base_dir / "workflow_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{w.id}{suffix}"
    with video_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    w.video_path = str(video_path)
    w.status = "running"
    w.current_step = "starting"
    db.add(w)
    db.commit()
    db.refresh(w)

    # Launch orchestrate.sh in background
    orchestrate_script = settings.base_dir / "nemoclaw" / "orchestrate.sh"
    if not orchestrate_script.exists():
        w.status = "failed"
        w.error_message = "orchestrate.sh not found"
        db.add(w)
        db.commit()
        db.refresh(w)
        return serialize_workflow(w)

    env_vars = {
        "WORKFLOW_ID": w.id,
        "WORKFLOW_API_URL": f"http://127.0.0.1:{settings.api_prefix.split(':')[-1] if ':' in settings.api_prefix else '8010'}",
        "AGENT_TIMEOUT": "600",
    }

    proc_env = os.environ.copy()
    proc_env.update(env_vars)
    proc_env["WORKFLOW_API_URL"] = "http://127.0.0.1:8010"

    proc = subprocess.Popen(
        [str(orchestrate_script), str(video_path), scene_name, str(max_iterations)],
        env=proc_env,
        stdout=open(video_dir / f"{w.id}.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    w.pid = proc.pid
    db.add(w)
    db.commit()
    db.refresh(w)

    return serialize_workflow(w)


@router.patch("/workflows/{workflow_id}/state", response_model=WorkflowDetail)
def update_workflow_state(
    workflow_id: str,
    body: WorkflowStateUpdate,
    db: Session = Depends(get_db),
) -> WorkflowDetail:
    w = db.get(Workflow, workflow_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if body.status is not None:
        w.status = body.status
    if body.current_agent is not None:
        w.current_agent = body.current_agent
    if body.current_step is not None:
        w.current_step = body.current_step
    if body.iteration is not None:
        w.iteration = body.iteration
    if body.last_verdict is not None:
        w.last_verdict = body.last_verdict
    if body.last_reason is not None:
        w.last_reason = body.last_reason
    if body.reconstruction_id is not None:
        w.reconstruction_id = body.reconstruction_id
    if body.error_message is not None:
        w.error_message = body.error_message

    db.add(w)
    db.commit()
    db.refresh(w)
    return serialize_workflow(w)


@router.post("/workflows/{workflow_id}/stop", response_model=WorkflowDetail)
def stop_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
) -> WorkflowDetail:
    w = db.get(Workflow, workflow_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if w.status not in ("running", "pending"):
        raise HTTPException(status_code=400, detail="Workflow is not running")

    # Kill the entire process group
    if w.pid:
        try:
            os.killpg(os.getpgid(w.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    w.status = "failed"
    w.error_message = "Stopped by user"
    w.current_step = "stopped"
    w.current_agent = None
    db.add(w)
    db.commit()
    db.refresh(w)
    return serialize_workflow(w)


@router.delete("/workflows/{workflow_id}", status_code=204)
def delete_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
) -> None:
    w = db.get(Workflow, workflow_id)
    if w is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if w.status in ("running", "pending"):
        raise HTTPException(status_code=400, detail="Stop the workflow before deleting it")

    # Remove log and video files
    video_dir = settings.base_dir / "workflow_videos"
    for suffix in (".log", ".mov", ".mp4", ".m4v"):
        p = video_dir / f"{workflow_id}{suffix}"
        if p.exists():
            p.unlink()

    db.delete(w)
    db.commit()
