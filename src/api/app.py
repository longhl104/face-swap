"""FastAPI production interface for face swap operations."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse  # pyright: ignore[reportAttributeAccessIssue] # nopep8

from src.config import StoragePaths, load_config
from src.data.update import add_faces_to_dataset
from src.inference.engine import FaceSwapEngine

app = FastAPI(
    title="Face Swap API",
    description="Production-ready API for AI-powered face swapping on images and videos.",
    version="1.0.0",
)

config = load_config()
paths = StoragePaths(config)
paths.ensure_dirs()

# Session state: store uploaded file paths per session
_sessions: dict[str, dict[str, Path]] = {}
_engine: FaceSwapEngine | None = None


def get_engine() -> FaceSwapEngine:
    global _engine
    if _engine is None:
        _engine = FaceSwapEngine()
    return _engine


def _save_upload(upload: UploadFile, subdir: str) -> Path:
    dest_dir = paths.uploads / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "file.jpg").suffix
    dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _get_session(session_id: str) -> dict[str, Path]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", status_code=201)
def create_session() -> dict[str, str]:
    """Create a new face-swap session."""
    session_id = uuid.uuid4().hex
    _sessions[session_id] = {}
    return {"id": session_id}


@app.put("/sessions/{session_id}/source")
async def set_source(
    session_id: str,
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Upload the source face image for a session."""
    session = _get_session(session_id)
    saved = _save_upload(file, "source")
    session["source"] = saved
    return {"id": session_id, "source_path": str(saved)}


@app.put("/sessions/{session_id}/target")
async def set_target(
    session_id: str,
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Upload the target image or video for a session."""
    session = _get_session(session_id)
    saved = _save_upload(file, "target")
    session["target"] = saved
    return {"id": session_id, "target_path": str(saved)}


@app.post("/sessions/{session_id}/swaps")
async def create_swap(session_id: str) -> FileResponse:
    """Run face swap for a session with uploaded source and target."""
    session = _get_session(session_id)
    if "source" not in session or "target" not in session:
        raise HTTPException(
            status_code=400,
            detail="Upload source and target before creating a swap.",
        )

    source_path = session["source"]
    target_path = session["target"]
    suffix = target_path.suffix.lower()

    engine = get_engine()

    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        from src.inference.video import swap_video

        output = paths.inference_output / f"{session_id}_swapped.mp4"
        result = swap_video(engine, source_path, target_path, output)
    else:
        result = engine.swap_from_paths(source_path, target_path)

    if result is None:
        raise HTTPException(
            status_code=422, detail="Face detection failed on source or target.")

    return FileResponse(str(result), media_type="application/octet-stream", filename=result.name)


@app.post("/datasets/faces", status_code=201)
async def add_dataset_faces(
    file: UploadFile = File(..., description="Face image to add to the dataset"),
    augment: bool = Query(True, description="Apply flip and brightness augmentations"),
) -> dict[str, object]:
    """Add a face image to the training dataset."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        suffix = Path(file.filename or "file.jpg").suffix or ".jpg"
        dest = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        added = add_faces_to_dataset(
            tmp_dir,
            augment=augment,
            show_progress=False,
        )

    if added == 0:
        raise HTTPException(
            status_code=422,
            detail="No faces could be detected in the uploaded image.",
        )

    return {"files_received": 1, "samples_added": added, "augmented": augment}
