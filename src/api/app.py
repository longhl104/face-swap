"""FastAPI production interface for face swap operations."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config import StoragePaths, load_config
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
        _engine = FaceSwapEngine(config=config)
    return _engine


def _save_upload(upload: UploadFile, subdir: str) -> Path:
    dest_dir = paths.uploads / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "file.jpg").suffix
    dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-source")
async def upload_source(
    file: UploadFile = File(...),
    session_id: str | None = None,
) -> dict[str, str]:
    """Upload the source face image (identity to transfer)."""
    sid = session_id or uuid.uuid4().hex
    saved = _save_upload(file, "source")
    _sessions.setdefault(sid, {})["source"] = saved
    return {"session_id": sid, "source_path": str(saved)}


@app.post("/upload-target")
async def upload_target(
    file: UploadFile = File(...),
    session_id: str | None = None,
) -> dict[str, str]:
    """Upload the target image or video (face to replace)."""
    sid = session_id or uuid.uuid4().hex
    saved = _save_upload(file, "target")
    _sessions.setdefault(sid, {})["target"] = saved
    return {"session_id": sid, "target_path": str(saved)}


@app.post("/swap")
async def swap(session_id: str) -> FileResponse:
    """Trigger face swap using previously uploaded source and target."""
    session = _sessions.get(session_id)
    if not session or "source" not in session or "target" not in session:
        raise HTTPException(
            status_code=400,
            detail="Upload both source and target first, or provide a valid session_id.",
        )

    source_path = session["source"]
    target_path = session["target"]
    suffix = target_path.suffix.lower()

    engine = get_engine()

    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(
            status_code=501,
            detail="Video swap not yet available. Use an image target or upgrade to v0.8.0.",
        )

    result = engine.swap_from_paths(source_path, target_path)

    if result is None:
        raise HTTPException(status_code=422, detail="Face detection failed on source or target.")

    return FileResponse(str(result), media_type="application/octet-stream", filename=result.name)
