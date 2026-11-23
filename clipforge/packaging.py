from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
from pydantic import BaseModel, Field

from .config import slugify


class QACheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class QAResult(BaseModel):
    passed: bool = False
    attempts: int = 0
    dropped: bool = False
    checks: list[QACheck] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class ClipMeta(BaseModel):
    index: int
    filename: str
    title_options: list[str]
    chosen_title: str
    hook: str
    hook_variants: list[dict]
    hook_start: float
    score: float
    score_breakdown: dict
    start: float
    end: float
    duration: float
    description: str
    hashtags: list[str]
    emphasis_words: list[str] = Field(default_factory=list)
    thumbnail: str = ""
    layout: str = ""
    keep_segments: list[list[float]] = Field(default_factory=list)
    qa: QAResult = Field(default_factory=QAResult)


class VideoMeta(BaseModel):
    video_id: str
    url: str
    source_title: str
    slug: str
    folder: str
    created_at: str
    mode: str
    settings: dict
    settings_fingerprint: str
    facecam: dict | None = None
    clips: list[ClipMeta]
    dropped: list[dict] = Field(default_factory=list)
    zip: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clip_filename(index: int, title: str) -> str:
    return f"clip_{index:02d}_{slugify(title, 48)}.mp4"


def thumbnail_path(folder: Path, index: int) -> Path:
    return folder / "thumbnails" / f"clip_{index:02d}.png"


def make_thumbnail(clip_path: Path | str, out_path: Path | str,
                   fallback_t: float = 1.0) -> Path:
    """Pick the frame with the largest detected face (expression proxy)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps else 0.0
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path)) if cascade_path.exists() else None
    use_faces = cascade is not None and not cascade.empty()
    times = [t for t in (0.6, 1.0, 1.4, 1.8, 2.2, 2.6) if t < max(1.0, duration)]
    if duration > 4:
        times += [duration * 0.4, duration * 0.7]
    best = None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if use_faces:
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            score = max([w * h for (_, _, w, h) in faces], default=0)
        else:
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if best is None or score > best[0]:
            best = (score, frame)
    cap.release()
    if best is None:
        cap = cv2.VideoCapture(str(clip_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, fallback_t * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a thumbnail frame from {clip_path}")
        best = (0, frame)
    cv2.imwrite(str(out_path), best[1])
    return out_path


def build_zip(folder: Path | str, out_zip: Path | str | None = None) -> Path:
    folder = Path(folder)
    out_zip = Path(out_zip) if out_zip else folder / "clips.zip"
    tmp = out_zip.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix != ".tmp" and path != out_zip:
                if "source" in path.relative_to(folder).parts:
                    continue
                zf.write(path, path.relative_to(folder).as_posix())
    shutil.move(str(tmp), str(out_zip))
    return out_zip
