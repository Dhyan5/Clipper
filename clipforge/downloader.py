from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, video_id_from_url
from .utils import read_json, run, write_json

_DL_LOCK = threading.RLock()

VIDEO_FORMAT = "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]"
MERGED_VIDEO_FORMAT = (
    "bv*[height<=720][ext=mp4]+ba[ext=m4a]/"
    "bv*[height<=720]+ba/"
    "b[height<=720]+ba/"
    "best[height<=720]/best"
)


def _folder(video_id: str) -> Path:
    p = CACHE_DIR / video_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find_audio(video_id: str) -> Path | None:
    folder = CACHE_DIR / video_id
    for p in sorted(folder.glob("audio.*")):
        if p.suffix not in {".json", ".part"} and p.stat().st_size > 0:
            return p
    return None


def _remux_to_mp4(raw: Path, dest: Path) -> Path:
    if raw.resolve() == dest.resolve():
        return dest
    try:
        run(["ffmpeg", "-y", "-i", str(raw), "-c", "copy", "-movflags", "+faststart", str(dest)], timeout=600)
    except Exception:
        run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "20",
             "-c:a", "aac", "-movflags", "+faststart", str(dest)], timeout=1800)
    return dest


def download_audio(url: str, progress=None) -> dict:
    with _DL_LOCK:  # concurrent Streamlit reruns must not fight over the file
        return _download_audio(url, progress)


def _download_audio(url: str, progress=None) -> dict:
    """Download only the audio track. This is the largest pre-render file."""
    vid = video_id_from_url(url)
    folder = _folder(vid)
    info_path = folder / "info.json"
    existing = _find_audio(vid)
    if existing and info_path.exists():
        return {"video_id": vid, "path": existing,
                "info": read_json(info_path, {}), "cached": True}

    if progress:
        progress("Downloading audio track")
    import yt_dlp

    opts = {
        "outtmpl": str(folder / "audio.%(ext)s"),
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 4,
        "fragment_retries": 4,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=True)
    audio = _find_audio(vid)
    if not audio:
        raise RuntimeError("Audio download produced no file")
    info = {
        "id": raw.get("id") or vid,
        "title": raw.get("title") or vid,
        "duration": raw.get("duration"),
        "uploader": raw.get("uploader"),
        "webpage_url": raw.get("webpage_url") or url,
        "thumbnail": raw.get("thumbnail"),
    }
    write_json(info_path, info)
    return {"video_id": vid, "path": audio, "info": info, "cached": False}


def download_video(url: str, video_id: str, progress=None) -> Path:
    with _DL_LOCK:
        return _download_video(url, video_id, progress)


def _download_video(url: str, video_id: str, progress=None) -> Path:
    """Download the 720p video. Call this only right before clip rendering."""
    folder = _folder(video_id)
    src = folder / "source.mp4"
    if src.exists() and src.stat().st_size > 0:
        return src
    if progress:
        progress("Downloading video for rendering")
    import yt_dlp

    opts = {
        "outtmpl": str(folder / "source.%(ext)s"),
        "format": MERGED_VIDEO_FORMAT,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 4,
        "fragment_retries": 4,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    if not src.exists():
        candidates = [
            p for p in folder.glob("source.*")
            if p.suffix not in {".part", ".json"} and p.stat().st_size > 0
        ]
        if not candidates:
            raise RuntimeError("Download finished but no video file was produced")
        _remux_to_mp4(max(candidates, key=lambda p: p.stat().st_size), src)
    for leftover in folder.glob("source.*"):
        if leftover != src and leftover.suffix != ".json":
            leftover.unlink(missing_ok=True)
    return src


def _timestamps(duration: float, count: int = 30) -> list[float]:
    if not duration or duration <= 0:
        return []
    return [round(float(t), 2) for t in np.linspace(duration * 0.02,
                                                    duration * 0.98, count)]


def _stream_urls(url: str) -> list[str]:
    out = run(
        [sys.executable, "-m", "yt_dlp", "-g", "-f", VIDEO_FORMAT,
         "--no-warnings", url],
        timeout=180,
    )
    return [ln.strip() for ln in out.splitlines() if ln.strip().startswith("http")]


def video_dimensions(url: str) -> tuple[int, int] | None:
    try:
        out = run(
            [sys.executable, "-m", "yt_dlp", "--print",
             "%(width)s %(height)s", "-f", VIDEO_FORMAT, "--no-warnings", url],
            timeout=180,
        )
        parts = out.strip().splitlines()[-1].split()
        w, h = int(float(parts[0])), int(float(parts[1]))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return None


def _extract_frames_from_stream(stream_url: str, timestamps: list[float],
                                tmp: Path) -> list[np.ndarray]:
    import cv2

    frames: list[np.ndarray] = []
    for i, t in enumerate(timestamps):
        out = tmp / f"f_{i:02d}.jpg"
        try:
            run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{t:.2f}", "-i", stream_url, "-frames:v", "1",
                 "-q:v", "3", str(out)],
                timeout=120,
            )
        except Exception:
            continue
        frame = cv2.imread(str(out))
        if frame is not None:
            frames.append(frame)
        out.unlink(missing_ok=True)
    return frames


def _extract_frames_from_sections(url: str, timestamps: list[float],
                                  tmp: Path) -> list[np.ndarray]:
    import cv2

    frames: list[np.ndarray] = []
    for i, t in enumerate(timestamps):
        if len(frames) >= 20:
            break
        pattern = str(tmp / f"s_{i:02d}.%(ext)s")
        try:
            run(
                [sys.executable, "-m", "yt_dlp",
                 "--download-sections", f"*{max(0.0, t - 0.5):.1f}-{t + 1.5:.1f}",
                 "--force-keyframes-at-cuts",
                 "-f", "bv*[height<=360]/b[height<=360]/b",
                 "-o", pattern, "--no-warnings", "--quiet", url],
                timeout=300,
            )
        except Exception:
            continue
        media = [p for p in tmp.glob(f"s_{i:02d}.*")
                 if p.suffix not in {".part", ".json", ".jpg"}]
        if not media:
            continue
        out = tmp / f"sec_{i:02d}.jpg"
        try:
            run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", "1.0", "-i", str(media[0]), "-frames:v", "1",
                 "-q:v", "3", str(out)],
                timeout=120,
            )
        except Exception:
            pass
        frame = cv2.imread(str(out)) if out.exists() else None
        if frame is not None:
            frames.append(frame)
        for p in media + [out]:
            p.unlink(missing_ok=True)
    return frames


def sample_face_frames(url: str, duration: float, count: int = 30) -> dict:
    """Grab ~30 stills without downloading the video; samples deleted here."""
    vid = video_id_from_url(url)
    folder = _folder(vid)
    tmp = Path(tempfile.mkdtemp(prefix="frames_", dir=str(folder)))
    timestamps = _timestamps(duration, count)
    frames: list[np.ndarray] = []
    method = "none"
    try:
        try:
            urls = _stream_urls(url)
        except Exception:
            urls = []
        if urls:
            frames = _extract_frames_from_stream(urls[0], timestamps, tmp)
            method = "stream-seek"
        if len(frames) < max(8, count // 3):
            frames = _extract_frames_from_sections(url, timestamps, tmp)
            method = "sections"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    width = int(frames[0].shape[1]) if frames else 0
    height = int(frames[0].shape[0]) if frames else 0
    return {"frames": frames, "width": width, "height": height,
            "method": method, "timestamps": timestamps}
