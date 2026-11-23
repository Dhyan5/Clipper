from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"
PROMPTS_DIR = ROOT / "prompts"
FONTS_DIR = ROOT / "fonts"
MODELS_DIR = ROOT / "models"
for _d in (CACHE_DIR, OUTPUT_DIR, PROMPTS_DIR, FONTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CACHE_VERSION = 1
FONT_NAME = "Anton"
DEFAULT_MODEL = "gemini-2.5-flash"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except Exception:  # dotenv is optional at runtime
    pass


@dataclass
class Settings:
    clip_count: int = 6
    min_len: float = 20.0
    max_len: float = 45.0
    preset: str = "Hormozi"
    cold_open: bool = False
    keep_source: bool = False
    model: str = DEFAULT_MODEL
    whisper_model: str = "small"
    facecam_override: str = ""

    def fingerprint(self) -> str:
        raw = json.dumps(
            {k: v for k, v in asdict(self).items() if k != "keep_source"},
            sort_keys=True,
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def get_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


def save_api_key(key: str) -> None:
    key = (key or "").strip()
    if not key:
        return
    env_path = ROOT / ".env"
    lines = []
    if env_path.exists():
        lines = [
            ln
            for ln in env_path.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("GEMINI_API_KEY")
        ]
    lines.append(f"GEMINI_API_KEY={key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = key


def check_ffmpeg() -> tuple[bool, str]:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        return False, (
            f"Missing {', '.join(missing)}. Install FFmpeg and add it to PATH:\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=30
        )
        first = out.stdout.splitlines()[0] if out.stdout else "ffmpeg"
        return True, first
    except Exception as exc:  # pragma: no cover - environment specific
        return False, f"FFmpeg found but not runnable: {exc}"


@lru_cache(maxsize=1)
def has_nvenc() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "h264_nvenc" not in (out.stdout or ""):
            return False
        probe = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.04",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True,
            timeout=30,
        )
        return probe.returncode == 0
    except Exception:
        return False


def has_cuda() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return (
            subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=15
            ).returncode
            == 0
        )
    except Exception:
        return False


def detect_gpu() -> str:
    """Return 'nvenc', 'cuda' or 'cpu' for the render/whisper decisions."""
    if has_nvenc():
        return "nvenc"
    if has_cuda():
        return "cuda"
    return "cpu"


def video_encoder_args() -> list[str]:
    if has_nvenc():
        return [
            "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20",
            "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p",
    ]


def whisper_config() -> tuple[str, str]:
    if has_cuda():
        return "cuda", "float16"
    return "cpu", "int8"


_YT_PATTERNS = [
    r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})",
]


def video_id_from_url(url: str) -> str:
    url = (url or "").strip()
    for pat in _YT_PATTERNS:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not find a YouTube video id in: {url!r}")


def slugify(text: str, max_len: int = 60) -> str:
    text = (text or "clip").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "clip")


def output_dir_for(title: str, video_id: str) -> Path:
    return OUTPUT_DIR / f"{slugify(title, 50)}_{video_id}"


def parse_facecam_override(raw: str) -> dict | None:
    if not raw or not raw.strip():
        return None
    try:
        parts = [float(p) for p in re.split(r"[,\s]+", raw.strip()) if p]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    x, y, w, h = (int(round(p)) for p in parts)
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}
