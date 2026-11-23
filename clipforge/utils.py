from __future__ import annotations

import json
import subprocess
from pathlib import Path


class CommandError(RuntimeError):
    pass


def run(cmd: list[str], cwd: Path | str | None = None, timeout: int | None = None) -> str:
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-4000:]
        raise CommandError(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return proc.stdout


def ffprobe(path: Path | str) -> dict:
    out = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        timeout=120,
    )
    return json.loads(out)


def duration(path: Path | str) -> float:
    try:
        return float(ffprobe(path)["format"]["duration"])
    except Exception:
        return 0.0


def read_json(path: Path | str, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: Path | str, data) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_frame(video: Path | str, t: float, out: Path | str) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(out)],
        timeout=120,
    )
    return out


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
