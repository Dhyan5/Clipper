from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .captions import CAPTION_Y, SAFE_BOTTOM, SAFE_RIGHT, SAFE_TOP, parse_ass_events
from .utils import ffprobe

TARGET_LUFS = -14.0
LUFS_TOLERANCE = 1.5


def _ass_seconds(stamp: str) -> float:
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def measure_loudness(path: Path | str) -> dict:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    text = proc.stderr or ""
    blocks = re.findall(r"\{[\s\S]*?\}", text)
    if not blocks:
        return {"input_i": None, "error": text[-500:]}
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError:
        return {"input_i": None, "error": text[-500:]}


def _audio_rms(path: Path | str, hop: float = 0.05, sr: int = 16000):
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, timeout=900, check=True,
        )
        y, real_sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    n = max(1, int(real_sr * hop))
    usable = (len(y) // n) * n
    if usable == 0:
        return np.array([0.0]), hop
    frames = y[:usable].reshape(-1, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return rms, hop


def dead_air(path: Path | str, min_run: float = 0.8) -> dict:
    rms, hop = _audio_rms(path)
    db = 20.0 * np.log10(rms + 1e-9)
    peak = float(np.quantile(db, 0.95))
    threshold = max(-55.0, peak - 38.0)
    quiet = db < threshold
    runs: list[tuple[float, float]] = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * hop >= min_run:
                runs.append((start * hop, i * hop))
            start = None
    if start is not None and (len(quiet) - start) * hop >= min_run:
        runs.append((start * hop, len(quiet) * hop))
    longest = max((e - s for s, e in runs), default=0.0)
    return {"max_dead_air": round(longest, 2), "runs": runs,
            "threshold_db": round(threshold, 1)}


def hook_timing(ass_path: Path | str) -> dict:
    text = Path(ass_path).read_text(encoding="utf-8")
    starts = [_ass_seconds(e["start"]) for e in parse_ass_events(text) if e["style"] == "Hook"]
    first = min(starts) if starts else None
    return {"first_hook": first, "ok": first is not None and first <= 2.0}


def _plain(text: str) -> str:
    return re.sub(r"\{[^}]*\}", "", text)


def captions_safe(ass_path: Path | str) -> dict:
    """Estimate each text box and compare with the safe zones."""
    text = Path(ass_path).read_text(encoding="utf-8")
    issues: list[str] = []
    events = parse_ass_events(text)
    for e in events:
        raw = e["text"]
        content = _plain(raw).replace("\\N", "\n")
        if not content.strip():
            continue
        sizes = [int(x) for x in re.findall(r"\\fs(\d+)", raw)]
        size = max(sizes) if sizes else (86 if e["style"] == "Caption" else 54)
        pos = re.search(r"\\pos\((\d+),(\d+)\)", raw)
        x = int(pos.group(1)) if pos else 540
        y = int(pos.group(2)) if pos else (CAPTION_Y if e["style"] == "Caption" else 180)
        an8 = "\\an8" in raw
        lines = content.split("\n")
        max_chars = max((len(ln.replace(" ", "")) for ln in lines), default=0)
        spaces = max((ln.count(" ") for ln in lines), default=0)
        width = max_chars * 0.47 * size + spaces * 0.24 * size
        if e["style"] == "Hook":
            width = min(width, 820)
            top = y
            bottom = y + len(lines) * size * 1.15
        else:
            width = min(width, 800)
            top = y - size * 0.8
            bottom = y + size * 0.8
        left, right = x - width / 2, x + width / 2
        if left < -1 or right > SAFE_RIGHT + 3:
            issues.append(f"{e['start']} {e['style']} horizontal {left:.0f}-{right:.0f}")
        if top < SAFE_TOP - 3:
            issues.append(f"{e['start']} {e['style']} too high ({top:.0f})")
        if bottom > SAFE_BOTTOM + 3:
            issues.append(f"{e['start']} {e['style']} too low ({bottom:.0f})")
        if an8 and top < SAFE_TOP - 3:
            issues.append(f"{e['start']} hook above safe top")
    return {"ok": not issues, "issues": issues[:20], "events": len(events)}


def _fps(stream: dict) -> float:
    raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        num, den = raw.split("/")
        return float(num) / float(den) if float(den) else 0.0
    except Exception:
        return 0.0


def ends_on_sentence(end: float, boundaries: list[float], tol: float = 0.5) -> bool:
    if not boundaries:
        return True
    return any(abs(float(b) - end) <= tol for b in boundaries)


def check_clip(path: Path | str, ass_path: Path | str, boundaries: list[float],
               planned_end: float) -> dict:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        info = ffprobe(path)
    except Exception as exc:
        return {"passed": False, "attempts": 0, "dropped": False,
                "checks": [{"name": "probe", "passed": False, "detail": str(exc)}],
                "metrics": {}}
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    width = int(video.get("width", 0)) if video else 0
    height = int(video.get("height", 0)) if video else 0
    fps = _fps(video) if video else 0.0
    duration = float(info.get("format", {}).get("duration") or 0.0)

    add("resolution", width == 1080 and height == 1920, f"{width}x{height}")
    add("video_codec", bool(video) and video.get("codec_name") == "h264",
        video.get("codec_name", "none") if video else "none")
    add("audio_codec", bool(audio) and audio.get("codec_name") == "aac",
        audio.get("codec_name", "none") if audio else "none")
    add("fps", abs(fps - 30.0) < 0.1, f"{fps:.2f}")
    add("duration", 15.0 <= duration <= 60.0, f"{duration:.2f}s")

    loud = measure_loudness(path)
    try:
        lufs = float(loud.get("input_i"))
    except (TypeError, ValueError):
        lufs = None
    add("loudness", lufs is not None and abs(lufs - TARGET_LUFS) <= LUFS_TOLERANCE,
        f"{lufs} LUFS" if lufs is not None else "unmeasured")

    air = dead_air(path)
    add("dead_air", air["max_dead_air"] <= 0.8,
        f"{air['max_dead_air']}s max")

    hook = hook_timing(ass_path)
    add("hook_within_2s", hook["ok"],
        f"first hook at {hook['first_hook']}")

    safe = captions_safe(ass_path)
    add("caption_safe_zones", safe["ok"],
        "; ".join(safe["issues"][:3]) or f"{safe['events']} events")

    add("ends_on_sentence", ends_on_sentence(planned_end, boundaries),
        f"end {planned_end:.2f}s")

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "attempts": 0,
        "dropped": False,
        "checks": checks,
        "metrics": {
            "duration": round(duration, 2), "lufs": lufs,
            "max_dead_air": air["max_dead_air"], "fps": round(fps, 2),
            "hook_first": hook["first_hook"],
        },
    }


def caption_frame_diff(path: Path | str, times: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict:
    """Mean pixel difference in the caption band between sampled frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame[int(1920 * 0.44):int(1920 * 0.60), :, :])
    cap.release()
    diffs = []
    for a, b in zip(frames, frames[1:]):
        if a.shape == b.shape:
            diffs.append(float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))))
    return {"frames": len(frames), "diffs": [round(d, 2) for d in diffs],
            "animated": any(d > 2.0 for d in diffs)}
