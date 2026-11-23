from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CACHE_DIR
from .utils import read_json, run, write_json


def audio_path(video_id: str) -> Path:
    return CACHE_DIR / video_id / "audio16k.wav"


def extract_audio(source: Path | str, video_id: str) -> Path:
    out = audio_path(video_id)
    if out.exists() and out.stat().st_size > 0:
        return out
    for stream in ("0:a:0",):
        try:
            run(
                ["ffmpeg", "-y", "-i", str(source), "-map", stream, "-vn",
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
                timeout=900,
            )
            if out.exists() and out.stat().st_size > 0:
                return out
        except Exception:
            continue
    raise RuntimeError("Source has no usable audio stream")


def envelope(video_id: str, hop_sec: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, normalised RMS 0..1) cached next to the wav."""
    cache = CACHE_DIR / video_id / f"envelope_v1_{hop_sec:.2f}.json"
    if cache.exists():
        data = read_json(cache)
        return np.array(data["times"]), np.array(data["rms"])
    import librosa

    wav = audio_path(video_id)
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    frame = max(256, int(sr * hop_sec))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=frame)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=frame)
    peak = float(rms.max()) if len(rms) else 0.0
    rms_n = (rms / peak).astype(float) if peak > 0 else rms.astype(float)
    write_json(cache, {"times": times.tolist(), "rms": rms_n.tolist()})
    return times, rms_n


def energy_between(times: np.ndarray, rms: np.ndarray, t0: float, t1: float) -> float:
    mask = (times >= t0) & (times < t1)
    if not mask.any():
        return 0.0
    return float(np.mean(np.square(rms[mask])) ** 0.5)


def loud_peak_times(times: np.ndarray, rms: np.ndarray, t0: float, t1: float, top: float = 0.9) -> list[float]:
    mask = (times >= t0) & (times < t1)
    if not mask.any():
        return []
    sub_t, sub_r = times[mask], rms[mask]
    if float(sub_r.max()) <= 0:
        return []
    thr = float(np.quantile(sub_r, top)) * float(sub_r.max())
    return [round(float(t), 2) for t, r in zip(sub_t, sub_r) if r >= thr]


def silence_intervals(
    times: np.ndarray,
    rms: np.ndarray,
    t0: float,
    t1: float,
    min_silence: float = 0.35,
    rel_threshold: float = 0.03,
) -> list[tuple[float, float]]:
    """Find runs quieter than an adaptive threshold, longer than min_silence."""
    mask = (times >= t0) & (times < t1)
    if not mask.any():
        return []
    sub_t, sub_r = times[mask], rms[mask]
    floor = max(0.006, float(np.quantile(sub_r, 0.10)) * 1.6)
    thr = max(floor, rel_threshold * float(np.quantile(sub_r, 0.95)))
    quiet = sub_r < thr
    if not quiet.any():
        return []
    step = float(sub_t[1] - sub_t[0]) if len(sub_t) > 1 else 0.05
    runs: list[tuple[float, float]] = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = float(sub_t[i])
        elif not q and start is not None:
            end = float(sub_t[i])
            if end - start >= min_silence:
                runs.append((start, end))
            start = None
    if start is not None:
        end = float(sub_t[-1]) + step
        if end - start >= min_silence:
            runs.append((start, end))
    return runs
