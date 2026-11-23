from __future__ import annotations

import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

from .audio import extract_audio
from .config import CACHE_DIR, CACHE_VERSION, has_cuda
from .utils import read_json, write_json

_LOCK = threading.Lock()


def transcript_path(video_id: str, model_size: str) -> Path:
    return CACHE_DIR / video_id / f"transcript_{model_size}_v{CACHE_VERSION}.json"


def _report(progress, msg: str, frac: float | None = None) -> None:
    """Call progress(msg, frac) if the callback accepts it, else progress(msg)."""
    if not progress:
        return
    try:
        progress(msg, frac)
    except TypeError:
        progress(msg)


def _register_cuda_dlls() -> None:
    """Expose pip-installed nvidia-cublas/cudnn DLLs to CTranslate2 on Windows."""
    if sys.platform != "win32":
        return
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except Exception:
        return
    for base in list(getattr(nvidia, "__path__", [])):
        for sub in Path(base).glob("*/bin"):
            try:
                os.add_dll_directory(str(sub))
            except Exception:
                pass
            os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=None)
def _load_model(model_size: str, device: str, compute: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute)


@lru_cache(maxsize=1)
def _gpu_works() -> bool:
    """Run a 1 s dummy transcription so missing cuBLAS/cuDNN fails fast."""
    if not has_cuda():
        return False
    _register_cuda_dlls()
    try:
        import numpy as np

        model = _load_model("tiny", "cuda", "float16")
        segs, _ = model.transcribe(np.zeros(16000, dtype="float32"), vad_filter=False)
        list(segs)
        return True
    except Exception:
        return False


def _pick_device() -> tuple[str, str]:
    if _gpu_works():
        return "cuda", "float16"
    return "cpu", "int8"


def _word_dict(w) -> dict | None:
    text = (w.word or "").strip()
    if not text:
        return None
    return {
        "start": round(float(w.start), 3),
        "end": round(float(w.end), 3),
        "word": text,
        "score": round(float(getattr(w, "probability", 0.0) or 0.0), 4),
    }


def _run(model, audio, device: str, duration: float, progress, label: str):
    from faster_whisper import BatchedInferencePipeline

    # Batched inference = big speedup (VAD-chunked, parallel decode).
    batch = 16 if device == "cuda" else 8
    pipe = BatchedInferencePipeline(model=model)
    segments_iter, info = pipe.transcribe(
        audio, word_timestamps=True, vad_filter=True, beam_size=5,
        batch_size=batch, condition_on_previous_text=False,
    )
    out: list[dict] = []
    t0 = time.time()
    last = 0.0
    for seg in segments_iter:
        words = [d for d in (_word_dict(w) for w in (seg.words or [])) if d]
        if words:
            out.append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": (seg.text or "").strip(),
                "words": words,
            })
        now = time.time()
        if duration > 0 and now - last > 0.5:
            last = now
            frac = min(1.0, float(seg.end) / duration)
            elapsed = now - t0
            eta = (elapsed / frac - elapsed) if frac > 0.02 else 0
            _report(progress,
                    f"{label}: {frac * 100:4.1f}% ({seg.end / 60:.1f}/{duration / 60:.1f} min)"
                    + (f", ~{eta:.0f}s left" if eta else ""), frac)
    out.sort(key=lambda s: s["start"])
    return out, info


def transcribe(source: Path | str, video_id: str, model_size: str = "small", progress=None) -> dict:
    """Word-level transcript cached under cache/<video_id>/. No-op on re-run."""
    path = transcript_path(video_id, model_size)
    cached = read_json(path)
    if cached is not None:
        cached["cached"] = True
        return cached

    import soundfile as sf

    with _LOCK:  # never transcribe twice at once (Streamlit reruns)
        cached = read_json(path)
        if cached is not None:
            cached["cached"] = True
            return cached

        _report(progress, "Preparing audio for transcription", 0.0)
        # Decode with FFmpeg + soundfile instead of PyAV: pyav 19 removed a kwarg
        # that faster-whisper still passes when given a file path.
        wav = extract_audio(source, video_id)
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, f"unexpected sample rate {sr}"
        duration = len(audio) / 16000.0

        _report(progress, "Checking GPU for Whisper", 0.0)
        device, compute = _pick_device()
        label = f"Transcribing with whisper-{model_size} on {device.upper()}"
        _report(progress, label, 0.0)
        try:
            model = _load_model(model_size, device, compute)
            segments, info = _run(model, audio, device, duration, progress, label)
        except Exception:
            if device == "cpu":
                raise
            label = f"Transcribing with whisper-{model_size} on CPU (GPU failed)"
            _report(progress, label, 0.0)
            model = _load_model(model_size, "cpu", "int8")
            segments, info = _run(model, audio, "cpu", duration, progress, label)
        _report(progress, "Transcription complete", 1.0)

        all_words = [w for s in segments for w in s["words"]]
        data = {
            "video_id": video_id,
            "language": getattr(info, "language", None),
            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
            "duration": round(float(getattr(info, "duration", 0.0) or duration), 3),
            "word_count": len(all_words),
            "device": device,
            "segments": segments,
            "cached": False,
        }
        if not all_words:
            raise RuntimeError("Transcription produced no words; the video has no usable speech")
        write_json(path, data)
        return data


def transcript_text(transcript: dict, max_chars: int = 24000) -> str:
    lines = [
        f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}"
        for seg in transcript.get("segments", [])
    ]
    return "\n".join(lines)[:max_chars]
