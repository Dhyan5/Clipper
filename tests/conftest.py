from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def transcript() -> dict:
    segments = []
    t = 0.0
    n = 0
    duration = 90.0
    while t < duration:
        end = min(t + 5.0, duration)
        words = []
        wt = t
        while wt < end - 0.1:
            n += 1
            word = f"word{n}"
            if wt + 0.5 >= end - 0.1:
                word += "."
            words.append({
                "start": round(wt, 2),
                "end": round(min(wt + 0.35, end), 2),
                "word": word,
                "score": 0.9,
            })
            wt += 0.5
        segments.append({
            "start": round(t, 2),
            "end": round(end, 2),
            "text": " ".join(w["word"] for w in words),
            "words": words,
        })
        t = end
    return {
        "video_id": "TESTVIDEO01",
        "language": "en",
        "language_probability": 0.99,
        "duration": duration,
        "word_count": sum(len(s["words"]) for s in segments),
        "segments": segments,
    }


@pytest.fixture
def envelope() -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0, 90, 0.05)
    rms = 0.2 + 0.15 * np.sin(times / 3.0) + 0.05 * np.cos(times)
    return times, np.clip(rms, 0.01, 1.0)
