import math

import numpy as np
import soundfile as sf

from clipforge.captions import build_ass
from clipforge.qa import dead_air, ends_on_sentence, measure_loudness


def test_dead_air_detection(tmp_path):
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    tone = 0.2 * np.sin(2 * math.pi * 220 * t)
    y = np.concatenate([tone[:sr], np.zeros(sr * 2), tone[:sr]])
    wav = tmp_path / "a.wav"
    sf.write(wav, y, sr)
    result = dead_air(wav)
    assert result["max_dead_air"] >= 1.8
    assert result["runs"]


def test_ends_on_sentence():
    assert ends_on_sentence(30.0, [0.0, 15.0, 30.0])
    assert not ends_on_sentence(29.0, [0.0, 15.0, 36.0])
    assert ends_on_sentence(30.0, [])


def test_caption_safe_zones(transcript, tmp_path):
    out = tmp_path / "c.ass"
    build_ass(transcript, [(0.0, 20.0)], out, hook_text="hook text here",
              duration=20.0)
    from clipforge.qa import captions_safe

    assert captions_safe(out)["ok"]


def test_loudness_parser_returns_dict():
    # No media file: parser must not raise, just return an error dict.
    result = measure_loudness("does-not-exist.mp4")
    assert isinstance(result, dict)
