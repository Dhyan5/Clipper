import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import soundfile as sf

import clipforge.transcriber as transcriber


def test_transcribe_caches_word_timestamps(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(transcriber, "CACHE_DIR", cache)
    wav = tmp_path / "audio16k.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    monkeypatch.setattr(transcriber, "extract_audio", lambda source, vid: wav)
    monkeypatch.setattr(transcriber, "_pick_device", lambda: ("cpu", "int8"))
    transcriber._load_model.cache_clear()

    calls = []

    class Word:
        def __init__(self, text, start, end, prob):
            self.word, self.start, self.end, self.probability = text, start, end, prob

    class Segment:
        start, end, text = 0.0, 2.0, "hello world"

        def __init__(self):
            self.words = [Word("hello", 0.0, 0.6, 0.9), Word("world", 0.7, 1.4, 0.8)]

    class Info:
        language, language_probability, duration = "en", 0.99, 2.0

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return [Segment()], Info()

    class FakeBatched:
        def __init__(self, model):
            self.model = model

        def transcribe(self, audio, **kwargs):
            return self.model.transcribe(audio, **kwargs)

    fake = ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    fake.BatchedInferencePipeline = FakeBatched
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    progress_msgs = []
    first = transcriber.transcribe(
        wav, "VID12345678", "small", lambda m, f=None: progress_msgs.append((m, f))
    )
    transcriber._load_model.cache_clear()
    assert any(f == 1.0 for _, f in progress_msgs)
    assert first["word_count"] == 2
    assert all("start" in w and "end" in w for s in first["segments"] for w in s["words"])
    assert calls and calls[0]["word_timestamps"] is True
    assert transcriber.transcript_path("VID12345678", "small").exists()

    second = transcriber.transcribe(wav, "VID12345678", "small")
    assert second["cached"] is True
    assert len(calls) == 1
