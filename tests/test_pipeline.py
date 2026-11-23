import numpy as np

import clipforge.config as config
import clipforge.pipeline as pipe
from clipforge.pipeline import _map_dead_air_to_source


def test_map_dead_air_to_source():
    assert _map_dead_air_to_source([(10.0, 20.0)], [(2.0, 4.0)]) == [(12.0, 14.0)]
    assert _map_dead_air_to_source([(10.0, 20.0), (22.0, 30.0)],
                                   [(8.0, 12.0)]) == [(18.0, 20.0), (22.0, 24.0)]


def test_process_uses_audio_first_and_defers_video(monkeypatch, tmp_path):
    config.OUTPUT_DIR = tmp_path / "output"
    config.CACHE_DIR = tmp_path / "cache"
    pipe.OUTPUT_DIR = config.OUTPUT_DIR
    pipe.CACHE_DIR = config.CACHE_DIR
    monkeypatch.setattr(pipe, "check_ffmpeg", lambda: (True, "ok"))
    monkeypatch.setattr(config, "get_api_key", lambda: "")

    order = []
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    info = {"id": "VID12345678", "title": "Test Video", "duration": 40.0}

    def fake_audio(url, progress=None):
        order.append("audio")
        return {"video_id": "VID12345678", "path": audio, "info": info,
                "cached": False}

    def fake_transcribe(path, *args, **kwargs):
        order.append("transcribe")
        assert path == audio
        words = [
            {"word": f"w{i}", "start": float(i), "end": float(i) + 0.5,
             "score": 0.9}
            for i in range(40)
        ]
        return {"segments": [{"start": 0.0, "end": 40.0,
                              "text": " ".join(w["word"] for w in words),
                              "words": words}],
                "word_count": len(words), "duration": 40.0}

    def fake_facecam(url, video_id, duration, override=None, progress=None):
        order.append("facecam")
        return {"box": {"x": 10, "y": 10, "w": 100, "h": 100},
                "width": 1280, "height": 720, "engine": "test",
                "hits": 30, "samples": 30}

    def fake_highlights(*args, **kwargs):
        order.append("highlights")
        return {"plans": [{"index": 1, "start": 2.0, "end": 22.0,
                           "score": 8.0, "breakdown": {}}],
                "boundaries": [22.0], "mode": "fallback"}

    def fake_enrich(plans, transcript, api_key, model, progress=None):
        return {"mode": "fallback", "clips": [{
            "index": 1,
            "hooks": [{"type": "curiosity", "text": "THIS IS HUGE"}],
            "best_hook_index": 0,
            "titles": ["Test Clip"],
            "description": "desc",
            "hashtags": ["#test"],
            "emphasis_words": [],
            "payoff_time_sec": 20.0,
        }]}

    def fake_video(url, video_id, progress=None):
        order.append("video")
        return source

    def fake_render(**kwargs):
        order.append("render")
        kwargs["out_path"].write_bytes(b"mp4")
        return kwargs["out_path"]

    monkeypatch.setattr(pipe, "download_audio", fake_audio)
    monkeypatch.setattr(pipe, "download_video", fake_video)
    monkeypatch.setattr(pipe, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipe, "extract_audio", lambda src, vid: audio)
    monkeypatch.setattr(pipe, "envelope",
                        lambda vid: (np.linspace(0, 40, 41), np.ones(41) * 0.1))
    monkeypatch.setattr(pipe, "loud_peak_times", lambda *a, **k: [])
    monkeypatch.setattr(pipe, "detect_facecam_remote", fake_facecam)
    monkeypatch.setattr(pipe, "detect_highlights", fake_highlights)
    monkeypatch.setattr(pipe.hooks, "enrich", fake_enrich)
    monkeypatch.setattr(pipe, "_source_size", lambda src, fc: (1280, 720))
    monkeypatch.setattr(pipe, "silence_intervals", lambda *a, **k: [])
    monkeypatch.setattr(pipe, "plan_retention",
                        lambda *a, **k: {"keep_segments": [(2.0, 22.0)]})
    monkeypatch.setattr(pipe, "render_clip", fake_render)
    monkeypatch.setattr(pipe, "check_clip", lambda *a, **k: {
        "passed": True, "attempts": 1, "dropped": False, "checks": [],
        "metrics": {"duration": 20.0, "lufs": -14.0},
    })
    monkeypatch.setattr(pipe, "make_thumbnail", lambda clip, thumb, **k: (
        thumb.parent.mkdir(parents=True, exist_ok=True),
        thumb.write_bytes(b"png"),
        thumb,
    )[2])
    monkeypatch.setattr(pipe, "build_zip", lambda folder: (
        folder / "clips.zip").write_bytes(b"zip"))

    result = pipe.process("https://www.youtube.com/watch?v=VID12345678",
                          config.Settings())
    assert order.index("audio") < order.index("facecam") < order.index("video")
    assert order.index("video") < order.index("render")
    assert result["cached"] is False
    folder = pipe.Path(result["folder"])
    assert (folder / "metadata.json").exists()
    assert (folder / "clips.zip").exists()
    assert (folder / result["metadata"]["clips"][0]["filename"]).exists()
