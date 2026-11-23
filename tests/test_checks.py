from pathlib import Path

from clipforge.checks import (
    assert_no_file_larger_than_audio, find_files_larger_than,
)
from clipforge.downloader import _timestamps


def test_timestamps_spread_and_count():
    ts = _timestamps(100.0, 30)
    assert len(ts) == 30
    assert ts == sorted(ts)
    assert ts[0] >= 2.0
    assert ts[-1] <= 98.0
    assert _timestamps(0, 30) == []


def test_no_file_larger_than_audio(tmp_path):
    cache = tmp_path / "cache" / "VID"
    cache.mkdir(parents=True)
    (cache / "audio.m4a").write_bytes(b"x" * 1000)
    (cache / "facecam_v1.json").write_text("{}")
    (tmp_path / "small.bin").write_bytes(b"x" * 500)
    report = assert_no_file_larger_than_audio(tmp_path, cache)
    assert report["limit"] == 1000
    assert report["violations"] == []

    (tmp_path / "source.mp4").write_bytes(b"x" * 5000)
    violations = find_files_larger_than(tmp_path, 1000)
    assert any(p.name == "source.mp4" for p, _ in violations)
