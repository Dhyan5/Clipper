from pathlib import Path

from clipforge.pipeline import _combined_mode
from clipforge.utils import clamp, read_json, write_json


def test_json_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"a": 1})
    assert read_json(p) == {"a": 1}
    assert read_json(tmp_path / "missing.json", default={}) == {}


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0


def test_combined_mode():
    assert _combined_mode("gemini", "gemini", "key") == "gemini"
    assert "no Gemini key" in _combined_mode("heuristic", "template", "")
    assert "fallback" in _combined_mode("heuristic", "template", "key")
