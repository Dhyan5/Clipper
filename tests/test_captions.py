from pathlib import Path

from clipforge.captions import (
    CAPTION_Y, PRESETS, build_ass, map_words, parse_ass_events,
)
from clipforge.qa import captions_safe, hook_timing


def _no_tags(text: str) -> str:
    import re

    return re.sub(r"\{[^}]*\}", "", text).replace("\\N", " ")


def test_map_words_edited_timeline(transcript):
    words, total = map_words(transcript, [(0.0, 10.0), (20.0, 30.0)])
    assert total == 20.0
    assert words
    assert all(0.0 <= w["start"] <= 20.0 for w in words)
    assert max(w["end"] for w in words) <= 20.4


def test_build_ass_events_safe_and_animated(transcript, tmp_path):
    keep = [(0.0, 20.0), (40.0, 60.0)]
    enable = ["word3", "word4"]
    out = tmp_path / "captions.ass"
    info = build_ass(
        transcript, keep, out, hook_text="this is a hook with eight words max ok",
        preset_name="Hormozi", duration=40.0, emphasis_words=enable,
        loud_times=[5.0, 25.0],
    )
    assert out.exists()
    assert info["caption_events"] > 0
    text = out.read_text(encoding="utf-8")
    events = parse_ass_events(text)
    hooks = [e for e in events if e["style"] == "Hook"]
    captions = [e for e in events if e["style"] == "Caption"]
    assert len(hooks) == 1
    assert hooks[0]["start"] == "0:00:00.10"
    assert len(captions) > 5
    assert "\\fscx70" in text and "\\t(0,80,\\fscx115" in text
    assert "\\c&H0000FFFF" in text
    for event in captions:
        plain = _no_tags(event["text"])
        assert 0 < len(plain.split()) <= 3
    assert captions_safe(out)["ok"], captions_safe(out)["issues"]
    assert hook_timing(out)["ok"]


def test_profanity_masked(transcript, tmp_path):
    transcript["segments"][0]["words"][0]["word"] = "fuck"
    out = tmp_path / "c.ass"
    build_ass(transcript, [(0.0, 6.0)], out, preset_name="Clean Minimal",
              duration=6.0)
    text = out.read_text(encoding="utf-8")
    assert "FUCK" not in text.upper().replace("****", "") or "****" in text
    assert "****" in text


def test_presets_exist():
    assert set(PRESETS) == {"Hormozi", "MrBeast", "Gaming Neon", "Clean Minimal", "Classic"}
    assert CAPTION_Y < 1632
