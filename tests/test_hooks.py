from clipforge.hooks import fallback_packaging, sanitize_packaging


def _plan(transcript):
    return {
        "index": 1, "start": 0.0, "end": 25.0, "duration": 25.0,
        "text": transcript["segments"][0]["text"] + " " + transcript["segments"][1]["text"],
        "keywords": ["word3", "word4", "12"],
    }


def test_fallback_constraints(transcript):
    pack = fallback_packaging(_plan(transcript), transcript)
    assert len(pack["hooks"]) == 3
    assert all(1 <= len(h["text"].split()) <= 8 for h in pack["hooks"])
    assert len(pack["titles"]) == 3
    assert all(len(t) < 60 for t in pack["titles"])
    assert 10 <= len(pack["hashtags"]) <= 15
    assert all(h.startswith("#") for h in pack["hashtags"])
    assert len(pack["emphasis_words"]) <= 8
    assert 0 < pack["payoff_time_sec"] < 25


def test_sanitize_bad_llm_output(transcript):
    bad = {
        "index": 1,
        "hooks": [{"type": "shock", "text": "one two three four five six seven eight nine ten eleven"}],
        "best_hook_index": 99,
        "titles": ["x" * 120],
        "description": "",
        "hashtags": ["bad tag!", "#ok"],
        "emphasis_words": ["notspoken"],
        "payoff_time_sec": 999,
    }
    pack = sanitize_packaging(bad, _plan(transcript), transcript)
    assert len(pack["hooks"]) == 3
    assert all(len(h["text"].split()) <= 8 for h in pack["hooks"])
    assert 0 <= pack["best_hook_index"] < 3
    assert len(pack["titles"]) == 3
    assert all(len(t) < 60 for t in pack["titles"])
    assert 10 <= len(pack["hashtags"]) <= 15
    assert pack["description"]
    assert 0 < pack["payoff_time_sec"] < 25
