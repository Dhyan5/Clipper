from clipforge.highlights import detect_highlights, sentence_units


def test_detect_highlights_fallback(transcript, envelope):
    times, rms = envelope
    result = detect_highlights(
        transcript, times, rms, api_key="", model="gemini-2.5-flash",
        clip_count=6, min_len=20, max_len=45,
    )
    assert result["mode"].startswith("fallback")
    plans = result["plans"]
    assert len(plans) >= 3
    boundaries = set(result["boundaries"])
    for p in plans:
        assert 20 - 0.01 <= p["duration"] <= 45 + 0.01, p
        assert p["start"] in boundaries
        assert p["end"] in boundaries
        assert 0 <= p["score"] <= 10
        assert "breakdown" in p
    for a in range(len(plans)):
        for b in range(a + 1, len(plans)):
            x, y = plans[a], plans[b]
            overlap = max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
            assert overlap <= 0.15 * min(x["duration"], y["duration"]) + 0.01


def test_sentence_units_use_punctuation(transcript):
    units = sentence_units(transcript)
    assert units
    assert all(u.end > u.start for u in units)
    assert units[0].text.endswith(".")
