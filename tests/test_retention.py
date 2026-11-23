from clipforge.retention import plan_retention, prepend_cold_open


def _words(start=0.0, end=30.0):
    words = []
    t = start + 0.2
    i = 0
    while t < end - 0.3:
        i += 1
        words.append({"start": round(t, 2), "end": round(t + 0.3, 2),
                      "word": "um" if i == 3 else f"w{i}"})
        t += 0.5
    return words


def test_cuts_silence_and_fillers():
    words = _words()
    result = plan_retention(
        words, 0.0, 30.0, silences=[(10.0, 12.0), (20.0, 21.0)],
        min_total=10.0, remove_fillers=True,
    )
    assert result["total_duration"] < 30.0
    assert result["total_duration"] >= 10.0
    segs = result["keep_segments"]
    assert segs[0][0] == 0.0
    for a, b in zip(segs, segs[1:]):
        assert b[0] > a[1]
    for s, e in segs:
        assert not (10.4 < s < 11.6), "silence interior should be cut"


def test_min_total_fallback_keeps_clip():
    result = plan_retention(
        _words(), 0.0, 30.0, silences=[(2.0, 25.0)],
        min_total=25.0, full_silence=True,
    )
    assert result["total_duration"] >= 25.0
    assert result["keep_segments"]


def test_cold_open_prepends_teaser():
    keep = [(10.0, 40.0)]
    out, teaser = prepend_cold_open(keep, 10.0, 40.0, payoff_source=36.0)
    assert teaser > 0.8
    assert len(out) == 2
    assert out[0][0] == 35.6
    assert out[1] == (10.0, 40.0)
