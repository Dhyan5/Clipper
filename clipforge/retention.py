from __future__ import annotations

import re

FILLERS = {"um", "uh", "erm", "hmm", "mmm", "ah", "eh"}


def _merge(cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cuts = sorted((max(0.0, s), max(0.0, e)) for s, e in cuts if e > s)
    merged: list[tuple[float, float]] = []
    for s, e in cuts:
        if merged and s <= merged[-1][1] + 0.001:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _segments_from_cuts(start: float, end: float,
                        cuts: list[tuple[float, float]],
                        min_segment: float = 0.20) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    cursor = start
    for s, e in _merge(cuts):
        s, e = max(start, s), min(end, e)
        if s > cursor + 0.01:
            segments.append((round(cursor, 3), round(s, 3)))
        cursor = max(cursor, e)
    if end > cursor + 0.01:
        segments.append((round(cursor, 3), round(end, 3)))
    return [(s, e) for s, e in segments if e - s >= min_segment]


def plan_retention(
    words: list[dict],
    start: float,
    end: float,
    silences: list[tuple[float, float]],
    breath: float = 0.1,
    min_silence: float = 0.35,
    min_total: float = 15.0,
    remove_fillers: bool = True,
    full_silence: bool = False,
) -> dict:
    """Cut long silences (keep a breath) and isolated filler words."""
    word_list = sorted(
        [w for w in words if start <= (w["start"] + w["end"]) / 2.0 < end],
        key=lambda w: w["start"],
    )
    lead_trim = 0.0
    if remove_fillers and word_list:
        first = re.sub(r"[^a-z]", "", word_list[0]["word"].lower())
        if first in FILLERS and len(word_list) > 1:
            lead_trim = max(0.0, word_list[1]["start"] - start - 0.05)

    def build(filler_cuts: bool, silence_min: float, strip_all: bool) -> list[tuple[float, float]]:
        cuts: list[tuple[float, float]] = []
        for s, e in silences:
            s, e = max(start, s), min(end, e)
            if e - s < silence_min:
                continue
            if strip_all:
                cuts.append((s, e))
            else:
                cs, ce = s + breath, e - breath
                if ce - cs > 0.08:
                    cuts.append((cs, ce))
        if filler_cuts and remove_fillers:
            for n, w in enumerate(word_list):
                core = re.sub(r"[^a-z]", "", w["word"].lower())
                if core not in FILLERS:
                    continue
                prev_end = word_list[n - 1]["end"] if n > 0 else start
                next_start = word_list[n + 1]["start"] if n + 1 < len(word_list) else end
                if w["start"] - prev_end >= 0.08 and next_start - w["end"] >= 0.08:
                    cuts.append((w["start"] - 0.02, w["end"] + 0.02))
        segs = _segments_from_cuts(start + lead_trim, end, cuts)
        return segs

    segments = build(True, min_silence, full_silence)
    if sum(e - s for s, e in segments) < min_total:
        segments = build(False, min_silence, full_silence)
    if sum(e - s for s, e in segments) < min_total:
        segments = build(False, min_silence * 1.6, False)
    if not segments or sum(e - s for s, e in segments) < min_total:
        segments = [(round(start, 3), round(end, 3))]

    total = round(sum(e - s for s, e in segments), 3)
    return {
        "keep_segments": segments,
        "total_duration": total,
        "silences": [[round(s, 3), round(e, 3)] for s, e in silences],
        "lead_trim": round(lead_trim, 3),
        "segments": len(segments),
    }


def prepend_cold_open(keep_segments: list[tuple[float, float]], source_start: float,
                      source_end: float, payoff_source: float,
                      teaser: float = 1.5) -> tuple[list[tuple[float, float]], float]:
    """Duplicate a short payoff teaser before the clip (no captions on it)."""
    ts = max(source_start, min(source_end - teaser, payoff_source - 0.4))
    te = min(source_end, ts + teaser)
    if te - ts < 0.8:
        return keep_segments, 0.0
    return [(round(ts, 3), round(te, 3))] + list(keep_segments), round(te - ts, 3)
