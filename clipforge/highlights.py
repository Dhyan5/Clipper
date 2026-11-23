from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .audio import energy_between
from .gemini import GeminiClient, GeminiError, load_prompt

POWER_WORDS = {
    "money", "free", "secret", "truth", "never", "always", "why", "how",
    "best", "worst", "insane", "crazy", "shocking", "mistake", "win", "lose",
    "death", "dead", "love", "hate", "fear", "impossible", "nobody",
    "everyone", "actually", "literally", "huge", "massive", "rich", "broke",
    "stop", "start", "change", "changed", "problem", "warning", "danger",
}
FILLERS = {"um", "uh", "erm", "hmm", "mmm", "ah", "eh"}


@dataclass
class SentenceUnit:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    index: int
    i: int
    j: int
    start: float
    end: float
    text: str
    score: float = 0.0
    breakdown: dict = field(default_factory=dict)
    reason: str = ""
    reject: bool = False
    keywords: list[str] = field(default_factory=list)
    offset_start: float = 0.0
    offset_end: float = 0.0


class WindowScore(BaseModel):
    index: int
    hook_strength: float = Field(5, ge=0, le=10)
    emotional_peak: float = Field(5, ge=0, le=10)
    payoff: float = Field(5, ge=0, le=10)
    self_contained: float = Field(5, ge=0, le=10)
    shareability: float = Field(5, ge=0, le=10)
    curiosity_gap: float = Field(5, ge=0, le=10)
    quotability: float = Field(5, ge=0, le=10)
    reject: bool = False
    reason: str = ""
    best_start_offset_sec: float = 0.0
    best_end_offset_sec: float = 0.0
    keywords: list[str] = Field(default_factory=list)

    def rubric_mean(self) -> float:
        vals = [self.hook_strength, self.emotional_peak, self.payoff,
                self.self_contained, self.shareability, self.curiosity_gap,
                self.quotability]
        return sum(vals) / len(vals)


class WindowBatch(BaseModel):
    windows: list[WindowScore]


def sentence_units(transcript: dict) -> list[SentenceUnit]:
    units: list[SentenceUnit] = []
    for seg in transcript.get("segments", []):
        current: list[dict] = []
        for w in seg.get("words", []):
            current.append(w)
            if re.search(r"[.!?][\"')\]]?$", w.get("word", "")) and len(current) >= 3:
                units.append(SentenceUnit(current[0]["start"], current[-1]["end"],
                                          " ".join(x["word"] for x in current)))
                current = []
        if current:
            units.append(SentenceUnit(current[0]["start"], current[-1]["end"],
                                      " ".join(x["word"] for x in current)))
    return units


def boundary_times(units: list[SentenceUnit]) -> list[float]:
    if not units:
        return []
    bounds = {round(u.start, 3) for u in units} | {round(u.end, 3) for u in units}
    return sorted(bounds)


def candidates_from_units(units: list[SentenceUnit], min_len: float,
                          max_len: float, cap: int = 90) -> list[Candidate]:
    raw: list[Candidate] = []
    for i in range(len(units)):
        for j in range(i + 1, len(units) + 1):
            dur = units[j - 1].end - units[i].start
            if dur < min_len - 0.01:
                continue
            if dur > max_len + 0.01:
                break
            raw.append(
                Candidate(
                    index=len(raw), i=i, j=j,
                    start=round(units[i].start, 3),
                    end=round(units[j - 1].end, 3),
                    text=" ".join(u.text for u in units[i:j]),
                )
            )
    if len(raw) <= cap:
        return raw
    keep = np.linspace(0, len(raw) - 1, cap).astype(int)
    sampled = [raw[k] for k in sorted(set(keep.tolist()))]
    for idx, cand in enumerate(sampled):
        cand.index = idx
    return sampled


def _window_words(transcript: dict, start: float, end: float) -> list[dict]:
    return [
        w for seg in transcript.get("segments", []) for w in seg.get("words", [])
        if start <= (w["start"] + w["end"]) / 2.0 < end
    ]


def _keyword_hits(words: list[dict]) -> int:
    hits = 0
    for w in words:
        core = re.sub(r"[^a-z]", "", w["word"].lower())
        if core in POWER_WORDS:
            hits += 1
        if re.search(r"\d", w["word"]):
            hits += 1
        if w["word"].isupper() and len(core) > 2:
            hits += 1
    return hits


def _rank(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return [7.0]
    order = arr.argsort().argsort().astype(float)
    return (order / (len(arr) - 1) * 10.0).tolist()


def heuristic_candidates(transcript: dict, units: list[SentenceUnit],
                         min_len: float, max_len: float,
                         times=None, rms=None, cap: int = 90) -> list[Candidate]:
    cands = candidates_from_units(units, min_len, max_len, cap=cap)
    energies: list[float] = []
    densities: list[float] = []
    keywords: list[float] = []
    for c in cands:
        words = _window_words(transcript, c.start, c.end)
        dur = max(0.1, c.end - c.start)
        energies.append(energy_between(times, rms, c.start, c.end)
                        if times is not None and rms is not None else 0.0)
        densities.append(len(words) / dur)
        keywords.append(float(_keyword_hits(words)))
    e_rank = _rank(energies)
    d_rank = _rank(densities)
    k_rank = _rank(keywords)
    for n, c in enumerate(cands):
        dur = max(0.1, c.end - c.start)
        wps = densities[n]
        reject = wps < 0.35 or (e_rank[n] < 2 and d_rank[n] < 2)
        score = 0.50 * e_rank[n] + 0.30 * d_rank[n] + 0.20 * k_rank[n]
        c.score = round(float(score), 2)
        c.reject = bool(reject)
        c.reason = "quiet / little speech" if reject else "audio energy + speech density + keywords"
        c.breakdown = {
            "gemini": None,
            "audio": round(float(e_rank[n]), 2),
            "density": round(float(d_rank[n]), 2),
            "keywords": round(float(k_rank[n]), 2),
            "final": c.score,
            "words_per_sec": round(wps, 2),
        }
        c.keywords = _top_keywords(_window_words(transcript, c.start, c.end), 5)
    return cands


def _top_keywords(words: list[dict], limit: int = 5) -> list[str]:
    stop = {
        "the", "and", "that", "this", "with", "have", "from", "your", "you",
        "for", "are", "was", "but", "not", "they", "what", "when", "how",
        "there", "their", "about", "just", "like", "know", "want", "going",
    }
    seen: dict[str, int] = {}
    for w in words:
        core = re.sub(r"[^a-z0-9]", "", w["word"].lower())
        if len(core) < 4 or core in stop:
            continue
        seen[core] = seen.get(core, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:limit]]


def _snap_window(cand: Candidate, units: list[SentenceUnit], min_len: float,
                 max_len: float) -> tuple[float, float]:
    starts = list(range(cand.i, cand.j))
    ends = list(range(cand.i + 1, cand.j + 1))
    target_s = cand.start + cand.offset_start
    target_e = cand.end + cand.offset_end
    si = min(starts, key=lambda k: abs(units[k].start - target_s))
    ej = min(ends, key=lambda k: abs(units[k - 1].end - target_e))
    if ej <= si:
        ej = si + 1
    while ej > si + 1 and units[ej - 1].end - units[si].start > max_len:
        ej -= 1
    while ej - 1 < len(units) - 1 and units[ej - 1].end - units[si].start < min_len:
        if ej >= cand.j:
            break
        ej += 1
    if units[ej - 1].end - units[si].start < min_len and si > cand.i:
        si -= 1
    return round(units[si].start, 3), round(units[ej - 1].end, 3)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    shortest = max(0.001, min(a_end - a_start, b_end - b_start))
    return inter / shortest


def gemini_candidates(client: GeminiClient, transcript: dict,
                      cands: list[Candidate], progress=None) -> bool:
    if not cands:
        return False
    if progress:
        progress("Scoring highlights with Gemini")
    payload = {
        "duration": transcript.get("duration"),
        "candidates": [
            {"index": c.index, "start": c.start, "end": c.end,
             "duration": round(c.end - c.start, 2), "text": c.text[:700]}
            for c in cands
        ],
    }
    prompt = load_prompt("highlights").replace("{payload}", json.dumps(payload))
    try:
        data = client.generate_json(prompt)
        batch = WindowBatch.model_validate(data)
    except (GeminiError, ValidationError, ValueError):
        return False
    by_index = {w.index: w for w in batch.windows}
    if not by_index:
        return False
    for c in cands:
        w = by_index.get(c.index)
        if not w:
            continue
        c.reject = w.reject
        c.reason = w.reason[:300]
        c.keywords = [str(k)[:40] for k in w.keywords[:5]]
        c.offset_start = float(w.best_start_offset_sec)
        c.offset_end = float(w.best_end_offset_sec)
        g = w.rubric_mean()
        c.breakdown = {
            "gemini": round(g, 2),
            "audio": c.breakdown.get("audio", 0.0),
            "density": c.breakdown.get("density", 0.0),
            "keywords": c.breakdown.get("keywords", 0.0),
            "reasons": c.reason,
        }
        c.score = round(
            0.60 * g + 0.25 * float(c.breakdown.get("audio") or 0)
            + 0.15 * float(c.breakdown.get("density") or 0), 2
        )
        c.breakdown["final"] = c.score
    return True


def detect_highlights(transcript: dict, times, rms, api_key: str,
                      model: str, clip_count: int, min_len: float,
                      max_len: float, progress=None) -> dict:
    units = sentence_units(transcript)
    if not units:
        raise RuntimeError("No sentence units in transcript")
    cands = heuristic_candidates(transcript, units, min_len, max_len, times, rms)
    if not cands:
        # Fall back to fixed sentence windows if the video is unusually paced.
        cands = candidates_from_units(units, 0.0, 1e9, cap=50)

    mode = "fallback (heuristic)" if not api_key else "heuristic"
    client = GeminiClient(api_key, model)
    if client.available:
        try:
            if gemini_candidates(client, transcript, cands, progress):
                mode = "gemini"
            else:
                mode = "fallback (Gemini response invalid)"
        except Exception as exc:  # never crash the pipeline for an LLM error
            mode = f"fallback (Gemini unavailable: {str(exc)[:120]})"

    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    chosen: list[Candidate] = []
    for threshold in (0.15, 0.35):
        for c in ranked:
            if len(chosen) >= clip_count:
                break
            if c.reject or any(
                _overlap(c.start, c.end, x.start, x.end) > threshold for x in chosen
            ):
                continue
            chosen.append(c)
        if len(chosen) >= clip_count:
            break

    plans = []
    for n, c in enumerate(chosen[:clip_count], start=1):
        start, end = _snap_window(c, units, min_len, max_len)
        plans.append(
            {
                "index": n,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "score": c.score,
                "breakdown": c.breakdown,
                "reason": c.reason,
                "keywords": c.keywords,
                "text": c.text,
                "source_window": [c.start, c.end],
            }
        )
    return {"mode": mode, "plans": plans, "units": units,
            "boundaries": boundary_times(units), "candidate_count": len(cands)}
