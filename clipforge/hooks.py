from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from .gemini import GeminiClient, GeminiError, load_prompt

GENERIC_TAGS = [
    "#shorts", "#viral", "#clips", "#interview", "#story", "#podcast",
    "#creator", "#trending", "#fyp", "#learn", "#facts", "#video",
]

STOP = {
    "the", "and", "that", "this", "with", "have", "from", "your", "you",
    "for", "are", "was", "but", "not", "they", "what", "when", "how", "our",
    "there", "their", "about", "just", "like", "know", "want", "going", "i",
    "we", "it", "to", "of", "in", "on", "is", "be", "a", "an", "do", "did",
}


class HookVariant(BaseModel):
    type: str = "curiosity"
    text: str = ""


class ClipPackaging(BaseModel):
    index: int
    hooks: list[HookVariant] = Field(default_factory=list)
    best_hook_index: int = 0
    titles: list[str] = Field(default_factory=list)
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    emphasis_words: list[str] = Field(default_factory=list)
    payoff_time_sec: float = 0.0


class PackagingBatch(BaseModel):
    clips: list[ClipPackaging] = Field(default_factory=list)


def _truncate_words(text: str, limit: int = 8) -> str:
    words = re.sub(r"\s+", " ", (text or "").strip()).split()
    return " ".join(words[:limit])


def _title(text: str, limit: int = 59) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut or text[:limit]


def _norm_tag(tag: str) -> str:
    core = re.sub(r"[^A-Za-z0-9]", "", tag or "")
    return f"#{core}" if core else ""


def _clip_words(plan: dict, transcript: dict) -> list[str]:
    out = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            mid = (w["start"] + w["end"]) / 2.0
            if plan["start"] <= mid < plan["end"]:
                core = re.sub(r"[^a-z0-9]", "", w["word"].lower())
                if core:
                    out.append(core)
    return out


def _top_terms(plan: dict, transcript: dict, limit: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for core in _clip_words(plan, transcript):
        if len(core) < 4 or core in STOP:
            continue
        counts[core] = counts.get(core, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    terms = [k for k, _ in ranked[:limit]]
    for k in plan.get("keywords", []):
        k = re.sub(r"[^a-z0-9]", "", str(k).lower())
        if k and k not in terms:
            terms.append(k)
    return terms[:limit] or ["story"]


def fallback_packaging(plan: dict, transcript: dict) -> dict:
    terms = _top_terms(plan, transcript)
    kw = terms[0]
    hooks = [
        {"type": "curiosity", "text": _truncate_words(f"The truth about {kw}")},
        {"type": "shock", "text": _truncate_words(f"You won't believe this {kw}")},
        {"type": "relatable", "text": _truncate_words(f"Everyone gets {kw} wrong")},
    ]
    text = re.sub(r"\s+", " ", plan.get("text", "")).strip()
    snippet = text[:80]
    titles = [
        _title(f"{kw.title()}: What Nobody Tells You"),
        _title(snippet.title() if snippet else f"This {kw.title()} Changes Everything"),
        _title(f"Why {kw.title()} Matters More Than You Think"),
    ]
    hashtags: list[str] = []
    for t in terms + ["shorts", "viral", "clips", "interview", "story"]:
        tag = _norm_tag(t)
        if tag and tag not in hashtags:
            hashtags.append(tag)
    for tag in GENERIC_TAGS:
        if len(hashtags) >= 12:
            break
        if tag not in hashtags:
            hashtags.append(tag)
    emphasis = list(dict.fromkeys(
        [w for w in plan.get("keywords", [])][:5] + [kw]
    ))
    payoff = max(1.0, min(plan["duration"] - 1.0, plan["duration"] - 2.0))
    return {
        "index": plan["index"],
        "hooks": hooks,
        "best_hook_index": 0,
        "titles": titles,
        "description": (snippet + ("..." if len(text) > 80 else "")).strip()
        + " Follow for more clips like this.",
        "hashtags": hashtags[:15],
        "emphasis_words": emphasis[:8],
        "payoff_time_sec": round(payoff, 2),
    }


def sanitize_packaging(pack: dict, plan: dict, transcript: dict) -> dict:
    fb = fallback_packaging(plan, transcript)
    hooks = [h for h in (pack.get("hooks") or []) if str(h.get("text", "")).strip()]
    hooks = [
        {"type": str(h.get("type") or "curiosity"),
         "text": _truncate_words(str(h.get("text", "")))}
        for h in hooks[:3]
    ]
    while len(hooks) < 3:
        hooks.append(fb["hooks"][len(hooks) % 3])
    hooks = [h for h in hooks if h["text"]][:3]
    if not hooks:
        hooks = fb["hooks"]

    try:
        best = int(pack.get("best_hook_index", 0))
    except (TypeError, ValueError):
        best = 0
    best = max(0, min(len(hooks) - 1, best))

    titles = [_title(str(t)) for t in (pack.get("titles") or []) if str(t).strip()]
    titles = list(dict.fromkeys([t for t in titles if t]))
    for cand in fb["titles"]:
        if len(titles) >= 3:
            break
        if cand not in titles:
            titles.append(cand)
    titles = titles[:3]

    tags: list[str] = []
    for tag in pack.get("hashtags") or []:
        n = _norm_tag(str(tag))
        if n and n.lower() not in {x.lower() for x in tags}:
            tags.append(n)
    for tag in fb["hashtags"] + GENERIC_TAGS:
        if len(tags) >= 12:
            break
        if tag.lower() not in {x.lower() for x in tags}:
            tags.append(tag)
    tags = tags[:15]

    spoken = set(_clip_words(plan, transcript))
    emph = []
    for w in pack.get("emphasis_words") or []:
        core = re.sub(r"[^a-z0-9]", "", str(w).lower())
        if core and (core in spoken or len(emph) < 3):
            emph.append(core)
    if not emph:
        emph = fb["emphasis_words"]
    emph = list(dict.fromkeys(emph))[:8]

    try:
        payoff = float(pack.get("payoff_time_sec") or fb["payoff_time_sec"])
    except (TypeError, ValueError):
        payoff = float(fb["payoff_time_sec"])
    payoff = max(0.5, min(float(plan["duration"]) - 0.5, payoff))

    description = re.sub(r"\s+", " ", str(pack.get("description") or "")).strip()
    if not description:
        description = fb["description"]

    return {
        "index": plan["index"],
        "hooks": hooks,
        "best_hook_index": best,
        "titles": titles,
        "description": description,
        "hashtags": tags,
        "emphasis_words": emph,
        "payoff_time_sec": round(payoff, 2),
    }


def enrich(plans: list[dict], transcript: dict, api_key: str, model: str,
           progress=None) -> dict:
    """One batched Gemini call for hooks + packaging; templates on failure."""
    mode = "template"
    by_index: dict[int, dict] = {}
    if api_key:
        if progress:
            progress("Writing hooks and packaging with Gemini")
        payload = {
            "clips": [
                {
                    "index": p["index"],
                    "start": p["start"],
                    "end": p["end"],
                    "duration": p["duration"],
                    "text": p.get("text", "")[:1800],
                    "keywords": p.get("keywords", []),
                }
                for p in plans
            ]
        }
        prompt = load_prompt("hooks").replace("{payload}", json.dumps(payload))
        try:
            data = GeminiClient(api_key, model).generate_json(prompt)
            batch = PackagingBatch.model_validate(data)
            for item in batch.clips:
                by_index[item.index] = item.model_dump()
            mode = "gemini"
        except (GeminiError, ValidationError, ValueError) as exc:
            mode = f"fallback (Gemini packaging failed: {str(exc)[:120]})"

    out = []
    for plan in plans:
        pack = by_index.get(plan["index"]) or fallback_packaging(plan, transcript)
        out.append(sanitize_packaging(pack, plan, transcript))
    return {"mode": mode, "clips": out}
