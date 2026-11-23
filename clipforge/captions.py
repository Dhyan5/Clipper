from __future__ import annotations

import re
from pathlib import Path

SAFE_TOP = 154          # 8% of 1920
SAFE_BOTTOM = 1632      # bottom 15% starts here
SAFE_RIGHT = 950        # right 12% starts at 950 of 1080
CAPTION_Y = 1000        # just under the seam (960)
HOOK_Y = 180

PRESETS = {
    "Hormozi": {
        "font": "Anton", "size": 86, "outline": 7, "shadow": 3,
        "primary": "&H00FFFFFF", "accent": ["&H0000FFFF", "&H0000FF00", "&H00FFFF00", "&H00FF00FF"],
        "uppercase": True, "emoji": True, "shake": True, "margin": 140,
    },
    "MrBeast": {
        "font": "Anton", "size": 92, "outline": 8, "shadow": 2,
        "primary": "&H00FFFFFF", "accent": ["&H0000FFFF", "&H0000FF00", "&H00FFFF00", "&H0080FFFF"],
        "uppercase": True, "emoji": True, "shake": True, "margin": 140,
    },
    "Gaming Neon": {
        "font": "Anton", "size": 90, "outline": 6, "shadow": 4,
        "primary": "&H00FFFFFF", "accent": ["&H00FFFF00", "&H00FF00FF", "&H0000FFFF", "&H0000FF00"],
        "uppercase": True, "emoji": False, "shake": True, "margin": 140,
    },
    "Clean Minimal": {
        "font": "Anton", "size": 72, "outline": 4, "shadow": 1,
        "primary": "&H00FFFFFF", "accent": ["&H0000FFFF", "&H00D7FF", "&H00F5F5F5", "&H00FFFFFF"],
        "uppercase": False, "emoji": False, "shake": False, "margin": 150,
    },
    "Classic": {
        "font": "Anton", "size": 78, "outline": 5, "shadow": 2,
        "primary": "&H00FFFFFF", "accent": ["&H0000FFFF", "&H0050FFFF", "&H0000FFFF", "&H0050FFFF"],
        "uppercase": True, "emoji": False, "shake": False, "margin": 145,
    },
}

PROFANITY = {
    "fuck", "fucking", "shit", "bitch", "asshole", "cunt", "dick", "bastard",
}

EMOTION_EMOJI = {
    "money": "💰", "cash": "💰", "rich": "💰", "love": "❤", "dead": "💀",
    "death": "💀", "kill": "💀", "win": "🔥", "best": "🔥", "insane": "🤯",
    "crazy": "🤯", "secret": "🤫", "fear": "😱", "scared": "😱", "free": "🤑",
}

EMOTIONAL_WORDS = {
    "never", "always", "hate", "love", "death", "dead", "kill", "money",
    "rich", "broke", "insane", "crazy", "shocking", "secret", "truth",
    "worst", "best", "win", "lose", "free", "fear", "scared", "impossible",
    "unbelievable", "wrong", "mistake", "everyone", "nobody", "huge", "massive",
}


def _ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _clean(text: str) -> str:
    text = (text or "").replace("{", "").replace("}", "").replace("\\", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _mask(text: str) -> str:
    core = re.sub(r"[^A-Za-z]", "", text).lower()
    return "****" if core in PROFANITY else text


def parse_ass_events(ass_text: str) -> list[dict]:
    events = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        events.append(
            {
                "layer": parts[0].split(":", 1)[1].strip(),
                "start": parts[1].strip(),
                "end": parts[2].strip(),
                "style": parts[3].strip(),
                "text": parts[9].strip(),
            }
        )
    return events


def map_words(transcript: dict, keep_segments: list[tuple[float, float]],
              offset: float = 0.0) -> tuple[list[dict], float]:
    """Remap source-time words onto the edited (cut) timeline."""
    out: list[dict] = []
    out_t = float(offset)
    for (s, e) in keep_segments:
        for seg in transcript.get("segments", []):
            for w in seg.get("words", []):
                mid = (w["start"] + w["end"]) / 2.0
                if s <= mid < e:
                    ns = max(w["start"], s) - s + out_t
                    ne = min(w["end"], e) - s + out_t
                    if ne > ns + 0.03:
                        out.append(
                            {"text": w["word"], "start": round(ns, 3),
                             "end": round(ne, 3), "orig_mid": mid}
                        )
        out_t += e - s
    out.sort(key=lambda w: w["start"])
    return out, out_t


def _group(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and (
            len(cur) >= 3
            or w["start"] - cur[-1]["end"] > 0.45
            or w["start"] - cur[0]["start"] > 1.9
        ):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def _is_emphasis(word: dict, emphasis: set[str], loud_times: list[float], base_size: int) -> bool:
    txt = _clean(word["text"])
    core = re.sub(r"[^A-Za-z0-9]", "", txt).lower()
    if re.search(r"\d", txt):
        return True
    if core in EMOTIONAL_WORDS or core in emphasis:
        return True
    if txt.isupper() and len(txt) > 2:
        return True
    return any(abs(word["orig_mid"] - t) < 0.35 for t in loud_times)


def _width_scale(chars: int, spaces: int, size: int) -> float:
    est = chars * 0.47 * size + spaces * 0.24 * size
    return min(1.0, 780.0 / est) if est > 0 else 1.0


def _compose(group: list[dict], active: int, preset: dict, emphasis: set[str],
             loud_times: list[float], accent: str) -> str:
    base = preset["size"]
    joined = " ".join(_clean(w["text"]) for w in group)
    scale = _width_scale(len(joined.replace(" ", "")), joined.count(" "), base)
    eff = max(40, int(base * scale))
    parts = []
    for i, w in enumerate(group):
        txt = _mask(_clean(w["text"]))
        if preset["uppercase"]:
            txt = txt.upper()
        emph = _is_emphasis(w, emphasis, loud_times, eff)
        if i == active:
            size = int(eff * (1.3 if emph else 1.0))
            tags = (
                f"\\c{accent}\\fs{size}\\fscx70\\fscy70"
                "\\t(0,80,\\fscx115\\fscy115)\\t(80,120,\\fscx100\\fscy100)"
            )
            if emph and preset["shake"]:
                tags += "\\frz3\\t(0,40,\\frz-3)\\t(40,80,\\frz3)\\t(80,120,\\frz0)"
            shown = txt
            if preset["emoji"] and emph:
                core = re.sub(r"[^A-Za-z]", "", w["text"]).lower()
                emoji = next((v for k, v in EMOTION_EMOJI.items() if k in core), "")
                shown = f"{txt} {emoji}".strip() if emoji else txt
            parts.append("{" + tags + "}" + shown)
        else:
            parts.append(f"{{\\c{preset['primary']}\\fs{int(eff * 0.92)}\\fscx100\\fscy100}}{txt}")
    return " ".join(parts)


def _hook_text(hook: str, max_chars: int = 46) -> str:
    hook = _clean(hook).upper()
    if len(hook) <= max_chars:
        return hook
    words = hook.split()
    half = len(words) // 2
    return " ".join(words[:half]) + "\\N" + " ".join(words[half:])


def build_ass(
    transcript: dict,
    keep_segments: list[tuple[float, float]],
    out_path: Path | str,
    hook_text: str = "",
    preset_name: str = "Hormozi",
    duration: float = 30.0,
    emphasis_words: list[str] | None = None,
    loud_times: list[float] | None = None,
    offset: float = 0.0,
    hide_caption_before: float = 0.0,
) -> dict:
    """Write a word-synced animated .ass file for the edited clip timeline."""
    preset = PRESETS.get(preset_name, PRESETS["Hormozi"])
    emphasis = {w.lower() for w in (emphasis_words or [])}
    loud_times = loud_times or []
    words, total = map_words(transcript, keep_segments, offset=offset)
    groups = _group(words)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 0\n"
        "YCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{preset['font']},{preset['size']},{preset['primary']},"
        f"&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,"
        f"{preset['outline']},{preset['shadow']},5,{preset['margin']},"
        f"{preset['margin']},0,1\n"
        f"Style: Hook,{preset['font']},54,&H0000FFFF,&H000000FF,&H00000000,"
        f"&H80000000,-1,0,0,0,100,100,1,0,1,7,4,8,130,130,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    if hook_text:
        hook = _hook_text(hook_text)
        events.append(
            f"Dialogue: 1,{_ass_time(0.10)},{_ass_time(2.50)},Hook,,0,0,0,,"
            "{\\an8\\pos(540,180)\\fad(0,250)\\fscx82\\fscy82"
            "\\t(0,120,\\fscx112\\fscy112)\\t(120,240,\\fscx100\\fscy100)}"
            f"{hook}"
        )

    captions = 0
    for gi, group in enumerate(groups):
        accent = preset["accent"][gi % len(preset["accent"])]
        for wi, word in enumerate(group):
            start = word["start"]
            if start < hide_caption_before:
                continue
            if wi + 1 < len(group):
                end = group[wi + 1]["start"]
            else:
                end = max(word["end"] + 0.12, group[-1]["end"] + 0.12)
            end = min(end, duration)
            if end <= start + 0.04:
                end = start + 0.12
            text = _compose(group, wi, preset, emphasis, loud_times, accent)
            events.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
                f"{{\\an5\\pos(540,{CAPTION_Y})}}{text}"
            )
            captions += 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return {
        "path": out_path,
        "hook_start": 0.10,
        "caption_events": captions,
        "groups": len(groups),
        "mapped_words": len(words),
        "timeline_duration": round(total, 3),
        "preset": preset_name,
    }
