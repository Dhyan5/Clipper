from __future__ import annotations

import shutil
from pathlib import Path

from . import hooks
from .audio import envelope, extract_audio, loud_peak_times, silence_intervals
from .captions import build_ass
from .config import (
    CACHE_DIR, OUTPUT_DIR, Settings, check_ffmpeg, output_dir_for,
    parse_facecam_override, video_id_from_url,
)
from .downloader import download_audio, download_video
from .facecam import detect_facecam, detect_facecam_remote, scale_box
from .highlights import detect_highlights
from .layout import render_clip
from .packaging import (
    ClipMeta, QAResult, VideoMeta, build_zip, clip_filename, make_thumbnail,
    now_iso, thumbnail_path,
)
from .qa import TARGET_LUFS, check_clip, dead_air
from .retention import plan_retention, prepend_cold_open
from .transcriber import transcribe
from .utils import read_json, write_json


def _say(progress, msg: str) -> None:
    if progress:
        progress(msg)


def _flat_words(transcript: dict) -> list[dict]:
    return [w for seg in transcript.get("segments", []) for w in seg.get("words", [])]


def _find_cached(video_id: str, fingerprint: str) -> tuple[Path, dict] | None:
    for meta_path in OUTPUT_DIR.glob(f"*_{video_id}/metadata.json"):
        data = read_json(meta_path, {})
        if data.get("settings_fingerprint") != fingerprint:
            continue
        folder = meta_path.parent
        files = [folder / c["filename"] for c in data.get("clips", [])]
        if files and all(p.exists() and p.stat().st_size > 0 for p in files):
            if not (folder / "clips.zip").exists():
                build_zip(folder)
            return folder, data
    return None


def _source_size(source: Path, facecam: dict) -> tuple[int, int]:
    if facecam.get("width") and facecam.get("height"):
        return int(facecam["width"]), int(facecam["height"])
    from .utils import ffprobe

    info = ffprobe(source)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    return int(v["width"]), int(v["height"])


def _map_dead_air_to_source(keep: list[tuple[float, float]],
                            runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    extra: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in keep:
        length = e - s
        for a, b in runs:
            aa, bb = max(a, cursor), min(b, cursor + length)
            if bb - aa > 0.08:
                extra.append((round(s + aa - cursor, 3),
                              round(s + bb - cursor, 3)))
        cursor += length
    return extra


def _build_assets(plan: dict, transcript: dict, times, rms, loud_times: list[float],
                  pack: dict, settings: Settings, work: Path, end_override: float | None,
                  silence_min: float, full_silence: bool, gain_db: float = 0.0,
                  extra_cuts: list[tuple[float, float]] | None = None) -> dict:
    start = float(plan["start"])
    end = float(end_override if end_override is not None else plan["end"])
    if end <= start + 1.0:
        end = start + max(8.0, settings.min_len)
    words = _flat_words(transcript)
    silences = silence_intervals(times, rms, start, end, min_silence=silence_min)
    silences = sorted(silences + list(extra_cuts or []))
    ret = plan_retention(
        words, start, end, silences,
        min_silence=silence_min, min_total=15.0, full_silence=full_silence,
    )
    keep = ret["keep_segments"]
    teaser = 0.0
    if settings.cold_open:
        payoff_src = start + float(pack.get("payoff_time_sec") or 1.0)
        keep, teaser = prepend_cold_open(keep, start, end, payoff_src)
    total = round(sum(e - s for s, e in keep), 3)
    ass_path = work / f"clip_{plan['index']:02d}.ass"
    hook_options = pack.get("hooks") or [{"text": ""}]
    best = int(pack.get("best_hook_index") or 0)
    hook_text = hook_options[max(0, min(len(hook_options) - 1, best))]["text"]
    build_ass(
        transcript, keep, ass_path, hook_text=hook_text,
        preset_name=settings.preset, duration=total,
        emphasis_words=pack.get("emphasis_words") or [],
        loud_times=loud_times, offset=0.0, hide_caption_before=teaser,
    )
    return {"keep": keep, "total": total, "ass": ass_path,
            "teaser": teaser, "retention": ret, "end": end, "gain_db": gain_db}


def _render_and_check(plan: dict, transcript: dict, times, rms, loud_times,
                      pack: dict, settings: Settings, source: Path, facecam: dict,
                      width: int, height: int, folder: Path, work: Path,
                      boundaries: list[float], progress=None) -> dict:
    index = plan["index"]
    filename = clip_filename(index, (pack.get("titles") or ["clip"])[0])
    out_path = folder / filename
    gain = 0.0
    silence_min = 0.35
    full_silence = False
    end_override = None
    extra_cuts: list[tuple[float, float]] = []
    qa: dict = {}
    assets: dict = {}
    attempts = 0
    while True:
        assets = _build_assets(
            plan, transcript, times, rms, loud_times, pack, settings, work,
            end_override, silence_min, full_silence, gain, extra_cuts,
        )
        _say(progress, f"Rendering clip {index} (attempt {attempts + 1})")
        render_clip(
            source=source, keep_segments=assets["keep"],
            duration=assets["total"], facecam_box=facecam.get("box"),
            width=width, height=height, ass_path=assets["ass"],
            out_path=out_path, work_dir=work / f"clip_{index:02d}",
            gain_db=gain, progress=progress,
        )
        _say(progress, f"Quality check clip {index}")
        qa = check_clip(out_path, assets["ass"], boundaries, assets["end"])
        qa["attempts"] = attempts + 1
        if qa["passed"] or attempts >= 2:
            break
        failed = {c["name"] for c in qa["checks"] if not c["passed"]}
        fixed = False
        if "loudness" in failed and qa.get("metrics", {}).get("lufs") is not None:
            gain += TARGET_LUFS - float(qa["metrics"]["lufs"])
            fixed = True
        if "dead_air" in failed:
            air = dead_air(out_path)
            extra_cuts = _map_dead_air_to_source(assets["keep"], air["runs"])
            silence_min = 0.22
            full_silence = True
            fixed = bool(extra_cuts)
        if "ends_on_sentence" in failed and boundaries:
            end_override = min(boundaries, key=lambda b: abs(float(b) - assets["end"]))
            fixed = True
        if not fixed:
            break
        attempts += 1

    thumb = thumbnail_path(folder, index)
    make_thumbnail(out_path, thumb)
    return {
        "plan": plan, "pack": pack, "filename": filename, "path": out_path,
        "qa": qa, "assets": assets, "thumbnail": thumb,
    }


def _result_clip(result: dict, folder: Path, layout: str) -> ClipMeta:
    pack = result["pack"]
    plan = result["plan"]
    qa = result["qa"]
    hooks_list = pack.get("hooks") or []
    best = int(pack.get("best_hook_index") or 0)
    hook = hooks_list[max(0, min(len(hooks_list) - 1, best))]["text"] if hooks_list else ""
    titles = pack.get("titles") or ["Clip"]
    thumb = result["thumbnail"]
    return ClipMeta(
        index=plan["index"],
        filename=result["filename"],
        title_options=titles,
        chosen_title=titles[0],
        hook=hook,
        hook_variants=hooks_list,
        hook_start=0.10,
        score=round(float(plan["score"]), 2),
        score_breakdown=plan.get("breakdown") or {},
        start=round(float(plan["start"]), 3),
        end=round(float(result["assets"]["end"]), 3),
        duration=round(float(qa.get("metrics", {}).get("duration") or result["assets"]["total"]), 3),
        description=pack.get("description", ""),
        hashtags=pack.get("hashtags") or [],
        emphasis_words=pack.get("emphasis_words") or [],
        thumbnail=thumb.relative_to(folder).as_posix(),
        layout=layout,
        keep_segments=[[round(s, 3), round(e, 3)] for s, e in result["assets"]["keep"]],
        qa=QAResult(**{k: qa[k] for k in ("passed", "attempts", "dropped", "checks", "metrics")
                       if k in qa}),
    )


def _combined_mode(highlight_mode: str, hook_mode: str, api_key: str) -> str:
    if not api_key:
        return "fallback (no Gemini key): audio energy + keywords + speech density"
    if highlight_mode == "gemini" and hook_mode == "gemini":
        return "gemini"
    return f"fallback (highlights: {highlight_mode}; packaging: {hook_mode})"


def process(url: str, settings: Settings, progress=None, force: bool = False) -> dict:
    """Run the full pipeline for one YouTube URL. Cached results return instantly."""
    ok, msg = check_ffmpeg()
    if not ok:
        raise RuntimeError(msg)
    video_id = video_id_from_url(url)
    if not force:
        cached = _find_cached(video_id, settings.fingerprint())
        if cached:
            folder, data = cached
            _say(progress, "Cache hit: nothing to do")
            return {"folder": str(folder), "metadata": data, "cached": True,
                    "video_id": video_id}

    audio_dl = download_audio(url, progress)
    audio_path = Path(audio_dl["path"])
    title = audio_dl["info"].get("title") or video_id
    duration = float(audio_dl["info"].get("duration") or 0.0)
    folder = output_dir_for(title, video_id)
    folder.mkdir(parents=True, exist_ok=True)
    work = folder / "source"
    work.mkdir(parents=True, exist_ok=True)

    transcript = transcribe(audio_path, video_id, settings.whisper_model, progress)
    _say(progress, "Analyzing audio energy")
    extract_audio(audio_path, video_id)
    times, rms = envelope(video_id)
    override = parse_facecam_override(settings.facecam_override)
    facecam = detect_facecam_remote(
        url, video_id, duration or float(transcript.get("duration") or 0.0),
        override, progress,
    )
    loud = loud_peak_times(
        times, rms, 0.0,
        float(transcript.get("duration") or duration or 0.0), 0.93,
    )

    _say(progress, "Finding viral windows")
    from .config import get_api_key

    api_key = get_api_key()
    hl = detect_highlights(
        transcript, times, rms, api_key, settings.model, settings.clip_count,
        settings.min_len, settings.max_len, progress,
    )
    return _run_with_key(url, settings, progress, video_id, title, folder,
                         work, transcript, times, rms, loud, facecam, hl, api_key)


def _run_with_key(url, settings, progress, video_id, title, folder, work,
                  transcript, times, rms, loud, facecam, hl,
                  api_key: str) -> dict:
    plans = hl["plans"]
    if not plans:
        raise RuntimeError("No usable highlight windows were found")
    _say(progress, "Writing hooks and titles")
    pk = hooks.enrich(plans, transcript, api_key, settings.model, progress)
    packs = {p["index"]: p for p in pk["clips"]}
    mode = _combined_mode(hl["mode"], pk["mode"], api_key)
    boundaries = hl["boundaries"]

    _say(progress, "Downloading video for rendering")
    source = download_video(url, video_id, progress)
    real_w, real_h = _source_size(source, {})
    if facecam.get("box") and facecam.get("width") and facecam.get("height"):
        box = scale_box(facecam["box"], facecam["width"], facecam["height"],
                        real_w, real_h)
        facecam = {**facecam, "box": box, "width": real_w, "height": real_h}
    width, height = real_w, real_h
    layout = "facecam+gameplay" if facecam.get("box") else "gameplay+blurred-fill"

    results = []
    for plan in plans:
        pack = packs.get(plan["index"]) or hooks.fallback_packaging(plan, transcript)
        results.append(
            _render_and_check(
                plan, transcript, times, rms, loud, pack, settings, source, facecam,
                width, height, folder, work, boundaries, progress,
            )
        )

    passed = [r for r in results if r["qa"]["passed"]]
    kept = passed
    dropped = [r for r in results if r not in kept]
    for r in dropped:
        r["qa"]["dropped"] = True
        r["path"].unlink(missing_ok=True)
        r["thumbnail"].unlink(missing_ok=True)

    clip_metas = [_result_clip(r, folder, layout) for r in kept]
    meta = VideoMeta(
        video_id=video_id,
        url=url,
        source_title=title,
        slug=folder.name,
        folder=str(folder),
        created_at=now_iso(),
        mode=mode,
        settings={k: v for k, v in settings.__dict__.items()},
        settings_fingerprint=settings.fingerprint(),
        facecam={"box": facecam.get("box"), "engine": facecam.get("engine"),
                 "hits": facecam.get("hits"), "samples": facecam.get("samples")},
        clips=clip_metas,
        dropped=[{"index": r["plan"]["index"], "qa": r["qa"]} for r in dropped],
    )
    write_json(folder / "metadata.json", meta.model_dump())
    _say(progress, "Packaging zip")
    build_zip(folder)
    if not settings.keep_source:
        for child in list(work.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    if not kept:
        raise RuntimeError("Every clip failed the quality gate")
    return {"folder": str(folder), "metadata": meta.model_dump(), "cached": False,
            "video_id": video_id}


def rerender_clip(folder: str | Path, index: int, start: float, end: float,
                  settings: Settings, progress=None) -> dict:
    """Re-render one clip with user-supplied start/end (source cache required)."""
    folder = Path(folder)
    meta = read_json(folder / "metadata.json")
    if not meta:
        raise RuntimeError("metadata.json missing; generate clips first")
    video_id = meta["video_id"]
    audio_dl = download_audio(meta.get("url", ""), progress)
    audio_path = Path(audio_dl["path"])
    source = CACHE_DIR / video_id / "source.mp4"
    if not source.exists():
        source = download_video(meta.get("url", ""), video_id, progress)
    clip = next((c for c in meta["clips"] if c["index"] == index), None)
    if clip is None:
        raise RuntimeError(f"Clip {index} not found")
    if end <= start + 2:
        raise ValueError("End must be at least 2s after start")

    transcript = transcribe(audio_path, video_id, settings.whisper_model, progress)
    extract_audio(audio_path, video_id)
    times, rms = envelope(video_id)
    loud = loud_peak_times(times, rms, start, end, 0.93)
    facecam = detect_facecam(source, video_id)
    real_w, real_h = _source_size(source, {})
    if facecam.get("box") and facecam.get("width") and facecam.get("height"):
        facecam = {**facecam,
                   "box": scale_box(facecam["box"], facecam["width"],
                                    facecam["height"], real_w, real_h)}
    width, height = real_w, real_h
    pack = {
        "hooks": clip.get("hook_variants") or [{"type": "curiosity", "text": clip.get("hook", "")}],
        "best_hook_index": 0,
        "titles": clip.get("title_options") or [clip.get("chosen_title", "clip")],
        "emphasis_words": clip.get("emphasis_words") or [],
        "payoff_time_sec": min(end - start - 1.0, max(1.0, (end - start) - 2.0)),
    }
    keep = [(round(start, 3), round(end, 3))]
    total = round(end - start, 3)
    work = folder / "source"
    work.mkdir(parents=True, exist_ok=True)
    ass_path = work / f"clip_{index:02d}.ass"
    build_ass(
        transcript, keep, ass_path, hook_text=pack["hooks"][0]["text"],
        preset_name=settings.preset, duration=total,
        emphasis_words=pack["emphasis_words"], loud_times=loud,
    )
    out_path = folder / clip["filename"]
    _say(progress, f"Re-rendering clip {index}")
    qa: dict = {}
    gain = 0.0
    for attempt in range(3):
        render_clip(
            source=source, keep_segments=keep, duration=total,
            facecam_box=facecam.get("box"), width=width, height=height,
            ass_path=ass_path, out_path=out_path,
            work_dir=work / f"clip_{index:02d}", gain_db=gain,
        )
        qa = check_clip(out_path, ass_path, [], end)
        qa["attempts"] = attempt + 1
        if qa["passed"]:
            break
        lufs = qa.get("metrics", {}).get("lufs")
        if lufs is None or abs(float(lufs) - TARGET_LUFS) <= 1.5:
            break
        gain += TARGET_LUFS - float(lufs)

    make_thumbnail(out_path, thumbnail_path(folder, index))
    clip.update({
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(float(qa.get("metrics", {}).get("duration") or total), 3),
        "qa": qa,
    })
    meta["clips"] = [c if c["index"] != index else clip for c in meta["clips"]]
    write_json(folder / "metadata.json", meta)
    build_zip(folder)
    if not settings.keep_source:
        shutil.rmtree(work, ignore_errors=True)
    return {"folder": str(folder), "metadata": meta, "cached": False}
