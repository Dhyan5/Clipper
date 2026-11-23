from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from clipforge.config import (
    DEFAULT_MODEL, Settings, check_ffmpeg, get_api_key, save_api_key,
)
from clipforge.pipeline import process, rerender_clip
from clipforge.utils import read_json

st.set_page_config(page_title="ClipForge", page_icon="🎬", layout="wide")

ok, ffmpeg_msg = check_ffmpeg()
if not ok:
    st.error(ffmpeg_msg)
    st.stop()

st.title("ClipForge")
st.caption("One YouTube link in, several viral vertical clips out. 1080x1920, burned-in animated captions.")
st.info("Only process content you own or have permission to clip.")

with st.sidebar:
    st.header("Settings")
    clip_count = st.slider("Clips", 3, 10, 6)
    min_len = st.number_input("Min length (s)", 10, 60, 20, step=1)
    max_len = st.number_input("Max length (s)", 15, 90, 45, step=1)
    preset = st.selectbox(
        "Caption style",
        ["Hormozi", "MrBeast", "Gaming Neon", "Clean Minimal", "Classic"],
    )
    cold_open = st.toggle("Cold open teaser", value=False)
    keep_source = st.toggle("Keep temporary source files", value=False)
    whisper_model = st.selectbox("Whisper model", ["small", "base", "medium"], index=0)
    with st.expander("Advanced"):
        model = st.text_input("Gemini model", DEFAULT_MODEL)
        facecam_override = st.text_input(
            "Facecam box override (x,y,w,h in source pixels)", ""
        )

if max_len <= min_len:
    st.warning("Max length must be greater than min length.")

url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
key_input = st.text_input(
    "Gemini API key (optional, saved to .env)",
    type="password",
    placeholder="Paste a key to enable LLM scoring; blank uses .env or fallback",
)
if get_api_key():
    st.caption("A Gemini key is already configured (.env or saved earlier).")

generate = st.button("Generate clips", type="primary", use_container_width=True)

if generate:
    if not url.strip():
        st.error("Paste a YouTube link first.")
    elif max_len <= min_len:
        st.error("Fix the min/max length settings.")
    else:
        if key_input.strip():
            save_api_key(key_input)
        settings = Settings(
            clip_count=clip_count,
            min_len=float(min_len),
            max_len=float(max_len),
            preset=preset,
            cold_open=cold_open,
            keep_source=keep_source,
            whisper_model=whisper_model,
            model=model.strip() or DEFAULT_MODEL,
            facecam_override=facecam_override.strip(),
        )
        try:
            with st.status("Generating clips...", expanded=True) as status:
                bar = st.progress(0.0, text="Starting...")

                def on_progress(msg: str, frac: float | None = None) -> None:
                    if frac is None:
                        # New stage: log it permanently and reset the bar.
                        status.write(f"▶ {msg}")
                        status.update(label=msg)
                        bar.progress(0.0, text=msg)
                    else:
                        status.update(label=msg)
                        bar.progress(max(0.0, min(1.0, float(frac))), text=msg)

                result = process(url.strip(), settings, progress=on_progress)
                bar.progress(1.0, text="Done")
                status.update(label="Done", state="complete")
            st.session_state["folder"] = result["folder"]
            st.session_state["last_result"] = {
                "cached": result["cached"],
                "mode": result["metadata"].get("mode", ""),
            }
        except Exception as exc:
            st.exception(exc)

folder = st.session_state.get("folder")
last = st.session_state.get("last_result") or {}
if folder:
    folder = Path(folder)
    meta = read_json(folder / "metadata.json")
    if meta:
        if last.get("cached"):
            st.success("Loaded instantly from cache.")
        mode = meta.get("mode", "")
        if mode.startswith("gemini"):
            st.success("Gemini scoring and packaging enabled.")
        else:
            st.warning(f"Non-LLM mode: {mode}")

        st.subheader(meta.get("source_title", "Clips"))
        clips = meta.get("clips", [])
        cols = st.columns(2)
        for n, clip in enumerate(clips):
            path = folder / clip["filename"]
            with cols[n % 2]:
                if path.exists():
                    st.video(str(path))
                st.markdown(f"**{clip.get('chosen_title', 'Clip')}**")
                st.caption(
                    f"Hook: {clip.get('hook', '')}  |  Score: {clip.get('score', 0)}"
                    f"  |  {clip.get('start', 0):.1f}s - {clip.get('end', 0):.1f}s"
                )
                qa = clip.get("qa") or {}
                if qa.get("passed"):
                    st.success("Quality gate passed")
                else:
                    st.warning("Quality gate flagged this clip")
                if path.exists():
                    st.download_button(
                        "Download MP4", path.read_bytes(), file_name=path.name,
                        mime="video/mp4", key=f"dl_{clip['index']}",
                    )
                with st.expander("Edit timing and re-render"):
                    new_start = st.number_input(
                        "Start (s)", 0.0, 7200.0, float(clip.get("start", 0.0)),
                        step=0.5, key=f"start_{clip['index']}",
                    )
                    new_end = st.number_input(
                        "End (s)", 0.0, 7200.0, float(clip.get("end", 0.0)),
                        step=0.5, key=f"end_{clip['index']}",
                    )
                    if st.button("Re-render clip", key=f"re_{clip['index']}"):
                        settings = Settings(
                            clip_count=clip_count,
                            min_len=float(min_len),
                            max_len=float(max_len),
                            preset=preset,
                            cold_open=cold_open,
                            keep_source=keep_source,
                            whisper_model=whisper_model,
                            model=model.strip() or DEFAULT_MODEL,
                        )
                        try:
                            with st.spinner("Re-rendering..."):
                                rerender_clip(folder, clip["index"], new_start,
                                              new_end, settings)
                            st.rerun()
                        except Exception as exc:
                            st.exception(exc)

        zip_path = folder / "clips.zip"
        if zip_path.exists():
            st.download_button(
                "Download all as ZIP", zip_path.read_bytes(),
                file_name=zip_path.name, mime="application/zip", key="dl_zip",
            )
        with st.expander("metadata.json"):
            st.code(json.dumps(meta, indent=2)[:12000], language="json")
        if meta.get("dropped"):
            st.warning(f"{len(meta['dropped'])} clip(s) dropped by the quality gate.")
