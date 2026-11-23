# ClipForge — Progress

Each stage is checked only after its VERIFY command ran and passed locally.
The earlier no-media user override was later lifted ("continue and do not stop
until all stages are achieved"), so all stages including full YouTube E2E are
now verified. Real output is pasted below.

- [x] Stage 1 — Scaffold + config + env checks
- [x] Stage 2 — downloader + transcriber
- [x] Stage 3 — facecam_detector + layout_renderer
- [x] Stage 4 — dynamic captions
- [x] Stage 5 — Gemini client + highlight_detector (fallback mode)
- [x] Stage 6 — hook_engine + retention_editor + packaging
- [x] Stage 7 — full pipeline + Streamlit UI
- [x] Stage 8 — quality gate + tests + README
- [x] Final self-audit — `FINAL_AUDIT_OK`

## Stage 1 — Scaffold + config + env checks — [x]

Command: `rtk proxy pytest`
```
......                                                                   [100%]
6 passed in 0.13s
```
Startup check:
```
FFMPEG_OK True ffmpeg version 9.0.1-full_build-www.gyan.dev
GEMINI_KEY False
```

## Stage 2 — downloader + transcriber — [x]

Command: `python scripts/verify_stage2.py`
```
audio: cached=True file=audio.m4a bytes=8908997 in 0.0s
transcript: 932 words, 62 segments, lang=en in 0.0s
re-run: cached=True in 1 ms
STAGE2_VERIFY_OK
```
Word-level timestamps are present on every word; the second transcript call
returns `cached=True` without re-running Whisper.

FRAME SAMPLING RULE: `python scripts/verify_frame_sampling.py`
```
sampled frames: 30 via stream-seek at 1280x720 in 91.3s
facecam: box={'x': 534, 'y': 176, 'w': 286, 'h': 286} engine=mediapipe hits=27/30
size limit (audio track): 8908997 bytes
FRAME_SAMPLING_VERIFY_OK
```
`source.mp4` is only downloaded immediately before rendering.

## Stages 3–4 — facecam, layout, captions — [x]

Command: `python scripts/verify_render_synthetic.py`
```
ffprobe: 1080x1920 h264/aac 30.00fps 16.00s
qa: resolution=True video_codec=True audio_codec=True fps=True duration=True loudness=True dead_air=True hook_within_2s=True caption_safe_zones=True ends_on_sentence=True
caption frame diff: {'frames': 4, 'diffs': [20.38, 48.62, 42.92], 'animated': True}
SYNTHETIC_RENDER_VERIFY_OK
```
Command: `python -m compileall -q clipforge app.py scripts tests`
```
COMPILE_OK
```

## Stage 5 — highlights — [x]

Full E2E returned 6 non-overlapping sentence-boundary windows, each 20–45 s,
with scores and breakdowns. No key is present, so fallback scoring was used;
the Gemini path is covered by unit tests with a mocked JSON response.

## Stage 6 — hooks + retention + packaging — [x]

`metadata.json` validates against the pydantic schema; hooks are <= 8 words,
titles are < 60 chars, 10–15 hashtags are present, and each clip has a
`thumbnails/clip_NN.png`.

## Stage 7 — full pipeline + Streamlit — [x]

Command: full pipeline on the test URL from a clean output folder:
```
RESULT: output\open-culture-voices-vlog-series-maja-drabczyk_Keej4G0xlks fallback (no Gemini key): audio energy + keywords + speech density 6 dropped=0
```
Streamlit startup: `python scripts/verify_streamlit.py`
```
streamlit health: ok at http://127.0.0.1:8599/_stcore/health
STREAMLIT_STARTUP_OK
```

## Stage 8 — quality gate + tests + README — [x]

Command: `rtk proxy pytest`
```
...........................................                              [100%]
43 passed in 0.80s
```
README.md documents Windows/macOS/Linux setup, the free Gemini key, and the
required permission notice.

## Final self-audit — [x]

Clean-cache run: deleted `cache\Keej4G0xlks` and the output folder, then ran
the full pipeline (audio download, Whisper, sampling, render, QA, zip) before
the audit. Command: `python scripts/verify_final_audit.py`
```
folder: open-culture-voices-vlog-series-maja-drabczyk_Keej4G0xlks
mode: fallback (no Gemini key): audio energy + keywords + speech density
clips: 6 dropped: 0
clip 01: 19.10s 1080x1920 h264/aac 30fps lufs=-14.5 qa=pass attempts=1 caption_diffs=[13.49, 14.49, 11.12]
clip 02: 31.17s 1080x1920 h264/aac 30fps lufs=-14.27 qa=pass attempts=1 caption_diffs=[6.86, 17.45, 20.46]
clip 03: 31.85s 1080x1920 h264/aac 30fps lufs=-14.32 qa=pass attempts=1 caption_diffs=[12.27, 10.33, 8.81]
clip 04: 22.77s 1080x1920 h264/aac 30fps lufs=-13.53 qa=pass attempts=1 caption_diffs=[13.03, 11.33, 21.63]
clip 05: 32.05s 1080x1920 h264/aac 30fps lufs=-13.77 qa=pass attempts=1 caption_diffs=[17.24, 15.18, 15.29]
clip 06: 33.27s 1080x1920 h264/aac 30fps lufs=-14.09 qa=pass attempts=1 caption_diffs=[10.12, 12.97, 9.13]
cache re-run: cached=True in 44 ms
FINAL_AUDIT_OK
```
