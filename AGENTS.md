# ClipForge — Agent Spec (verbatim) + Working Notes

You are a senior Python engineer working FULLY AUTONOMOUSLY. I am busy and will
not answer questions or say "continue". Do not stop to ask, do not wait for
approval, do not end your turn until the DEFINITION OF DONE is met or you hit
a genuine blocker (see BLOCKERS). Make decisions yourself and log them in
DECISIONS.md.

=====================================================================
PROJECT: "ClipForge"
=====================================================================
A free, local tool with a Streamlit UI. I paste ONE YouTube link, it processes
that ONE video, and produces several viral, attention-grabbing vertical clips
(1080x1920), each saved as its own separate MP4. One video at a time.

TEST VIDEO for all self-tests: <TEST_URL>
Gemini key is in .env as GEMINI_API_KEY (also let the UI accept a pasted key
in a password field and save it to .env).

=====================================================================
AUTONOMY PROTOCOL (follow strictly)
=====================================================================
1. First write this entire spec to AGENTS.md, then create PROGRESS.md
   (checklist of stages with [ ] / [x]) and DECISIONS.md.
2. Work through the stages in order WITHOUT pausing. After finishing each
   stage, run its VERIFY step. A stage is only [x] when its verify command
   actually ran and passed. Paste the real output into PROGRESS.md.
3. If a verify step fails: read the error, fix the root cause, re-run. Try up
   to 5 different fixes per problem. Do not repeat the same failed fix. If
   still failing, choose the simplest fallback, note it in DECISIONS.md, and
   keep moving.
4. Never fake results. Never write placeholders, mocks standing in for real
   features, or "TODO". Never claim success without command output.
5. Make minimal edits to working files. Run `pytest` after every stage; do not
   advance with failing tests.
6. If your context resets, re-read AGENTS.md, PROGRESS.md, DECISIONS.md and
   resume at the first unchecked stage.
7. Install missing Python packages yourself with pip. If FFmpeg is missing and
   cannot be installed automatically, that is a BLOCKER.

=====================================================================
CONSTRAINTS
=====================================================================
- Python 3.11+, free tools only: yt-dlp, FFmpeg, faster-whisper, OpenCV,
  MediaPipe, librosa, pydantic, Streamlit, pytest.
- LLM: Gemini API only (default model gemini-2.5-flash, configurable). Force
  JSON output, validate with pydantic, retry with exponential backoff on
  rate limits. Send the transcript in FEW large batched calls, never one call
  per tiny window. Prompts live in /prompts/*.txt.
- If Gemini is unavailable or the key is missing, the pipeline must degrade
  gracefully to a non-LLM mode (audio energy + keyword + speech density
  scoring, template hooks/titles) and say so clearly in the UI. It must
  never crash.
- GPU auto-detect (NVENC/CUDA), else CPU (libx264 -crf 20, whisper small int8).
- Cache every stage keyed by video_id; re-running the same link repeats nothing.
- Check FFmpeg at startup with install instructions if missing.

=====================================================================
USER FLOW
=====================================================================
Streamlit: link box, Gemini key field, settings (clip count default 6, min/max
length 20-45s, caption style, cold-open toggle) -> "Generate clips" -> live
progress with stage names -> grid of clip players with title, hook, score,
per-clip download, "download all as zip", and per-clip start/end edit +
re-render.

STORAGE: output/<video_title_slug>_<video_id>/
    clip_01_<title-slug>.mp4, clip_02_...mp4  (each clip a separate file)
    thumbnails/clip_NN.png
    metadata.json (title options, chosen title, hook, score + breakdown,
                   start/end, description, hashtags, QA report)
    source/ (temp segments, deleted after render unless "keep source")

=====================================================================
LAYOUT (1080x1920, H.264/AAC, 30fps, audio -14 LUFS)
=====================================================================
Top half 1080x960 = facecam (cropped, scaled to fill). Bottom half 1080x960 =
gameplay (center-cropped, excluding the facecam area where possible). Captions
at the seam. Thin progress bar along the seam. No facecam found: gameplay-only
over a blurred background fill. Facecam found via MediaPipe on ~30 sampled
frames, stable box, cached, with manual override.

=====================================================================
DYNAMIC CAPTIONS (most important, make them very eye-catching)
=====================================================================
.ass from Whisper word timestamps, burned in via FFmpeg:
- 1-3 words on screen at a time, huge bold font (bundle Anton or Montserrat
  ExtraBold, OFL), thick black outline + shadow.
- Each word POPS: scale 70% -> 115% -> 100% over ~120ms.
- Active word in a bright color; cycle accents (yellow, green, cyan, pink).
- Auto-emphasis: numbers, names, loud/shouted words, emotional words get 130%
  size, accent color, short shake (use librosa loud peaks + Gemini keywords).
- Optional emoji after emotional words; profanity -> ****.
- Safe zones: avoid bottom 15%, right 12%, top 8%.
- 5 presets: Hormozi (default), MrBeast, Gaming Neon, Clean Minimal, Classic.

=====================================================================
VIRAL ENGINE
=====================================================================
- Candidate windows from sentence boundaries only, never mid-sentence.
- Gemini rubric per window (0-10): hook_strength, emotional_peak, payoff,
  self_contained, shareability, curiosity_gap, quotability, + reasons,
  best_start_offset_sec, best_end_offset_sec.
- Score = 0.60 Gemini + 0.25 audio energy + 0.15 speech density/keywords
  (add chat-spike only if live chat replay exists).
- Reject silence, lobby/menu time, clips needing prior context. Start on the
  strongest moment, end right after the payoff. NMS: no overlaps, keep top N.
- Hooks: 3 variants (curiosity / shock / relatable), max 8 words, must pay off,
  Gemini picks best; shown as animated on-screen text in the first 2.5s.
  Optional cold open (1-2s payoff teaser, then cut back). Trim leading filler.
- Retention: cut silences > 0.35s (keep 0.1s breath, audio crossfades), remove
  isolated filler words, eased punch-in zooms 105-115% (max 1 per ~3s), a
  visual change every 4-6s.
- Packaging: 3 title options (<60 chars), description, 10-15 hashtags,
  thumbnail from the best face-expression frame.

=====================================================================
STAGES (each ends with VERIFY; run it, don't skip)
=====================================================================
1 Scaffold + config + env checks. VERIFY: `pytest` runs, startup check passes.
2 downloader + transcriber. VERIFY: process <TEST_URL>; assert transcript JSON
  exists with word-level timestamps and >0 words; assert cache hit on re-run.
3 facecam_detector + layout_renderer. VERIFY: render one clip; ffprobe asserts
  1080x1920, h264, aac, 30fps; extract a frame and check with OpenCV that the
  top half contains a detected face (or fallback layout was used).
4 captions. VERIFY: generate .ass, render, extract 3 frames at different times,
  confirm via pixel diff that caption pixels differ between frames and fall
  inside safe zones.
5 Gemini client + highlight_detector. VERIFY: returns N non-overlapping
  windows with scores, all start/end on sentence boundaries, durations within
  limits. Also verify the no-key fallback mode works.
6 hook_engine + retention_editor + packaging. VERIFY: metadata.json validates
  against a pydantic schema; hooks <= 8 words; thumbnail PNG exists.
7 pipeline + Streamlit UI + per-video folders + zip + single-clip re-render.
  VERIFY: run the full pipeline headlessly on <TEST_URL>; assert N separate
  clip files exist in the correct folder; run Streamlit headless and confirm
  it starts without errors.
8 Quality gate + tests + README. Gate checks every clip and auto re-renders
  (max 2 tries) or drops it if: hook text appears after 2s, >0.8s dead air,
  captions overlap unsafe zones, loudness outside -14 LUFS +/-1.5 (measure
  with ffmpeg loudnorm), or the clip ends mid-sentence. QA report goes into
  metadata.json. README covers Windows/macOS/Linux setup and getting a free
  Gemini key, plus the notice: "Only process content you own or have
  permission to clip."

=====================================================================
FINAL SELF-AUDIT (mandatory before finishing)
=====================================================================
Run a final end-to-end test on <TEST_URL> from a clean cache and confirm:
[ ] at least 3 separate clip MP4s exist, each 1080x1920 with audio
[ ] every clip is 15-60s and passes the quality gate
[ ] captions are visibly animated (frame diff check)
[ ] metadata.json is complete and valid
[ ] `pytest` is fully green
[ ] app starts with `streamlit run app.py`
[ ] re-running the same link is instant (cache)
Write FINAL_REPORT.md: what works, what fell back and why, known limits,
exact run commands. Fix every failed item and repeat the audit until all pass.

=====================================================================
BLOCKERS (the ONLY reasons to stop and ask me)
=====================================================================
FFmpeg or Python cannot be installed, the test URL is unreachable/private, or
the Gemini key is rejected (then continue in non-LLM fallback mode and tell me).

=====================================================================
START NOW
=====================================================================
Begin with AGENTS.md, PROGRESS.md, DECISIONS.md, then Stage 1. Keep going
through all 8 stages and the final audit without stopping.

=====================================================================
WORKING NOTES (added by agent)
=====================================================================
- The literal token `<TEST_URL>` was supplied, not a URL. Resolved test URL:
  https://www.youtube.com/watch?v=Keej4G0xlks (Open Culture Voices Vlog
  Series: Maja Drabczyk, official Creative Commons channel, CC-BY, 551 s,
  dense talking-head speech + face). Logged in DECISIONS.md.
- No `.env` or GEMINI_API_KEY was present at start; non-LLM fallback is the
  default path until a key is saved through the UI/.env.
- Python 3.12.10, FFmpeg 9.0.1, yt-dlp, faster-whisper, OpenCV, MediaPipe,
  librosa, Streamlit installed. GPU: RTX 3050 (NVENC available).

- USER OVERRIDE: no video downloading/clipping while building. All cached
  media and outputs were deleted. Tests cover modules without processing a
  source video.
- FRAME SAMPLING RULE: the full video must never be downloaded for facecam
  detection. Detection samples ~30 360p stills from `yt-dlp -g` stream URLs
  (FFmpeg `-ss` seeks) with a `--download-sections` fallback; temp frames are
  deleted and only the facecam box is cached. `download_video()` is called only
  immediately before clip rendering. Verify:
  `python scripts/verify_frame_sampling.py` asserts no project file is larger
  than the audio track before render. PASSED on 2026-11-18 (30/30 frames,
  engine=mediapipe, 27/30 face hits, largest file = audio.m4a 8,908,997 B).
- Audio-only cache re-run is instant: `download_audio()` returned
  `cached=True` in ~16 ms for the same link.
- Synthetic render verify (no YouTube clip): `python scripts/verify_render_synthetic.py`
  passed on 2026-11-18. It builds a local 16 s testsrc+sine source, renders a
  1080x1920 h264/aac 30 fps clip through the real layout/caption/QA code,
  confirms all QA checks pass, and confirms animated caption pixels stay
  inside safe zones. FFmpeg 9 here has no `-filter_complex_script`, so the
  filter graph is passed inline via `-filter_complex`; NVENC is listed but the
  installed driver is too old, so render_clip retries with libx264.
