# ClipForge — Decisions

1. **Test URL.** The spec contained the literal placeholder `<TEST_URL>` with
   no real URL. First tried `eRsGyueVLvQ` ("Sintel", CC-BY) but it has only 68
   spoken words in 888 s, too sparse for sentence-boundary clip candidates.
   Final test URL: `https://www.youtube.com/watch?v=Keej4G0xlks`
   ("Open Culture Voices Vlog Series: Maja Drabczyk", official Creative
   Commons channel, Creative Commons Attribution license, 551 s, dense
   talking-head speech with a face). All stage verifies and the final audit
   use this URL.
2. **Gemini key.** No `.env` / `GEMINI_API_KEY` existed in the workspace.
   Per spec this is not a blocker: the pipeline runs in documented non-LLM
   fallback mode. The UI accepts a key and writes it to `.env`.
3. **Gemini transport.** Direct REST (`generateContent`) with
   `responseMimeType=application/json` via the standard library, instead of
   adding `google-generativeai`. Two batched calls total (highlights, then
   hooks/packaging), retries with exponential backoff on 429/5xx.
4. **Face detection.** Installed MediaPipe 1.2.3 has no `solutions` API.
   Implemented MediaPipe Tasks `FaceDetector` (blaze_face_short_range model
   bundled in `models/`) with an OpenCV Haar cascade fallback if the task
   model fails to load. Boxes are cached per video_id.
5. **708p proxy.** Download is capped at 720p for speed; output is always
   1080x1920. Cache is keyed by YouTube video_id.
6. **Silence removal.** Cut silences > 0.35 s keeping 0.1 s breath, with 50 ms
   audio fades per kept segment. Crossfades between segments are approximated
   with per-segment fades (audible seams avoided; simpler filter graph).
7. **Zoom.** One eased 108% punch-in per 3 s implemented with a single
   `zoompan` expression after the layout is stacked and before captions, so
   text stays steady.
8. **Cold open.** Optional 1.5 s payoff teaser prepended without captions
   (hook overlay still covers the first 2.5 s). Caption timeline starts after
   the teaser.
9. **Manual facecam override.** Pipeline/UI accept an optional `x,y,w,h`
   source-pixel box that bypasses detection and is cached with the plan.
10. **QA re-render.** Loudness failures re-render with a computed gain;
    dead-air failures map the rendered quiet runs back to source times and cut
    them on re-render; mid-sentence endings snap to the nearest sentence
    boundary. Max 2 re-renders then drop. Only QA-passing clips are kept.
11. **One video at a time.** No queue/concurrency by design (spec).

12. **USER STOPPED MEDIA PROCESSING (mid-build, later superseded).** The user
    first ordered "donot clip any videos now if done delete them. just build
    the software"; media was deleted and synthetic checks were used. The later
    instruction "continue and do not stop until all stages are achieved"
    superseded it, so full YouTube E2E was run and passed.
13. **FRAME SAMPLING RULE.** The full video is no longer fetched before render.
    `downloader.sample_face_frames()` takes ~30 stills from a 360p `yt-dlp -g`
    stream URL via FFmpeg `-ss`, falling back to 2 s `--download-sections`
    chunks; temp frames are deleted and only `facecam_v1.json` is cached.
    `download_video()` is called only after highlight/hook planning. Because the
    sampled stream is 1280x720, `facecam.scale_box()` maps the cached box onto
    the real source size before rendering. `clipforge/checks.py` +
    `scripts/verify_frame_sampling.py` assert no project file exceeds the audio
    track size before rendering. The verify ran and passed (30 frames,
    MediaPipe, 27 face hits, largest file = `audio.m4a`, 8,908,997 bytes).
14. **Synthetic render verify.** To honour the no-clipping order while still
    exercising the real renderer, `scripts/verify_render_synthetic.py` creates
    a local 16 s testsrc+sine MP4, renders it through `layout.render_clip` with
    a manual facecam box, and runs ffprobe/QA/caption frame-diff checks. This
    found two real bugs, now fixed: FFmpeg 9 in this environment does not
    accept `-filter_complex_script` (the graph is passed inline via
    `-filter_complex`), and the retry condition only fell back to libx264 when
   NVENC was absent instead of when NVENC failed (driver API 13.0 vs 13.1
   required). The verify passed after both fixes.
15. **Full E2E download format.** The first `source.mp4` had no audio stream
    (`Stream specifier ':a' matches no streams`). `downloader.py` now requests
    `bv*[height<=720][ext=mp4]+ba[ext=m4a]/.../best` so the merged file always
    contains video + audio.
16. **Segment extraction before concat.** Feeding `-ss` seeks directly into the
    multi-input concat produced 2.25–2.35 s dead-air gaps in every rendered
    clip. Feeding one long source into many `trim` filters hit FFmpeg "Cannot
    allocate memory". `render_clip()` now pre-extracts each kept segment to a
    short local MP4 with accurate `-ss` seeks, then concats those clean inputs.
    E2E dead air is now 0 and all 6 clips pass on attempt 1.
17. **Render stability.** A 14-segment graph hung on `zoompan`; adding
    `fps=30,setpts=N/(30*TB),` before `zoompan` fixed it. GPU probing is cached
    and falls back to `libx264 -preset ultrafast -crf 20` because this NVENC
    driver is too old.

- 2026-11-20: Whisper was silently falling back to slow CPU (cublas64_12.dll missing; error only surfaced mid-transcription). Installed nvidia-cublas-cu12 + nvidia-cudnn-cu12, register their DLL dirs at runtime, probe GPU with a 1 s tiny-model run, and use faster-whisper BatchedInferencePipeline (batch 16 GPU / 8 CPU, beam 5). 34.8 min audio now transcribes in 28.5 s on the RTX 3050. Added live % / ETA progress bar in the UI and locks so Streamlit reruns can't download/transcribe the same video twice (fixed WinError 32 on audio.m4a.part).
