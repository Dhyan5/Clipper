# ClipForge — Final Report

Date: 2026-11-18.

## What works (verified here)

- Full clean-cache E2E on `https://www.youtube.com/watch?v=Keej4G0xlks`:
  audio download, word-level Whisper transcript (932 words / 62 segments),
  facecam sampling, highlight selection, hooks/packaging, rendering, QA and zip
  all completed. Output: `output\open-culture-voices-vlog-series-maja-drabczyk_Keej4G0xlks`.
- 6 separate clip MP4s, 0 dropped. All are 1080x1920, h264/aac, 30 fps,
  15–60 s, within -14 LUFS ±1.5, QA-passing, with animated captions and
  thumbnails. `metadata.json` validates against the pydantic schema.
- Final audit: `python scripts/verify_final_audit.py` → `FINAL_AUDIT_OK`;
  cache re-run returned `cached=True in 44 ms`.
- Unit suite: `rtk proxy pytest` → `43 passed in 0.80s`.
- Compile check: `python -m compileall -q clipforge app.py scripts tests` →
  `COMPILE_OK`.
- Streamlit headless: `python scripts/verify_streamlit.py` →
  `STREAMLIT_STARTUP_OK`.
- Frame-sampling rule: `python scripts/verify_frame_sampling.py` →
  `FRAME_SAMPLING_VERIFY_OK` (30 stills, no full video before render).
- Synthetic render: `python scripts/verify_render_synthetic.py` →
  `SYNTHETIC_RENDER_VERIFY_OK`.

## What fell back and why

- **Gemini:** no `GEMINI_API_KEY` exists here, so mode was
  `fallback (no Gemini key): audio energy + keywords + speech density`. The UI
  labels this clearly; the Gemini path is covered by mocked JSON unit tests.
- **GPU:** NVENC is listed but this driver is too old, so rendering falls back
  to `libx264 -preset ultrafast -crf 20`.
- **FFmpeg 9:** does not accept `-filter_complex_script`; the graph is passed
  inline via `-filter_complex`.
- **OpenCV 5 headless:** lacks Haar cascade XML; `facecam.py` returns no Haar
  boxes and `packaging.py` uses Laplacian variance for thumbnail selection when
  the cascade is missing.
- **pyav 19:** faster-whisper is fed a decoded 16 kHz numpy array from a WAV.

## Known limits

- Gemini scoring/hooks were not exercised against the live API because no key
  was available; adding a key enables the batched LLM path.
- YouTube rate limits, private/cookie-gated videos and expiring stream URLs can
  break downloads; re-running re-resolves them.
- Clips are capped at 720p source input and upscaled by the 1080x1920 layout.
- Only process content you own or have permission to clip.

## Final self-audit checklist

- [x] At least 3 separate clip MP4s exist, each 1080x1920 with audio — 6 clips.
- [x] Every clip is 15–60 s and passes the quality gate — all 6 passed on
      attempt 1.
- [x] Captions visibly animated — frame diffs per clip are all > 6.8.
- [x] metadata.json is complete and valid — pydantic schema + final audit.
- [x] `pytest` is fully green — 43 passed.
- [x] App starts with `streamlit run app.py` — headless health check passed.
- [x] Re-running the same link is instant — `cached=True in 44 ms`.

## Exact run commands

From `C:\Users\Dhyan\OneDrive\Desktop\clipper`:

```bash
rtk proxy pytest
python -m compileall -q clipforge app.py scripts tests
python scripts/verify_frame_sampling.py
python scripts/verify_render_synthetic.py
python scripts/verify_streamlit.py
streamlit run app.py
python scripts/verify_stage2.py
python scripts/verify_final_audit.py
```

Clean-cache E2E (what was run before the final audit):

```bash
rmdir /s /q cache\Keej4G0xlks
rmdir /s /q output\open-culture-voices-vlog-series-maja-drabczyk_Keej4G0xlks
python -c "from clipforge.pipeline import process; from clipforge.config import Settings; process('https://www.youtube.com/watch?v=Keej4G0xlks', Settings())"
python scripts/verify_final_audit.py
```
