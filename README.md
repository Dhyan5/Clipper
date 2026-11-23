# ClipForge

Local Streamlit app: paste one YouTube link, get several viral vertical clips
(1080x1920 H.264/AAC, 30 fps, -14 LUFS) with animated burned-in captions. One
video at a time. All processing is local except optional Gemini scoring.

> Only process content you own or have permission to clip.

## Requirements

- Python 3.11+
- FFmpeg + ffprobe on `PATH`
- Optional: free Gemini API key for LLM highlight scoring, hooks and packaging

FFmpeg install:

```bash
# Windows
winget install Gyan.FFmpeg
# macOS
brew install ffmpeg
# Linux (Debian/Ubuntu)
sudo apt install ffmpeg
```

Verify: `ffmpeg -version && ffprobe -version`.

## Install

```bash
# Windows
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The bundled Anton font (`fonts/Anton-Regular.ttf`) and MediaPipe face model
(`models/blaze_face_short_range.tflite`) are already in the repo.

## Gemini key (optional)

1. Create a free key in Google AI Studio: <https://aistudio.google.com/app/apikey>
2. Paste it into the password field in the UI (saved to `.env`), or create
   `.env` in this folder:

```env
GEMINI_API_KEY=your_key_here
```

Default model: `gemini-2.5-flash`; change it in the sidebar. With no key,
ClipForge runs in non-LLM mode (audio energy + keywords + speech density,
template hooks/titles) and the UI says so. It never crashes because of a
missing key.

## Run

```bash
streamlit run app.py
```

Then open the printed local URL. Headless/server check:

```bash
streamlit run app.py --server.headless true --server.port 8599
```

## What the UI does

- Link box, optional key field, clip count (3–10, default 6), min/max length
  (default 20–45 s), caption preset, cold-open toggle, Whisper model,
  facecam box override.
- Live stage progress while generating.
- Clip grid with player, chosen title, hook, score, timing, QA status,
  per-clip MP4 download, timing edit + re-render, and "Download all as ZIP".
- A warning is shown when non-LLM mode is active.

## Output

```text
output/<video_title_slug>_<video_id>/
  clip_01_<title-slug>.mp4
  clip_02_<title-slug>.mp4
  ...
  thumbnails/clip_NN.png
  metadata.json
  clips.zip
  source/            # temporary segments, removed unless "keep source" is on
```

Caches live in `cache/<video_id>/` (`audio.*`, `audio16k.wav`,
`transcript_*_v1.json`, `facecam_v1.json`, `info.json`, and `source.mp4` only
once rendering starts). Re-running the same link with the same settings is
instant.

## Frame-sampling rule

The full video is **never** downloaded for facecam detection. `downloader.py`
takes ~30 stills from a 360p `yt-dlp -g` stream URL using FFmpeg `-ss`, with a
short `--download-sections` fallback. Frames are deleted; only the stable
facecam box is cached. `source.mp4` is fetched only immediately before clip
rendering. The verify script asserts that nothing larger than the audio track
is created before that point:

```bash
python scripts/verify_frame_sampling.py
```

## Pipeline (short version)

1. Audio-only download, Whisper word-level transcript, audio envelope.
2. Facecam detected from sampled stills (MediaPipe Tasks / Haar fallback) or
   manual `x,y,w,h` override; box cached and rescaled to the real source.
3. Sentence-boundary candidate windows scored with Gemini rubric or the
   no-key heuristic; NMS removes overlaps.
4. Three hooks per clip, 3 titles, description, 10–15 hashtags.
5. Silence/filler trimming, cold open, zoom, 1080x1920 stacked layout, .ass
   captions burned in, loudnorm to -14 LUFS (NVENC when available, else x264).
6. QA checks hook timing, dead air, caption safe zones, loudness and
   sentence endings; up to 2 re-renders, then drop the clip.

## Verification

```bash
pytest                              # or: rtk proxy pytest in this workspace
python -m compileall -q clipforge app.py scripts tests
python scripts/verify_streamlit.py  # starts Streamlit headless, then stops it
python scripts/verify_frame_sampling.py
python scripts/verify_render_synthetic.py
python scripts/verify_final_audit.py
```

`scripts/verify_stage2.py` runs audio-only download + a real Whisper
transcript + cache check when media processing is allowed.

## Notes / limits

- YouTube availability, rate limits and cookie-gated/private videos can break
  downloads; use content you can access and are allowed to clip.
- MediaPipe's installed version has no `solutions` API, so the Tasks
  `FaceDetector` model is used with a Haar fallback.
- pyav 19 is incompatible with faster-whisper file input; transcription
  extracts a 16 kHz WAV with FFmpeg and feeds a numpy array.
- Live clip-level E2E was deferred initially by user instruction, then run and
  passed after the instruction was lifted; see FINAL_REPORT.md.
