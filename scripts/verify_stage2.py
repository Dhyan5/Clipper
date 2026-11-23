"""Stage 2 verify (frame-sampling rule): audio-only download + transcript + cache."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clipforge.config import video_id_from_url
from clipforge.downloader import download_audio
from clipforge.transcriber import transcribe, transcript_path

URL = "https://www.youtube.com/watch?v=Keej4G0xlks"

t0 = time.perf_counter()
dl = download_audio(URL)
source = dl["path"].parent / "source.mp4"
source_before = (source.stat().st_mtime_ns, source.stat().st_size) if source.exists() else None
t1 = time.perf_counter()
print(f"audio: cached={dl['cached']} file={dl['path'].name} "
      f"bytes={dl['path'].stat().st_size} in {t1-t0:.1f}s")

tr = transcribe(dl["path"], dl["video_id"], "small")
t2 = time.perf_counter()
print(f"transcript: {tr['word_count']} words, {len(tr['segments'])} segments, "
      f"lang={tr['language']} in {t2-t1:.1f}s")

t3 = time.perf_counter()
tr2 = transcribe(dl["path"], dl["video_id"], "small")
t4 = time.perf_counter()
print(f"re-run: cached={tr2.get('cached')} in {(t4-t3)*1000:.0f} ms")

path = transcript_path(video_id_from_url(URL), "small")
assert path.exists(), path
assert tr["word_count"] > 0
assert all("start" in w and "end" in w for s in tr["segments"] for w in s["words"])
assert tr2.get("cached") is True
source_after = (source.stat().st_mtime_ns, source.stat().st_size) if source.exists() else None
assert source_after == source_before, "full video downloaded too early"
print("STAGE2_VERIFY_OK")
