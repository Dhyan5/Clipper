"""Final self-audit for the generated ClipForge output."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import OUTPUT_DIR, Settings, video_id_from_url
from clipforge.packaging import VideoMeta
from clipforge.pipeline import process
from clipforge.qa import caption_frame_diff
from clipforge.utils import ffprobe, read_json

URL = "https://www.youtube.com/watch?v=Keej4G0xlks"


def main() -> None:
    vid = video_id_from_url(URL)
    folders = list(OUTPUT_DIR.glob(f"*_{vid}"))
    assert folders, f"no output folder for {vid}"
    folder = folders[0]
    meta = VideoMeta.model_validate(read_json(folder / "metadata.json"))
    assert meta.video_id == vid
    assert len(meta.clips) >= 3, len(meta.clips)
    print(f"folder: {folder.name}")
    print(f"mode: {meta.mode}")
    print(f"clips: {len(meta.clips)} dropped: {len(meta.dropped)}")
    assert not meta.dropped, meta.dropped
    assert (folder / "clips.zip").stat().st_size > 0

    for clip in meta.clips:
        path = folder / clip.filename
        assert path.exists() and path.stat().st_size > 0, path
        info = ffprobe(path)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        fps = video["avg_frame_rate"].split("/")
        fps = float(fps[0]) / float(fps[1])
        duration = float(info["format"]["duration"])
        assert (video["width"], video["height"]) == (1080, 1920)
        assert video["codec_name"] == "h264"
        assert audio["codec_name"] == "aac"
        assert abs(fps - 30.0) < 0.1
        assert 15.0 <= duration <= 60.0, duration
        assert clip.qa.passed, clip.qa
        assert all(c.passed for c in clip.qa.checks), clip.qa.checks
        assert clip.qa.attempts <= 3
        assert len(clip.hook.split()) <= 8, clip.hook
        assert all(len(t) < 60 for t in clip.title_options)
        assert 10 <= len(clip.hashtags) <= 15, len(clip.hashtags)
        assert (folder / clip.thumbnail).exists(), clip.thumbnail
        diff = caption_frame_diff(path, times=(2.5, 3.5, 4.5, 5.5))
        assert diff["frames"] >= 3 and diff["animated"], diff
        lufs = clip.qa.metrics.get("lufs")
        assert lufs is not None and abs(float(lufs) + 14.0) <= 1.5, lufs
        print(
            f"clip {clip.index:02d}: {duration:.2f}s {video['width']}x{video['height']} "
            f"{video['codec_name']}/{audio['codec_name']} {fps:.0f}fps "
            f"lufs={lufs} qa=pass attempts={clip.qa.attempts} "
            f"caption_diffs={diff['diffs']}"
        )

    t0 = time.perf_counter()
    cached = process(URL, Settings(), progress=None)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert cached["cached"] is True
    assert elapsed_ms < 3000, elapsed_ms
    print(f"cache re-run: cached=True in {elapsed_ms:.0f} ms")
    print("FINAL_AUDIT_OK")


if __name__ == "__main__":
    main()
