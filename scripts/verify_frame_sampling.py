"""Verify FRAME SAMPLING RULE: no full video before the render stage."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.checks import assert_no_file_larger_than_audio, largest_files
from clipforge.config import CACHE_DIR, OUTPUT_DIR, video_id_from_url
from clipforge.downloader import (download_audio, sample_face_frames,
                                  _timestamps)
from clipforge.facecam import detect_facecam_remote

URL = "https://www.youtube.com/watch?v=Keej4G0xlks"


def main() -> None:
    vid = video_id_from_url(URL)
    cache = CACHE_DIR / vid
    cached_video = cache / "source.mp4"
    if cached_video.exists():
        cached_video.unlink()
        print(f"removed stale full video {cached_video}")
    for mp4 in OUTPUT_DIR.glob(f"*_{vid}/*.mp4"):
        print(f"NOTE existing rendered clip present: {mp4}")

    t0 = time.perf_counter()
    audio = download_audio(URL)
    print(f"audio: {audio['path'].name} {audio['path'].stat().st_size} bytes "
          f"(cached={audio['cached']}) in {time.perf_counter() - t0:.1f}s")
    duration = float(audio["info"].get("duration") or 0)
    ts = _timestamps(duration, 30)
    print(f"timestamps: {len(ts)} spread {ts[0]}..{ts[-1]} over {duration}s")

    t1 = time.perf_counter()
    sample = sample_face_frames(URL, duration, count=30)
    print(f"sampled frames: {len(sample['frames'])} via {sample['method']} "
          f"at {sample['width']}x{sample['height']} in {time.perf_counter() - t1:.1f}s")
    leftover = list(cache.glob("frames_*"))
    assert not leftover, f"sample temp dir leaked: {leftover}"

    facecam = detect_facecam_remote(URL, vid, duration, progress=print)
    print(f"facecam: box={facecam.get('box')} engine={facecam.get('engine')} "
          f"hits={facecam.get('hits')}/{facecam.get('samples')}")
    assert (cache / "facecam_v1.json").exists(), "facecam cache missing"
    assert not cached_video.exists(), "full video exists before rendering"

    report = assert_no_file_larger_than_audio(ROOT, cache)
    print(f"size limit (audio track): {report['limit']} bytes")
    print("largest project files:")
    for path, size in report["largest"]:
        print(f"  {size:>12} {path.relative_to(ROOT)}")
    if report["violations"]:
        for path, size in report["violations"]:
            print(f"  VIOLATION {size} {path}")
        raise AssertionError("file larger than audio track exists before rendering")
    print("FRAME_SAMPLING_VERIFY_OK")


if __name__ == "__main__":
    main()
