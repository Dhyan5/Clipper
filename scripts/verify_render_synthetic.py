"""Render a local synthetic clip to verify layout, captions and QA without YouTube."""
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.captions import SAFE_BOTTOM, SAFE_RIGHT, SAFE_TOP, build_ass
from clipforge.layout import render_clip
from clipforge.qa import caption_frame_diff, check_clip
from clipforge.utils import ffprobe, run

DURATION = 16.0
WORDS = ["THIS", "IS", "HUGE", "YOU", "NEED", "TO", "SEE", "THE", "SECRET", "NOW"]


def _synthetic_source(path: Path) -> None:
    run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
         "-t", str(DURATION), "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        timeout=300,
    )


def _transcript() -> dict:
    words = []
    for i in range(32):
        start = i * 0.5
        words.append({"word": WORDS[i % len(WORDS)], "start": start,
                      "end": round(start + 0.42, 3), "score": 0.9})
    text = " ".join(w["word"] for w in words)
    return {"segments": [{"start": 0.0, "end": DURATION, "text": text, "words": words}]}


def _frames(path: Path, times: tuple[float, ...]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            out.append(frame)
    cap.release()
    return out


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="clipforge_render_verify_"))
    try:
        src = tmp_root / "source.mp4"
        work = tmp_root / "work"
        out = tmp_root / "clip.mp4"
        plain_out = tmp_root / "no_captions.mp4"
        print(f"generating synthetic source: {src}")
        _synthetic_source(src)
        tr = _transcript()
        keep = [(0.0, DURATION)]
        box = {"x": 120, "y": 80, "w": 320, "h": 320}

        ass = build_ass(
            tr, keep, work / "captions.ass", hook_text="THIS IS HUGE",
            preset_name="Hormozi", duration=DURATION,
            emphasis_words=["huge"], loud_times=[1.0, 4.0, 8.0],
        )
        print(f"ass events: {ass['caption_events']} captions over {ass['groups']} groups")
        assert ass["caption_events"] > 10

        render_clip(src, keep, DURATION, box, 1280, 720,
                    work / "captions.ass", out, work / "render")
        empty_ass = build_ass({"segments": []}, keep, work / "empty.ass",
                              duration=DURATION)
        render_clip(src, keep, DURATION, box, 1280, 720,
                    empty_ass["path"], plain_out, work / "plain")

        info = ffprobe(out)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        fps = video["avg_frame_rate"].split("/")
        fps_val = float(fps[0]) / float(fps[1])
        print(f"ffprobe: {video['width']}x{video['height']} "
              f"{video['codec_name']}/{audio['codec_name']} {fps_val:.2f}fps "
              f"{float(info['format']['duration']):.2f}s")
        assert (video["width"], video["height"]) == (1080, 1920)
        assert video["codec_name"] == "h264"
        assert audio["codec_name"] == "aac"
        assert abs(fps_val - 30.0) < 0.1

        qa = check_clip(out, work / "captions.ass", [DURATION], DURATION)
        print("qa:", " ".join(f"{c['name']}={c['passed']}" for c in qa["checks"]))
        assert qa["passed"], qa["checks"]

        diff = caption_frame_diff(out, times=(1.2, 2.4, 3.6, 5.0))
        print("caption frame diff:", diff)
        assert diff["frames"] >= 3 and diff["animated"]

        times = (1.2, 2.4, 3.6, 5.0)
        captioned, plain = _frames(out, times), _frames(plain_out, times)
        assert len(captioned) == len(plain) == len(times)
        for t, cap_frame, plain_frame in zip(times, captioned, plain):
            delta = np.abs(cap_frame.astype(np.int16) - plain_frame.astype(np.int16))
            band = delta[int(SAFE_TOP):int(SAFE_BOTTOM), :int(SAFE_RIGHT)]
            mask = np.max(band, axis=2) > 50
            if mask.sum() < 50:
                continue
            ys, xs = np.where(mask)
            ylo, yhi = np.percentile(ys, [1, 99])
            xhi = np.percentile(xs, 99)
            print(f"t={t:.1f}s caption pixels {mask.sum()} "
                  f"y={ylo + SAFE_TOP:.0f}..{yhi + SAFE_TOP:.0f} x<={xhi:.0f}")
            assert SAFE_TOP - 5 <= ylo + SAFE_TOP
            assert yhi + SAFE_TOP <= SAFE_BOTTOM + 5
            assert xhi <= SAFE_RIGHT + 5
        print("SYNTHETIC_RENDER_VERIFY_OK")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
