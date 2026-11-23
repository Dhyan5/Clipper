from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CACHE_DIR, CACHE_VERSION, MODELS_DIR
from .utils import read_json, write_json

FACE_MODEL = MODELS_DIR / "blaze_face_short_range.tflite"


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def expand_box(box: dict, width: int, height: int, factor: float = 1.35) -> dict:
    cx = box["x"] + box["w"] / 2.0
    cy = box["y"] + box["h"] / 2.0
    w = min(width, box["w"] * factor)
    h = min(height, box["h"] * factor)
    x = max(0.0, min(width - w, cx - w / 2.0))
    y = max(0.0, min(height - h, cy - h / 2.0))
    return {
        "x": _even(x),
        "y": _even(y),
        "w": _even(min(w, width - x)),
        "h": _even(min(h, height - y)),
    }


def scale_box(box: dict | None, from_w: int, from_h: int,
              to_w: int, to_h: int) -> dict | None:
    if not box or not from_w or not from_h or (from_w, from_h) == (to_w, to_h):
        return box
    sx, sy = to_w / from_w, to_h / from_h
    return {
        "x": _even(box["x"] * sx), "y": _even(box["y"] * sy),
        "w": _even(box["w"] * sx), "h": _even(box["h"] * sy),
    }


def facecam_crop(box: dict | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not box:
        return None
    b = expand_box(box, width, height)
    return b["x"], b["y"], b["w"], b["h"]


def gameplay_crop(box: dict | None, width: int, height: int) -> tuple[int, int, int, int]:
    """Pick a source region that avoids the facecam box where possible."""
    if not box:
        return 0, 0, _even(width), _even(height)
    top_gap = box["y"]
    bottom_gap = height - (box["y"] + box["h"])
    if bottom_gap >= top_gap and bottom_gap > height * 0.18:
        y = _even(box["y"] + box["h"] + 4)
        return 0, y, _even(width), _even(height - y)
    if top_gap > height * 0.18:
        return 0, 0, _even(width), _even(box["y"] - 4)
    return 0, 0, _even(width), _even(height)


def _sample_frames(path: Path | str, count: int = 30) -> tuple[list[np.ndarray], int, int]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    if total > 0 and width > 0:
        idxs = np.linspace(int(total * 0.02), max(1, int(total * 0.98) - 1), count).astype(int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) > count:
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, count).astype(int)]
    cap.release()
    return frames, width, height


def _detect_mediapipe(frames: list[np.ndarray]) -> list[dict]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not FACE_MODEL.exists():
        raise FileNotFoundError(FACE_MODEL)
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    detector = vision.FaceDetector.create_from_options(options)
    found: list[dict] = []
    try:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(image)
            best = None
            for det in result.detections or []:
                b = det.bounding_box
                score = float(det.categories[0].score) if det.categories else 0.0
                cand = {"x": int(b.origin_x), "y": int(b.origin_y),
                        "w": int(b.width), "h": int(b.height), "score": score}
                if best is None or cand["w"] * cand["h"] > best["w"] * best["h"]:
                    best = cand
            if best:
                found.append(best)
    finally:
        detector.close()
    return found


def _detect_haar(frames: list[np.ndarray]) -> list[dict]:
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.exists():
        return []
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        return []
    found: list[dict] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.12,
                                         minNeighbors=6, minSize=(60, 60))
        best = None
        for (x, y, w, h) in faces:
            cand = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": 1.0}
            if best is None or w * h > best["w"] * best["h"]:
                best = cand
        if best:
            found.append(best)
    return found


def _core(frames: list[np.ndarray], width: int, height: int, video_id: str,
          override: dict | None, cache: bool = True) -> dict:
    engine = "none"
    found: list[dict] = []
    if frames:
        try:
            found = _detect_mediapipe(frames)
            engine = "mediapipe"
        except Exception:
            found = _detect_haar(frames)
            engine = "haar"

    if override is not None:
        box = expand_box(override, width, height) if width and height else override
        data = {"box": box, "samples": len(frames), "hits": len(found),
                "engine": "manual", "width": width, "height": height,
                "cached": False}
    elif found and len(found) >= max(5, int(0.25 * len(frames))):
        med = {
            "x": int(np.median([b["x"] for b in found])),
            "y": int(np.median([b["y"] for b in found])),
            "w": int(np.median([b["w"] for b in found])),
            "h": int(np.median([b["h"] for b in found])),
        }
        box = expand_box(med, width, height)
        data = {"box": box, "samples": len(frames), "hits": len(found),
                "engine": engine, "width": width, "height": height,
                "confidence": round(float(np.mean([b.get("score", 1.0) for b in found])), 3),
                "cached": False}
    else:
        data = {"box": None, "samples": len(frames), "hits": len(found),
                "engine": engine, "width": width, "height": height,
                "cached": False}
    if cache and (frames or override is not None):
        write_json(CACHE_DIR / video_id / f"facecam_v{CACHE_VERSION}.json", data)
    return data


def detect_facecam_from_frames(frames: list[np.ndarray], video_id: str,
                               override: dict | None = None,
                               width: int = 0, height: int = 0) -> dict:
    if not width or not height:
        width = int(frames[0].shape[1]) if frames else 0
        height = int(frames[0].shape[0]) if frames else 0
    return _core(frames, width, height, video_id, override)


def detect_facecam(source: Path | str, video_id: str, override: dict | None = None,
                   progress=None) -> dict:
    """Local-file fallback (re-render): 30 frames read from the cached video."""
    cache = CACHE_DIR / video_id / f"facecam_v{CACHE_VERSION}.json"
    if override is None:
        cached = read_json(cache)
        if cached is not None:
            cached["cached"] = True
            return cached
    if progress:
        progress("Detecting facecam")
    frames, width, height = _sample_frames(source)
    return _core(frames, width, height, video_id, override)


def detect_facecam_remote(url: str, video_id: str, duration: float,
                          override: dict | None = None, progress=None) -> dict:
    """Detect facecam from ~30 remote stills; no full video download."""
    cache = CACHE_DIR / video_id / f"facecam_v{CACHE_VERSION}.json"
    if override is None:
        cached = read_json(cache)
        if cached is not None:
            cached["cached"] = True
            return cached

    if override is not None:
        from .downloader import video_dimensions

        dims = video_dimensions(url) or (1920, 1080)
        return _core([], dims[0], dims[1], video_id, override)

    if progress:
        progress("Sampling ~30 frames for facecam (no video download)")
    from .downloader import sample_face_frames

    sample = sample_face_frames(url, duration, count=30)
    frames = sample["frames"]
    data = _core(frames, sample["width"], sample["height"], video_id, None,
                 cache=bool(frames))
    data["sampling"] = sample["method"]
    return data
