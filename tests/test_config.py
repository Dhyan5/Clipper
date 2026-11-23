from pathlib import Path

from clipforge.config import (
    Settings,
    check_ffmpeg,
    detect_gpu,
    parse_facecam_override,
    slugify,
    video_id_from_url,
    video_encoder_args,
)


def test_video_id_forms():
    assert video_id_from_url("https://www.youtube.com/watch?v=eRsGyueVLvQ") == "eRsGyueVLvQ"
    assert video_id_from_url("https://youtu.be/eRsGyueVLvQ") == "eRsGyueVLvQ"
    assert video_id_from_url("https://www.youtube.com/shorts/eRsGyueVLvQ") == "eRsGyueVLvQ"
    assert video_id_from_url("eRsGyueVLvQ") == "eRsGyueVLvQ"


def test_slugify():
    assert slugify("Sintel - Open Movie by Blender Foundation") == "sintel-open-movie-by-blender-foundation"
    assert slugify("!!!") == "clip"
    assert len(slugify("x" * 200)) <= 60


def test_ffmpeg_startup_check():
    ok, msg = check_ffmpeg()
    assert ok, msg
    assert "ffmpeg" in msg.lower()


def test_gpu_and_encoder_args():
    gpu = detect_gpu()
    assert gpu in {"nvenc", "cuda", "cpu"}
    args = video_encoder_args()
    assert "-c:v" in args


def test_facecam_override():
    assert parse_facecam_override("10,20,300,200") == {"x": 10, "y": 20, "w": 300, "h": 200}
    assert parse_facecam_override("") is None
    assert parse_facecam_override("1,2,3") is None
    assert parse_facecam_override("a,b,c,d") is None
    assert parse_facecam_override("1,2,0,4") is None


def test_settings_fingerprint_changes():
    a = Settings()
    b = Settings()
    assert a.fingerprint() == b.fingerprint()
    b.clip_count = 3
    assert a.fingerprint() != b.fingerprint()
