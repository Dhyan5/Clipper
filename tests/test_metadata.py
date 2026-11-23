from clipforge.packaging import (
    ClipMeta, QAResult, VideoMeta, clip_filename, thumbnail_path,
)
from pydantic import ValidationError
import pytest


def _clip(index=1):
    return ClipMeta(
        index=index,
        filename=clip_filename(index, "A Great Title"),
        title_options=["A Great Title", "Second", "Third"],
        chosen_title="A Great Title",
        hook="The truth about this",
        hook_variants=[{"type": "curiosity", "text": "The truth about this"}],
        hook_start=0.1,
        score=8.2,
        score_breakdown={"gemini": 8.0, "audio": 7.0, "density": 6.0},
        start=10.0,
        end=40.0,
        duration=29.0,
        description="desc",
        hashtags=[f"#t{i}" for i in range(12)],
        emphasis_words=["truth"],
        thumbnail="thumbnails/clip_01.png",
        layout="facecam+gameplay",
        keep_segments=[[10.0, 40.0]],
        qa=QAResult(passed=True, attempts=1),
    )


def test_video_metadata_validates():
    meta = VideoMeta(
        video_id="TESTVIDEO01",
        url="https://example.com",
        source_title="Source",
        slug="source_TESTVIDEO01",
        folder="output/source_TESTVIDEO01",
        created_at="2026-01-01T00:00:00+00:00",
        mode="fallback (no Gemini key)",
        settings={"clip_count": 6},
        settings_fingerprint="abc123",
        clips=[_clip()],
    )
    dumped = meta.model_dump()
    assert dumped["clips"][0]["duration"] == 29.0
    assert VideoMeta.model_validate(dumped).clips[0].hook == "The truth about this"


def test_invalid_metadata_rejected():
    with pytest.raises(ValidationError):
        ClipMeta(index="not-an-int", filename="x")


def test_filenames_and_thumbnails():
    assert clip_filename(1, "Hello, World!") == "clip_01_hello-world.mp4"
    assert thumbnail_path(__import__("pathlib").Path("out"), 3).as_posix() == "out/thumbnails/clip_03.png"
