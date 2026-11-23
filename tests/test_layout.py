import clipforge.layout as layout
from clipforge.layout import build_filter_script, render_clip
from clipforge.utils import CommandError


def test_facecam_layout_chain():
    script = build_filter_script(
        keep_segments=[(0.0, 5.0), (7.0, 10.0)], duration=8.0,
        facecam_box={"x": 80, "y": 40, "w": 320, "h": 240},
        width=1280, height=720, face_src="fonts",
    )
    assert "[0:v]setpts=PTS-STARTPTS" in script
    assert "[1:v]setpts=PTS-STARTPTS" in script
    assert "[1:a]asetpts=PTS-STARTPTS" in script
    assert "concat=n=2:v=1:a=1" in script
    assert "vstack=inputs=2" in script
    assert "fps=30,setpts=N/(30*TB),zoompan" in script
    assert "subtitles=filename=captions.ass:fontsdir=fonts" in script
    assert "loudnorm=I=-14" in script
    assert "drawbox" in script


def test_fallback_layout_chain():
    script = build_filter_script(
        keep_segments=[(0.0, 20.0)], duration=20.0,
        facecam_box=None, width=1280, height=720,
    )
    assert "boxblur" in script
    assert "overlay=(W-w)/2:(H-h)/2" in script
    assert "vstack" not in script
    assert "[v0]null[cv]" in script
    assert "[a0]anull[ca]" in script


def test_gain_and_single_segment():
    script = build_filter_script(
        keep_segments=[(2.0, 12.0)], duration=10.0, facecam_box=None,
        width=640, height=360, gain_db=3.5,
    )
    assert "volume=3.50dB" in script


def _ass_file(tmp_path):
    path = tmp_path / "captions.ass"
    path.write_text("[Script Info]\n", encoding="utf-8")
    return path


def test_render_uses_inline_filter_complex(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(layout, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(layout, "video_encoder_args", lambda: ["-c:v", "libx264"])
    out = render_clip(
        tmp_path / "source.mp4", [(0.0, 5.0)], 5.0, None, 1280, 720,
        _ass_file(tmp_path), tmp_path / "clip.mp4", tmp_path / "work",
    )
    assert out == tmp_path / "clip.mp4"
    final = calls[-1]
    assert "-filter_complex" in final
    assert "-filter_complex_script" not in final
    assert final.count("-i") == 1


def test_render_falls_back_from_nvenc(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 2:
            raise CommandError("nvenc failed")

    monkeypatch.setattr(layout, "run", fake_run)
    monkeypatch.setattr(
        layout, "video_encoder_args",
        lambda: ["-c:v", "h264_nvenc", "-preset", "p5"],
    )
    render_clip(
        tmp_path / "source.mp4", [(0.0, 5.0)], 5.0, None, 1280, 720,
        _ass_file(tmp_path), tmp_path / "clip.mp4", tmp_path / "work",
    )
    assert len(calls) == 3
    assert "libx264" in calls[-1]
