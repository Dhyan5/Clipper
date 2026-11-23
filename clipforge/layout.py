from __future__ import annotations

import shutil
from pathlib import Path

from .config import FONTS_DIR, video_encoder_args
from .facecam import facecam_crop, gameplay_crop
from .utils import CommandError, run

OUT_W, OUT_H = 1080, 1920
HALF_H = 960
FPS = 30


def _progress_bar(duration: float) -> str:
    d = max(0.1, float(duration))
    return (
        f"drawbox=x=0:y=957:w='min(iw*max(t,0)/{d:.3f},iw)':h=6:"
        "color=0xFFD400@0.9:t=fill"
    )


def _layout_chain(facecam_box: dict | None, width: int, height: int) -> str:
    if facecam_box:
        fx, fy, fw, fh = facecam_crop(facecam_box, width, height)
        gx, gy, gw, gh = gameplay_crop(facecam_box, width, height)
        return (
            f"[cv]crop={fw}:{fh}:{fx}:{fy},"
            "scale=1080:960:force_original_aspect_ratio=increase,"
            "crop=1080:960,setsar=1[top];"
            f"[cv]crop={gw}:{gh}:{gx}:{gy},"
            "scale=1080:960:force_original_aspect_ratio=increase,"
            "crop=1080:960,setsar=1[bottom];"
            "[top][bottom]vstack=inputs=2[stack]"
        )
    return (
        "[cv]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1,boxblur=24:2,eq=brightness=-0.10[bg];"
        "[cv]scale=1080:960:force_original_aspect_ratio=decrease,setsar=1[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[stack]"
    )


def build_filter_script(
    keep_segments: list[tuple[float, float]],
    duration: float,
    facecam_box: dict | None,
    width: int,
    height: int,
    ass_name: str = "captions.ass",
    gain_db: float = 0.0,
    face_src: str | None = None,
) -> str:
    if not keep_segments:
        raise ValueError("keep_segments is empty")
    lines: list[str] = []
    for i, (s, e) in enumerate(keep_segments):
        seg_d = max(0.12, e - s)
        fade_start = max(0.0, seg_d - 0.05)
        lines.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        lines.append(
            f"[{i}:a]asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d=0.05,afade=t=out:st={fade_start:.3f}:d=0.05[a{i}]"
        )
    if len(keep_segments) == 1:
        lines.append("[v0]null[cv]")
        lines.append("[a0]anull[ca]")
    else:
        pairs = "".join(f"[v{i}][a{i}]" for i in range(len(keep_segments)))
        lines.append(f"{pairs}concat=n={len(keep_segments)}:v=1:a=1[cv][ca]")

    lines.append(_layout_chain(facecam_box, width, height))
    lines.append(f"[stack]{_progress_bar(duration)}[bar]")
    lines.append(
        "[bar]fps=30,setpts=N/(30*TB),"
        "zoompan=z='1+0.08*if(lt(mod(on/30,3),0.4),mod(on/30,3)/0.4,"
        "if(lt(mod(on/30,3),2.6),1,(3-mod(on/30,3))/0.4))':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={OUT_W}x{OUT_H}:fps={FPS}[zoom]"
    )
    font_dir = "fonts" if face_src else None
    sub = f"subtitles=filename={ass_name}"
    if font_dir:
        sub += f":fontsdir={font_dir}"
    lines.append(f"[zoom]{sub},format=yuv420p[v]")
    gain = f"volume={gain_db:.2f}dB," if abs(gain_db) > 0.01 else ""
    lines.append(
        f"[ca]{gain}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000:async=1[a]"
    )
    return ";\n".join(lines) + "\n"


def render_clip(
    source: Path | str,
    keep_segments: list[tuple[float, float]],
    duration: float,
    facecam_box: dict | None,
    width: int,
    height: int,
    ass_path: Path | str,
    out_path: Path | str,
    work_dir: Path | str,
    gain_db: float = 0.0,
    progress=None,
) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ass_src = Path(ass_path)
    ass_local = work_dir / "captions.ass"
    if ass_src.resolve() != ass_local.resolve():
        shutil.copyfile(ass_src, ass_local)
    face_src = None
    if FONTS_DIR.exists() and any(FONTS_DIR.glob("*.ttf")):
        fdir = work_dir / "fonts"
        fdir.mkdir(exist_ok=True)
        for f in FONTS_DIR.glob("*.ttf"):
            shutil.copyfile(f, fdir / f.name)
        face_src = "fonts"

    script = build_filter_script(
        keep_segments, duration, facecam_box, width, height,
        ass_name="captions.ass", gain_db=gain_db, face_src=face_src,
    )
    script_path = work_dir / "filter.txt"
    script_path.write_text(script, encoding="utf-8")

    source_path = str(Path(source).resolve())
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: list[Path] = []
    for i, (s, e) in enumerate(keep_segments):
        seg_path = seg_dir / f"seg_{i:02d}.mp4"
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, s):.3f}", "-t", f"{max(0.12, e - s):.3f}",
            "-i", source_path, "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-avoid_negative_ts", "make_zero", str(seg_path),
        ], cwd=work_dir, timeout=600)
        seg_paths.append(seg_path)

    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for seg_path in seg_paths:
        cmd += ["-i", str(seg_path)]
    cmd += [
        "-filter_complex", script,
        "-map", "[v]", "-map", "[a]",
        "-t", f"{duration:.3f}",
        "-r", str(FPS),
    ]
    tail = [
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", "-shortest", str(out_path),
    ]
    enc = video_encoder_args()
    try:
        run(cmd + enc + tail, cwd=work_dir, timeout=3600)
    except CommandError:
        if "h264_nvenc" not in enc:
            raise
        run(
            cmd + ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                   "-pix_fmt", "yuv420p"] + tail,
            cwd=work_dir,
            timeout=3600,
        )
    return out_path
