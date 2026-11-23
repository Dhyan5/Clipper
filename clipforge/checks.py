from __future__ import annotations

from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}


def audio_track_size(video_cache: Path | str) -> int:
    folder = Path(video_cache)
    sizes = [p.stat().st_size for p in folder.glob("audio.*")
             if p.suffix not in {".json", ".part"}]
    # A derived 16k wav is still the audio track and is allowed to be biggest.
    wav = folder / "audio16k.wav"
    if wav.exists():
        sizes.append(wav.stat().st_size)
    return max(sizes, default=0)


def walk_files(root: Path | str):
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS):
            yield p


def find_files_larger_than(root: Path | str, limit: int) -> list[tuple[Path, int]]:
    out = []
    for p in walk_files(root):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > limit:
            out.append((p, size))
    return sorted(out, key=lambda x: -x[1])


def largest_files(root: Path | str, n: int = 10) -> list[tuple[Path, int]]:
    files = []
    for p in walk_files(root):
        try:
            files.append((p, p.stat().st_size))
        except OSError:
            continue
    return sorted(files, key=lambda x: -x[1])[:n]


def assert_no_file_larger_than_audio(root: Path | str, video_cache: Path | str) -> dict:
    limit = audio_track_size(video_cache)
    if limit <= 0:
        raise AssertionError("No audio track found; download audio before verifying")
    violations = find_files_larger_than(root, limit)
    return {"limit": limit, "violations": violations,
            "largest": largest_files(root, 8)}
