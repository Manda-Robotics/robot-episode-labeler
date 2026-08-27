import subprocess, sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from rel.video.contact_sheet import build_sheets
from rel.video.decode import VideoError, probe, sample_frames


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 6s clip whose colour changes every second, so time is checkable."""
    d = tmp_path_factory.mktemp("clip")
    parts = []
    for i, c in enumerate(["red", "green", "blue", "yellow", "magenta", "cyan"]):
        p = d / f"{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"color=c={c}:s=160x120:r=25:d=1", "-pix_fmt", "yuv420p", str(p)],
                       check=True)
        parts.append(f"file '{p}'")
    lst = d / "l.txt"; lst.write_text("\n".join(parts))
    out = d / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(out)], check=True)
    return out


def test_probe_reads_duration(clip):
    info = probe(clip)
    assert 5.5 < info.duration < 6.5 and info.width == 160


def test_missing_file_raises_video_error():
    with pytest.raises(VideoError):
        probe("/nonexistent/nope.mp4")


def test_sampling_grid_matches_wall_clock(clip):
    frames = sample_frames(clip, interval=0.5, width=64)
    assert len(frames) >= 11
    assert [f.t for f in frames[:4]] == [0.0, 0.5, 1.0, 1.5]
    # second 0 is red, second 2 is blue
    r = frames[0].image.getpixel((32, 24))
    b = frames[4].image.getpixel((32, 24))
    assert r[0] > 180 and r[2] < 80
    assert b[2] > 180 and b[0] < 80


def test_window_sampling_is_aligned_to_absolute_time(clip):
    win = sample_frames(clip, interval=0.5, width=64, start=2.0, end=4.0)
    assert win[0].t == 2.0
    assert all(2.0 <= f.t <= 4.0 for f in win)


def test_zero_interval_rejected(clip):
    with pytest.raises(ValueError):
        sample_frames(clip, interval=0)


def test_sheets_tile_and_carry_their_times(clip):
    frames = sample_frames(clip, interval=0.5, width=64)
    sheets = build_sheets(frames, per_sheet=4, columns=2)
    assert len(sheets) == (len(frames) + 3) // 4
    assert sheets[0].times == [f.t for f in frames[:4]]
    assert sheets[0].image.width > 64  # tiled, not a single frame
    assert build_sheets([], per_sheet=4, columns=2) == []


@pytest.fixture(scope="module")
def av1_clip(tmp_path_factory):
    """AV1 is the case that broke a full benchmark pass: Homebrew's macOS ffmpeg
    ships a videotoolbox-only AV1 decoder, so every AV1 episode failed on a host
    that cannot decode AV1 in hardware."""
    d = tmp_path_factory.mktemp("av1")
    out = d / "clip.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=160x120:r=10:d=3",
         "-c:v", "libsvtav1", "-pix_fmt", "yuv420p", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip("no AV1 encoder available to build the fixture")
    return out


def test_av1_is_decodable(av1_clip):
    import subprocess as sp
    codec = sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(av1_clip)],
                   capture_output=True, text=True).stdout.strip()
    assert codec == "av1"
    frames = sample_frames(av1_clip, interval=0.5, width=64)
    assert len(frames) >= 5
