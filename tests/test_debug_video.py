from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from main.video.hands.debug_video import DebugVideoWriter, draw_debug_frame, sample_indices


def test_debug_samples_keep_fifteen_second_timeline():
    indices, fps = sample_indices(450, 30.0)
    assert fps == 5.0
    assert len(indices) == 75
    assert indices[0] == 0
    assert indices[-1] == 444


def test_debug_overlay_draws_grid_and_fingertip():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    atlas = SimpleNamespace(
        quality={"status": "grid"},
        source_nodes=np.array([
            [[10, 10], [100, 10], [190, 10]],
            [[10, 50], [100, 50], [190, 50]],
            [[10, 90], [100, 90], [190, 90]],
        ], dtype=float),
    )
    output = draw_debug_frame(
        frame, atlas,
        [{"finger": "index", "x": 120, "y": 55, "distal": (125, 70),
          "row": 2.0, "fret_region": 5}],
        time_sec=1.0, tracking={"status": "tracked", "method": "orb"},
    )
    assert not np.array_equal(output, frame)
    assert output[50, 50, 1] > 0  # string grid
    assert output[55, 120].any()  # fingertip marker


def test_debug_video_preserves_source_audio(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is unavailable")
    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=4:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source),
    ], check=True)
    output = tmp_path / "visual_overlay.mp4"
    writer = DebugVideoWriter(output, source, 64, 48, 4.0)
    for index in range(4):
        writer.write(np.full((48, 64, 3), index * 50, dtype=np.uint8))
    writer.close()
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,nb_frames",
        "-of", "json", str(output),
    ], check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    assert {stream["codec_type"] for stream in info["streams"]} == {"video", "audio"}
    assert next(stream for stream in info["streams"] if stream["codec_type"] == "video")["nb_frames"] == "4"
    assert float(info["format"]["duration"]) == pytest.approx(1.0, abs=.1)
