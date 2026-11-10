"""Synchronized, annotated video for inspecting visual transcription runs."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


DEBUG_FPS = 5.0
FINGER_COLORS = {
    "index": (80, 100, 255),
    "middle": (80, 220, 80),
    "ring": (255, 180, 70),
    "pinky": (230, 100, 220),
}


def sample_indices(frame_count: int, source_fps: float, output_fps: float = DEBUG_FPS) -> tuple[tuple[int, ...], float]:
    """Return source indices for a constant-rate video with source-time duration."""
    if frame_count < 1 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Debug video needs valid frame count and source FPS")
    if not math.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("Debug video FPS must be positive")
    fps = min(float(output_fps), float(source_fps))
    count = max(1, math.ceil(frame_count / source_fps * fps - 1e-9))
    indices = tuple(min(frame_count - 1, round(index * source_fps / fps)) for index in range(count))
    return indices, fps


class DebugVideoWriter:
    """Stream annotated BGR frames to FFmpeg and retain the source audio."""

    def __init__(self, output: str | Path, source: str | Path, width: int, height: int, fps: float):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.shape = (height, width, 3)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.8f}", "-i", "pipe:0",
            "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-shortest", "-movflags", "+faststart", str(self.output),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError("Debug frame has the wrong size or pixel type")
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames_written += 1

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        assert self.process.stderr is not None
        error = self.process.stderr.read().decode("utf-8", errors="replace").strip()
        status = self.process.wait()
        if status != 0 or not self.output.is_file() or self.output.stat().st_size == 0:
            raise RuntimeError(f"Could not encode visual debug video: {error or f'FFmpeg exit {status}'}")


def draw_debug_frame(
    bgr: np.ndarray,
    atlas: Any,
    fingertips: Sequence[Mapping[str, Any]],
    *,
    time_sec: float,
    tracking: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Draw the observed fret/string grid and four tracked fingertips per hand."""
    output = bgr.copy()
    height, width = output.shape[:2]
    quality = getattr(atlas, "quality", {}) if atlas is not None else {}
    grid_status = quality.get("status", "unavailable")
    nodes = np.asarray(getattr(atlas, "source_nodes", []), dtype=np.float64)
    if grid_status == "grid" and nodes.ndim == 3 and nodes.shape[2] == 2:
        for row in nodes[1:-1]:
            cv2.polylines(output, [np.rint(row).astype(np.int32)], False, (80, 230, 80), 2, cv2.LINE_AA)
        for column in nodes.transpose(1, 0, 2)[1:-1]:
            cv2.polylines(output, [np.rint(column).astype(np.int32)], False, (50, 180, 255), 2, cv2.LINE_AA)
    elif nodes.ndim == 3 and nodes.shape[2] == 2:
        outline = np.concatenate((nodes[0], nodes[1:, -1], nodes[-1, -2::-1], nodes[-2:0:-1, 0]))
        cv2.polylines(output, [np.rint(outline).astype(np.int32)], True, (160, 160, 160), 2, cv2.LINE_AA)

    for tip in fingertips:
        x, y = int(round(tip["x"])), int(round(tip["y"]))
        color = FINGER_COLORS.get(str(tip["finger"]), (255, 255, 255))
        distal = tip.get("distal")
        if distal is not None:
            cv2.line(output, (int(round(distal[0])), int(round(distal[1]))), (x, y), color, 2, cv2.LINE_AA)
        cv2.circle(output, (x, y), 7, color, -1, cv2.LINE_AA)
        label = str(tip["finger"])
        if tip.get("row") is not None:
            label += f" s{float(tip['row']):.1f}"
        if tip.get("fret_region") is not None:
            label += f" f{int(tip['fret_region'])}"
        cv2.putText(output, label, (x + 10, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, label, (x + 10, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, color, 2, cv2.LINE_AA)

    method = (tracking or {}).get("method") or (tracking or {}).get("status", "off")
    status = f"{time_sec:5.2f}s  board: {grid_status}  tracking: {method}  fingertips: {len(fingertips)}"
    text_width = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, .7, 2)[0][0]
    cv2.rectangle(output, (0, 0), (min(width - 1, text_width + 24), 42), (0, 0, 0), -1)
    cv2.putText(output, status, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return output
