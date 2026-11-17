"""Render a stable, synchronized presentation of a saved AV transcription.

The demo draws the measured, frame-tracked calibration and the actual final
note assignments. It does not visualize raw per-frame YOLO atlas proposals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from main.video.fretboard.audio_visual_fusion import BoardCalibration, track_calibration
from main.video.hands.debug_video import DebugVideoWriter
from main.audio.guitar_tab import STANDARD_TUNING_MIDI


STRINGS_HIGH_TO_LOW = (1, 2, 3, 4, 5, 6)
HAND_MODEL_PATH = Path(__file__).resolve().parents[1] / "main/models/hands/hand_landmarker.task"
COLORS = {
    "cyan": (255, 221, 70),
    "green": (40, 255, 0),
    "string": (40, 255, 0),
    "white": (245, 248, 250),
    "orange": (70, 173, 255),
}


def midi_name(pitch: int) -> str:
    names = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def draw_board(frame: np.ndarray, board: BoardCalibration, active_notes: list[dict]) -> None:
    """Draw stable calibrated strings and measured fret wires in source pixels."""
    for string in STRINGS_HIGH_TO_LOW:
        y = board.string_y(string)
        points = board.board_to_image([[0.0, y], [board.visible_x_max, y]])
        cv2.line(frame, tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)),
                 COLORS["string"], 3, cv2.LINE_AA)

    wire_numbers = sorted({0, 1, 3, 5, 7, 8, board.far_fret})
    for fret in wire_numbers:
        x = board.fret_x(fret)
        endpoints = board.board_to_image([[x, 0.0], [x, 1.0]])
        color = COLORS["orange"] if fret == 0 else COLORS["green"]
        cv2.line(frame, tuple(np.rint(endpoints[0]).astype(int)), tuple(np.rint(endpoints[1]).astype(int)),
                 color, 2 if fret == 0 else 1, cv2.LINE_AA)

    # A short pulse identifies the selected physical string/fret at the
    # corresponding audio attack. This marks the final assignment, not a raw
    # detector prediction.
    for note in active_notes:
        fret, string_idx = int(note["fret_idx"]), int(note["string_idx"])
        if fret <= 0:
            continue
        x = (board.fret_x(fret - 1) + board.fret_x(fret)) / 2
        string_number = 6 - string_idx
        y = board.string_y(string_number)
        point = tuple(np.rint(board.board_to_image([[x, y]])[0]).astype(int))
        cv2.circle(frame, point, 16, COLORS["cyan"], 3, cv2.LINE_AA)
        cv2.circle(frame, point, 5, COLORS["white"], -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, event: dict | None) -> None:
    height, width = frame.shape[:2]
    # Cover the printed score. Only the requested attack and tab position text
    # appears while an audio event is active.
    top = height - 330
    cv2.rectangle(frame, (0, top), (width, height), (12, 19, 28), -1)
    if event:
        pitches = "  +  ".join(
            midi_name(STANDARD_TUNING_MIDI[int(n["string_idx"])] + int(n["fret_idx"]))
            for n in event["notes"]
        )
        positions = "   ·   ".join(
            f"string {6 - int(n['string_idx'])}, fret {n['fret_idx']}"
            for n in event["notes"]
        )
        cv2.putText(frame, f"AUDIO ATTACK   {pitches}", (48, top + 125),
                    cv2.FONT_HERSHEY_SIMPLEX, .76, COLORS["white"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"FINAL TAB POSITION   {positions}", (48, top + 205),
                    cv2.FONT_HERSHEY_SIMPLEX, .72, COLORS["cyan"], 2, cv2.LINE_AA)


def detect_fingertips(landmarker, rgb: np.ndarray, timestamp_ms: int) -> list[dict]:
    """Return thumb and four fingertip landmarks for the right hand only."""
    import mediapipe as mp

    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms,
    )
    tips = []
    names = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}
    distal = {4: 3, 8: 7, 12: 11, 16: 15, 20: 19}
    hands = getattr(result, "hand_landmarks", ())
    handedness = getattr(result, "handedness", ())
    for hand, labels in zip(hands, handedness):
        is_right_hand = any(
            label.category_name == "Right" and float(label.score) >= 0.5
            for label in labels
        )
        if not is_right_hand:
            continue
        for index, name in names.items():
            tip, joint = hand[index], hand[distal[index]]
            tips.append({
                "name": name,
                "x": float(tip.x * rgb.shape[1]), "y": float(tip.y * rgb.shape[0]),
                "joint": (float(joint.x * rgb.shape[1]), float(joint.y * rgb.shape[0])),
            })
    return tips


def draw_fingertips(frame: np.ndarray, tips: list[dict]) -> None:
    """Show all tracked fingertip paths at each sampled frame."""
    colors = {
        "thumb": (255, 120, 50), "index": (80, 100, 255), "middle": (80, 220, 80),
        "ring": (255, 180, 70), "pinky": (230, 100, 220),
    }
    for tip in tips:
        x, y = int(round(tip["x"])), int(round(tip["y"]))
        jx, jy = (int(round(value)) for value in tip["joint"])
        color = colors[tip["name"]]
        cv2.line(frame, (jx, jy), (x, y), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 11, (255, 255, 255), 1, cv2.LINE_AA)


def render(source: Path, calibration_path: Path, transcription_path: Path, output: Path) -> int:
    data = json.loads(transcription_path.read_text())
    calibration_data = json.loads(calibration_path.read_text())
    calibration = BoardCalibration.from_dict(calibration_data)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or count < 1:
        cap.release()
        raise ValueError("Source video has invalid frame metadata")
    if calibration.image_size and tuple(calibration.image_size) != (width, height):
        cap.release()
        raise ValueError("Calibration image size does not match source video")
    reference_frame = int(calibration_data.get("reference_frame", 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
    ok, reference_bgr = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read calibration reference frame")
    reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    events = sorted(data["events"], key=lambda e: e["onset_sec"])
    duration = count / fps
    frame_indices = list(range(0, count, max(1, round(fps / 5))))
    if frame_indices[-1] != count - 1:
        frame_indices.append(count - 1)
    if not HAND_MODEL_PATH.is_file():
        cap.release()
        raise ValueError(f"MediaPipe hand model is missing: {HAND_MODEL_PATH}")
    import mediapipe as mp

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    writer = DebugVideoWriter(output, source, width, height, len(frame_indices) / duration)
    event_index = 0
    tracking_counts = {"tracked": 0, "failed": 0}
    last_timestamp_ms = -1
    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = cap.read()
            if not ok:
                continue
            time_sec = frame_index / fps
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = max(round(time_sec * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            fingertip_tracks = detect_fingertips(landmarker, rgb, timestamp_ms)
            board, status = track_calibration(
                reference, rgb, calibration,
            )
            is_tracked = board is not None and status.get("status") in {"reference", "tracked"}
            tracking = "tracked" if is_tracked else "failed"
            tracking_counts[tracking] += 1
            while event_index + 1 < len(events) and events[event_index + 1]["onset_sec"] <= time_sec:
                event_index += 1
            event = events[event_index] if events and 0 <= time_sec - events[event_index]["onset_sec"] <= .42 else None
            if is_tracked:
                draw_board(bgr, board, event["notes"] if event else [])
            draw_fingertips(bgr, fingertip_tracks)
            draw_hud(bgr, event)
            writer.write(bgr)
    finally:
        cap.release()
        landmarker.close()
        writer.close()
    return tracking_counts["tracked"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--transcription", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        count = render(args.source, args.calibration, args.transcription, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Rendered {args.output} with tracked fretboard geometry on {count} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
