from __future__ import annotations

import cv2
import numpy as np

from main.video.fretboard.audio_visual_fusion import BoardCalibration, track_calibration
from main.video.hands.visual_fingering import _track_frame_calibration


def _moving_board():
    rng = np.random.default_rng(7)
    reference = np.full((480, 640, 3), 35, dtype=np.uint8)
    reference[150:270, 90:460] = rng.integers(25, 230, (120, 370, 3), dtype=np.uint8)
    cv2.rectangle(reference, (90, 150), (460, 270), (210, 210, 210), 2)
    calibration = BoardCalibration(((90, 150), (460, 150), (460, 270), (90, 270)))
    matrix = cv2.getRotationMatrix2D((320, 240), 6.0, 1.0)
    matrix[:, 2] += (72, 18)
    moved = cv2.warpAffine(reference, matrix, (640, 480), borderValue=(35, 35, 35))
    expected = cv2.transform(np.float32(calibration.corners)[None], matrix)[0]
    return reference, moved, calibration, expected


def test_large_neck_motion_uses_feature_matching_when_flow_fails(monkeypatch):
    reference, moved, calibration, expected = _moving_board()
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda *_args, **_kwargs: (None, None, None))

    tracked, info = track_calibration(reference, moved, calibration)

    assert tracked is not None
    assert info["status"] == "tracked"
    assert info["method"] == "orb"
    assert np.max(np.linalg.norm(np.asarray(tracked.corners) - expected, axis=1)) < 5.0


def test_unrelated_frame_does_not_reuse_static_calibration():
    reference, _moved, calibration, _expected = _moving_board()
    unrelated = np.full_like(reference, 35)

    tracked, info = track_calibration(reference, unrelated, calibration)

    assert tracked is None
    assert info["status"] == "failed"


def test_nearby_frame_recovers_short_occlusion_without_unbounded_chaining():
    def tracker(source, target, _board):
        if source == "reference" and target != "frame0":
            return None, {"status": "failed", "reason": "occluded_reference"}
        return target, {"status": "tracked", "inliers": 20}

    previous = None
    strategies = []
    for index in range(5):
        board, info, previous = _track_frame_calibration(
            "reference", f"frame{index}", "board", index, 30.0,
            tracker, previous,
        )
        strategies.append(info["strategy"])
        if index < 4:
            assert board == f"frame{index}"
        else:
            assert board is None
    assert strategies == ["reference_frame", "nearby_frame", "nearby_frame", "nearby_frame", "failed"]
