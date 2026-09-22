"""Small, model-free checks for the fretboard geometry helpers.

The tests deliberately use a synthetic mask so they do not need the bundled
YOLO checkpoint or an image fixture.
"""

import json

import cv2
import numpy as np

from pipeline import fret_detect_yolo as detector
from pipeline import gradio_demo


def _synthetic_board():
    """Return an RGB image and a slightly skewed, filled board mask."""
    image = np.zeros((140, 420, 3), dtype=np.uint8)
    image[..., 0] = np.arange(image.shape[1], dtype=np.uint8)
    image[..., 1] = np.arange(image.shape[0], dtype=np.uint8)[:, None]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    polygon = np.array([[40, 35], [350, 20], [370, 95], [20, 110]], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return image, mask


def test_synthetic_mask_produces_ordered_fret_geometry():
    """PCA corners and projected fret lines should be finite and ordered."""
    _, mask = _synthetic_board()

    corners = detector._pca_corners(mask)
    assert corners is not None
    assert corners.shape == (4, 2)
    assert np.isfinite(corners).all()

    fret_lines = detector.detect_frets(mask)
    assert fret_lines, "a filled synthetic board should yield fret lines"
    points = np.asarray(fret_lines, dtype=np.float32)
    assert points.shape[1:] == (2, 2)
    assert np.isfinite(points).all()

    # The implementation emits lines from the nut toward the bridge.  Check
    # order in the board's principal (top-edge) direction rather than relying
    # on image x, since real boards can be rotated.
    axis = corners[1] - corners[0]
    axis /= np.linalg.norm(axis)
    midpoints = points.mean(axis=1)
    positions = midpoints @ axis
    assert np.all(np.diff(positions) > -1e-3)


def test_rectify_fretboard_honors_output_size():
    """Exercise the public warp helper without requiring a model checkpoint."""
    image, mask = _synthetic_board()
    rectified = detector.rectify_fretboard(mask, image=image, output_size=(96, 32))
    assert rectified.shape[:2] == (32, 96)
    assert rectified.shape[2] == 3
    assert np.isfinite(rectified).all()


def test_build_json_report_round_trips_through_json(tmp_path):
    """Reports and their output-path metadata remain JSON-native on disk."""
    image, mask = _synthetic_board()
    fret_lines = detector.detect_frets(mask)
    result = detector.build_json_report(
        "synthetic.png",
        (image.shape[1], image.shape[0]),
        "weights.pt",
        0.25,
        mask,
        fret_lines,
        rectified_dimensions=(96, 32),
        rectified_output="rectified.png",
    )

    encoded = json.dumps(result, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == "1.0"
    assert decoded["image"]["dimensions"] == [420, 140]
    assert decoded["model"]["type"] == "yolov8-seg"
    assert decoded["fret_count"] == len(fret_lines)
    assert decoded["frets"][0]["number"] == 1
    assert decoded["rectified"]["output_path"] == "rectified.png"

    report_path = tmp_path / "nested" / "result.json"
    detector._write_json_report(report_path, result)
    assert json.loads(report_path.read_text(encoding="utf-8")) == decoded


def test_gradio_processing_returns_all_outputs_without_loading_a_model(monkeypatch):
    """The UI adapter should connect the shared detector outputs end to end."""
    image, mask = _synthetic_board()
    monkeypatch.setattr(gradio_demo, "_get_model", lambda _path: object())
    monkeypatch.setattr(detector, "predict_mask", lambda _model, _image: mask)

    annotated, rectified, summary, report = gradio_demo.process_image(image)

    assert annotated.shape == image.shape
    assert rectified.shape == (256, 1024, 3)
    assert "Detection summary" in summary
    assert report["schema_version"] == "1.0"
    assert report["fret_count"] == len(report["frets"])
    assert len(report["rectified"]["source_corners"]) == 4
    assert len(report["rectified"]["perspective_transform"]) == 3
