"""Geometry checks for robust four-corner proposals from segmentation masks."""

import cv2
import numpy as np

from pipeline.board_geometry import estimate_board_corners


def _make_board(angle_degrees=14.0, length=270.0, width=74.0, center=(210.0, 120.0)):
    """Create a filled, slightly tapered board in a local coordinate frame."""
    theta = np.deg2rad(angle_degrees)
    longitudinal = np.array([np.cos(theta), np.sin(theta)])
    across = np.array([-np.sin(theta), np.cos(theta)])
    if across[1] < 0:
        across = -across
    c = np.asarray(center, dtype=np.float64)
    start, end = c - longitudinal * length / 2, c + longitudinal * length / 2
    polygon = np.array([
        start - across * width * 0.43,
        end - across * width * 0.31,
        end + across * width * 0.50,
        start + across * width * 0.39,
    ])
    mask = np.zeros((320, 440), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask, polygon.astype(np.float32)


def _assert_order_and_finite(corners, near_vertical=False):
    assert corners is not None
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32
    assert np.isfinite(corners).all()

    longitudinal = corners[[1, 2]].mean(axis=0) - corners[[0, 3]].mean(axis=0)
    across = corners[[2, 3]].mean(axis=0) - corners[[0, 1]].mean(axis=0)
    if near_vertical:
        assert longitudinal[1] > 0
    else:
        assert longitudinal[0] > 0
    assert across[1] > 0


def test_clean_skew_trapezoid_returns_ordered_corners_close_to_outline():
    mask, expected = _make_board(angle_degrees=14.0)

    corners = estimate_board_corners(mask)

    _assert_order_and_finite(corners)
    # The axes are image-oriented, so compare the corner sets without assuming
    # semantic nut/bridge labels for the polygon's local start and end.
    distances = np.linalg.norm(corners[:, None, :] - expected[None, :, :], axis=2)
    assert np.max(np.min(distances, axis=1)) < 8.0


def test_near_vertical_board_uses_positive_image_y_for_longitudinal_axis():
    mask, _ = _make_board(angle_degrees=83.0, length=260.0, width=58.0,
                          center=(220.0, 160.0))

    corners = estimate_board_corners(mask)

    _assert_order_and_finite(corners, near_vertical=True)


def test_holes_notches_and_small_detached_spur_do_not_move_corners_much():
    clean, _ = _make_board(angle_degrees=9.0, length=290.0, width=82.0,
                           center=(215.0, 150.0))
    damaged = clean.copy()
    # An interior void, plus local boundary notches on opposite rails.
    cv2.rectangle(damaged, (190, 126), (216, 145), 0, thickness=-1)
    cv2.rectangle(damaged, (246, 79), (267, 94), 0, thickness=-1)
    cv2.rectangle(damaged, (114, 208), (137, 222), 0, thickness=-1)
    # A small detached spur outside the board envelope.
    cv2.fillPoly(damaged, [np.array([[82, 72], [93, 67], [99, 77]], dtype=np.int32)], 255)

    expected = estimate_board_corners(clean)
    actual = estimate_board_corners(damaged)

    _assert_order_and_finite(expected)
    _assert_order_and_finite(actual)
    assert np.max(np.linalg.norm(actual - expected, axis=1)) < 7.0


def test_empty_tiny_and_isotropic_masks_return_none():
    empty = np.zeros((100, 160), dtype=np.uint8)
    tiny = np.zeros_like(empty)
    tiny[10, 10] = 255
    square = np.zeros_like(empty)
    cv2.rectangle(square, (45, 25), (100, 80), 255, thickness=-1)

    assert estimate_board_corners(empty) is None
    assert estimate_board_corners(tiny) is None
    assert estimate_board_corners(square) is None

