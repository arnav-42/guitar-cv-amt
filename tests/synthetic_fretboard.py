"""Deterministic synthetic fretboard scenes for geometry benchmarks.

The scenes are rendered on a tapered board plane, then projected into a camera
frame with a homography.  Their node coordinates are retained as floating
point ground truth so rectifiers can be compared without involving a detector.
No model weights or private images are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticFretboard:
    """One rendered board and the geometry used to make it.

    ``source_nodes`` has shape ``(fret_count + 1, num_strings, 2)``.  Its first
    row is the nut (station zero); rows 1..fret_count are the equal-temperament
    fret stations, with the last station at the bridge end.  Coordinates are
    floating point ``(x, y)`` pixel locations in ``image``.

    ``visible_nodes`` marks nodes in-frame, on the visible board mask, and not
    covered by the known occluder.  This makes the scoreable subset explicit
    for crop and hand-occlusion cases.
    """

    name: str
    image: np.ndarray
    mask: np.ndarray
    occlusion_mask: np.ndarray
    source_nodes: np.ndarray
    visible_nodes: np.ndarray
    fret_x: np.ndarray
    homography: np.ndarray
    board_corners: np.ndarray
    num_strings: int
    fret_count: int
    description: str = ""


_PLANE_W = 1280
_PLANE_H = 320
_FRAME_SIZE = (760, 500)  # (width, height)


def equal_temperament_positions(fret_count: int) -> np.ndarray:
    """Return normalized nut-to-bridge stations x_0..x_N.

    The formula is x_n=(1-2**(-n/12))/(1-2**(-N/12)); x_0 is the nut and x_N
    is the bridge-side end of the modeled scale.
    """
    if fret_count < 1:
        raise ValueError("fret_count must be positive")
    n = np.arange(fret_count + 1, dtype=np.float64)
    denominator = 1.0 - 2.0 ** (-float(fret_count) / 12.0)
    return (1.0 - 2.0 ** (-n / 12.0)) / denominator


def _plane_geometry(fret_count: int, num_strings: int):
    # The board is a physical trapezoid: it widens toward the bridge.  Its top
    # and bottom boundaries are straight, so each constant-fraction string is
    # also a straight line and the strings fan apart longitudinally.
    x0, x1 = 32.0, _PLANE_W - 32.0
    top0, bot0 = 91.0, 229.0
    top1, bot1 = 29.0, 291.0
    u = equal_temperament_positions(fret_count)
    xs = x0 + u * (x1 - x0)
    t = np.linspace(0.135, 0.865, num_strings, dtype=np.float64)
    top = top0 + u * (top1 - top0)
    bottom = bot0 + u * (bot1 - bot0)
    ys = top[:, None] + t[None, :] * (bottom - top)[:, None]
    nodes = np.stack((np.broadcast_to(xs[:, None], ys.shape), ys), axis=-1)

    corners = np.asarray(
        [[x0, top0], [x1, top1], [x1, bot1], [x0, bot0]], dtype=np.float32
    )
    return xs, nodes, corners, (x0, x1, top0, bot0, top1, bot1)


def _render_plane(fret_count: int, num_strings: int, seed: int):
    """Render a dark, lightly grained fretboard with fretwire and strings."""
    rng = np.random.default_rng(seed)
    h, w = _PLANE_H, _PLANE_W
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Warm dark wood with subdued grain.  Texture is deliberately smooth and
    # low contrast so the geometry comes from fretwire and strings.
    noise = rng.normal(0.0, 3.3, (h, w)).astype(np.float32)
    grain = 3.3 * np.sin(xx * 0.046 + yy * 0.014) + 1.7 * np.sin(xx * 0.013 - yy * 0.035)
    vignette = -7.0 * np.square((yy - h / 2.0) / (h * 0.72))
    base = np.empty((h, w, 3), dtype=np.float32)
    for ch, value in enumerate((52.0, 42.0, 32.0)):
        base[..., ch] = value + noise + grain + vignette
    image = np.clip(base, 0, 255).astype(np.uint8)

    _, _, board_corners, geom = _plane_geometry(fret_count, num_strings)
    x0, x1, top0, bot0, top1, bot1 = geom
    board_poly = np.rint(board_corners).astype(np.int32)
    board_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(board_mask, [board_poly], 255)

    # Add faint lengthwise grain streaks clipped to the board.
    for _ in range(190):
        x = int(rng.integers(int(x0), int(x1)))
        y = int(rng.integers(38, h - 38))
        if board_mask[y, x] == 0:
            continue
        length = int(rng.integers(24, 180))
        color = tuple(int(v) for v in rng.integers(35, 61, size=3))
        cv2.line(image, (x, y), (min(w - 1, x + length), y + int(rng.integers(-2, 3))), color, 1, cv2.LINE_AA)

    # Dark, narrow edge binding helps the board read as a tapered object.
    cv2.polylines(image, [board_poly], True, (24, 23, 22), 5, cv2.LINE_AA)

    fret_x, nodes, _, _ = _plane_geometry(fret_count, num_strings)
    # Frets are transverse straight metal wires between the two board edges.
    for index, x in enumerate(fret_x[1:], start=1):
        u = (x - x0) / (x1 - x0)
        top = int(round(top0 + u * (top1 - top0)))
        bottom = int(round(bot0 + u * (bot1 - bot0)))
        cv2.line(image, (int(round(x)) + 1, top), (int(round(x)) + 1, bottom), (20, 20, 20), 6, cv2.LINE_AA)
        cv2.line(image, (int(round(x)), top), (int(round(x)), bottom), (168, 174, 177), 3, cv2.LINE_AA)
        cv2.line(image, (int(round(x)) - 1, top), (int(round(x)) - 1, bottom), (218, 222, 218), 1, cv2.LINE_AA)

    # Six individual strings, with clear margins from the board sides.  Each
    # is a true straight segment; slight paired strokes provide a metal glint.
    for s in range(num_strings):
        p0 = tuple(np.rint(nodes[0, s]).astype(int))
        p1 = tuple(np.rint(nodes[-1, s]).astype(int))
        cv2.line(image, p0, p1, (28, 27, 26), 3, cv2.LINE_AA)
        cv2.line(image, p0, p1, (182, 183, 177), 1, cv2.LINE_AA)
        cv2.line(image, (p0[0], p0[1] - 1), (p1[0], p1[1] - 1), (222, 219, 207), 1, cv2.LINE_AA)

    # Mother-of-pearl style dots between the strings at familiar positions.
    # Their presence makes the scene read as a guitar board without carrying
    # the fret geometry itself.
    for fret in (3, 5, 7, 9, 12):
        if fret >= fret_count:
            continue
        cx = int(round((fret_x[fret] + fret_x[fret + 1]) / 2.0))
        u = (cx - x0) / (x1 - x0)
        top = top0 + u * (top1 - top0)
        bottom = bot0 + u * (bot1 - bot0)
        cy = int(round((top + bottom) / 2.0))
        radius = 12 if fret == 12 else 7
        cv2.circle(image, (cx, cy), radius + 2, (30, 31, 33), -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), radius, (166, 171, 168), -1, cv2.LINE_AA)
        cv2.circle(image, (cx - 2, cy - 2), max(2, radius // 3), (205, 208, 202), -1, cv2.LINE_AA)

    return image, board_mask, fret_x, nodes, board_corners


def _camera_homography(case: str, out_size: Tuple[int, int]) -> np.ndarray:
    width, height = out_size
    if case == "clean_frontal":
        dst = np.asarray([[72, 142], [688, 142], [688, 358], [72, 358]], np.float32)
        return cv2.getPerspectiveTransform(
            np.asarray([[0, 0], [_PLANE_W - 1, 0], [_PLANE_W - 1, _PLANE_H - 1], [0, _PLANE_H - 1]], np.float32),
            dst,
        )
    if case == "perspective":
        dst = np.asarray([[111, 70], [670, 119], [622, 378], [48, 310]], np.float32)
        return cv2.getPerspectiveTransform(
            np.asarray([[0, 0], [_PLANE_W - 1, 0], [_PLANE_W - 1, _PLANE_H - 1], [0, _PLANE_H - 1]], np.float32),
            dst,
        )
    if case == "rotated_negative_steep":
        # Build a strongly clockwise camera-plane rotation (long axis rises to
        # the right), with a small projective skew superimposed.
        center = np.asarray([width / 2.0, height / 2.0], np.float64)
        theta = np.deg2rad(-34.0)
        along = np.asarray([np.cos(theta), np.sin(theta)])
        across = np.asarray([-np.sin(theta), np.cos(theta)])
        c = center
        half_l, half_w = 300.0, 92.0
        quad = np.asarray([
            c - half_l * along - half_w * across,
            c + half_l * along - half_w * across,
            c + half_l * along + half_w * across,
            c - half_l * along + half_w * across,
        ], np.float32)
        # Mild skew keeps this case rotated while avoiding an affine-only easy
        # transform.  The offset is chosen inward so all corners remain in frame.
        quad += np.asarray([[0, 0], [3, 5], [-4, 3], [-1, -4]], np.float32)
        return cv2.getPerspectiveTransform(
            np.asarray([[0, 0], [_PLANE_W - 1, 0], [_PLANE_W - 1, _PLANE_H - 1], [0, _PLANE_H - 1]], np.float32),
            quad,
        )
    if case == "partial_longitudinal_crop":
        # The nut-side end lies beyond the left image boundary.  The last
        # roughly 15% of the board remains outside the visible frame.
        dst = np.asarray([[-92, 153], [618, 116], [618, 363], [-92, 334]], np.float32)
        return cv2.getPerspectiveTransform(
            np.asarray([[0, 0], [_PLANE_W - 1, 0], [_PLANE_W - 1, _PLANE_H - 1], [0, _PLANE_H - 1]], np.float32),
            dst,
        )
    return cv2.getPerspectiveTransform(
        np.asarray([[0, 0], [_PLANE_W - 1, 0], [_PLANE_W - 1, _PLANE_H - 1], [0, _PLANE_H - 1]], np.float32),
        np.asarray([[72, 142], [688, 142], [688, 358], [72, 358]], np.float32),
    )


def _project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(len(flat)))) @ homography.T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    return projected.reshape(np.asarray(points).shape)


def _apply_mask_distortions(mask: np.ndarray, case: str) -> np.ndarray:
    result = mask.copy()
    h, w = result.shape
    if case == "mask_notch_spur":
        # A deep but localized side notch and a thin outward spur model common
        # segmentation artifacts while preserving most of the board support.
        cv2.rectangle(result, (w // 2 - 10, h // 2 - 30), (w // 2 + 9, h // 2 + 10), 0, -1)
        cv2.fillPoly(result, [np.asarray([[w // 3, h // 2 - 3], [w // 3 - 22, h // 2 - 5], [w // 3, h // 2 + 4]], np.int32)], 255)
    return result


def make_case(
    case: str,
    *,
    fret_count: int = 12,
    num_strings: int = 6,
    seed: int = 11,
    output_size: Tuple[int, int] = _FRAME_SIZE,
) -> SyntheticFretboard:
    """Render a named deterministic scene.

    Supported cases are ``clean_frontal``, ``perspective``,
    ``rotated_negative_steep``, ``hand_occlusion``, ``mask_notch_spur``,
    ``partial_longitudinal_crop``, and ``blank_no_evidence``.
    """
    supported = {
        "clean_frontal", "perspective", "rotated_negative_steep",
        "hand_occlusion", "mask_notch_spur", "partial_longitudinal_crop",
        "blank_no_evidence",
    }
    if case not in supported:
        raise ValueError(f"unknown synthetic case {case!r}; choose from {sorted(supported)}")
    if num_strings < 2:
        raise ValueError("num_strings must be at least two")

    plane_image, plane_mask, fret_x, plane_nodes, _ = _render_plane(fret_count, num_strings, seed)
    width, height = (int(output_size[0]), int(output_size[1]))
    hmat = _camera_homography(case, (width, height))
    image = cv2.warpPerspective(plane_image, hmat, (width, height), flags=cv2.INTER_LINEAR, borderValue=(26, 27, 29))
    mask = cv2.warpPerspective(plane_mask, hmat, (width, height), flags=cv2.INTER_NEAREST)
    nodes = _project_points(plane_nodes, hmat).astype(np.float32)
    source_frets = _project_points(np.stack((fret_x, np.full_like(fret_x, 0.0)), axis=-1), hmat)  # x reference only

    occlusion = np.zeros((height, width), dtype=np.uint8)
    if case == "hand_occlusion":
        # Draw one palm plus two broad fingers in source-plane coordinates,
        # then warp both color and the exact binary occlusion label.
        hand = np.zeros((_PLANE_H, _PLANE_W), dtype=np.uint8)
        cv2.ellipse(hand, (690, 160), (108, 61), -8, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.fillPoly(hand, [np.asarray([[632, 119], [687, 74], [719, 82], [684, 132]], np.int32)], 255)
        cv2.fillPoly(hand, [np.asarray([[702, 119], [761, 83], [789, 99], [739, 145]], np.int32)], 255)
        plane_occlusion = cv2.bitwise_and(hand, plane_mask)
        occlusion = cv2.warpPerspective(plane_occlusion, hmat, (width, height), flags=cv2.INTER_NEAREST)
        hand_rgb = np.zeros_like(plane_image)
        # Warm skin-like shading over the foreground region.
        hand_rgb[:] = (137, 119, 105)
        cv2.ellipse(hand_rgb, (660, 138), (95, 50), -8, 0, 360, (158, 137, 121), -1, cv2.LINE_AA)
        cv2.fillPoly(hand_rgb, [np.asarray([[632, 119], [687, 74], [719, 82], [684, 132]], np.int32)], (164, 142, 126))
        cv2.fillPoly(hand_rgb, [np.asarray([[702, 119], [761, 83], [789, 99], [739, 145]], np.int32)], (151, 130, 116))
        plane_image[hand > 0] = hand_rgb[hand > 0]
        image = cv2.warpPerspective(plane_image, hmat, (width, height), flags=cv2.INTER_LINEAR, borderValue=(26, 27, 29))

    if case == "mask_notch_spur":
        mask = _apply_mask_distortions(mask, case)
    if case == "blank_no_evidence":
        image[:] = (31, 32, 34)
        mask[:] = 0

    xcoords = nodes[..., 0]
    ycoords = nodes[..., 1]
    visible = (
        (xcoords >= 0) & (xcoords < width) & (ycoords >= 0) & (ycoords < height)
    )
    ix = np.clip(np.rint(xcoords).astype(int), 0, width - 1)
    iy = np.clip(np.rint(ycoords).astype(int), 0, height - 1)
    visible &= mask[iy, ix] > 0
    visible &= occlusion[iy, ix] == 0
    if case == "blank_no_evidence":
        visible[:] = False

    description = {
        "clean_frontal": "frontal view of tapered board with unobstructed strings and frets",
        "perspective": "strong perspective foreshortening with converging transverse geometry",
        "rotated_negative_steep": "strong clockwise rotation with a small projective skew",
        "hand_occlusion": "foreground hand occludes a known interior region",
        "mask_notch_spur": "otherwise clean view with a segmentation notch and outward spur",
        "partial_longitudinal_crop": "nut-side length of board is cropped by the camera frame",
        "blank_no_evidence": "blank frame and empty segmentation mask",
    }[case]

    return SyntheticFretboard(
        name=case,
        image=image,
        mask=mask,
        occlusion_mask=occlusion,
        source_nodes=nodes,
        visible_nodes=visible,
        fret_x=fret_x.astype(np.float32),
        homography=hmat.astype(np.float64),
        board_corners=_project_points(_plane_geometry(fret_count, num_strings)[2], hmat).astype(np.float32),
        num_strings=num_strings,
        fret_count=fret_count,
        description=description,
    )


def make_cases(
    cases: Iterable[str] | None = None,
    *,
    fret_count: int = 12,
    num_strings: int = 6,
    seed: int = 11,
    output_size: Tuple[int, int] = _FRAME_SIZE,
) -> Dict[str, SyntheticFretboard]:
    """Return all standard scenes, or the requested subset, in stable order."""
    names = list(cases) if cases is not None else [
        "clean_frontal", "perspective", "rotated_negative_steep", "hand_occlusion",
        "mask_notch_spur", "partial_longitudinal_crop", "blank_no_evidence",
    ]
    return {
        name: make_case(name, fret_count=fret_count, num_strings=num_strings, seed=seed, output_size=output_size)
        for name in names
    }

