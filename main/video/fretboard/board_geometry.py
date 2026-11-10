"""Robust, image-coordinate-neutral coarse geometry for a fretboard mask.

The returned corners are ordered ``[tl, tr, br, bl]``.  The longitudinal
axis points toward positive image x (or positive image y for a near-vertical
board); the across-board axis points downward when it has a vertical
component.  This ordering is geometric and does not guess which end is the
nut.
"""

from __future__ import annotations

import cv2
import numpy as np


_MIN_MASK_PIXELS = 20
_MAX_BINS = 64


def _oriented_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return robust center, longitudinal, and across axes for x/y points."""
    if len(points) < _MIN_MASK_PIXELS:
        return None

    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return None

    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    if values[order[-1]] <= 1e-6:
        return None
    # A nearly round component does not define a stable board direction.
    minor = max(float(values[order[0]]), 1e-6)
    if float(values[order[-1]]) / minor < 1.2:
        return None

    longitudinal = vectors[:, order[-1]].astype(np.float64)
    if abs(longitudinal[0]) >= 0.12:
        if longitudinal[0] < 0:
            longitudinal = -longitudinal
    elif longitudinal[1] < 0:
        longitudinal = -longitudinal

    across = np.array([-longitudinal[1], longitudinal[0]], dtype=np.float64)
    if across[1] < -1e-8 or (abs(across[1]) <= 1e-8 and across[0] < 0):
        across = -across
    return center, longitudinal, across


def _significant_mask(mask: np.ndarray) -> np.ndarray | None:
    """Keep the main board and nearby, substantial disconnected fragments."""
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    min_area = max(8, int(np.ceil(largest_area * 0.001)))

    ys, xs = np.where(labels == largest_label)
    reference = np.column_stack((xs, ys)).astype(np.float64)
    axes = _oriented_axes(reference)
    if axes is None:
        return None
    origin, longitudinal, across = axes
    ref_u = (reference - origin) @ longitudinal
    ref_v = (reference - origin) @ across
    u_min, u_max = np.percentile(ref_u, [0.5, 99.5])
    v_min, v_max = np.percentile(ref_v, [1.0, 99.0])
    ref_width = max(float(v_max - v_min), 2.0)
    ref_length = max(float(u_max - u_min), 2.0)
    ref_v_center = float(np.median(ref_v))

    keep = np.zeros(count, dtype=bool)
    keep[largest_label] = True
    for label in range(1, count):
        if label == largest_label or stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, width, height = stats[label, :4]
        comp_corners = np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float64,
        )
        projected = (comp_corners - origin) @ np.column_stack((longitudinal, across))
        comp_u_min, comp_u_max = projected[:, 0].min(), projected[:, 0].max()
        comp_v = (centers[label] - origin) @ across
        gap = max(u_min - comp_u_max, comp_u_min - u_max, 0.0)
        if (
            abs(comp_v - ref_v_center) <= 1.25 * ref_width
            and gap <= max(1.5 * ref_width, 0.15 * ref_length)
        ):
            keep[label] = True

    return (keep[labels]).astype(np.uint8)


def _bin_envelope(primary: np.ndarray, secondary: np.ndarray, bins: int,
                  low_q: float, high_q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize low/high secondary-coordinate envelopes in primary bins."""
    lo, hi = float(primary.min()), float(primary.max())
    if hi - lo <= 1e-6:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    edges = np.linspace(lo, hi, bins + 1)
    indices = np.searchsorted(edges, primary, side="right") - 1
    indices = np.clip(indices, 0, bins - 1)

    centers_out: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for index in range(bins):
        selected = indices == index
        if np.count_nonzero(selected) < 2:
            continue
        p = primary[selected]
        s = secondary[selected]
        centers_out.append(float(np.median(p)))
        lows.append(float(np.quantile(s, low_q)))
        highs.append(float(np.quantile(s, high_q)))
    return (np.asarray(centers_out), np.asarray(lows), np.asarray(highs))


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """Fit ``y = slope*x + intercept`` with median initialization and Huber IRLS."""
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[good], dtype=np.float64), np.asarray(y[good], dtype=np.float64)
    if len(x) < 4 or np.ptp(x) <= 1e-6:
        return None

    # Theil-Sen initialization resists isolated mask notches and spurs.
    slopes: list[np.ndarray] = []
    span = float(np.ptp(x))
    min_dx = max(1e-6, span * 0.12)
    for i in range(len(x) - 1):
        dx = x[i + 1:] - x[i]
        usable = np.abs(dx) >= min_dx
        if np.any(usable):
            slopes.append((y[i + 1:][usable] - y[i]) / dx[usable])
    if not slopes:
        return None
    slope = float(np.median(np.concatenate(slopes)))
    intercept = float(np.median(y - slope * x))

    design = np.column_stack((x, np.ones_like(x)))
    for _ in range(12):
        residual = y - (slope * x + intercept)
        median_residual = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median_residual)))
        delta = max(1.0, 1.345 * scale)
        weights = np.ones_like(residual)
        outside = np.abs(residual - median_residual) > delta
        weights[outside] = delta / np.abs(residual[outside] - median_residual)
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_y = y * np.sqrt(weights)
        try:
            next_slope, next_intercept = np.linalg.lstsq(
                weighted_design, weighted_y, rcond=None
            )[0]
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite([next_slope, next_intercept]).all():
            return None
        if abs(next_slope - slope) + abs(next_intercept - intercept) < 1e-5:
            slope, intercept = float(next_slope), float(next_intercept)
            break
        slope, intercept = float(next_slope), float(next_intercept)

    residual = np.abs(y - (slope * x + intercept))
    median_residual = float(np.median(residual))
    mad = 1.4826 * float(np.median(np.abs(residual - median_residual)))
    inliers = residual <= max(2.0, median_residual + 2.5 * mad)
    if np.count_nonzero(inliers) < 4:
        return None
    return slope, intercept, inliers


def _intersect_lines(rail: tuple[float, float], cap: tuple[float, float]) -> np.ndarray | None:
    """Intersect a rail ``v=a*u+b`` with an end cap ``u=c*v+d``."""
    rail_slope, rail_intercept = rail
    cap_slope, cap_intercept = cap
    denominator = 1.0 - cap_slope * rail_slope
    if abs(denominator) < 1e-5:
        return None
    u = (cap_slope * rail_intercept + cap_intercept) / denominator
    v = rail_slope * u + rail_intercept
    point = np.array([u, v], dtype=np.float64)
    return point if np.isfinite(point).all() else None


def estimate_board_corners(mask: np.ndarray) -> np.ndarray | None:
    """Estimate four robust fretboard corners from a binary segmentation mask.

    The function tolerates interior holes, local boundary notches, and small
    detached components. Significant fragments near the main board are kept;
    isolated tiny components and outlying envelope bins are suppressed. The
    result is a ``float32`` array ordered ``[tl, tr, br, bl]`` in image x/y
    coordinates, or ``None`` when the mask does not define stable geometry.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2 or min(arr.shape, default=0) < 2:
        return None
    binary = (arr > 0).astype(np.uint8)
    if int(binary.sum()) < _MIN_MASK_PIXELS:
        return None

    selected = _significant_mask(binary)
    if selected is None:
        return None
    ys, xs = np.where(selected > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)
    axes = _oriented_axes(points)
    if axes is None:
        return None
    origin, longitudinal, across = axes

    # A well-supported four-sided convex outline contains better endpoint
    # information than PCA-coordinate envelopes on a strongly tapered board.
    # In particular, rows outside the narrow end only see a side rail, not an
    # end cap. Avoid fitting those side-rail samples as an end cap.
    hull = cv2.convexHull(points.astype(np.float32))
    perimeter = cv2.arcLength(hull, True)
    for fraction in (.003, .006, .01, .015):
        poly = cv2.approxPolyDP(hull, fraction * perimeter, True).reshape(-1, 2)
        if len(poly) != 4:
            continue
        uv = (poly - origin) @ np.column_stack((longitudinal, across))
        start = np.argsort(uv[:, 0])[:2]
        end = np.argsort(uv[:, 0])[2:]
        start = start[np.argsort(uv[start, 1])]
        end = end[np.argsort(uv[end, 1])]
        ordered = poly[[start[0], end[0], end[1], start[1]]]
        area = abs(cv2.contourArea(ordered.astype(np.float32)))
        if area < 20 or not cv2.isContourConvex(ordered.astype(np.float32)):
            continue
        # Check hull discrepancy and actual mask coverage, so a large spur
        # cannot cheaply become a new corner of the fretboard.
        hull_area = cv2.contourArea(hull)
        if abs(hull_area-area)/max(area, 1) > .025:
            continue
        filled = np.zeros_like(binary)
        cv2.fillConvexPoly(filled, np.rint(ordered).astype(np.int32), 1)
        intersection = np.count_nonzero(filled & selected)
        if intersection / max(np.count_nonzero(filled), 1) < .83:
            continue
        if intersection / max(np.count_nonzero(selected), 1) < .97:
            continue
        return ordered.astype(np.float32)

    # Limit work for very large masks without favoring one image region.
    if len(points) > 120_000:
        sample_indices = np.linspace(0, len(points) - 1, 120_000, dtype=np.int64)
        fit_points = points[sample_indices]
    else:
        fit_points = points

    projected = (points - origin) @ np.column_stack((longitudinal, across))
    fit_projected = (fit_points - origin) @ np.column_stack((longitudinal, across))
    u, v = projected[:, 0], projected[:, 1]
    fit_u, fit_v = fit_projected[:, 0], fit_projected[:, 1]
    length = float(np.ptp(u))
    width = float(np.ptp(v))
    if length < 5.0 or width < 2.0 or length / max(width, 1.0) < 1.25:
        return None

    longitudinal_bins = min(_MAX_BINS, max(12, int(np.ceil(length / max(width * 0.08, 2.0)))))
    rail_u, top_values, bottom_values = _bin_envelope(
        fit_u, fit_v, longitudinal_bins, 0.02, 0.98
    )
    if len(rail_u) < 6:
        return None
    widths = bottom_values - top_values
    median_width = float(np.median(widths))
    if median_width <= 1.0:
        return None
    rail_bins = widths >= max(2.0, 0.35 * median_width)
    top_fit = _robust_line(rail_u[rail_bins], top_values[rail_bins])
    bottom_fit = _robust_line(rail_u[rail_bins], bottom_values[rail_bins])
    if top_fit is None or bottom_fit is None:
        return None
    top_line = top_fit[:2]
    bottom_line = bottom_fit[:2]

    # Fit the two end caps from the opposite pair of binned silhouette
    # envelopes. This lets a supported slanted end remain slanted.
    across_bins = min(_MAX_BINS, max(12, int(np.ceil(width / 2.0))))
    cap_v, start_values, end_values = _bin_envelope(
        fit_v, fit_u, across_bins, 0.01, 0.99
    )
    if len(cap_v) < 6:
        return None
    start_fit = _robust_line(cap_v, start_values)
    end_fit = _robust_line(cap_v, end_values)
    if start_fit is None or end_fit is None:
        return None
    start_line = (start_fit[0], start_fit[1])  # u = c*v + d
    end_line = (end_fit[0], end_fit[1])

    tl_uv = _intersect_lines(top_line, start_line)
    tr_uv = _intersect_lines(top_line, end_line)
    br_uv = _intersect_lines(bottom_line, end_line)
    bl_uv = _intersect_lines(bottom_line, start_line)
    if any(point is None for point in (tl_uv, tr_uv, br_uv, bl_uv)):
        return None

    uv = np.stack((tl_uv, tr_uv, br_uv, bl_uv))
    corners = origin + uv[:, :1] * longitudinal + uv[:, 1:] * across
    if not np.isfinite(corners).all():
        return None

    edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    if np.min(edge_lengths) < 2.0:
        return None
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if area < 20.0:
        return None

    return corners.astype(np.float32)
