"""Topology-preserving piecewise-bilinear maps for fretboard grids.

``source_nodes[j, i]`` stores the observed image position of grid node
``(i, j)``.  ``target_x`` and ``target_y`` give the corresponding rectified
pixel coordinates.  The target axes must span the complete output image so
each output pixel belongs to a grid cell and no extrapolation is needed.

An occlusion mask is a source-image mask in which nonzero pixels are occluded.
When no such mask is supplied, ``visible_mask`` equals ``valid_mask``; this
means source sampling is in frame, while occlusion status is unknown.
"""

from __future__ import annotations

import operator
from typing import Any

import cv2
import numpy as np


_FLOAT_EPS = np.finfo(np.float64).eps


def _as_nodes(source_nodes: Any) -> np.ndarray:
    try:
        nodes = np.asarray(source_nodes, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_nodes must be a finite numeric (ny, nx, 2) array") from exc
    if nodes.ndim != 3 or nodes.shape[2] != 2 or nodes.shape[0] < 2 or nodes.shape[1] < 2:
        raise ValueError("source_nodes must have shape (ny, nx, 2) with ny,nx >= 2")
    if not np.isfinite(nodes).all():
        raise ValueError("source_nodes must contain only finite coordinates")
    return nodes


def _as_axis(values: Any, name: str, expected_length: int) -> np.ndarray:
    try:
        axis = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric 1D array") from exc
    if axis.ndim != 1 or axis.size != expected_length:
        raise ValueError(f"{name} must be 1D with length {expected_length}")
    if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} must be finite and strictly ascending")
    return axis


def _as_output_size(output_size: Any) -> tuple[int, int]:
    if not isinstance(output_size, (tuple, list, np.ndarray)):
        raise ValueError("output_size must be a pair (width, height) of integers")
    try:
        if len(output_size) != 2:
            raise ValueError("output_size must be a pair (width, height) of integers")
    except TypeError as exc:
        raise ValueError("output_size must be a pair (width, height) of integers") from exc
    try:
        width, height = (operator.index(v) for v in output_size)
    except TypeError as exc:
        raise ValueError("output_size must be a pair (width, height) of integers") from exc
    if width < 2 or height < 2:
        raise ValueError("output_size dimensions must be at least 2")
    return width, height


def _cell_coefficients(nodes: np.ndarray):
    """Return bilinear coefficients and verify every cell is well oriented."""
    p00 = nodes[:-1, :-1]
    p10 = nodes[:-1, 1:]
    p01 = nodes[1:, :-1]
    p11 = nodes[1:, 1:]
    b = p10 - p00
    c = p01 - p00
    d = p11 - p10 - p01 + p00

    # The Jacobian determinant is affine over a bilinear cell, so checking it
    # at its four corners proves it has a nonzero, constant sign everywhere.
    du0, du1 = b, b + d
    dv0, dv1 = c, c + d
    dets = np.stack(
        (
            _cross(du0, dv0),
            _cross(du0, dv1),
            _cross(du1, dv0),
            _cross(du1, dv1),
        ),
        axis=-1,
    )
    cell_edges = np.stack((b, c, b + d, c + d), axis=-2)
    scale = np.max(np.linalg.norm(cell_edges, axis=-1), axis=-1)
    tol = 64.0 * _FLOAT_EPS * np.maximum(scale * scale, np.finfo(np.float64).tiny)
    positive = np.all(dets > tol[..., None], axis=-1)
    negative = np.all(dets < -tol[..., None], axis=-1)
    if not np.all(positive | negative):
        raise ValueError("each source cell must be convex with a nonzero Jacobian (fold detected)")
    orientations = np.where(positive, 1, -1)
    if np.any(orientations != orientations.flat[0]):
        raise ValueError("source cells must have a consistent orientation (fold detected)")
    return p00, b, c, d


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _validate_grid(source_nodes: Any, target_x: Any, target_y: Any):
    nodes = _as_nodes(source_nodes)
    ny, nx, _ = nodes.shape
    x_axis = _as_axis(target_x, "target_x", nx)
    y_axis = _as_axis(target_y, "target_y", ny)
    coeffs = _cell_coefficients(nodes)
    # Positive local Jacobians alone do not exclude a mesh wrapping around
    # and overlapping itself. Require a simple outer boundary as well.
    boundary = np.concatenate((nodes[0], nodes[1:, -1], nodes[-1, -2::-1], nodes[-2:0:-1, 0]))
    scale = max(float(np.ptp(boundary, axis=0).max()), 1.)
    eps = 1e-10 * scale * scale
    for i in range(len(boundary)):
        a, b = boundary[i], boundary[(i+1) % len(boundary)]
        for j in range(i+2, len(boundary)):
            if i == 0 and j == len(boundary)-1:
                continue
            c, d = boundary[j], boundary[(j+1) % len(boundary)]
            if np.any(np.maximum(np.minimum(a, b), np.minimum(c, d)) >
                      np.minimum(np.maximum(a, b), np.maximum(c, d)) + 1e-9*scale):
                continue
            ab_c, ab_d = _cross(b-a, c-a), _cross(b-a, d-a)
            cd_a, cd_b = _cross(d-c, a-c), _cross(d-c, b-c)
            if ((ab_c <= eps and ab_d >= -eps) or (ab_d <= eps and ab_c >= -eps)) and \
               ((cd_a <= eps and cd_b >= -eps) or (cd_b <= eps and cd_a >= -eps)):
                raise ValueError("source boundary intersects itself (global fold detected)")
    return nodes, x_axis, y_axis, coeffs


def _check_full_coverage(x_axis: np.ndarray, y_axis: np.ndarray, width: int, height: int) -> None:
    if x_axis[0] != 0.0 or x_axis[-1] != width - 1:
        raise ValueError("target_x must start at 0 and end at output width - 1")
    if y_axis[0] != 0.0 or y_axis[-1] != height - 1:
        raise ValueError("target_y must start at 0 and end at output height - 1")


def _evaluate_cells(p00: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
                    cell_i: np.ndarray, cell_j: np.ndarray, u: np.ndarray,
                    v: np.ndarray) -> np.ndarray:
    origin = p00[cell_j, cell_i]
    along_u = b[cell_j, cell_i]
    along_v = c[cell_j, cell_i]
    twist = d[cell_j, cell_i]
    return origin + along_u * u[..., None] + along_v * v[..., None] + twist * (u * v)[..., None]


def build_grid_maps(source_nodes, target_x, target_y, output_size):
    """Build OpenCV destination-to-source maps for a bilinear grid warp.

    Args:
        source_nodes: Observed image nodes with shape ``(ny, nx, 2)``.
        target_x: Strictly ascending target x pixel coordinates, beginning at
            0 and ending at ``width - 1``.
        target_y: Strictly ascending target y pixel coordinates, beginning at
            0 and ending at ``height - 1``.
        output_size: ``(width, height)`` integers, each at least 2.

    Returns:
        ``(map_x, map_y)`` float32 arrays of shape ``(height, width)``.
    """
    width, height = _as_output_size(output_size)
    nodes, x_axis, y_axis, (p00, b, c, d) = _validate_grid(source_nodes, target_x, target_y)
    _check_full_coverage(x_axis, y_axis, width, height)

    x = np.arange(width, dtype=np.float64)
    y = np.arange(height, dtype=np.float64)
    cell_i = np.searchsorted(x_axis, x, side="right") - 1
    cell_j = np.searchsorted(y_axis, y, side="right") - 1
    cell_i = np.clip(cell_i, 0, x_axis.size - 2)
    cell_j = np.clip(cell_j, 0, y_axis.size - 2)
    u = (x - x_axis[cell_i]) / (x_axis[cell_i + 1] - x_axis[cell_i])
    v = (y - y_axis[cell_j]) / (y_axis[cell_j + 1] - y_axis[cell_j])
    ii, jj = np.meshgrid(cell_i, cell_j)
    uu, vv = np.meshgrid(u, v)
    mapped = _evaluate_cells(p00, b, c, d, ii, jj, uu, vv)
    return mapped[..., 0].astype(np.float32), mapped[..., 1].astype(np.float32)


def warp_grid(image, source_nodes, target_x, target_y, output_size, occlusion_mask=None):
    """Warp a grayscale or RGB image through the observed bilinear grid.

    The returned ``valid_mask`` records whether the source sampling coordinate
    lies inside the source image bounds, including exact border pixels.
    ``visible_mask`` additionally removes pixels mapped to nonzero entries of
    ``occlusion_mask``. With no occlusion mask, visibility is unknown and the
    returned visible mask equals the geometric valid mask.
    """
    source = np.asarray(image)
    if source.ndim not in (2, 3) or source.shape[0] < 1 or source.shape[1] < 1:
        raise ValueError("image must be a nonempty grayscale or RGB image")
    if source.ndim == 3 and source.shape[2] != 3:
        raise ValueError("color image must have exactly 3 RGB channels")
    width, height = _as_output_size(output_size)
    map_x, map_y = build_grid_maps(source_nodes, target_x, target_y, (width, height))
    ih, iw = source.shape[:2]
    valid = (map_x >= 0.0) & (map_x <= iw - 1) & (map_y >= 0.0) & (map_y <= ih - 1)
    warped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if occlusion_mask is None:
        visible = valid.copy()
    else:
        occ = np.asarray(occlusion_mask)
        if occ.ndim != 2 or occ.shape != (ih, iw):
            raise ValueError("occlusion_mask must be a 2D mask matching the source image")
        try:
            finite_mask = np.isfinite(occ).all()
        except TypeError as exc:
            raise ValueError("occlusion_mask must contain boolean or numeric values") from exc
        if not finite_mask:
            raise ValueError("occlusion_mask must contain only finite values")
        # Any occluded contributor to the bilinear sampling footprint makes
        # that output pixel unavailable, even if its nearest pixel is clear.
        x0 = np.clip(np.floor(map_x).astype(int), 0, iw-1)
        x1 = np.clip(np.ceil(map_x).astype(int), 0, iw-1)
        y0 = np.clip(np.floor(map_y).astype(int), 0, ih-1)
        y1 = np.clip(np.ceil(map_y).astype(int), 0, ih-1)
        warped_occ = (occ[y0, x0] != 0) | (occ[y0, x1] != 0) | (occ[y1, x0] != 0) | (occ[y1, x1] != 0)
        visible = valid & ~warped_occ
    return {
        "image": warped,
        "map_x": map_x,
        "map_y": map_y,
        "valid_mask": valid,
        "visible_mask": visible,
    }


def _point_array(points: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be a numeric array with final dimension 2") from exc
    if arr.ndim == 0 or arr.shape[-1] != 2:
        raise ValueError("points must have final dimension 2")
    return arr.reshape((-1, 2)), arr.shape


def grid_to_image(points, source_nodes, target_x, target_y):
    """Map arbitrary target-grid coordinates to source-image coordinates.

    Points outside the target grid or with non-finite coordinates map to NaN.
    The input shape is preserved, with the final coordinate dimension 2.
    """
    flat, shape = _point_array(points)
    nodes, x_axis, y_axis, (p00, b, c, d) = _validate_grid(source_nodes, target_x, target_y)
    result = np.full((flat.shape[0], 2), np.nan, dtype=np.float64)
    finite = np.isfinite(flat).all(axis=1)
    inside = finite & (flat[:, 0] >= x_axis[0]) & (flat[:, 0] <= x_axis[-1])
    inside &= (flat[:, 1] >= y_axis[0]) & (flat[:, 1] <= y_axis[-1])
    indices = np.flatnonzero(inside)
    if indices.size:
        pts = flat[indices]
        ii = np.searchsorted(x_axis, pts[:, 0], side="right") - 1
        jj = np.searchsorted(y_axis, pts[:, 1], side="right") - 1
        ii = np.clip(ii, 0, x_axis.size - 2)
        jj = np.clip(jj, 0, y_axis.size - 2)
        u = (pts[:, 0] - x_axis[ii]) / (x_axis[ii + 1] - x_axis[ii])
        v = (pts[:, 1] - y_axis[jj]) / (y_axis[jj + 1] - y_axis[jj])
        result[indices] = _evaluate_cells(p00, b, c, d, ii, jj, u, v)
    return result.reshape(shape)


def _invert_cell(point: np.ndarray, p00: np.ndarray, b: np.ndarray,
                 c: np.ndarray, d: np.ndarray, residual_tol: float):
    """Find a point's bilinear parameters; return None if outside this cell."""
    # The affine part gives a strong starting point for mildly skewed cells;
    # clipping still provides a safe seed for points near a quadrilateral tip.
    try:
        seed = np.linalg.solve(np.column_stack((b, c)), point - p00)
    except np.linalg.LinAlgError:
        seed = np.array((0.5, 0.5), dtype=np.float64)
    uv = np.clip(seed, -0.25, 1.25)
    for _ in range(24):
        u, v = uv
        estimate = p00 + b * u + c * v + d * u * v
        residual = estimate - point
        if np.linalg.norm(residual, ord=np.inf) <= residual_tol:
            break
        du = b + d * v
        dv = c + d * u
        jac = np.column_stack((du, dv))
        try:
            step = np.linalg.solve(jac, residual)
        except np.linalg.LinAlgError:
            return None
        uv -= step
        if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e6:
            return None
    u, v = uv
    estimate = p00 + b * u + c * v + d * u * v
    scale = max(
        float(np.linalg.norm(b)),
        float(np.linalg.norm(c)),
        float(np.linalg.norm(d)),
        np.finfo(np.float64).tiny,
    )
    uv_tol = max(2e-10, 8.0 * residual_tol / scale)
    if np.linalg.norm(estimate - point, ord=np.inf) > residual_tol:
        return None
    if u < -uv_tol or u > 1.0 + uv_tol or v < -uv_tol or v > 1.0 + uv_tol:
        return None
    return min(1.0, max(0.0, u)), min(1.0, max(0.0, v))


def image_to_grid(points, source_nodes, target_x, target_y):
    """Invert the piecewise-bilinear map from image points to target grid.

    Bilinear inversion is solved independently in candidate convex cells using
    Newton iterations. Points outside every source cell or with non-finite
    coordinates map to NaN. The input shape is preserved.
    """
    flat, shape = _point_array(points)
    nodes, x_axis, y_axis, (p00, b, c, d) = _validate_grid(source_nodes, target_x, target_y)
    ny, nx, _ = nodes.shape
    result = np.full((flat.shape[0], 2), np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(flat).all(axis=1))
    # Candidate cells are screened by their bounding boxes before Newton
    # inversion. Grid sizes in fretboards are small, keeping this transparent.
    for point_index in finite_indices:
        point = flat[point_index]
        for j in range(ny - 1):
            for i in range(nx - 1):
                corners = np.array((nodes[j, i], nodes[j, i + 1], nodes[j + 1, i], nodes[j + 1, i + 1]))
                scale = max(float(np.ptp(corners[:, 0])), float(np.ptp(corners[:, 1])))
                coordinate_scale = float(np.max(np.abs(corners)))
                residual_tol = max(
                    2e-10 * scale,
                    64.0 * _FLOAT_EPS * coordinate_scale,
                    np.finfo(np.float64).tiny,
                )
                if np.any(point < corners.min(axis=0) - residual_tol) or np.any(point > corners.max(axis=0) + residual_tol):
                    continue
                uv = _invert_cell(point, p00[j, i], b[j, i], c[j, i], d[j, i], residual_tol)
                if uv is not None:
                    u, v = uv
                    result[point_index] = (
                        x_axis[i] + u * (x_axis[i + 1] - x_axis[i]),
                        y_axis[j] + v * (y_axis[j + 1] - y_axis[j]),
                    )
                    break
            if np.isfinite(result[point_index]).all():
                break
    return result.reshape(shape)
