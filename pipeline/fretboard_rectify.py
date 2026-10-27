"""Evidence-constrained fretboard atlas (EFA).

Segmentation proposes a region, image line evidence supplies the internal grid,
and a checked bilinear mesh maps that grid into a rectangle. This module does
not assume that a mask endpoint is the nut, or hallucinate 21 observed frets.
Images are RGB uint8; masks are nonzero foreground. No model is loaded here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares

from .board_geometry import estimate_board_corners
from .grid_warp import warp_grid, image_to_grid, grid_to_image
from .string_evidence import detect_strings


@dataclass
class RectificationResult:
    image: np.ndarray
    source_nodes: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    valid_mask: np.ndarray
    visible_mask: np.ndarray
    quality: dict
    coarse_transform: np.ndarray

    def image_to_grid(self, points):
        return image_to_grid(points, self.source_nodes, self.target_x, self.target_y)

    def grid_to_image(self, points):
        return grid_to_image(points, self.source_nodes, self.target_x, self.target_y)

    def to_dict(self):
        return dict(schema_version="2.0", method="evidence_constrained_atlas",
                    size=[self.image.shape[1], self.image.shape[0]],
                    source_nodes=self.source_nodes.tolist(),
                    target_x=self.target_x.tolist(), target_y=self.target_y.tolist(),
                    quality=self.quality,
                    coarse_transform=self.coarse_transform.tolist(),
                    mapping="piecewise_bilinear; coarse_transform is NOT the final mapping")


def _robust_line(x, y, weights=None):
    """Huber fit of y = slope*x + intercept with deterministic initialization."""
    x, y = np.asarray(x), np.asarray(y)
    w = np.ones(len(x)) if weights is None else np.asarray(weights)
    design = np.column_stack([x, np.ones(len(x))])
    coef = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    for _ in range(8):
        residual = y - design @ coef
        scale = max(0.002, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        rw = w * np.minimum(1, 1.5 * scale / np.maximum(np.abs(residual), 1e-9))
        coef = np.linalg.lstsq(design * np.sqrt(rw[:, None]), y * np.sqrt(rw), rcond=None)[0]
    return coef


def _line_candidates(gray, usable, axis):
    """Aggregate both sides of metal wires, preserving their transverse slope.

    Each candidate is z(t)=a*(t-.5)+b in unit coarse coordinates. Frets use
    (t,z)=(y,x), strings use (x,y). Coverage is union support, not line count.
    """
    h, w = gray.shape
    segments = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if segments is None:
        return []
    groups = []
    candidates = []
    for segment in segments[:, 0]:
        p = segment.reshape(2, 2) / [w - 1, h - 1]
        t, z = (p[:, 1], p[:, 0]) if axis == "frets" else (p[:, 0], p[:, 1])
        dt = t[1] - t[0]
        if abs(dt) < (0.075 if axis == "frets" else 0.035):
            continue
        a = (z[1] - z[0]) / dt
        if abs(a) > (0.30 if axis == "frets" else 0.50):
            continue
        b = z[0] - a * (t[0] - .5)
        if not .015 < b < .985:
            continue
        samples = np.linspace(p[0], p[1], 15) * [w - 1, h - 1]
        indices = np.rint(samples).astype(int)
        indices[:, 0] = np.clip(indices[:, 0], 0, w - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, h - 1)
        if np.mean(usable[indices[:, 1], indices[:, 0]] > 0) < .8:
            continue
        candidates.append((float(b), float(a), t.copy(), z.copy()))
    # A fixed coarse pixel tolerance merges the two edges of a single wire.
    tolerance = (12. / (w - 1)) if axis == "frets" else (9. / (h - 1))
    for c in sorted(candidates, key=lambda v: v[0]):
        compatible = [g for g in groups if abs(c[0] - np.median([v[0] for v in g])) < tolerance
                      and abs(c[1] - np.median([v[1] for v in g])) < .09]
        if compatible:
            min(compatible, key=lambda g: abs(c[0] - np.median([v[0] for v in g]))).append(c)
        else:
            groups.append([c])
    output = []
    for group in groups:
        ts = np.concatenate([v[2] for v in group])
        zs = np.concatenate([v[3] for v in group])
        # Estimate direction from segment DIRECTIONS. Regressing all pooled
        # endpoints creates a false tilt when one edge of a thick wire is
        # visible on the left and its other edge is visible on the right.
        weights = np.array([abs(v[2][1]-v[2][0]) for v in group])
        slopes = np.array([v[1] for v in group])
        order = np.argsort(slopes)
        a = slopes[order[np.searchsorted(np.cumsum(weights[order]), weights.sum()/2)]]
        intercept = np.average([v[0] for v in group], weights=weights)
        occupied = np.zeros(100, bool)
        for c in group:
            lo, hi = sorted(np.clip(np.rint(c[2] * 99).astype(int), 0, 99))
            occupied[lo:hi + 1] = True
        coverage = float(occupied.mean())
        if coverage < (.28 if axis == "frets" else .20):
            continue
        output.append(dict(a=float(a), b=float(intercept), support=coverage,
                           residual=float(np.median(np.abs(zs - (a * (ts - .5) + intercept))))))
    return sorted(output, key=lambda v: v["b"])


def _fret_consensus(lines):
    """Find a common transverse line pencil; hand contours are outliers."""
    if len(lines) < 3:
        return []
    x = np.array([p["b"] for p in lines])
    slopes = np.array([p["a"] for p in lines])
    weights = np.array([p["support"] for p in lines])
    best = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if abs(x[j] - x[i]) < .10:
                continue
            slope = (slopes[j] - slopes[i]) / (x[j] - x[i])
            intercept = slopes[i] - slope * x[i]
            inliers = np.abs(slopes - (slope * x + intercept)) < .018
            score = weights[inliers].sum()
            if best is None or score > best[0]:
                best = score, inliers
    if best is None or best[1].sum() < 3:
        return []
    keep = best[1]
    model = _robust_line(x[keep], slopes[keep], weights[keep])
    result = []
    for i in np.flatnonzero(keep):
        item = dict(lines[i])
        # Pool line direction but keep individually measured fret locations.
        item["a"] = float(model[0] * item["b"] + model[1])
        result.append(item)
    return result


def _refine_fret_centers(gray, usable, lines):
    """Refit wire centers from independent horizontal strips, not their edges.

    LSD provides capture ranges. Narrow-minus-broad horizontal filtering then
    measures bright metal ridges in separate height bands, reducing the bias
    from a different wire edge being visible in each part of the image.
    """
    if len(lines) < 3:
        return lines
    h, w = gray.shape
    floating = gray.astype(np.float32)
    ridge = cv2.GaussianBlur(floating, (0, 0), 0.7, sigmaY=.5) - cv2.GaussianBlur(floating, (0, 0), 5., sigmaY=.5)
    refined = []
    for i, line in enumerate(lines):
        neighbors = [abs(line["b"]-other["b"]) for k, other in enumerate(lines) if k != i]
        radius = max(3, min(10, int(.30*min(neighbors)*(w-1))))
        ts, zs, weights = [], [], []
        for yn in np.linspace(.06, .94, 22):
            yc = int(round(yn*(h-1)))
            xc = int(round((line["a"]*(yn-.5)+line["b"])*(w-1)))
            x0, x1 = max(1, xc-radius), min(w-1, xc+radius+1)
            y0, y1 = max(0, yc-3), min(h, yc+4)
            if x1-x0 < 3 or usable[yc, np.clip(xc, 0, w-1)] == 0:
                continue
            profile = np.median(ridge[y0:y1, x0:x1], axis=0)
            peak = int(np.argmax(profile))
            if peak == 0 or peak == len(profile)-1 or profile[peak] < 3.:
                continue
            left, mid, right = profile[peak-1:peak+2]
            denom = left-2*mid+right
            offset = float(np.clip(.5*(left-right)/denom, -.5, .5)) if abs(denom)>1e-6 else 0.
            ts.append(yn-.5)
            zs.append((x0+peak+offset)/(w-1))
            weights.append(float(np.clip(mid, 3, 30)))
        if len(ts) >= 10:
            a, b = _robust_line(ts, zs, weights)
            residual = float(np.median(abs(np.asarray(zs)-(a*np.asarray(ts)+b))))
            if residual < .004 and abs(a-line["a"]) < .04:
                line = dict(line, a=float(a), b=float(b), residual=residual)
        refined.append(line)
    return _fret_consensus(refined)


def _string_consensus(lines, count):
    """Select an ordered equal-spacing comb with independent line support.

    No forced K-means clusters and no synthetic string lines: every selected
    string must have substantial image evidence. Ambiguous/missing strings
    degrade the output to a coarse preview instead of certifying a grid.
    """
    if len(lines) < count:
        return []
    b = np.array([v["b"] for v in lines])
    a = np.array([v["a"] for v in lines])
    supports = np.array([v["support"] for v in lines])
    best = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            step = (b[j] - b[i]) / (count - 1)
            if not .50 < b[j] - b[i] < .96:
                continue
            targets = b[i] + np.arange(count) * step
            inds = np.argmin(abs(targets[:, None] - b[None, :]), axis=1)
            residuals = abs(b[inds] - targets)
            if len(set(inds)) != count or np.max(residuals) > .20 * step:
                continue
            # A perspective line family has slope affine in center intercept.
            fit = _robust_line(b[inds], a[inds], supports[inds])
            slope_error = abs(a[inds] - (fit[0] * b[inds] + fit[1]))
            if np.max(slope_error) > .06:
                continue
            score = supports[inds].sum() - 8 * residuals.sum() - 3 * slope_error.sum()
            if best is None or score > best[0]:
                best = score, inds
    if best is None:
        return []
    selected = [dict(lines[i]) for i in best[1]]
    # Pool the string direction pencil, avoiding independent noisy tilts from
    # one-sided specular glints or small fragments near fret crossings.
    bs = np.array([v["b"] for v in selected])
    slopes = np.array([v["a"] for v in selected])
    fit = _robust_line(bs, slopes, [v["support"] for v in selected])
    for item in selected:
        item["a"] = float(fit[0]*item["b"]+fit[1])
    return selected


def _fret_lattice(lines):
    """Fit a projective equal-temperament sequence, allowing internal gaps.

    Absolute fret number is unidentifiable without a nut/number anchor. Both
    directions are tried; these indices describe a relative sequence only.
    Fits are accepted only when residuals are small relative to local spacing.
    """
    if len(lines) < 5:
        return lines, np.arange(len(lines), dtype=float), False, None
    x = np.array([v["b"] for v in lines])
    gaps = np.diff(x)
    # Estimate local spacing from neighbors so a single occluded run can skip.
    base = np.array([np.median(gaps[max(0, i-2):min(len(gaps), i+3)]) for i in range(len(gaps))])
    steps = np.clip(np.rint(gaps / np.maximum(base, .001)), 1, 4).astype(int)
    indices = np.r_[0, np.cumsum(steps)]
    best = None
    for sign in [-1, 1]:
        q = 2. ** (sign * indices / 12.)
        # x=(a+b*q)/(1+c*q); bounded pole avoids invalid projectivities.
        cmin = -.90 / q.max()
        def predict(p, q=q):
            return (p[0] + p[1] * q) / (1 + p[2] * q)
        initial = np.linalg.lstsq(np.column_stack([np.ones(len(x)), q]), x, rcond=None)[0]
        fit = least_squares(lambda p: predict(p) - x, [*initial, 0],
                            bounds=([-100, -100, cmin], [100, 100, 20]),
                            loss="soft_l1", f_scale=.003, max_nfev=150)
        error = float(np.sqrt(np.mean((predict(fit.x)-x)**2)))
        if best is None or error < best[0]:
            best = error, sign, fit.x
    accepted = best[0] < min(.008, .12 * np.median(gaps))
    if not accepted:
        return lines, np.arange(len(lines), dtype=float), False, best[0]
    full = []
    direction_model = _robust_line(x, np.array([v["a"] for v in lines]))
    for n in range(int(indices[-1]) + 1):
        found = np.flatnonzero(indices == n)
        if found.size:
            item = dict(lines[found[0]], observed=True)
        else:
            q = 2. ** (best[1] * n / 12.)
            p = best[2]
            b = (p[0] + p[1]*q)/(1+p[2]*q)
            item = dict(a=float(direction_model[0]*b+direction_model[1]),
                        b=float(b), support=0., residual=0., observed=False)
        full.append(item)
    return full, np.arange(len(full), dtype=float), True, best[0]


def _intersections(frets, strings):
    points = np.empty((len(strings), len(frets), 2), dtype=np.float64)
    for j, s in enumerate(strings):
        for i, f in enumerate(frets):
            # x = af*(y-.5)+bf; y=as*(x-.5)+bs.
            denom = 1 - f["a"] * s["a"]
            if abs(denom) < 1e-6:
                raise ValueError("Degenerate fret/string intersection")
            x = (f["a"]*(s["b"]-.5*s["a"]-.5)+f["b"])/denom
            y = s["a"]*(x-.5)+s["b"]
            points[j, i] = [x, y]
    return points


def rectify_fretboard(image, mask, output_size=(1024, 256), num_strings=6,
                      spacing="uniform", occlusion_mask=None):
    """Return a standardized atlas, or None when the mask has no usable neck.

    ``spacing='uniform'`` equalizes fret intervals only when a relative lattice
    fit passes. Otherwise evidence locations retain their coarse x positions.
    ``spacing='observed'`` always preserves those positions (not metric units).
    Quality.status='grid' requires all requested strings and >=3 fret wires;
    otherwise a clearly marked 'coarse' preview is returned. No absolute fret
    or string pitch labels are inferred. The full mask ROI is retained, with
    partial cells at each longitudinal end and margins outside outer strings.
    """
    image = np.asarray(image)
    mask = np.asarray(mask)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an RGB uint8 HxWx3 array")
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all():
        raise ValueError("mask must be a finite HxW array matching image")
    if len(output_size) != 2 or any(int(v) != v or v < 2 for v in output_size):
        raise ValueError("output_size must contain two integers >=2")
    output_size = tuple(map(int, output_size))
    if not isinstance(num_strings, (int, np.integer)) or not 2 <= num_strings <= 12:
        raise ValueError("num_strings must be an integer between 2 and 12")
    if spacing not in ("uniform", "observed"):
        raise ValueError("spacing must be uniform or observed")
    if occlusion_mask is not None:
        occlusion_mask = np.asarray(occlusion_mask)
        if occlusion_mask.shape != mask.shape or not np.isfinite(occlusion_mask).all():
            raise ValueError("occlusion_mask must match image dimensions and be finite")
    corners = estimate_board_corners(mask)
    if corners is None:
        return None
    cw, ch = 1024, 256
    dest = np.float32([[0, 0], [cw-1, 0], [cw-1, ch-1], [0, ch-1]])
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), dest)
    coarse = cv2.warpPerspective(image, transform, (cw, ch))
    usable = cv2.warpPerspective((mask > 0).astype(np.uint8), transform, (cw, ch), flags=cv2.INTER_NEAREST)
    if occlusion_mask is not None:
        occ = cv2.warpPerspective((occlusion_mask > 0).astype(np.uint8), transform, (cw, ch), flags=cv2.INTER_NEAREST)
        usable[cv2.dilate(occ, np.ones((5, 5), np.uint8)) > 0] = 0
    usable = cv2.erode(usable, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(coarse, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2., tileGridSize=(16, 4)).apply(gray)
    frets = _fret_consensus(_line_candidates(gray, usable, "frets"))
    frets = _refine_fret_centers(gray, usable, frets)
    strings = _string_consensus(_line_candidates(gray, usable, "strings"), num_strings)
    string_detector = "line_segments"
    if len(strings) != num_strings:
        strings = _string_consensus(detect_strings(gray, usable, num_strings), num_strings)
        string_detector = "strip_projection"
    quality = dict(status="coarse", observed_frets=len(frets), observed_strings=len(strings),
                   requested_strings=int(num_strings), absolute_fret_numbers_known=False,
                   orientation="image-axis; nut direction and string pitches unknown",
                   occlusion_known=occlusion_mask is not None,
                   warnings=[], lattice_accepted=False, string_detector=string_detector)
    # Coarse sampling is subdivided to approximate the projective map well.
    xs, ys = np.linspace(0, 1, 25), np.linspace(0, 1, 9)
    xx, yy = np.meshgrid(xs, ys)
    unit_nodes = np.stack([xx, yy], axis=-1)
    target_x, target_y = xs * (output_size[0]-1), ys * (output_size[1]-1)
    if len(frets) >= 3 and len(strings) == num_strings:
        frets, numbers, accepted, residual = _fret_lattice(frets)
        quality.update(lattice_accepted=accepted, lattice_rmse=residual,
                       inferred_frets=sum(not v.get("observed", True) for v in frets))
        # Boundary rails stay part of the mesh; real line evidence controls its
        # interior. End caps remain ROI boundaries, never called fret 0 or 21.
        edge_frets = [dict(a=0., b=0.), *frets, dict(a=0., b=1.)]
        edge_strings = [dict(a=0., b=0.), *strings, dict(a=0., b=1.)]
        unit_nodes = _intersections(edge_frets, edge_strings)
        # Intersections with rail/endcap must lie inside the coarse ROI. An
        # extrapolated outer line that crosses it is not a valid cell boundary.
        if np.any(unit_nodes < -.001) or np.any(unit_nodes > 1.001):
            quality["warnings"].append("Line extrapolation crosses ROI boundary; using coarse preview.")
        else:
            target_x = np.array([v["b"] for v in edge_frets])
            if spacing == "uniform" and accepted:
                left_gap = (frets[1]["b"] - frets[0]["b"])
                right_gap = (frets[-1]["b"] - frets[-2]["b"])
                coordinates = np.r_[-frets[0]["b"]/left_gap, numbers,
                                    numbers[-1]+(1-frets[-1]["b"])/right_gap]
                target_x = (coordinates-coordinates[0]) / (coordinates[-1]-coordinates[0])
            # Fixed output margins give the same string coordinates across
            # instruments; margins in the source remain available as pixels.
            target_y = np.r_[0., np.linspace(.08, .92, num_strings), 1.]
            target_x *= output_size[0]-1
            target_y *= output_size[1]-1
            quality["status"] = "grid"
            quality["fret_support"] = [v["support"] for v in frets]
            quality["string_support"] = [v["support"] for v in strings]
    if quality["status"] != "grid":
        unit_nodes = np.stack([xx, yy], axis=-1)
        target_x, target_y = xs * (output_size[0]-1), ys * (output_size[1]-1)
        quality["warnings"].append("Insufficient consistent internal grid evidence; rectangular preview is not a verified grid.")
    if not quality["lattice_accepted"]:
        quality["warnings"].append("Fret intervals are not certified; no equal-width fret numbering was imposed.")
    coarse_nodes = unit_nodes * [cw-1, ch-1]
    nodes = cv2.perspectiveTransform(coarse_nodes.reshape(1, -1, 2), np.linalg.inv(transform)).reshape(coarse_nodes.shape)
    try:
        warped = warp_grid(image, nodes, target_x, target_y, output_size, occlusion_mask=occlusion_mask)
    except ValueError:
        if quality["status"] != "grid":
            raise
        quality["status"] = "coarse"
        quality["warnings"].append("Rejected folded evidence mesh; using coarse preview.")
        coarse_nodes = np.stack([xx, yy], axis=-1)*[cw-1, ch-1]
        nodes = cv2.perspectiveTransform(coarse_nodes.reshape(1, -1, 2), np.linalg.inv(transform)).reshape(coarse_nodes.shape)
        target_x, target_y = xs * (output_size[0]-1), ys * (output_size[1]-1)
        warped = warp_grid(image, nodes, target_x, target_y, output_size, occlusion_mask=occlusion_mask)
    quality["valid_fraction"] = float(np.mean(warped["valid_mask"] > 0))
    quality["visible_fraction"] = float(np.mean(warped["visible_mask"] > 0)) if occlusion_mask is not None else None
    quality["spacing"] = "uniform_relative_frets" if quality["status"] == "grid" and quality["lattice_accepted"] and spacing == "uniform" else "observed"
    return RectificationResult(warped["image"], nodes, target_x, target_y,
                               warped["valid_mask"], warped["visible_mask"], quality, transform)
