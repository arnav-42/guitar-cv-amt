"""Evidence-based string-line proposals in a coarse fretboard view.

The detector combines a shared-slope projection search with independent
longitudinal strip checks. It returns no line unless image evidence supports
that string over a useful part of the visible board.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks


def _gray_float(gray: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(gray)
    if arr.ndim != 2 or min(arr.shape, default=0) < 16:
        return None
    values = arr.astype(np.float32)
    if not np.isfinite(values).all():
        return None
    if np.issubdtype(arr.dtype, np.integer) or float(values.max(initial=0)) > 1.5:
        values /= 255.0
    return np.clip(values, 0.0, 1.0)


def _ridge_response(gray: np.ndarray, usable: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Bright and dark horizontal ridges from a vertical narrow/broad DoG."""
    small = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.55, sigmaY=0.55)
    broad = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.55, sigmaY=2.1)
    response = np.abs(small - broad).astype(np.float32)
    response *= usable
    maximum = float(response.max(initial=0.0))
    if maximum < 1e-4:
        return None
    # A small amount of smoothing joins antialiased string glints without
    # broadening fret-wire crossings into plausible horizontal lines.
    response = cv2.GaussianBlur(response, (0, 0), sigmaX=0.45, sigmaY=0.45)
    return response, maximum


def _projected_profile(response: np.ndarray, usable: np.ndarray,
                       normalized_slope: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = response.shape
    shear = float(normalized_slope) * (h - 1) / max(w - 1, 1)
    matrix = np.float32([[1.0, 0.0, 0.0],
                         [-shear, 1.0, shear * (w - 1) * 0.5]])
    numerator = cv2.warpAffine(response, matrix, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpAffine(usable.astype(np.float32), matrix, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    count = coverage.sum(axis=1)
    profile = numerator.sum(axis=1) / np.maximum(count, 1e-4)
    profile[count < max(8.0, 0.12 * w)] = 0.0
    return profile.astype(np.float32), count.astype(np.float32)


def _profile_peaks(profile: np.ndarray, count: np.ndarray, image_shape: tuple[int, int]):
    """Return significant peaks and their local prominence/noise ratios."""
    h, w = image_shape
    smooth = cv2.GaussianBlur(profile[:, None], (1, 0), sigmaX=0, sigmaY=0.9)[:, 0]
    background = cv2.GaussianBlur(profile[:, None], (1, 0), sigmaX=0, sigmaY=7.0)[:, 0]
    signal = smooth - background
    usable_rows = (count > max(8.0, 0.12 * w))
    if np.count_nonzero(usable_rows) < 12:
        return np.empty(0, int), np.empty(0, np.float64)
    values = signal[usable_rows]
    med = float(np.median(values))
    noise = 1.4826 * float(np.median(np.abs(values - med)))
    noise = max(noise, 4e-4)
    absolute_floor = max(0.0018, 1.2 * noise)
    prominence = max(0.0013, 1.0 * noise)
    peak_indices, props = find_peaks(signal, distance=max(3, int(round(h * 0.018))),
                                     prominence=prominence)
    if len(peak_indices) == 0:
        return np.empty(0, int), np.empty(0, np.float64)
    in_range = ((peak_indices >= int(round(0.015 * (h - 1)))
                 ) & (peak_indices <= int(round(0.985 * (h - 1))))
                & (signal[peak_indices] >= absolute_floor)
                & usable_rows[peak_indices])
    peak_indices = peak_indices[in_range]
    prominences = props["prominences"][in_range]
    strengths = np.maximum(signal[peak_indices], prominences) / noise
    return peak_indices, strengths.astype(np.float64)


def _comb_hypothesis(peaks: np.ndarray, strengths: np.ndarray, h: int,
                     count: int) -> tuple[np.ndarray, float] | None:
    """Pick a supported six-line-like comb with only a weak spacing prior."""
    if len(peaks) < max(3, count - 2):
        return None
    # A faint string need not survive the GLOBAL peak threshold. Propose its
    # slot from the comb, then require independent LOCAL strip observations
    # below. A proposed slot by itself is never returned as a detected line.
    best_partial = None
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            for steps in range(1, count):
                step = (peaks[j]-peaks[i])/steps
                if not .50*(h-1) < step*(count-1) < .95*(h-1):
                    continue
                for slot in range(count-steps):
                    targets = peaks[i]+(np.arange(count)-slot)*step
                    if targets[0] < .015*(h-1) or targets[-1] > .985*(h-1):
                        continue
                    nearest = np.argmin(abs(targets[:, None]-peaks[None, :]), axis=1)
                    error = abs(targets-peaks[nearest])/step
                    supported = error < .24
                    if supported.sum() < count-1:
                        continue
                    score = float(np.log1p(strengths[nearest[supported]]).sum()
                                  - .8*np.sum(~supported)-2*error[supported].sum())
                    fitted = targets.copy()
                    fitted[supported] = peaks[nearest[supported]]
                    if best_partial is None or score > best_partial[1]:
                        best_partial = fitted, score
    if best_partial is not None:
        return best_partial
    if len(peaks) < count:
        return None
    min_span, max_span = 0.50 * (h - 1), 0.95 * (h - 1)
    best: tuple[np.ndarray, float] | None = None
    for start in range(len(peaks) - count + 1):
        last_start = min(len(peaks) - 1, start + max(count + 2, 18))
        for end in range(start + count - 1, last_start + 1):
            span = float(peaks[end] - peaks[start])
            if span < min_span or span > max_span:
                continue
            step = span / (count - 1)
            selected = [start]
            possible = True
            for slot in range(1, count - 1):
                target = peaks[start] + slot * step
                lo = target - 0.47 * step
                hi = target + 0.47 * step
                choices = [idx for idx in range(selected[-1] + 1, end)
                           if lo <= peaks[idx] <= hi]
                if not choices:
                    possible = False
                    break
                # Evidence is primary; the target position contributes only
                # a small regularizer, allowing perspective spacing changes.
                choice = max(choices, key=lambda idx: (
                    strengths[idx] - 0.12 * ((peaks[idx] - target) / max(step, 1)) ** 2
                ))
                selected.append(choice)
            if not possible:
                continue
            selected.append(end)
            indices = np.asarray(selected, dtype=int)
            gaps = np.diff(peaks[indices]).astype(np.float64)
            regularity = float(np.mean(((gaps - step) / max(step, 1.0)) ** 2))
            evidence = float(np.sum(np.log1p(np.maximum(strengths[indices], 0))))
            score = evidence - 0.18 * regularity
            if best is None or score > best[1]:
                best = indices, score
    if best is None:
        return None
    return peaks[best[0]], best[1]


def _robust_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """Small deterministic Huber regression used for each string's line."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-3)
    design = np.column_stack((x, np.ones_like(x)))
    coef = np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                           y * np.sqrt(weights), rcond=None)[0]
    for _ in range(10):
        residual = y - design @ coef
        median = float(np.median(residual))
        scale = max(0.0015, 1.4826 * float(np.median(np.abs(residual - median))))
        robust = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual - median), 1e-9))
        combined = weights * robust
        coef = np.linalg.lstsq(design * np.sqrt(combined[:, None]),
                               y * np.sqrt(combined), rcond=None)[0]
    residual = y - design @ coef
    median = float(np.median(residual))
    scale = max(0.0015, 1.4826 * float(np.median(np.abs(residual - median))))
    inliers = np.abs(residual - median) <= max(0.004, 2.5 * scale)
    if not np.isfinite(coef).all() or np.count_nonzero(inliers) < 4:
        return None
    return coef, residual, inliers


def _strip_observations(response: np.ndarray, usable: np.ndarray,
                        base_slope: float, center_y: float,
                        spacing_px: float):
    """Measure independent local ridge support in longitudinal image strips."""
    h, w = response.shape
    strip_count = 14
    strip_edges = np.linspace(0, w, strip_count + 1, dtype=int)
    radius = int(np.clip(round(0.32 * spacing_px), 4, 17))
    xs, ys, strengths = [], [], []
    visible_count = 0
    for index in range(strip_count):
        x0, x1 = strip_edges[index:index + 2]
        if x1 <= x0:
            continue
        xc = 0.5 * (x0 + x1 - 1)
        xc_norm = xc / max(w - 1, 1)
        prediction = center_y + base_slope * (h - 1) * (xc_norm - 0.5)
        y0 = max(0, int(np.floor(prediction)) - radius)
        y1 = min(h, int(np.ceil(prediction)) + radius + 1)
        if y1 - y0 < 5:
            continue
        strip_mask = usable[y0:y1, x0:x1]
        visible_fraction = float(np.mean(strip_mask > 0))
        if visible_fraction < 0.20:
            continue
        visible_count += 1

        mask_strip = usable[:, x0:x1].astype(np.float32)
        row_count = mask_strip.sum(axis=1)
        row_sum = (response[:, x0:x1] * mask_strip).sum(axis=1)
        profile = row_sum / np.maximum(row_count, 1e-4)
        profile[row_count < max(2.0, 0.12 * (x1 - x0))] = 0.0
        baseline = cv2.GaussianBlur(profile[:, None], (1, 0), sigmaX=0, sigmaY=5.0)[:, 0]
        signal = cv2.GaussianBlur((profile - baseline)[:, None], (1, 0),
                                  sigmaX=0, sigmaY=0.8)[:, 0]

        search_lo = max(2, int(np.floor(prediction)) - radius)
        search_hi = min(h - 2, int(np.ceil(prediction)) + radius + 1)
        if search_hi <= search_lo:
            continue
        local = signal[search_lo:search_hi]
        peak_index = int(np.argmax(local)) + search_lo
        # Fit a parabolic peak for subpixel observations.
        offset = 0.0
        if 0 < peak_index < h - 1:
            left, mid, right = signal[peak_index - 1:peak_index + 2]
            denom = left - 2.0 * mid + right
            if abs(denom) > 1e-8:
                offset = float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
        peak = float(signal[peak_index])

        # Estimate local noise away from the peak so a globally bright fret
        # crossing cannot make every strip look like string evidence.
        lo = max(2, peak_index - max(12, radius * 2))
        hi = min(h - 2, peak_index + max(12, radius * 2) + 1)
        local_y = np.arange(lo, hi)
        background = signal[lo:hi][np.abs(local_y - peak_index) >= 3]
        if len(background) < 8:
            background = signal[row_count > 0]
        bg_median = float(np.median(background)) if len(background) else 0.0
        noise = 1.4826 * float(np.median(np.abs(background - bg_median))) if len(background) else 0.0
        noise = max(noise, 0.0012)
        contrast = peak - bg_median
        snr = contrast / noise
        if contrast < 0.0030 or snr < 2.0:
            continue

        xs.append(xc_norm)
        ys.append((peak_index + offset) / max(h - 1, 1))
        strengths.append(float(np.clip(snr, 0.5, 20.0)))

    if visible_count == 0 or len(xs) < max(4, int(np.ceil(0.40 * visible_count))):
        return None
    fit = _robust_fit(np.asarray(xs) - 0.5, np.asarray(ys), np.asarray(strengths))
    if fit is None:
        return None
    coef, residuals, inliers = fit
    minimum_evidence = max(4, int(np.ceil(0.40 * visible_count)))
    if np.count_nonzero(inliers) < minimum_evidence:
        return None
    support = float(np.count_nonzero(inliers) / visible_count)
    residual = float(np.median(np.abs(residuals[inliers])))
    return dict(a=float(coef[0]), b=float(coef[1]), support=support,
                residual=residual)


def detect_strings(gray: np.ndarray, usable: np.ndarray, num_strings: int = 6) -> list[dict]:
    """Detect evidence-supported string lines in a coarse grayscale board.

    Parameters
    ----------
    gray:
        Enhanced grayscale image, conventionally ``256 x 1024``.
    usable:
        A matching binary mask where line evidence is allowed (0/1 or 0/255).
    num_strings:
        Number of strings expected. The output is empty unless every requested
        line has independent support over at least about 40% of visible length.

    Returns
    -------
    list[dict]
        Ordered line records with ``a``, ``b``, ``support``, and ``residual``;
        each line follows ``y = a * (x - 0.5) + b`` in normalized coordinates.
    """
    if not isinstance(num_strings, (int, np.integer)) or not 2 <= num_strings <= 12:
        return []
    gray_float = _gray_float(gray)
    if gray_float is None:
        return []
    mask = np.asarray(usable)
    if mask.ndim != 2 or mask.shape != gray_float.shape:
        return []
    valid = (mask > 0).astype(np.uint8)
    h, w = gray_float.shape
    if np.count_nonzero(valid) < 0.12 * h * w:
        return []
    response_result = _ridge_response(gray_float, valid)
    if response_result is None:
        return []
    response, response_max = response_result
    if response_max < 0.003:
        return []

    # Shared-slope shears align long thin string ridges before projection.
    # Keep several nearby hypotheses: actual strings may fan slightly.
    hypotheses = []
    slopes = np.linspace(-0.20, 0.20, 81)
    for slope in slopes:
        profile, row_coverage = _projected_profile(response, valid, float(slope))
        peaks, strengths = _profile_peaks(profile, row_coverage, (h, w))
        comb = _comb_hypothesis(peaks, strengths, h, int(num_strings))
        if comb is None:
            continue
        peak_y, score = comb
        hypotheses.append((float(score), float(slope), peak_y.astype(np.float64)))
    if not hypotheses:
        return []
    hypotheses.sort(key=lambda item: item[0], reverse=True)

    # Refine a few distinct shear hypotheses with independent local evidence.
    candidates = []
    for _, base_slope, peak_y in hypotheses:
        if any(abs(base_slope - used) < 0.012 for used in candidates):
            continue
        candidates.append(base_slope)
        if len(candidates) >= 7:
            break

    for base_slope in candidates:
        profile, row_coverage = _projected_profile(response, valid, base_slope)
        peaks, strengths = _profile_peaks(profile, row_coverage, (h, w))
        comb = _comb_hypothesis(peaks, strengths, h, int(num_strings))
        if comb is None:
            continue
        peak_y, _ = comb
        spacing_px = float(np.median(np.diff(peak_y)))
        records = []
        for y0 in peak_y:
            record = _strip_observations(
                response, valid, base_slope, float(y0), spacing_px
            )
            if record is None:
                records = []
                break
            records.append(record)
        if len(records) != num_strings:
            continue
        records.sort(key=lambda item: item["b"])
        line_spacing = np.diff([v["b"] for v in records])
        if len(line_spacing) and np.any(line_spacing <= 0.02):
            continue
        # Validate that adjacent fitted lines still form a broad ordered comb.
        if len(line_spacing) and (line_spacing.max() > 0.40 or line_spacing.min() < 0.035):
            continue
        return records
    return []
