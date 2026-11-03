"""
Guitar string detection (6 straight lines per frame) and a per-frame confidence flag.

  A  comb_band_slope_strip_fan   our detector, run on ROI A (`roi_a`: YOLO quad from
                                 fret_detect_yolo.rectify_perspective + 30% padding)
  B  lsd_strip_cascade           LSD segments -> strip projection (fretboard_rectify /
                                 string_evidence), run on ROI A for fusion or on its own
                                 board-corner ROI (`roi_b`)
  detect_strings_with_confidence A and B on ROI A; lines averaged when they agree within
                                 1/4 string spacing, else the set with more image evidence.
                                 confident=True when they agree.

Every detector takes an ROI (BGR) and returns lines (x1, y1, x2, y2) in ROI coordinates
(6 lines, or [] when it gives no answer); the ROI dict's `to_frame` maps them back.

Results (string pass = all 6 lines within 1/4 spacing):
  Roboflow test: A 42.5%, B 15.0%.  GuiFiR test (unseen): A 41.3%, B 22.7%;
  frames where A and B agree (18%): 91.7% pass, 99.4% within 1/2 spacing.

Idea of A: thin-line ridge response; one shared tilt (shear that sharpens the row
profile); six equally spaced comb teeth whose centre and spacing are tied to the band
crossed by frets; per-string refinement in 8 strips with slopes linear in string rank
(perspective fan). Parameters tuned on Roboflow valid + 14 own frames only.
"""

import cv2
import numpy as np

from pipeline.board_geometry import estimate_board_corners
from pipeline.fret_detect_yolo import load_yolo_model, predict_mask, rectify_perspective
from pipeline.fretboard_rectify import _line_candidates, _string_consensus
from pipeline.string_evidence import detect_strings


N = 6
SLOPES = np.linspace(-0.12, 0.12, 49)      # shear search range (dy/dx)
SPACING_RANGE = (0.08, 0.22)               # comb spacing as a fraction of ROI height
N_STRIPS = 8
SNAP = 0.0                                 # comb tooth -> local peak window, +/- spacings (0 = keep the comb)
SEARCH = 0.25                              # strip search window, +/- spacings
MIN_PROM = 0.6                             # drop strip peaks below this x median prominence
MAX_COLS = 512                             # columns used for projections
BAND_FRAC = 0.5                            # fret-energy threshold (fraction of range)
BAND_SPACINGS = (4.5, 7.0)                 # band height / string spacing
BAND_CENTER_TOL = 0.5                      # comb center vs band center, in spacings
RIDGE_SIGMA = 0.01                         # ridge scale as a fraction of ROI height
AGG = "median"                             # column aggregation for projections (robust to fingers)


ROI_PAD = 0.3                              # ROI A: rows added above and below the YOLO quad (x 256)
ATLAS_SIZE = (1024, 256)                   # ROI B size
AGREE_TOL = 0.25                           # A and B agree if no line differs by more (in A's spacing)


# ── ROIs ──────────────────────────────────────────────────────────────────


def fretboard_mask(model, frame_bgr):
    """Cleaned YOLO fretboard mask (uint8 0/255, frame size)."""
    from PIL import Image
    return predict_mask(model, Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))


def roi_a(frame_bgr, mask, pad=ROI_PAD):
    """ROI A: YOLO quad warped to 1024x256 plus pad*256 rows above and below.
    Returns {"image", "usable" (warped mask, 0/1), "to_frame", "quad"} or None."""
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    rect = rectify_perspective(pil, mask, return_details=True)
    if rect is None:
        return None
    transform = rect["transform"]
    h, w = rect["image"].shape[:2]
    mh = int(round(pad * h))
    transform = np.array([[1, 0, 0], [0, 1, mh], [0, 0, 1]], float) @ transform
    image = cv2.warpPerspective(frame_bgr, transform, (w, h + 2 * mh))
    usable = cv2.warpPerspective((mask > 0).astype(np.uint8), transform, (w, h + 2 * mh), flags=cv2.INTER_NEAREST)
    return {"image": image, "usable": usable, "to_frame": _inverse(transform),
            "quad": rect["corners"].astype(np.float32)}


def roi_b(frame_bgr, mask):
    """ROI B: estimate_board_corners quad warped to 1024x256, no padding (B's own ROI)."""
    corners = estimate_board_corners(mask)
    if corners is None:
        return None
    w, h = ATLAS_SIZE
    dest = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), dest)
    return {"image": cv2.warpPerspective(frame_bgr, transform, (w, h)),
            "usable": cv2.warpPerspective((mask > 0).astype(np.uint8), transform, (w, h), flags=cv2.INTER_NEAREST),
            "to_frame": _inverse(transform), "quad": corners.astype(np.float32)}


def _inverse(transform):
    inv = np.linalg.inv(transform)

    def to_frame(lines):
        lines = np.asarray(lines, float).reshape(-1, 4)
        pts = np.column_stack([lines.reshape(-1, 2), np.ones(2 * len(lines))]) @ inv.T
        return (pts[:, :2] / pts[:, 2:]).reshape(-1, 4)
    return to_frame


# ── A: comb-band ──────────────────────────────────────────────────────────


def _gray(roi_bgr):
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _raw(roi_bgr):
    """Same preprocessing as auto_tune (CLAHE + 5x5 blur)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0).astype(np.float32)


def _ridge(roi_bgr):
    """Negative 2nd y-derivative of a Gaussian-smoothed image (thin bright
    horizontal lines -> positive peak, brightness steps -> ~0)."""
    sig = max(0.7, RIDGE_SIGMA * roi_bgr.shape[0])
    g = cv2.GaussianBlur(_gray(roi_bgr), (0, 0), sigmaX=2.0, sigmaY=sig)
    return -cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * sig ** 2


def _shear(img, m):
    """out[y, x] = img[y + m*(x - xc), x]: a line y = y0 + m*(x - xc) becomes row y0."""
    h, w = img.shape[:2]
    if m == 0:
        return img
    xs = np.arange(w, dtype=np.float32)
    map_x = np.tile(xs, (h, 1))
    map_y = (np.arange(h, dtype=np.float32)[:, None] + m * (xs - (w - 1) / 2)).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _cols(w):
    return np.linspace(0, w - 1, min(w, MAX_COLS)).astype(int)


def _agg(a):
    return np.median(a, axis=1) if AGG == "median" else a.mean(axis=1)


def _profile(img, cols=None):
    return _agg(img[:, cols] if cols is not None else img)


def best_slope(img):
    """Idea 1: the shear whose row projection has the most gradient energy."""
    cols = _cols(img.shape[1])
    sub = img[:, cols]
    scores = []
    for m in SLOPES:
        # shear the column subset; x offsets must use the full-width coordinates
        h = sub.shape[0]
        xs = cols.astype(np.float32)
        map_x = np.tile(np.arange(len(cols), dtype=np.float32), (h, 1))
        map_y = (np.arange(h, dtype=np.float32)[:, None] + m * (xs - (img.shape[1] - 1) / 2)).astype(np.float32)
        p = cv2.remap(sub, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).mean(axis=1)
        scores.append(float(np.sum(np.diff(p) ** 2)))
    return float(SLOPES[int(np.argmax(scores))])


def fret_band(roi_bgr):
    """(top, bottom) rows of the fretboard: contiguous band of high fret
    (vertical-edge) energy around its maximum. None if degenerate."""
    h = roi_bgr.shape[0]
    g = cv2.GaussianBlur(_gray(roi_bgr), (0, 0), 1.0)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gx = cv2.blur(gx, (1, max(3, int(h * 0.05))))      # keep vertically consistent edges
    e = cv2.GaussianBlur(gx.mean(axis=1).reshape(-1, 1), (0, 0), max(1.0, 0.03 * h)).ravel()
    thr = e.min() + BAND_FRAC * (e.max() - e.min())
    t = b = int(np.argmax(e))
    while t > 0 and e[t - 1] >= thr:
        t -= 1
    while b < h - 1 and e[b + 1] >= thr:
        b += 1
    return (t, b) if b - t >= 0.2 * h else None


def _subpixel(p, i):
    if 0 < i < len(p) - 1:
        a, b, c = p[i - 1], p[i], p[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def comb_fit(p, h, band=None):
    """Equal-spacing constraint: best (y0, s) for 6 teeth, then snap each tooth.
    band = (top, bottom) restricts spacing and comb center (see module doc)."""
    p = np.asarray(p, dtype=float)
    p = p - np.median(p)
    lo, hi = SPACING_RANGE[0] * h, SPACING_RANGE[1] * h
    if band is not None:
        bh = band[1] - band[0]
        lo, hi = max(lo, bh / BAND_SPACINGS[1]), min(hi, bh / BAND_SPACINGS[0])
        if lo > hi:
            lo, hi = bh / BAND_SPACINGS[1], bh / BAND_SPACINGS[0]
    best = (-np.inf, None, None)
    k = np.arange(N)
    y = np.arange(len(p), dtype=float)
    for s in np.arange(lo, hi + 1e-9, 0.5):
        y0s = np.arange(0, len(p) - 1 - (N - 1) * s, 0.5)
        if band is not None:
            c = (band[0] + band[1]) / 2 - (N - 1) / 2 * s
            y0s = y0s[np.abs(y0s - c) <= BAND_CENTER_TOL * s]
        if len(y0s) == 0:
            continue
        teeth = y0s[:, None] + k[None, :] * s
        score = np.interp(teeth, y, p).sum(axis=1)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (score[j], y0s[j], s)
    _, y0, s = best
    if y0 is None:
        return None, None
    out = []
    for t in y0 + k * s:
        if SNAP <= 0:
            out.append(float(t))
            continue
        a, b = int(max(0, np.floor(t - SNAP * s))), int(min(len(p) - 1, np.ceil(t + SNAP * s)))
        i = a + int(np.argmax(p[a:b + 1]))
        out.append(_subpixel(p, i))
    return np.array(out), s


def strip_refine(img, lines_y, m, s, fan=False):
    """lines_y: y of each string at the ROI center (after shear by m).
    Returns (6, 4) lines in ROI coords."""
    h, w = img.shape
    xc = (w - 1) / 2
    sheared = _shear(img, m)
    edges = np.linspace(0, w, N_STRIPS + 1).astype(int)
    det = []  # (string, x_center, y_frame, prominence)
    for a, b in zip(edges[:-1], edges[1:]):
        p = _agg(sheared[:, a:b])
        x_mid = (a + b - 1) / 2
        for i, y in enumerate(lines_y):
            lo, hi = int(max(0, np.floor(y - SEARCH * s))), int(min(h - 1, np.ceil(y + SEARCH * s)))
            if hi - lo < 2:
                continue
            win = p[lo:hi + 1]
            j = int(np.argmax(win))
            if j in (0, len(win) - 1):          # no interior peak in the window
                continue
            prom = win[j] - win.min()
            ys = _subpixel(p, lo + j)
            det.append((i, x_mid, ys + m * (x_mid - xc), prom))
    det = np.array(det) if det else np.zeros((0, 4))
    if len(det):
        det = det[det[:, 3] >= MIN_PROM * np.median(det[:, 3])]

    fits = []
    for i, y in enumerate(lines_y):
        d = det[det[:, 0] == i] if len(det) else det
        a_, b_ = y - m * xc, m                    # fallback: global line  y = a + b*x
        n_pts = len(d)
        if n_pts >= 3:
            x, yy = d[:, 1], d[:, 2]
            keep = np.ones(n_pts, bool)
            for _ in range(3):
                b_, a_ = np.polyfit(x[keep], yy[keep], 1)
                r = np.abs(yy - (a_ + b_ * x))
                new = r <= max(0.15 * s, 1.0)
                if new.sum() < 3 or np.array_equal(new, keep):
                    break
                keep = new
            n_pts = int(keep.sum())
        fits.append((a_, b_, n_pts, d))

    if fan:
        slopes = np.array([f[1] for f in fits])
        wts = np.array([max(f[2], 0.5) for f in fits])
        k = np.arange(N)
        c1, c0 = np.polyfit(k, slopes, 1, w=np.sqrt(wts))
        new = []
        for i, (a_, b_, n_pts, d) in enumerate(fits):
            b2 = c0 + c1 * i
            a2 = float(np.median(d[:, 2] - b2 * d[:, 1])) if n_pts >= 1 else a_ + (b_ - b2) * xc
            new.append((a2, b2, n_pts, d))
        fits = new

    return [(0.0, a_, float(w - 1), a_ + b_ * (w - 1)) for a_, b_, _, _ in fits]


def _lines(ys, m, w):
    xc = (w - 1) / 2
    return [(0.0, float(y - m * xc), float(w - 1), float(y + m * xc)) for y in ys]


def _comb(img, m, band=None):
    ys, s = comb_fit(_profile(_shear(img, m), _cols(img.shape[1])), img.shape[0], band)
    return ys, s


def comb_band_slope_strip_fan(roi):
    """Equal spacing + fret band + ideas 1 + 2 + fan."""
    r = _ridge(roi)
    m = best_slope(r)
    ys, s = _comb(r, m, fret_band(roi))
    if ys is None:
        return []
    return strip_refine(r, ys, m, s, fan=True)


# ── B: LSD -> strip cascade ───────────────────────────────────────────────




def _prep(roi_bgr, usable):
    gray = cv2.createCLAHE(clipLimit=2., tileGridSize=(16, 4)).apply(
        cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY))
    return gray, cv2.erode(usable.astype(np.uint8), np.ones((3, 3), np.uint8))


def _to_lines(strings, shape):
    """y = a*(x - .5) + b in coordinates normalized by (w-1, h-1) -> ROI pixels."""
    h, w = shape[:2]
    return [(0.0, (s["b"] - s["a"] / 2) * (h - 1), float(w - 1), (s["b"] + s["a"] / 2) * (h - 1))
            for s in strings]


def _lsd(gray, usable):
    return _string_consensus(_line_candidates(gray, usable, "strings"), N)


def _strip(gray, usable):
    return _string_consensus(detect_strings(gray, usable, N), N)


def lsd_strip_cascade(roi_bgr, usable):
    gray, use = _prep(roi_bgr, usable)
    strings, lsd_strip_cascade.last_stage = _lsd(gray, use), "line_segments"
    if len(strings) != N:
        strings, lsd_strip_cascade.last_stage = _strip(gray, use), "strip_projection"
    return _to_lines(strings, roi_bgr.shape)


# ── Fusion and confidence ─────────────────────────────────────────────────


def _center_sorted(lines, w):
    lines = np.asarray(lines, float).reshape(-1, 4)
    yc = lines[:, 1] + (lines[:, 3] - lines[:, 1]) * ((w - 1) / 2 - lines[:, 0]) / (lines[:, 2] - lines[:, 0])
    return lines[np.argsort(yc)], np.sort(yc)


def _y_at(lines, xs):
    x1, y1, x2, y2 = lines.T
    return y1[:, None] + ((y2 - y1) / (x2 - x1))[:, None] * (xs[None, :] - x1[:, None])


def _evidence(ridge, lines):
    """Mean over strings of the median |ridge| sampled along each line."""
    h, w = ridge.shape
    xs = _cols(w).astype(np.float32)
    ys = _y_at(lines, xs).astype(np.float32)
    vals = cv2.remap(np.abs(ridge), np.tile(xs, (len(lines), 1)), ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return float(np.median(vals, axis=1).mean())


def fuse_arbitrate(roi_bgr, usable):
    w = roi_bgr.shape[1]
    a = comb_band_slope_strip_fan(roi_bgr)
    b = lsd_strip_cascade(roi_bgr, usable)
    if len(b) != N or len(a) != N:
        fuse_arbitrate.last_stage = "A_only" if len(a) == N else "B_only" if len(b) == N else "none"
        return a if len(a) == N else b
    a, ya = _center_sorted(a, w)
    b, _ = _center_sorted(b, w)
    s = float(np.median(np.diff(ya)))
    xs = np.array([0.0, (w - 1) / 2, w - 1.0])
    gap = np.abs(_y_at(a, xs) - _y_at(b, xs)).max() / s
    if gap <= AGREE_TOL:
        fuse_arbitrate.last_stage = "agree_avg"
        return list(map(tuple, (a + b) / 2))
    r = _ridge(roi_bgr)
    pick_a = _evidence(r, a) >= _evidence(r, b)
    fuse_arbitrate.last_stage = "disagree_A" if pick_a else "disagree_B"
    return list(map(tuple, a if pick_a else b))


def detect_strings_with_confidence(roi_bgr, usable):
    """Official output: (lines, confident).

    lines      fuse_arbitrate's 6 lines (x1, y1, x2, y2) in ROI coords, or [].
    confident  True when A and B independently agree within AGREE_TOL spacings.
               On held-out test 83% of such frames pass (<= 0.25 spacing) and
               100% are within 0.5 spacing, vs 42% passing otherwise
               (eval/compare_fusion, 2026-10-26).
    """
    lines = fuse_arbitrate(roi_bgr, usable)
    return lines, fuse_arbitrate.last_stage == "agree_avg"


lsd_strip_cascade.last_stage = None
fuse_arbitrate.last_stage = None


def detect_frame(frame_bgr, yolo_model):
    """One frame -> {"lines": (6, 4) string lines in frame coords, top to bottom (or None),
    "confident": A and B agree, "roi": ROI A}. Lines may be partial-width segments."""
    mask = fretboard_mask(yolo_model, frame_bgr)
    roi = roi_a(frame_bgr, mask) if mask.any() else None
    if roi is None:
        return {"lines": None, "confident": False, "roi": None}
    lines, confident = detect_strings_with_confidence(roi["image"], roi["usable"])
    if len(lines) != N:
        return {"lines": None, "confident": False, "roi": roi}
    frame_lines = roi["to_frame"](lines)
    mid = (frame_lines[:, 1] + frame_lines[:, 3]) / 2
    return {"lines": frame_lines[np.argsort(mid)], "confident": confident, "roi": roi}
