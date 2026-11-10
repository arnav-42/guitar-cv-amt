"""
Shared paths and scoring for the GuiFiR benchmark scripts (scripts/guifir/).

Run every script from the repository root as a module, e.g.
  python -m scripts.guifir.compare_strings --split test
Generated files go under output/guifir/ (git-ignored):
  masks/        cached YOLO fretboard masks
  preds/        hand-model predictions, <model>_<split>.jsonl
  lines_<split>.jsonl, audio_basicpitch_<split>.jsonl, wilor_det_<split>.jsonl
  eval/         summaries and per-frame CSVs
  figures/      plots
External models (not in the repo) live under third_party/ (see scripts/guifir/README.md).
"""

from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "guifir"
MASK_DIR = OUT / "masks"
PRED_DIR = OUT / "preds"
EVAL_DIR = OUT / "eval"
FIG_DIR = OUT / "figures"
THIRD_PARTY = REPO / "third_party"
YOLO_WEIGHTS = REPO / "main" / "models" / "fretboard" / "yolo_weights_best.pt"


def short_path(path):
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def y_at(lines, x):
    """y of each near-horizontal line at x. lines: (n, 4); x: scalar -> (n,), (m,) -> (n, m)."""
    lines = np.asarray(lines, dtype=float).reshape(-1, 4)
    x1, y1, x2, y2 = lines.T
    dx = x2 - x1
    slope = np.divide(y2 - y1, dx, out=np.zeros_like(dx), where=np.abs(dx) > 1e-9)
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return y1 + slope * (x - x1)
    return y1[:, None] + slope[:, None] * (x[None, :] - x1[:, None])


def _sorted_at(lines, x):
    order = np.argsort(y_at(lines, x))
    return lines[order], order


def score_lines(gt, pred, xs):
    """Compare predicted string lines with GT lines at x positions xs (frame coords).
    Errors are in units of the local GT string spacing; a method must return
    exactly as many lines as GT (count_ok) to be scored."""
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    pred = np.asarray(pred, dtype=float).reshape(-1, 4)
    x_mid = (xs[0] + xs[-1]) / 2
    gt, g_order = _sorted_at(gt, x_mid)
    yg = y_at(gt, xs)
    gaps = np.diff(yg, axis=0)
    if np.any(gaps <= 0):
        raise ValueError("GT strings cross inside the evaluation range")
    spacing = np.median(gaps, axis=0)
    out = {"spacing_px": float(np.median(spacing)), "top_to_bottom_ids": (g_order + 1).tolist(),
           "n_pred": len(pred), "count_ok": len(pred) == len(gt)}
    if not out["count_ok"]:
        return out
    pred, _ = _sorted_at(pred, x_mid)
    err = y_at(pred, xs) - yg
    norm = err / spacing
    out.update(err_px=err, err_norm=norm,
               max_norm=float(np.abs(norm).max()), mean_norm=float(np.abs(norm).mean()),
               max_px=float(np.abs(err).max()), mean_px=float(np.abs(err).mean()))
    return out


def x_range(quad, gt):
    """Scoring x range of one ROI: inner x of its quad, clipped to the labelled extent."""
    qx = np.sort(np.asarray(quad, float)[:, 0])
    lo, hi = max(qx[1], gt[:, [0, 2]].min()), min(qx[2], gt[:, [0, 2]].max())
    return (lo, hi) if hi > lo else None


_YOLO = {}


def cached_mask(key, frame_bgr):
    """YOLO fretboard mask for a frame, cached as output/guifir/masks/<key>.png."""
    from main.video.fretboard.string_detection import fretboard_mask, load_yolo_model
    path = MASK_DIR / f"{key}.png"
    if path.exists():
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if "model" not in _YOLO:
        _YOLO["model"] = load_yolo_model(str(YOLO_WEIGHTS))
    mask = fretboard_mask(_YOLO["model"], frame_bgr)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    return mask
