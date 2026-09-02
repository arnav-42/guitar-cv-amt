#!/usr/bin/env python3
"""
Fret Detection using YOLO Segmentation + Method B (PCA + Warp)
==============================================================
Standalone script that loads a trained YOLOv8 segmentation model,
predicts the fretboard mask, and detects fret positions using the
PCA + Perspective Warp method (best-performing method from evaluation).

Usage:
  python fret_detect_yolo.py --image path/to/guitar.jpg
  python fret_detect_yolo.py --image path/to/guitar.jpg --weights yolo_weights_best.pt
  python fret_detect_yolo.py --image path/to/guitar.jpg --visualize
"""

import os, sys, argparse
import numpy as np
import cv2
from PIL import Image

# ── CONFIG ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(SCRIPT_DIR, "yolo_weights_best.pt")
YOLO_CONF_THRESH = 0.25
NUM_FRETS = 21


# ═══════════════════════════════════════
# YOLO MODEL
# ═══════════════════════════════════════
def load_yolo_model(weights_path):
    """Load a YOLOv8 segmentation model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found")
        sys.exit(1)
    return YOLO(weights_path)


def predict_mask(model, img_pil):
    """YOLO inference → cleaned binary mask (uint8, 0/255)."""
    W, H = img_pil.size
    results = model.predict(img_pil, conf=YOLO_CONF_THRESH, verbose=False)
    combined = np.zeros((H, W), dtype=np.uint8)
    if results and results[0].masks is not None:
        for mask_data in results[0].masks.data:
            m = mask_data.cpu().numpy()
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            combined = np.logical_or(combined, m > 0.5).astype(np.uint8)
    raw = (combined * 255).astype(np.uint8)
    return _clean_mask(raw)


def _clean_mask(mask, ks=5):
    """Morphological cleanup: close, open, keep largest component."""
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    nl, lab, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if nl <= 1:
        return (mask > 0).astype(np.uint8) * 255
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == largest).astype(np.uint8) * 255


# ═══════════════════════════════════════
# FRET DETECTION — Method B: PCA + Warp
# ═══════════════════════════════════════
def _rule18(length, nf=NUM_FRETS):
    """Rule of 18 fret positions along a normalized scale length."""
    c = 17.817
    r = 1 - 1 / c
    sl = length / (1 - r**nf)
    pos, cur, rem = [], 0.0, sl
    for _ in range(nf):
        d = rem / c
        cur += d
        rem -= d
        pos.append(cur)
    return pos


def _pca_corners(mask):
    """PCA-based oriented bounding box with tapered edge fitting."""
    ys, xs = np.where(mask > 0)
    if len(ys) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(0)
    c = pts - mean
    vals, vecs = np.linalg.eig(np.cov(c.T))
    idx = np.argsort(vals)[::-1]
    maj, mnr = vecs[:, idx[0]], vecs[:, idx[1]]
    u, v = c @ maj, c @ mnr
    um, uM = u.min(), u.max()
    mg = (uM - um) * 0.1
    wmin = np.std(v[u < um + mg]) if np.any(u < um + mg) else 1e9
    wmax = np.std(v[u > uM - mg]) if np.any(u > uM - mg) else 1e9
    if wmax < wmin:
        maj = -maj
        u = c @ maj
        um, uM = u.min(), u.max()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c_max = max(contours, key=cv2.contourArea)
    c_pts = c_max.reshape(-1, 2).astype(np.float32)

    c_v = (c_pts - mean) @ mnr
    top_mask = c_v > 0
    bot_mask = c_v <= 0
    top_pts = c_pts[top_mask]
    bot_pts = c_pts[bot_mask]

    if len(top_pts) > 10 and len(bot_pts) > 10:
        lt = cv2.fitLine(top_pts, cv2.DIST_L1, 0, 0.01, 0.01)
        lb = cv2.fitLine(bot_pts, cv2.DIST_L1, 0, 0.01, 0.01)
        vx_t, vy_t, x_t, y_t = lt[0][0], lt[1][0], lt[2][0], lt[3][0]
        vx_b, vy_b, x_b, y_b = lb[0][0], lb[1][0], lb[2][0], lb[3][0]

        def intersect(ray_p, ray_v, line_p0, line_n):
            denom = np.dot(ray_v, line_n)
            if abs(denom) < 1e-6:
                return ray_p
            t = -np.dot(ray_p - line_p0, line_n) / denom
            return ray_p + t * ray_v

        pt_L = mean + um * maj
        pt_R = mean + uM * maj

        vt = np.array([vx_t, vy_t])
        pt = np.array([x_t, y_t])
        vb = np.array([vx_b, vy_b])
        pb = np.array([x_b, y_b])

        tr = intersect(pt, vt, pt_R, maj)
        br = intersect(pb, vb, pt_R, maj)
        tl = intersect(pt, vt, pt_L, maj)
        bl = intersect(pb, vb, pt_L, maj)

        v_tl = np.dot(tl - mean, mnr)
        v_bl = np.dot(bl - mean, mnr)
        if v_tl > v_bl:
            tl, bl = bl, tl
            tr, br = br, tr
        return np.array([tl, tr, br, bl], dtype=np.float32)
    else:
        vm, vM = v.min(), v.max()
        tl = mean + um * maj + vm * mnr
        tr = mean + uM * maj + vm * mnr
        br = mean + uM * maj + vM * mnr
        bl = mean + um * maj + vM * mnr
        return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_frets(mask):
    """
    Method B: PCA + Perspective Warp fret detection.

    Uses PCA to find oriented fretboard corners, warps to a canonical
    rectangle, places frets via the Rule of 18, then projects back to
    image coordinates.

    Args:
        mask: Binary mask (uint8, 0/255) of the fretboard region.

    Returns:
        List of fret lines as ((x1, y1), (x2, y2)) tuples in image coords.
    """
    src = _pca_corners(mask)
    if src is None:
        return []
    OW, OH = 1024, 256
    dst = np.array([[0, 0], [OW - 1, 0], [OW - 1, OH - 1], [0, OH - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    Hi = np.linalg.inv(H)
    lines = []
    for fx in _rule18(OW):
        if fx >= OW:
            break
        p1 = Hi @ [fx, 0, 1]
        p1 /= p1[2]
        p2 = Hi @ [fx, OH - 1, 1]
        p2 /= p2[2]
        lines.append(((p1[0], p1[1]), (p2[0], p2[1])))
    return lines


# ═══════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════
def visualize(img_path, mask, fret_lines, output_path=None):
    """Draw fret lines on the image and optionally save."""
    img = cv2.imread(img_path)
    if img is None:
        return
    # Semi-transparent green mask overlay
    overlay = img.copy()
    overlay[mask > 0] = (0, 200, 0)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    # Draw fret lines
    for i, (p1, p2) in enumerate(fret_lines):
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        cv2.putText(img, str(i + 1), (mid_x - 5, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")
    else:
        cv2.imshow("Fret Detection (PCA + Warp) — YOLO Mask", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Detect frets using YOLO segmentation + PCA+Warp (Method B)"
    )
    parser.add_argument("--image", required=True, help="Path to guitar image")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to YOLO segmentation weights")
    parser.add_argument("--visualize", action="store_true",
                        help="Show/save visualization of detected frets")
    parser.add_argument("--output", default=None,
                        help="Output path for visualization image (default: display)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    # Load model
    print(f"Loading YOLO model from {args.weights}...")
    model = load_yolo_model(args.weights)

    # Predict mask
    img_pil = Image.open(args.image).convert("RGB")
    print("Predicting fretboard mask...")
    mask = predict_mask(model, img_pil)
    mask_coverage = np.count_nonzero(mask) / mask.size * 100
    print(f"  Mask coverage: {mask_coverage:.1f}% of image")

    # Detect frets
    print("Detecting frets (PCA + Warp)...")
    fret_lines = detect_frets(mask)
    print(f"  Detected {len(fret_lines)} frets")

    # Print fret positions
    for i, (p1, p2) in enumerate(fret_lines):
        mid_x = (p1[0] + p2[0]) / 2
        print(f"  Fret {i+1:2d}: x={mid_x:.1f}px")

    # Visualize
    if args.visualize:
        visualize(args.image, mask, fret_lines, args.output)


if __name__ == "__main__":
    main()
