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

import argparse
import json
import os
import sys
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


def predict_mask(model, img_pil, conf=YOLO_CONF_THRESH):
    """YOLO inference → cleaned binary mask (uint8, 0/255)."""
    W, H = img_pil.size
    results = model.predict(img_pil, conf=conf, verbose=False)
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


def _as_rgb_array(image):
    """Convert a PIL image or numpy image to an RGB uint8 array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as loaded:
            return np.asarray(loaded.convert("RGB"), dtype=np.uint8).copy()

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must be a PIL image or an HxW/HxWx3/HxWx4 numpy array")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.copy()


def _as_mask_array(mask):
    """Convert a mask-like value to a binary uint8 mask (0/255)."""
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask.convert("L"))
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be a single-channel HxW array")
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def create_annotated_output(image, mask, fret_lines, *, as_pil=False):
    """Create an annotated RGB image in memory.

    ``image`` may be a PIL image or an RGB numpy array (a path is also
    accepted for convenience).  The returned value is an RGB ``uint8`` array
    by default, or a PIL image when ``as_pil=True``.  This function never
    writes a file or opens a display window.
    """
    rgb = _as_rgb_array(image)
    binary_mask = _as_mask_array(mask)
    if binary_mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"mask dimensions {binary_mask.shape[::-1]} do not match image "
            f"dimensions {rgb.shape[1::-1]}"
        )

    overlay = rgb.copy()
    overlay[binary_mask > 0] = (0, 200, 0)
    annotated = cv2.addWeighted(overlay, 0.3, rgb, 0.7, 0)
    for i, (p1, p2) in enumerate(fret_lines):
        start = (int(round(float(p1[0]))), int(round(float(p1[1]))))
        end = (int(round(float(p2[0]))), int(round(float(p2[1]))))
        cv2.line(annotated, start, end, (255, 0, 0), 2)
        mid_x = int(round((float(p1[0]) + float(p2[0])) / 2))
        mid_y = int(round((float(p1[1]) + float(p2[1])) / 2))
        cv2.putText(annotated, str(i + 1), (mid_x - 5, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if as_pil:
        return Image.fromarray(annotated, mode="RGB")
    return annotated


def rectify_perspective(image_or_mask, mask=None, output_size=(1024, 256), *,
                        return_details=False):
    """Warp a detected fretboard to a clean rectangle using PCA corners.

    Pass ``(image, mask)`` to warp an RGB image, or pass just ``mask`` to
    obtain a warped mask.  ``output_size`` is ``(width, height)``.  A return
    value of ``None`` indicates that PCA could not estimate four corners.  If
    ``return_details=True``, a dictionary containing the warped source,
    warped mask, source corners, and transform matrix is returned instead.
    """
    if mask is None:
        mask_arr = _as_mask_array(image_or_mask)
        source = mask_arr
        source_is_mask = True
    else:
        mask_arr = _as_mask_array(mask)
        source = _as_rgb_array(image_or_mask)
        if source.shape[:2] != mask_arr.shape:
            raise ValueError(
                f"mask dimensions {mask_arr.shape[::-1]} do not match image "
                f"dimensions {source.shape[1::-1]}"
            )
        source_is_mask = False

    try:
        out_width, out_height = (int(output_size[0]), int(output_size[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError("output_size must be a (width, height) pair") from None
    if out_width <= 0 or out_height <= 0:
        raise ValueError("output_size dimensions must be positive")

    src = _pca_corners(mask_arr)
    if src is None:
        return None
    dst = np.array(
        [[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1],
         [0, out_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    interpolation = cv2.INTER_NEAREST if source_is_mask else cv2.INTER_LINEAR
    warped_source = cv2.warpPerspective(
        source, transform, (out_width, out_height), flags=interpolation
    )
    if return_details:
        warped_mask = cv2.warpPerspective(
            mask_arr, transform, (out_width, out_height), flags=cv2.INTER_NEAREST
        )
        return {
            "image": None if source_is_mask else warped_source,
            "mask": warped_mask,
            "warped": warped_source,
            "corners": src,
            "transform": transform,
            "size": (out_width, out_height),
        }
    return warped_source


def rectify_fretboard(mask, image=None, output_size=(1024, 256), *,
                      return_details=False):
    """Convenience wrapper around :func:`rectify_perspective`.

    ``mask`` is the cleaned binary fretboard mask.  When ``image`` is
    supplied, the image is warped; otherwise the mask itself is warped.
    """
    source = mask if image is None else image
    return rectify_perspective(
        source, None if image is None else mask, output_size,
        return_details=return_details,
    )


# ═══════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════
def visualize(img_path, mask, fret_lines, output_path=None):
    """Draw fret lines on the image and optionally save."""
    try:
        img = create_annotated_output(img_path, mask, fret_lines)
    except (OSError, ValueError):
        return
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if output_path:
        cv2.imwrite(output_path, img_bgr)
        print(f"Saved visualization to {output_path}")
    else:
        cv2.imshow("Fret Detection (PCA + Warp) — YOLO Mask", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _point_json(point):
    """Convert a point containing numpy scalars into JSON-native values."""
    return {"x": round(float(point[0]), 3), "y": round(float(point[1]), 3)}


def _fret_records(fret_lines):
    records = []
    for index, (start, end) in enumerate(fret_lines, start=1):
        start_json = _point_json(start)
        end_json = _point_json(end)
        midpoint = {
            "x": round((start_json["x"] + end_json["x"]) / 2.0, 3),
            "y": round((start_json["y"] + end_json["y"]) / 2.0, 3),
        }
        records.append({
            "number": index,
            "start": start_json,
            "end": end_json,
            "midpoint": midpoint,
        })
    return records


def build_json_report(image_path, image_size, weights_path, conf_threshold,
                      mask, fret_lines, rectified_dimensions=None,
                      rectified_output=None, rectified_corners=None,
                      perspective_transform=None):
    """Build a machine-readable inference report using JSON-native values."""
    width, height = (int(image_size[0]), int(image_size[1]))
    mask_pixels = int(np.count_nonzero(mask))
    total_pixels = int(np.asarray(mask).size)
    coverage = (mask_pixels / total_pixels * 100.0) if total_pixels else 0.0
    report = {
        "schema_version": "1.0",
        "image": {
            "path": os.fspath(image_path),
            "width": width,
            "height": height,
            "dimensions": [width, height],
            "mode": "RGB",
            "channels": 3,
        },
        "model": {
            "type": "yolov8-seg",
            "weights": os.fspath(weights_path),
            "confidence_threshold": float(conf_threshold),
        },
        "mask": {
            "coverage_percent": float(coverage),
            "foreground_pixels": mask_pixels,
            "total_pixels": total_pixels,
        },
        "fret_count": int(len(fret_lines)),
        "frets": _fret_records(fret_lines),
    }
    if rectified_dimensions is not None:
        rect_width, rect_height = (
            int(rectified_dimensions[0]), int(rectified_dimensions[1])
        )
        report["rectified"] = {
            "width": rect_width,
            "height": rect_height,
            "dimensions": [rect_width, rect_height],
        }
        if rectified_output is not None:
            report["rectified"]["output_path"] = os.fspath(rectified_output)
        if rectified_corners is not None:
            report["rectified"]["source_corners"] = [
                _point_json(point) for point in rectified_corners
            ]
        if perspective_transform is not None:
            report["rectified"]["perspective_transform"] = [
                [round(float(value), 6) for value in row]
                for row in perspective_transform
            ]
    return report


def _write_json_report(path, report):
    """Write a report, creating its parent directory when needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


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
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Write machine-readable inference results to this JSON path")
    parser.add_argument("--rectified-output", default=None,
                        help="Save the PCA-perspective-rectified fretboard image")
    parser.add_argument("--conf", type=float, default=YOLO_CONF_THRESH,
                        help=f"YOLO confidence threshold (default: {YOLO_CONF_THRESH})")
    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    # Load model
    print(f"Loading YOLO model from {args.weights}...")
    model = load_yolo_model(args.weights)

    # Predict mask
    try:
        img_pil = Image.open(args.image).convert("RGB")
    except (OSError, ValueError) as exc:
        print(f"ERROR: Could not read image {args.image}: {exc}")
        sys.exit(1)
    print("Predicting fretboard mask...")
    mask = predict_mask(model, img_pil, conf=args.conf)
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

    rectification = None
    if args.rectified_output:
        try:
            rectification = rectify_perspective(
                img_pil, mask, return_details=True
            )
            if rectification is None:
                print("ERROR: Could not estimate fretboard corners for rectification")
                sys.exit(1)
            parent = os.path.dirname(os.path.abspath(args.rectified_output))
            if parent:
                os.makedirs(parent, exist_ok=True)
            Image.fromarray(rectification["image"], mode="RGB").save(
                args.rectified_output
            )
            print(f"Saved rectified fretboard to {args.rectified_output}")
        except (OSError, ValueError) as exc:
            print(f"ERROR: Could not create rectified output: {exc}")
            sys.exit(1)

    if args.json_path:
        report = build_json_report(
            args.image,
            img_pil.size,
            args.weights,
            args.conf,
            mask,
            fret_lines,
            rectified_dimensions=(
                None if rectification is None else rectification["size"]
            ),
            rectified_output=args.rectified_output,
            rectified_corners=(
                None if rectification is None else rectification["corners"]
            ),
            perspective_transform=(
                None if rectification is None else rectification["transform"]
            ),
        )
        try:
            _write_json_report(args.json_path, report)
        except (OSError, TypeError, ValueError) as exc:
            print(f"ERROR: Could not write JSON report to {args.json_path}: {exc}")
            sys.exit(1)
        print(f"Saved JSON report to {args.json_path}")

    # Visualize
    if args.visualize:
        visualize(args.image, mask, fret_lines, args.output)


if __name__ == "__main__":
    main()
