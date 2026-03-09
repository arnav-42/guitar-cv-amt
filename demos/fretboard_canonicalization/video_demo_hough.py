"""
python demos/fretboard_canonicalization/video_demo_hough.py \
  --input demos/fretboard_canonicalization/guitar_25s.mp4 \
  --weights demos/fretboard_canonicalization/model_weights.pt \
  --output demos/fretboard_canonicalization/guitar_25s_result.mp4 \
  --calib 10
"""

import os
import sys
import argparse
import time
from itertools import product

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
OUT_WIDTH    = 1024
OUT_HEIGHT   = 256
CONF_THRESH  = 0.5
CALIB_FRAMES = 10       
SKIP_FRAMES  = 2  
STRING_NAMES = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"[INFO] Using device: {device}")

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
def get_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model

def load_model(weights_path):
    model = get_model()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    # 1) 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # 2) largest component만 먼저 추출
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_opened > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        cleaned = (mask_opened > 0).astype(np.uint8) * 255
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        cleaned = (labels == largest_label).astype(np.uint8) * 255

    # 3) 왼쪽 경계만 저장 (dilate 전에 미리 기록)
    cols_with_pixels = np.where((cleaned > 0).any(axis=0))[0]
    left_boundary = cols_with_pixels[0] if len(cols_with_pixels) > 0 else 0

    # 4) 가로 방향으로만 dilate
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=3)

    # 5) 왼쪽 경계만 clamp (왼쪽은 원본, 오른쪽은 dilate된 것 그대로)
    cleaned[:, :left_boundary] = 0

    return cleaned

transform = T.Compose([T.ToTensor()])

# ─────────────────────────────────────────────
# Frame → warped_bgr
# ─────────────────────────────────────────────
def frame_to_warped(model, frame_bgr):
    from PIL import Image
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 1) Mask R-CNN segmentation
    img_tensor = transform(img_pil).to(device)
    with torch.no_grad():
        pred = model([img_tensor])[0]

    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    masks  = pred["masks"].cpu().numpy()

    H, W = frame_bgr.shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)
    for i, score in enumerate(scores):
        if score < CONF_THRESH or labels[i] != 1:
            continue
        m = (masks[i, 0] > 0.5).astype(np.uint8)
        combined = np.logical_or(combined, m).astype(np.uint8)

    if combined.sum() == 0:
        return None, None

    combined = (combined * 255).astype(np.uint8)

    # 2) Clean mask (custom pipeline)
    mask = clean_mask(combined)

    # largest component only (after cleaning, ensure non-empty)
    n, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ((labels_cc == largest).astype(np.uint8) * 255)

    # 3) Min-area rect → corners
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box  = cv2.boxPoints(rect).astype(np.float32)

    s    = box.sum(axis=1)
    diff = np.diff(box, axis=1)[:, 0]
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    src = np.array([tl, tr, bl, br], dtype=np.float32)

    # 4) warpPerspective
    dst = np.array([[0,0],[OUT_WIDTH-1,0],[0,OUT_HEIGHT-1],[OUT_WIDTH-1,OUT_HEIGHT-1]],
                   dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame_bgr, Hmat, (OUT_WIDTH, OUT_HEIGHT))
    return warped, Hmat

# ─────────────────────────────────────────────
# Fret detection
# ─────────────────────────────────────────────
def detect_frets_in_warped(warped_bgr):
    H, W = warped_bgr.shape[:2]
    gray  = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=80,
                             minLineLength=int(H * 0.5),
                             maxLineGap=30)
    if lines is None:
        return []

    vertical_xs = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y2 == y1:
            continue
        angle = abs(np.degrees(np.arctan2(x2-x1, y2-y1)))
        if angle < 5 or angle > 175:
            vertical_xs.append((x1+x2)//2)

    if not vertical_xs:
        return []

    vertical_xs = sorted(vertical_xs)
    merged, cluster = [], []
    for x in vertical_xs:
        if not cluster:
            cluster = [x]
        elif x - cluster[-1] <= 15:
            cluster.append(x)
        else:
            merged.append(int(np.mean(cluster)))
            cluster = [x]
    if cluster:
        merged.append(int(np.mean(cluster)))

    return sorted(merged)

# ─────────────────────────────────────────────
# String detection
# ─────────────────────────────────────────────
def detect_strings_in_warped(warped_bgr, expected_strings=6):
    H, W = warped_bgr.shape[:2]
    gray  = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=80,
                             minLineLength=int(W * 0.5),
                             maxLineGap=30)
    if lines is None:
        return []

    horizontal_ys = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if angle < 5 or angle > 175:
            horizontal_ys.append((y1+y2)//2)

    if not horizontal_ys:
        return []

    horizontal_ys = sorted(horizontal_ys)
    merged, cluster = [], []
    for y in horizontal_ys:
        if not cluster:
            cluster = [y]
        elif y - cluster[-1] <= 12:
            cluster.append(y)
        else:
            merged.append(int(np.mean(cluster)))
            cluster = [y]
    if cluster:
        merged.append(int(np.mean(cluster)))

    return sorted(merged)[:expected_strings]

# ─────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────
COLORS = [
    (255,  50,  50), ( 50, 255,  50), ( 50,  50, 255),
    (255, 255,  50), ( 50, 255, 255), (255,  50, 255),
]

def draw_overlay(warped, fret_xs, string_ys, calibrated=False):
    vis = warped.copy()
    H, W = vis.shape[:2]

    # frets (세로선)
    for i, x in enumerate(fret_xs):
        color = COLORS[i % len(COLORS)]
        cv2.line(vis, (x, 0), (x, H), color, 2)
        cv2.putText(vis, str(i), (x-6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # strings (가로선)
    for i, y in enumerate(string_ys):
        cv2.line(vis, (0, y), (W, y), (255, 255, 255), 1)
        cv2.putText(vis, STRING_NAMES[i] if i < 6 else f"S{i}",
                    (5, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # 캘리브레이션 상태
    label = f"Frets: {len(fret_xs)}  Strings: {len(string_ys)}"
    if calibrated:
        label += "  [CALIBRATED - occlusion safe]"
    cv2.rectangle(vis, (0, H-20), (W, H), (0,0,0), -1)
    cv2.putText(vis, label, (5, H-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,200), 1)
    return vis

# ─────────────────────────────────────────────
# Main video loop
# ─────────────────────────────────────────────
def run_video_pipeline(video_path: str, weights_path: str, output_path: str = None):
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    assert os.path.exists(weights_path), f"Weights not found: {weights_path}"

    print(f"[INFO] Loading model from {weights_path}...")
    model = load_model(weights_path)

    cap = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {vid_w}x{vid_h} @ {fps:.1f}fps, {total} frames")


    out_h = vid_h + OUT_HEIGHT + 10
    out_w = max(vid_w, OUT_WIDTH)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        print(f"[INFO] Output: {output_path}")


    calib_fret_xs   = [] 
    calib_string_ys = []
    calibrated       = False 
    calib_count      = 0
    calib_fret_pool  = []

    last_warped   = None
    last_fret_xs  = []
    last_string_ys= []
    frame_idx     = 0

    print("\n[INFO] Starting processing... Press Q to quit (if showing window)")
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        show_progress = frame_idx % 30 == 0
        if show_progress:
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx}/{total} ({frame_idx/total*100:.1f}%) | "
                  f"{elapsed:.1f}s elapsed | calib={'✓' if calibrated else f'{calib_count}/{CALIB_FRAMES}'}")


        run_rcnn = (frame_idx % SKIP_FRAMES == 0)

        if run_rcnn:
            warped, _ = frame_to_warped(model, frame)

            if warped is not None:
                last_warped = warped

                # fret / string 
                fret_xs   = detect_frets_in_warped(warped)
                string_ys = detect_strings_in_warped(warped)

                
                if not calibrated:
                    if len(fret_xs) >= 5: 
                        calib_fret_pool.append(fret_xs)
                        calib_count += 1
                        print(f"  [CALIB {calib_count}/{CALIB_FRAMES}] "
                              f"{len(fret_xs)} frets detected")

                    if calib_count >= CALIB_FRAMES:

                        target_n = max(set(len(x) for x in calib_fret_pool),
                                       key=lambda n: sum(1 for x in calib_fret_pool if len(x)==n))
                        pool = [x for x in calib_fret_pool if len(x) == target_n]

                        calib_fret_xs = [int(np.mean([p[i] for p in pool]))
                                         for i in range(target_n)]
                        calib_string_ys = string_ys if string_ys else []
                        calibrated = True
                        print(f"\n[CALIB DONE] Locked {len(calib_fret_xs)} frets: {calib_fret_xs}\n")


                if calibrated:
                    if len(fret_xs) >= len(calib_fret_xs) * 0.6:

                        last_fret_xs  = fret_xs
                        last_string_ys= string_ys if string_ys else calib_string_ys
                    else:

                        last_fret_xs  = calib_fret_xs
                        last_string_ys= calib_string_ys
                        if show_progress:
                            print(f"  [OCCLUSION] Using calibrated frets "
                                  f"(detected {len(fret_xs)}, expected ~{len(calib_fret_xs)})")
                else:
                    last_fret_xs  = fret_xs
                    last_string_ys= string_ys


        if last_warped is not None:
            warped_vis = draw_overlay(last_warped, last_fret_xs, last_string_ys, calibrated)
        else:
            warped_vis = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
            cv2.putText(warped_vis, "Waiting for fretboard detection...",
                        (20, OUT_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)


        frame_vis = frame.copy()
        status = f"Frame {frame_idx} | Frets: {len(last_fret_xs)} | {'CALIBRATED' if calibrated else 'CALIBRATING...'}"
        cv2.rectangle(frame_vis, (0,0), (len(status)*10+10, 28), (0,0,0), -1)
        cv2.putText(frame_vis, status, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,200), 1)


        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:vid_h, :vid_w] = frame_vis

        x_off = (out_w - OUT_WIDTH) // 2
        canvas[vid_h+10:vid_h+10+OUT_HEIGHT, x_off:x_off+OUT_WIDTH] = warped_vis

        if writer:
            writer.write(canvas)


        display_frame = cv2.resize(canvas, (out_w//2, out_h//2))
        cv2.imshow("Guitar Fret Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n[DONE] Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps)")
    if output_path:
        print(f"[DONE] Saved to: {output_path}")

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guitar fret detection from video")
    parser.add_argument("--input",   required=True,  help="Input video path (e.g. guitar.mp4)")
    parser.add_argument("--weights", required=True,  help="Mask R-CNN weights path (model_weights.pt)")
    parser.add_argument("--output",  default=None,   help="Output video path (optional, e.g. output.mp4)")
    parser.add_argument("--calib",   type=int, default=10,
                        help=f"Calibration frames (default: {CALIB_FRAMES})")
    parser.add_argument("--skip",    type=int, default=2,
                        help=f"Run Mask R-CNN every N frames (default: {SKIP_FRAMES})")
    args = parser.parse_args()

    CALIB_FRAMES = args.calib
    SKIP_FRAMES  = args.skip

    run_video_pipeline(
        video_path   = args.input,
        weights_path = args.weights,
        output_path  = args.output,
    )