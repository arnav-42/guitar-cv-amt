#!/usr/bin/env python3
"""
Fret Detection Evaluation Pipeline (Optimized)
================================================
Runs Mask R-CNN inference ONCE per image, caches masks, then runs all 5 fret
detection methods on the cached results. Produces standardized metrics for a
research paper comparison.

Usage:
  python evaluate_all.py                        # uses model_weights.pt
  python evaluate_all.py --weights model_weights_new.pt  # uses retrained model
"""

import os, sys, math, warnings, json, argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_util
from scipy.optimize import linear_sum_assignment
import scipy.signal

warnings.filterwarnings("ignore")

# ── CONFIG ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
NUM_CLASSES = 2  # background + fretboard
CONF_THRESH = 0.5
NUM_FRETS = 21
IOU_THRESH = 0.5
PX_THRESH = 20.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = T.Compose([T.ToTensor()])
YOLO_CONF_THRESH = 0.25
# Zone categories represent individual fret zones (Zone1-zone12) + Frets (overall)
ZONE_CAT_IDS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# Only individual zone categories (not the overall "Frets" bbox) for IoU matching
INDIV_ZONE_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# ═══════════════════════════════════════
# MODEL
# ═══════════════════════════════════════
def build_model(num_classes):
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    in_fm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, num_classes)
    return m

def load_model(weights_path):
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found"); sys.exit(1)
    m = build_model(NUM_CLASSES)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    m.to(DEVICE); m.eval()
    return m

# ═══════════════════════════════════════
# YOLO MODEL
# ═══════════════════════════════════════
def load_yolo_model(weights_path):
    """Load a YOLO segmentation model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found"); sys.exit(1)
    m = YOLO(weights_path)
    return m

def predict_mask_yolo(model, img_pil):
    """YOLO inference → combined binary mask (uint8, 0/255)."""
    import cv2
    W, H = img_pil.size
    results = model.predict(img_pil, conf=YOLO_CONF_THRESH, verbose=False)
    combined = np.zeros((H, W), dtype=np.uint8)
    if results and results[0].masks is not None:
        for mask_data in results[0].masks.data:
            m = mask_data.cpu().numpy()
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            combined = np.logical_or(combined, m > 0.5).astype(np.uint8)
    return (combined * 255).astype(np.uint8)

def precompute_masks_yolo(model, split_dir):
    """Run YOLO inference once per image, return dict {filename: cleaned_mask}."""
    coco = COCO(os.path.join(split_dir, "_annotations.coco.json"))
    cache = {}
    imgs = coco.getImgIds()
    print(f"  Pre-computing YOLO masks for {len(imgs)} images...", end=" ", flush=True)
    for img_id in imgs:
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(split_dir, info["file_name"])
        if not os.path.exists(path): continue
        pil = Image.open(path).convert("RGB")
        raw = predict_mask_yolo(model, pil)
        cache[info["file_name"]] = clean_mask(raw)
    print(f"done ({len(cache)} cached)")
    return cache, coco

# ═══════════════════════════════════════
# MASK PREDICTION & CACHE
# ═══════════════════════════════════════
def predict_mask_raw(model, img_pil):
    """Single inference → combined binary mask (uint8, 0/255)."""
    t = TRANSFORM(img_pil).to(DEVICE)
    with torch.no_grad():
        pred = model([t])[0]
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    masks  = pred["masks"].cpu().numpy()
    H, W = img_pil.size[1], img_pil.size[0]
    combined = np.zeros((H, W), dtype=np.uint8)
    for i, s in enumerate(scores):
        if s < CONF_THRESH or labels[i] != 1:
            continue
        combined = np.logical_or(combined, masks[i, 0] > 0.5).astype(np.uint8)
    return (combined * 255).astype(np.uint8)

def clean_mask(mask, ks=5):
    if mask.max() <= 1: mask = (mask * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    nl, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if nl <= 1: return (mask > 0).astype(np.uint8) * 255
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == largest).astype(np.uint8) * 255

def precompute_masks(model, split_dir):
    """Run inference once per image, return dict {filename: cleaned_mask}."""
    coco = COCO(os.path.join(split_dir, "_annotations.coco.json"))
    cache = {}
    imgs = coco.getImgIds()
    print(f"  Pre-computing masks for {len(imgs)} images...", end=" ", flush=True)
    for img_id in imgs:
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(split_dir, info["file_name"])
        if not os.path.exists(path): continue
        pil = Image.open(path).convert("RGB")
        raw = predict_mask_raw(model, pil)
        cache[info["file_name"]] = clean_mask(raw)
    print(f"done ({len(cache)} cached)")
    return cache, coco

# ═══════════════════════════════════════
# DICE SCORE
# ═══════════════════════════════════════
def compute_dice(mask_cache, coco, split_dir):
    scores = []
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        fn = info["file_name"]
        if fn not in mask_cache: continue
        H, W = info["height"], info["width"]
        pred = (mask_cache[fn] > 0).astype(np.uint8)
        # GT from polygon annotations
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=ZONE_CAT_IDS))
        gt = np.zeros((H, W), dtype=np.uint8)
        for ann in anns:
            if "segmentation" in ann and ann["segmentation"]:
                rle = coco_mask_util.frPyObjects(ann["segmentation"], H, W)
                m = coco_mask_util.decode(rle)
                if m.ndim == 3: m = m.max(axis=2)
                gt = np.logical_or(gt, m).astype(np.uint8)
        inter = np.sum(pred & gt)
        total = np.sum(pred) + np.sum(gt)
        scores.append((2.0 * inter / total) if total > 0 else 1.0)
    return scores

# ═══════════════════════════════════════
# FRET DETECTION METHODS
# Each takes (img_path, mask) → list of ((x1,y1),(x2,y2))
# ═══════════════════════════════════════
def rule18(length, nf=NUM_FRETS):
    c = 17.817; r = 1 - 1/c
    sl = length / (1 - r**nf)
    pos, cur, rem = [], 0.0, sl
    for _ in range(nf):
        d = rem / c; cur += d; rem -= d; pos.append(cur)
    return pos

def order_corners(box):
    s = box.sum(1); d = np.diff(box, axis=1)[:,0]
    return np.array([box[np.argmin(s)], box[np.argmin(d)],
                     box[np.argmax(d)], box[np.argmax(s)]], dtype=np.float32)

# ── A: Arnav's Canonicalization ──
def method_a(img_path, mask):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return []
    cnt = max(cnts, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    src = order_corners(box)
    OW, OH = 1024, 256
    dst = np.array([[0,0],[OW-1,0],[0,OH-1],[OW-1,OH-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    Hi = np.linalg.inv(H)
    lines = []
    for fx in rule18(OW):
        if fx >= OW: break
        p1 = Hi @ [fx, 0, 1]; p1 /= p1[2]
        p2 = Hi @ [fx, OH-1, 1]; p2 /= p2[2]
        lines.append(((p1[0], p1[1]), (p2[0], p2[1])))
    return lines

# ── B: PCA + Warp ──
def _pca_corners(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) < 10: return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(0); c = pts - mean
    vals, vecs = np.linalg.eig(np.cov(c.T))
    idx = np.argsort(vals)[::-1]
    maj, mnr = vecs[:, idx[0]], vecs[:, idx[1]]
    u, v = c @ maj, c @ mnr
    um, uM = u.min(), u.max()
    mg = (uM - um) * 0.1
    wmin = np.std(v[u < um + mg]) if np.any(u < um + mg) else 1e9
    wmax = np.std(v[u > uM - mg]) if np.any(u > uM - mg) else 1e9
    if wmax < wmin:
        maj = -maj; u = c @ maj; um, uM = u.min(), u.max()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
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
            if abs(denom) < 1e-6: return ray_p
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
        tl = mean + um*maj + vm*mnr
        tr = mean + uM*maj + vm*mnr
        br = mean + uM*maj + vM*mnr
        bl = mean + um*maj + vM*mnr
        return np.array([tl, tr, br, bl], dtype=np.float32)

def method_b(img_path, mask):
    src = _pca_corners(mask)
    if src is None: return []
    OW, OH = 1024, 256
    dst = np.array([[0,0],[OW-1,0],[OW-1,OH-1],[0,OH-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    Hi = np.linalg.inv(H)
    lines = []
    for fx in rule18(OW):
        if fx >= OW: break
        p1 = Hi @ [fx, 0, 1]; p1 /= p1[2]
        p2 = Hi @ [fx, OH-1, 1]; p2 /= p2[2]
        lines.append(((p1[0], p1[1]), (p2[0], p2[1])))
    return lines

# ── C: PCA no warp ──
def method_c(img_path, mask):
    ys, xs = np.where(mask > 0)
    if len(ys) < 50: return []
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(0); c = pts - mean
    vals, vecs = np.linalg.eig(np.cov(c.T))
    idx = np.argsort(vals)[::-1]
    maj, mnr = vecs[:, idx[0]], vecs[:, idx[1]]
    u, v = c @ maj, c @ mnr
    um, uM = u.min(), u.max()
    mg = (uM - um) * 0.1
    mm = u < um + mg; mM = u > uM - mg
    wn = (v[mm].max() - v[mm].min()) if mm.any() else 0
    wb = (v[mM].max() - v[mM].min()) if mM.any() else 0
    if wb < wn and wb > 0:
        maj = -maj; u = c @ maj; um, uM = u.min(), u.max()
        wn, wb = wb, wn
    nut = mean + um * maj
    vl = uM - um
    rc = 17.817; rat = 1 - 1/rc
    sl = vl / (1 - rat**NUM_FRETS)
    lines, cur, rem = [], 0.0, sl
    for i in range(1, NUM_FRETS+1):
        d = rem / rc; cur += d; rem -= d
        if cur > vl: break
        prog = cur / vl
        cw = wn + (wb - wn) * prog
        pc = nut + cur * maj
        lines.append(((pc - cw*0.5*mnr)[0:2].tolist(), (pc + cw*0.5*mnr)[0:2].tolist()))
    return [((a[0],a[1]),(b[0],b[1])) for a,b in lines]

# ── D: Paleari & Huet ──
def method_d(img_path, mask):
    img = cv2.imread(img_path)
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.count_nonzero(mask) == 0: return []
    h, w = gray.shape; ctr = (w//2, h//2)
    # Hough angle
    edges = cv2.Canny(cv2.bitwise_and(gray, gray, mask=mask), 50, 150, apertureSize=3)
    hl = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    angs = []
    if hl is not None:
        for l in hl:
            x1,y1,x2,y2 = l[0]
            if x2 != x1:
                a = np.degrees(np.arctan((y2-y1)/(x2-x1)))
                if -60 < a < 60: angs.append(a)
    angle = np.median(angs) if angs else 0.0
    # Rotate image AND mask together
    M = cv2.getRotationMatrix2D(ctr, angle, 1.0)
    rg = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC)
    rm = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    # Mask-aware Sobel: only within fretboard
    masked_gray = cv2.bitwise_and(rg, rg, mask=rm)
    enh = np.uint8(np.absolute(cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)))
    # Mask-aware H-proj crop
    ph = np.sum(rm, axis=1)
    ci = np.where(ph > 0)[0]
    if len(ci) < 10: return []
    nh = ci[-1] - ci[0]; mg = int(nh * 0.05)
    y0, y1 = max(0, ci[0]+mg), min(h, ci[-1]-mg)
    crop = enh[y0:y1, :]; crop_mask = rm[y0:y1, :]; ch = y1 - y0
    if ch < 10: return []
    # Skew optimization
    bs, mv = 0, 0
    for sk in np.linspace(-0.5, 0.5, 20):
        Ms = np.float32([[1,sk,0],[0,1,0]]); Ms[0,2] = -sk*ch/2
        skd = cv2.warpAffine(crop, Ms, (w, ch))
        v = np.var(np.sum(skd, axis=0))
        if v > mv: mv, bs = v, sk
    Ms = np.float32([[1,bs,0],[0,1,0]]); Ms[0,2] = -bs*ch/2
    sf = cv2.warpAffine(crop, Ms, (w, ch))
    sm = cv2.warpAffine(crop_mask, Ms, (w, ch), flags=cv2.INTER_NEAREST)
    # Peaks — only within mask columns
    pv = np.sum(sf, axis=0).astype(np.float64)
    # Zero out columns outside fretboard mask
    mask_col_coverage = np.sum(sm > 0, axis=0)
    pv[mask_col_coverage < ch * 0.1] = 0  # ignore columns with <10% mask
    pv /= (pv.max() + 1e-9)
    pks, _ = scipy.signal.find_peaks(pv, height=0.2, distance=15, prominence=0.05)
    # Inverse transform
    lines, so = [], bs*(ch/2)
    ar = -np.radians(angle); ox, oy = ctr
    for xd in pks:
        xt, xb = xd + so, xd - bs*ch + so
        px1, py1 = xt, float(y0); px2, py2 = xb, float(y1)
        q1 = (ox+math.cos(ar)*(px1-ox)-math.sin(ar)*(py1-oy),
              oy+math.sin(ar)*(px1-ox)+math.cos(ar)*(py1-oy))
        q2 = (ox+math.cos(ar)*(px2-ox)-math.sin(ar)*(py2-oy),
              oy+math.sin(ar)*(px2-ox)+math.cos(ar)*(py2-oy))
        lines.append((q1, q2))
    return lines

# ── E: Paleari & Huet Refined (mask-aware) ──
def method_e(img_path, mask):
    img = cv2.imread(img_path)
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.count_nonzero(mask) == 0: return []
    h, w = gray.shape; ctr = (w//2, h//2)
    # Robust orientation: Hough with fallback to minAreaRect
    edges = cv2.Canny(cv2.bitwise_and(gray, gray, mask=mask), 50, 150, apertureSize=3)
    hl = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    angs = []
    if hl is not None:
        for l in hl:
            x1,y1,x2,y2 = l[0]
            if x2 != x1: angs.append(np.degrees(np.arctan((y2-y1)/(x2-x1))))
    if angs:
        angle = np.median(angs)
    else:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            rect = cv2.minAreaRect(cnts[0])
            angle = rect[-1] + (90 if rect[1][0] < rect[1][1] else 0)
        else: angle = 0.0
    M = cv2.getRotationMatrix2D(ctr, angle, 1.0)
    rg = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC)
    rm = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    # Mask-aware crop
    ph = np.sum(rm, axis=1); ni = np.where(ph > 0)[0]
    if len(ni) < 5: return []
    nh = ni[-1] - ni[0]; mg = int(nh * 0.05)
    y0, y1 = max(0, ni[0]+mg), min(h, ni[-1]-mg)
    cg, cm = rg[y0:y1,:], rm[y0:y1,:]; ch = y1 - y0
    if ch < 10: return []
    # Mask-aware skew optimization
    mg2 = cv2.bitwise_and(cg, cg, mask=cm)
    bs, mv = 0, 0
    for sk in np.linspace(-0.5, 0.5, 30):
        Ms = np.float32([[1,sk,0],[0,1,0]]); Ms[0,2] = -sk*ch/2
        skd = cv2.warpAffine(mg2, Ms, (w, ch))
        v = np.var(np.sum(skd, axis=0))
        if v > mv: mv, bs = v, sk
    Ms = np.float32([[1,bs,0],[0,1,0]]); Ms[0,2] = -bs*ch/2
    sf = cv2.warpAffine(mg2, Ms, (w, ch))
    sm = cv2.warpAffine(cm, Ms, (w, ch), flags=cv2.INTER_NEAREST)
    # Sobel + mask-filtered peaks
    enh = np.absolute(cv2.Sobel(sf, cv2.CV_64F, 1, 0, ksize=3))
    pv = np.sum(enh, axis=0)
    # Zero out columns outside fretboard mask
    mask_col_cov = np.sum(sm > 0, axis=0)
    pv[mask_col_cov < ch * 0.1] = 0
    pv /= (pv.max() + 1e-9)
    pks, _ = scipy.signal.find_peaks(pv, height=0.15, distance=12, prominence=0.08)
    # Inverse transform
    lines, so = [], bs*(ch/2)
    ar = -np.radians(angle); ox, oy = ctr
    for xd in pks:
        xt, xb = xd + so, xd - bs*ch + so
        px1, py1 = xt, float(y0); px2, py2 = xb, float(y1)
        q1 = (ox+math.cos(ar)*(px1-ox)-math.sin(ar)*(py1-oy),
              oy+math.sin(ar)*(px1-ox)+math.cos(ar)*(py1-oy))
        q2 = (ox+math.cos(ar)*(px2-ox)-math.sin(ar)*(py2-oy),
              oy+math.sin(ar)*(px2-ox)+math.cos(ar)*(py2-oy))
        lines.append((q1, q2))
    return lines

# ── F: Hybrid PCA Warp + Peak Finding ──
def method_f(img_path, mask):
    """
    Hybrid approach: uses PCA perspective warp (Method B) for geometric
    normalization, then Sobel peak-finding (Method E) to detect only
    frets where actual edges exist. Avoids phantom frets from Rule of 18.
    """
    img = cv2.imread(img_path)
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.count_nonzero(mask) == 0: return []

    # 1. PCA corners → perspective warp to canonical space
    src = _pca_corners(mask)
    if src is None: return []
    OW, OH = 1024, 256
    dst = np.array([[0,0],[OW-1,0],[OW-1,OH-1],[0,OH-1]], dtype=np.float32)
    Hw = cv2.getPerspectiveTransform(src, dst)
    Hi = np.linalg.inv(Hw)

    warped = cv2.warpPerspective(gray, Hw, (OW, OH))
    wmask = cv2.warpPerspective(mask, Hw, (OW, OH), flags=cv2.INTER_NEAREST)

    # 2. Mask-aware processing
    mg = cv2.bitwise_and(warped, warped, mask=wmask)

    # 3. Fine skew correction in warped space
    bs, mv = 0, 0
    for sk in np.linspace(-0.3, 0.3, 25):
        Ms = np.float32([[1, sk, 0], [0, 1, 0]])
        Ms[0, 2] = -sk * OH / 2
        skd = cv2.warpAffine(mg, Ms, (OW, OH))
        v = np.var(np.sum(skd, axis=0))
        if v > mv:
            mv, bs = v, sk
    Ms = np.float32([[1, bs, 0], [0, 1, 0]])
    Ms[0, 2] = -bs * OH / 2
    sf = cv2.warpAffine(mg, Ms, (OW, OH))
    sm = cv2.warpAffine(wmask, Ms, (OW, OH), flags=cv2.INTER_NEAREST)

    # 4. Sobel vertical edges + projection profile
    enh = np.absolute(cv2.Sobel(sf, cv2.CV_64F, 1, 0, ksize=3))
    pv = np.sum(enh, axis=0)
    # Zero out columns with insufficient mask coverage
    col_cov = np.sum(sm > 0, axis=0)
    pv[col_cov < OH * 0.1] = 0
    pv /= (pv.max() + 1e-9)

    # 5. Find peaks — only where actual fret edges exist
    pks, _ = scipy.signal.find_peaks(pv, height=0.12, distance=8, prominence=0.04)

    # 6. Rule-of-18 spacing validation:
    #    Remove peaks whose spacing deviates too much from expected fret ratios.
    #    Fret spacing should decrease monotonically (nut→bridge).
    if len(pks) >= 3:
        spacings = np.diff(pks).astype(float)
        # Fret spacing ratios should be roughly constant (~0.9439 = 1 - 1/17.817)
        # Remove peaks that create anomalously small gaps (noise) or huge gaps
        median_sp = np.median(spacings)
        valid = [pks[0]]
        for i in range(1, len(pks)):
            gap = pks[i] - valid[-1]
            # Allow gaps between 30% and 300% of median spacing
            if 0.3 * median_sp <= gap <= 3.0 * median_sp:
                valid.append(pks[i])
        pks = np.array(valid)
        
    lines = []
    so = bs * (OH / 2)
    for xd in pks:
        # Undo skew
        xt = xd + so
        xb = xd - bs * OH + so
        # Undo perspective warp
        p1 = Hi @ [xt, 0, 1]; p1 /= p1[2]
        p2 = Hi @ [xb, OH - 1, 1]; p2 /= p2[2]
        lines.append(((p1[0], p1[1]), (p2[0], p2[1])))

    return lines

# ═══════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════
def calc_iou(a, b):
    x1A,y1A,wA,hA = a; x1B,y1B,wB,hB = b
    ix1,iy1 = max(x1A,x1B), max(y1A,y1B)
    ix2,iy2 = min(x1A+wA,x1B+wB), min(y1A+hA,y1B+hB)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    i = (ix2-ix1)*(iy2-iy1); u = wA*hA + wB*hB - i
    return i/u if u > 0 else 0.0

def line_to_bbox(p1, p2, pad=5.0):
    xn,yn = min(p1[0],p2[0])-pad, min(p1[1],p2[1])-pad
    xx,yx = max(p1[0],p2[0])+pad, max(p1[1],p2[1])+pad
    return [xn, yn, xx-xn, yx-yn]

def gt_fret_targets(boxes, merge=15.0):
    """Convert zone bboxes to fret-line x-targets by extracting and merging edges."""
    edges = []
    for x,y,w,h in boxes:
        edges.append({"x":x,"ymin":y,"ymax":y+h})
        edges.append({"x":x+w,"ymin":y,"ymax":y+h})
    if not edges: return []
    edges.sort(key=lambda e: e["x"])
    merged, cl = [], [edges[0]]
    for e in edges[1:]:
        if e["x"] - cl[-1]["x"] < merge: cl.append(e)
        else:
            merged.append({"x":sum(c["x"] for c in cl)/len(cl),
                          "ymin":min(c["ymin"] for c in cl), "ymax":max(c["ymax"] for c in cl)})
            cl = [e]
    merged.append({"x":sum(c["x"] for c in cl)/len(cl),
                  "ymin":min(c["ymin"] for c in cl), "ymax":max(c["ymax"] for c in cl)})
    return merged


def eval_fret_lines(coco, split_dir, lines_cache, rel_thresholds=[0.5, 1.0, 2.0]):
    """
    Fret-line evaluation via Hungarian matching on pixel distance,
    using relative thresholds (% of image width) to fairly compare
    across mixed resolutions (640px to 3840px).

    Thresholds are given as percentages of image width.
    E.g. 1.0 means 1% of width → 6.4px at 640px, 38.4px at 3840px.

    Returns: P, R, F1 at each threshold, plus Mean Pixel Error (normalized).
    """
    results = {t: {"tp": 0, "fp": 0, "fn": 0} for t in rel_thresholds}
    all_norm_dists = []  # distances normalized by image width

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        fname = info["file_name"]
        img_w = info["width"]
        # GT: zone edges → merged fret targets (scale merge distance by image width)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=ZONE_CAT_IDS))
        gt_boxes = [a["bbox"] for a in anns]
        merge_dist = max(img_w * 0.005, 5)  # 0.5% of width, min 5px
        gts = gt_fret_targets(gt_boxes, merge=merge_dist)
        # Pred: line midpoints
        preds = []
        for l in lines_cache.get(fname, []):
            x1, y1 = l[0]; x2, y2 = l[1]
            preds.append({"x": (x1 + x2) / 2.0, "ymin": min(y1, y2), "ymax": max(y1, y2)})

        if not gts:
            for t in rel_thresholds:
                results[t]["fp"] += len(preds)
            continue
        if not preds:
            for t in rel_thresholds:
                results[t]["fn"] += len(gts)
            continue

        # Cost matrix: horizontal pixel distance with vertical overlap check
        vert_tol = max(img_w * 0.02, 15)  # scaled vertical tolerance
        cost = np.zeros((len(gts), len(preds)))
        for r, g in enumerate(gts):
            for c, p in enumerate(preds):
                d = abs(g["x"] - p["x"])
                if p["ymax"] < g["ymin"] - vert_tol or p["ymin"] > g["ymax"] + vert_tol:
                    d = 9999.0
                cost[r, c] = d

        ri, ci = linear_sum_assignment(cost)

        # Record matched distances (normalized by image width)
        matched_dists = {}
        for r, c in zip(ri, ci):
            matched_dists[(r, c)] = cost[r, c]
            norm_d = cost[r, c] / img_w * 100  # as % of width
            if norm_d <= 5.0:  # within 5% of width — reasonable match
                all_norm_dists.append(norm_d)

        # Apply each threshold (relative to image width)
        for t in rel_thresholds:
            px_thresh = img_w * t / 100.0  # convert % to pixels
            tp = sum(1 for d in matched_dists.values() if d <= px_thresh)
            results[t]["tp"] += tp
            results[t]["fp"] += len(preds) - tp
            results[t]["fn"] += len(gts) - tp

    # Compute metrics per threshold
    out = {}
    for t in rel_thresholds:
        tp, fp, fn = results[t]["tp"], results[t]["fp"], results[t]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        out[f"P@{t}%"] = round(p, 4)
        out[f"R@{t}%"] = round(r, 4)
        out[f"F1@{t}%"] = round(f1, 4)

    out["MPE(%w)"] = round(np.mean(all_norm_dists), 3) if all_norm_dists else float("nan")
    return out



# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
METHODS = [
    ("A: Arnav Canon.", method_a),
    ("B: PCA + Warp",   method_b),
    ("C: PCA (no warp)",method_c),
    ("D: Paleari&Huet", method_d),
    ("E: Paleari Ref.", method_e),
    ("F: Hybrid",       method_f),
]

def run_evaluation(model_name, split_data, prefix=""):
    """Run all fret detection methods on cached masks and return rows + dice."""
    # Dice scores
    all_dice = []
    for sname, (cache, coco, spath) in split_data.items():
        ds = compute_dice(cache, coco, spath)
        if ds:
            all_dice.extend(ds)
            print(f"  {sname:>6s}: Dice = {np.mean(ds):.4f} ± {np.std(ds):.4f}  (n={len(ds)})")
    overall_dice = np.mean(all_dice) if all_dice else 0
    print(f"  {'ALL':>6s}: Dice = {overall_dice:.4f} ± {np.std(all_dice):.4f}  (n={len(all_dice)})")

    # Run methods
    rows = []
    for mname, mfn in METHODS:
        label = f"{prefix}{mname}" if prefix else mname
        print(f"\nMethod: {label}")
        for sname, (cache, coco, spath) in split_data.items():
            print(f"  {sname}...", end=" ", flush=True)
            lines_cache = {}
            for fname, cmask in cache.items():
                fpath = os.path.join(spath, fname)
                lines_cache[fname] = mfn(fpath, cmask)
            res = eval_fret_lines(coco, spath, lines_cache)
            row = {"Method": label, "Split": sname}
            row.update(res)
            rows.append(row)
            print("done")

    return rows, overall_dice


def main():
    parser = argparse.ArgumentParser(description="Fret Detection Evaluation")
    parser.add_argument("--weights", default=os.path.join(SCRIPT_DIR, "model_weights.pt"),
                        help="Path to Mask R-CNN weights file")
    parser.add_argument("--yolo-weights", default=None,
                        help="Path to YOLO segmentation weights (optional)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Mask R-CNN Weights: {args.weights}")
    if args.yolo_weights:
        print(f"YOLO Weights: {args.yolo_weights}")
    print()

    splits = [(n, os.path.join(DATASET_DIR, n)) for n in ["test","valid"]
              if os.path.isdir(os.path.join(DATASET_DIR, n))]

    all_rows = []
    dice_results = {}

    # ═══════════════════════════════════════
    # MASK R-CNN EVALUATION
    # ═══════════════════════════════════════
    print("=" * 72)
    print("MASK R-CNN EVALUATION")
    print("=" * 72)
    model = load_model(args.weights)
    print("Mask R-CNN loaded.\n")

    mrcnn_data = {}
    for sname, spath in splits:
        cache, coco = precompute_masks(model, spath)
        mrcnn_data[sname] = (cache, coco, spath)

    print("\nMask R-CNN Dice Scores:")
    mrcnn_rows, mrcnn_dice = run_evaluation("Mask R-CNN", mrcnn_data, prefix="")
    all_rows.extend(mrcnn_rows)
    dice_results["Mask R-CNN"] = mrcnn_dice

    # ═══════════════════════════════════════
    # YOLO EVALUATION (if weights provided)
    # ═══════════════════════════════════════
    if args.yolo_weights:
        print("\n" + "=" * 72)
        print("YOLO EVALUATION")
        print("=" * 72)
        yolo_model = load_yolo_model(args.yolo_weights)
        print("YOLO loaded.\n")

        yolo_data = {}
        for sname, spath in splits:
            cache, coco = precompute_masks_yolo(yolo_model, spath)
            yolo_data[sname] = (cache, coco, spath)

        print("\nYOLO Dice Scores:")
        yolo_rows, yolo_dice = run_evaluation("YOLO", yolo_data, prefix="YOLO: ")
        all_rows.extend(yolo_rows)
        dice_results["YOLO"] = yolo_dice

    # ═══════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 72)
    print("COMBINED EVALUATION RESULTS")
    print("=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)

    csv_path = os.path.join(SCRIPT_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Dice summary
    print("\n--- Dice Score Summary ---")
    for model_name, dice in dice_results.items():
        status = "✅" if dice >= 0.70 else "⚠️"
        print(f"  {status} {model_name}: Dice = {dice:.4f}")

    return dice_results, df

if __name__ == "__main__":
    main()
