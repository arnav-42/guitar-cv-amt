"""
GuiFiR-Vid (An et al., IEEE Access 2026) loader and ground-truth geometry.

Data (not in this repo; non-commercial research use, no redistribution):
  https://github.com/kmu-ee-ailab/guifir_dataset -> unpack GuiFiR.tar so that
  data/guifir/GuiFiR/{raw_clips,annotations,metadata} exists, or set GUIFIR_DIR.

Annotation rows (one per annotated frame, only frames with a picked note):
  filename      "<clip>/<clip>_<frame index>.png", 0-based index into the clip
  frame_tab     6 frets, 20 = not played, 0 = open; index 0 = high E = the BOTTOM
                string in the image; frets relative to the capo. Both checked on test:
                GT fingertips land within one fret for 96% of notes (49% without +capo);
                Basic Pitch hears 85% of tab pitches with index 0 = high E vs 10% reversed
  frame_capo    capo fret
  frame_gt_pose fingertips "4" "8" "12" "16" "20" (thumb..pinky, [x, y, vis]) and
                fretboard corners lt lb rt rb, all in 640x360 coordinates
                (scale by video width / 640). The corners are the outer strings at
                the nut and at fret 20; the nut is the shorter end.

Geometry: a homography maps the unit template (u: nut 0 -> fret 20 1,
v: top string 0 -> bottom string 1) to the frame. Fret n sits at
u_n = (1 - 2^(-n/12)) / (1 - 2^(-20/12)); string k (0 = top) at v = k/5.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
GUIFIR_DIR = Path(os.environ.get("GUIFIR_DIR", REPO / "data" / "guifir" / "GuiFiR"))
OPEN_MIDI = (64, 59, 55, 50, 45, 40)          # tab index 0..5 = E4 B3 G3 D3 A2 E2 (standard tuning)
ANN = {"train": "guifir_train_coco_anno_box.jsonl", "val": "guifir_val_coco_anno_box.jsonl",
       "test": "guifir_test_coco_anno_box.jsonl"}
TIPS = ("4", "8", "12", "16", "20")          # thumb, index, middle, ring, pinky tips
NOT_PLAYED = 20
LABEL_WIDTH = 640.0
N_FRETS = 20


def fret_u(n):
    n = np.asarray(n, float)
    return (1 - 2 ** (-n / 12)) / (1 - 2 ** (-N_FRETS / 12))


def load_rows(split):
    rows = []
    for line in open(GUIFIR_DIR / "annotations" / ANN[split]):
        r = json.loads(line)
        clip, fname = r["filename"].split("/")
        r["clip"] = clip
        r["frame_index"] = int(Path(fname).stem.rsplit("_", 1)[-1])
        rows.append(r)
    return rows


class ClipReader:
    """Random access to frames of one clip (sequential reads are cached)."""

    def __init__(self):
        self.clip, self.cap, self.pos = None, None, -1

    def read(self, clip, index):
        if clip != self.clip:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(str(GUIFIR_DIR / "raw_clips" / f"{clip}.mp4"))
            self.clip, self.pos = clip, -1
        if index != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        self.pos = index if ok else -1
        return frame if ok else None


def gt_geometry(row, width):
    """Ground truth in frame pixels: fingertips (5, 3) [x, y, vis] and the
    template->frame homography H (nut at u=0, top string at v=0)."""
    g = row["frame_gt_pose"]
    s = width / LABEL_WIDTH
    tips = np.array([[g[k][0] * s, g[k][1] * s, g[k][2]] for k in TIPS])
    lt, lb, rt, rb = (np.array(g[k][:2], float) * s for k in ("lt", "lb", "rt", "rb"))
    if np.linalg.norm(rt - rb) <= np.linalg.norm(lt - lb):    # right end narrower -> nut on the right
        nut_top, nut_bot, far_top, far_bot = rt, rb, lt, lb
    else:
        nut_top, nut_bot, far_top, far_bot = lt, lb, rt, rb
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    dst = np.float32([nut_top, far_top, far_bot, nut_bot])
    return tips, cv2.getPerspectiveTransform(src, dst)


def to_template(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)


def gt_string_lines(H):
    """6 GT strings as (x1, y1, x2, y2) segments from the nut to fret 20, top first."""
    v = np.arange(6) / 5
    ends = np.stack([np.column_stack([np.zeros(6), v]), np.column_stack([np.ones(6), v])], axis=1)
    img = cv2.perspectiveTransform(ends.reshape(1, -1, 2), H).reshape(6, 2, 2)
    return img.reshape(6, 4)


def tab_targets(row, top_is_index0=False, capo_relative=True):
    """Pressed notes as (string row k from the top, absolute fret n >= 1)."""
    out = []
    for i, f in enumerate(row["frame_tab"]):
        if f == NOT_PLAYED or f < 0:
            continue
        n = f + row["frame_capo"] if capo_relative else f
        if f == 0 or n < 1:          # open string (or capo-only): no finger
            continue
        out.append((i if top_is_index0 else 5 - i, n))
    return out


def cell_hits(H, tips_xy, targets, string_tol=0.5):
    """For each pressed (k, n): is any fingertip inside its cell, i.e. within
    string_tol spacings of string k and between frets n-1 and n?"""
    if not targets:
        return np.zeros(0, bool)
    uv = to_template(H, tips_xy)
    hits = []
    for k, n in targets:
        lo, hi = fret_u(n - 1), fret_u(n)
        ok = (uv[:, 0] >= lo) & (uv[:, 0] <= hi) & (np.abs(uv[:, 1] * 5 - k) <= string_tol)
        hits.append(bool(ok.any()))
    return np.array(hits)
