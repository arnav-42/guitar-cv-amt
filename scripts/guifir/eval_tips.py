#!/usr/bin/env python3
"""
Fingertip accuracy of hand models on GuiFiR-Vid, against GuiFiR's manual
fingertips and fretboard grid (guifir.py).

Per frame the fretting hand is the predicted hand whose fingertip centroid
lies inside GuiFiR's left-hand detection box (hand_annotation, +20% margin);
the closest such hand to the GT tips if several. No hand there = not found
(like GuiFiR's "LH Det."); errors are averaged over found frames only.
Reported per model:
  found         frames with any predicted hand
  MAE-H         mean fingertip error over the 5 tips in each video's native
                pixels - the unit that reproduces GuiFiR Table 1 (its MediaPipe:
                33.8% / 29.1 px; ours 38.6% / 25.1 px). MAE@640 = same in
                640-wide pixels (resolution-independent, ours only)
  err/spacing   fingertip error divided by the local GT string spacing
  across        |error across the strings| in string spacings (what decides
                which string a finger is assigned to), fretting fingers only
  <=0.5 across  share of fretting fingertips within half a spacing across strings
  cell recall   pressed notes whose cell (string +-0.5, fret n-1..n) holds a
                predicted fingertip, GT grid; the GT-fingertip row is the ceiling

Usage: python -m scripts.guifir.eval_tips --split test --models mediapipe rtmw hand_crop
"""

import argparse
import json

import numpy as np
import pandas as pd

from pipeline.guifir import LABEL_WIDTH, cell_hits, gt_geometry, load_rows, tab_targets, to_template
from scripts.guifir.common import EVAL_DIR, PRED_DIR

TIP_IDX = [4, 8, 12, 16, 20]


def local_spacing(H, uv):
    """String spacing (px) at template points uv: distance to one string below."""
    import cv2
    a = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)
    b = cv2.perspectiveTransform((uv + [0, 0.2]).reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(b - a, axis=1)


def evaluate(split, model, rows):
    preds = {}
    if model != "gt":
        for line in open(PRED_DIR / f"{model}_{split}.jsonl"):
            d = json.loads(line)
            preds[(d["clip"], d["frame"])] = d
    recs = []
    for r in rows:
        key = (r["clip"], r["frame_index"])
        width = preds[key]["width"] if model != "gt" else None
        if model != "gt" and width is None:
            continue
        if model == "gt":
            width = LABEL_WIDTH
        tips, H = gt_geometry(r, width)
        gt_xy = tips[:, :2]
        rec = {"clip": r["clip"], "frame": r["frame_index"], "found": True}
        if model == "gt":
            pred_xy = gt_xy
        else:
            box = (r.get("hand_annotation") or {}).get("boxes")
            if not box:                      # GuiFiR's detector found no left hand
                rec["found"] = False
                recs.append(rec)
                continue
            x1, y1, x2, y2 = np.asarray(box, float) * width / r["frame_width"]
            mx, my = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
            hands = [p for p in (np.asarray(h["keypoints"], float)[TIP_IDX] for h in preds[key]["hands"])
                     if x1 - mx <= p[:, 0].mean() <= x2 + mx and y1 - my <= p[:, 1].mean() <= y2 + my]
            if not hands:
                rec["found"] = False
                recs.append(rec)
                continue
            pred_xy = min(hands, key=lambda p: np.linalg.norm(p - gt_xy, axis=1).mean())
        err = np.linalg.norm(pred_xy - gt_xy, axis=1)
        uv_gt, uv_pred = to_template(H, gt_xy), to_template(H, pred_xy)
        spacing = local_spacing(H, uv_gt)
        across = np.abs(uv_pred[:, 1] - uv_gt[:, 1]) * 5          # in string spacings
        hits = cell_hits(H, pred_xy[1:], tab_targets(r))           # fretting fingers only
        rec.update({"mae_native": float(err.mean()), "mae640": float(err.mean() * LABEL_WIDTH / width),
                    "err_spacing": float(np.median(err / spacing)),
                    "across": float(np.median(across[1:])),
                    "across_le05": float((across[1:] <= 0.5).mean()),
                    "across_tips": json.dumps(np.round(across[1:], 4).tolist()),
                    "cell_hits": int(hits.sum()), "cell_n": len(hits)})
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--models", nargs="+", default=["mediapipe"])
    args = p.parse_args()
    rows = load_rows(args.split)
    out_dir = EVAL_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"GuiFiR {args.split} fingertips ({len(rows)} annotated frames). Fretting hand = prediction inside GuiFiR's left-hand box; errors over found frames."]
    for model in ["gt"] + args.models:
        if model != "gt" and not (PRED_DIR / f"{model}_{args.split}.jsonl").exists():
            lines.append(f"{model:10s} no predictions")
            continue
        df = evaluate(args.split, model, rows)
        df.to_csv(out_dir / f"tips_{model}.csv", index=False)
        ok = df[df["found"]]
        recall = ok["cell_hits"].sum() / max(ok["cell_n"].sum(), 1)
        lines.append(f"{'GT tips' if model == 'gt' else model:10s} found {df['found'].mean():.1%}  "
                     f"MAE-H {ok['mae_native'].mean():6.2f} px native  MAE@640 {ok['mae640'].mean():6.2f}  err/spacing (median) {ok['err_spacing'].median():.2f}  "
                     f"across-string error (median) {ok['across'].median():.2f}  <=0.5 across {ok['across_le05'].mean():.1%}  "
                     f"cell recall {recall:.1%}")
    text = "\n".join(lines)
    print(text)
    (out_dir / "tips_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
