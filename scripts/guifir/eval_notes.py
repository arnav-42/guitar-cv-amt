#!/usr/bin/env python3
"""
Note-level, video-only evaluation on GuiFiR-Vid: for every pressed note in the
manual tablature, is there a fretting fingertip at its (string, fret)?

Ground truth: guifir.tab_targets (string row from the top, absolute fret >= 1;
open strings and unplayed strings excluded) on GuiFiR's annotated frames.

A fingertip (index..pinky) is assigned to
  string = nearest string    (GT grid: round(v * 5); predicted lines: nearest
                              line at the fingertip's x, lines sorted top->bottom)
  fret   = the fret cell its position along the neck falls in (GT fret grid,
           guifir.fret_u boundaries; cell n = between frets n-1 and n)
Per note the best-matching fingertip is used (several notes may share a finger,
e.g. a barre), and we report: string correct, fret correct, both correct.

Contact calibration: fingertip keypoints mark the nail tip, not the contact
point. A per-finger median offset (du, dv) in fretboard-template coordinates,
from each finger to the cell centre of the note it plays, is learned on the
val split and applied on test. Learned separately for GT fingertips and for
each model's predictions.

Conditions:
  GT tips  + GT grid              ceiling of the fingertip labels themselves
  <model>  + GT grid              hand model alone
  <model>  + A strings, GT frets  realistic string geometry (compare_strings --save-lines)

Usage: python -m scripts.guifir.eval_notes --models hamer hand_crop mediapipe
Outputs: output/guifir/eval/test/notes_summary.txt, notes_<condition>.csv
"""

import argparse
import json

import numpy as np
import pandas as pd

from main.video.fretboard.guifir import fret_u, gt_geometry, load_rows, tab_targets, to_template
from scripts.guifir.common import EVAL_DIR, OUT, PRED_DIR

CACHE = OUT
TIP_IDX = [8, 12, 16, 20]                     # index, middle, ring, pinky (fretting fingers)
FRET_EDGES = fret_u(np.arange(21))            # u of nut (0) .. fret 20 (1)


def load_preds(model, split):
    out = {}
    for line in open(PRED_DIR / f"{model}_{split}.jsonl"):
        d = json.loads(line)
        out[(d["clip"], d["frame"])] = d
    return out


def load_lines(split):
    out = {}
    path = CACHE / f"lines_{split}.jsonl"
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            if d["method"] == "A":
                out[(d["clip"], d["frame"])] = np.asarray(d["lines"], float)
    return out


def fretting_tips(row, pred, width, tips_gt):
    """Predicted index..pinky tips (4, 2) of the hand inside GuiFiR's left-hand box
    (closest to the GT tips if several), or None. Same rule as eval_guifir_tips."""
    box = (row.get("hand_annotation") or {}).get("boxes")
    if not box or pred is None:
        return None
    x1, y1, x2, y2 = np.asarray(box, float) * width / row["frame_width"]
    mx, my = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    hands = []
    for h in pred["hands"]:
        k = np.asarray(h["keypoints"], float)
        c = k[[4, 8, 12, 16, 20]].mean(axis=0)
        if x1 - mx <= c[0] <= x2 + mx and y1 - my <= c[1] <= y2 + my:
            hands.append(k)
    if not hands:
        return None
    k = min(hands, key=lambda k: np.linalg.norm(k[[4, 8, 12, 16, 20]] - tips_gt, axis=1).mean())
    return k[TIP_IDX]


def fret_cell(u):
    return np.searchsorted(FRET_EDGES, u)      # n: FRET_EDGES[n-1] < u <= FRET_EDGES[n]


def strings_from_lines(lines, xy):
    """String row (0 = top) of each point: nearest predicted line at the point's x."""
    x1, y1, x2, y2 = lines.T
    out = []
    for x, y in xy:
        yl = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        order = np.argsort(yl)
        out.append(int(np.argmin(np.abs(yl[order] - y))))
    return np.array(out)


def score_frame(targets, uv, string_of):
    """Per note: a note is hit if ANY fingertip is on its string with the fret
    exact (both) or within ±1 (both_pm1). The reported finger is the first one
    that hits it, else the closest by |fret error| + |string error|."""
    cells = fret_cell(uv[:, 0])
    rows = []
    for k, n in targets:
        on = string_of == k
        exact, near = np.flatnonzero(on & (cells == n)), np.flatnonzero(on & (np.abs(cells - n) <= 1))
        cost = np.abs(cells - n) + np.abs(string_of - k)
        j = int(exact[0]) if len(exact) else int(near[0]) if len(near) else int(np.argmin(cost))
        rows.append({"string_ok": string_of[j] == k, "fret_ok": cells[j] == n,
                     "fret_pm1": abs(cells[j] - n) <= 1, "finger": j})
    return rows


def learn_offsets(rows, widths, tips_of):
    """Per-finger median (du, dv) from fingertip to the cell centre of the note
    it plays (nearest finger in template space), on the calibration split."""
    d = {j: [] for j in range(4)}
    for r in rows:
        w = widths.get((r["clip"], r["frame_index"]))
        if w is None:
            continue
        tips_gt, H = gt_geometry(r, w)
        tips = tips_of(r, w, tips_gt[:, :2])
        if tips is None:
            continue
        uv = to_template(H, tips)
        for k, n in tab_targets(r):
            centre = np.array([(FRET_EDGES[n - 1] + FRET_EDGES[n]) / 2, k / 5])
            j = int(np.argmin(np.linalg.norm((uv - centre) * [1, 5], axis=1)))
            d[j].append(centre - uv[j])
    return np.array([np.median(d[j], axis=0) if d[j] else [0.0, 0.0] for j in range(4)])


def evaluate(rows, widths, tips_of, lines=None, offsets=None):
    recs = []
    for r in rows:
        key = (r["clip"], r["frame_index"])
        w = widths.get(key)
        if w is None:
            continue
        targets = tab_targets(r)
        if not targets:
            continue
        tips_gt, H = gt_geometry(r, w)
        tips = tips_of(r, w, tips_gt[:, :2])
        miss = tips is None or (lines is not None and (key not in lines or len(lines[key]) != 6))
        if miss:
            recs += [{"clip": r["clip"], "frame": r["frame_index"], "string_ok": False, "fret_ok": False,
                      "fret_pm1": False, "answered": False} for _ in targets]
            continue
        uv = to_template(H, tips)
        if offsets is not None:
            uv = uv + offsets
        if lines is None:
            string_of = np.clip(np.rint(uv[:, 1] * 5), 0, 5).astype(int)
        else:
            import cv2
            xy = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)   # calibrated points back in the frame
            string_of = strings_from_lines(lines[key], xy)
        for s in score_frame(targets, uv, string_of):
            recs.append({"clip": r["clip"], "frame": r["frame_index"], "answered": True, **s})
    df = pd.DataFrame(recs)
    df["both_ok"] = df["string_ok"] & df["fret_ok"]
    df["both_pm1"] = df["string_ok"] & df["fret_pm1"]      # GT fret grid is itself only ~±1 fret precise
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="+", default=["hamer"])
    p.add_argument("--calib-split", default="val")
    p.add_argument("--no-lines", action="store_true", help="skip the A-strings conditions")
    args = p.parse_args()

    complete = lambda rs: [r for r in rs if all(k in r["frame_gt_pose"] for k in ("4", "8", "12", "16", "20", "lt", "lb", "rt", "rb"))]
    test, val = complete(load_rows("test")), complete(load_rows(args.calib_split))   # val has 1 corners-only frame
    ref = load_preds("hamer", "test")
    widths = {k: d["width"] for k, d in ref.items() if d["width"]}
    ref_val = load_preds("hamer", args.calib_split)
    widths_val = {k: d["width"] for k, d in ref_val.items() if d["width"]}
    lines = {} if args.no_lines else load_lines("test")

    def gt_tips(r, w, tips_gt):
        return tips_gt[1:]

    def model_tips(model, split):
        preds = load_preds(model, split)
        return lambda r, w, tips_gt: fretting_tips(r, preds.get((r["clip"], r["frame_index"])), w, tips_gt)

    conds = [("GT tips + GT grid", gt_tips, None, gt_tips)]
    for m in args.models:
        conds.append((f"{m} + GT grid", model_tips(m, "test"), None,
                      model_tips(m, args.calib_split) if (PRED_DIR / f"{m}_{args.calib_split}.jsonl").exists() else None))
        if lines:
            conds.append((f"{m} + A strings", model_tips(m, "test"), lines,
                          model_tips(m, args.calib_split) if (PRED_DIR / f"{m}_{args.calib_split}.jsonl").exists() else None))

    out_dir = EVAL_DIR / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = [f"Note-level, video-only, GuiFiR test. Pressed notes from manual tabs (open strings excluded). "
            f"Calibration learned on {args.calib_split}. Fret grid = GT for all conditions."]
    rows_out = []
    for name, tips_of, lns, calib_tips_of in conds:
        for calib in (False, True):
            if calib and calib_tips_of is None:
                continue
            off = learn_offsets(val, widths_val, calib_tips_of) if calib else None
            df = evaluate(test, widths, tips_of, lns, off)
            label = name + (" + contact calibration" if calib else "")
            df.to_csv(out_dir / f"notes_{label.replace(' ', '_').replace('+', 'plus')}.csv", index=False)
            rows_out.append({"condition": label, "notes": len(df), "answered": df["answered"].mean(),
                             "both": df["both_ok"].mean(), "string": df["string_ok"].mean(),
                             "fret": df["fret_ok"].mean(), "fret_pm1": df["fret_pm1"].mean(),
                             "both_pm1": df["both_pm1"].mean()})
            text.append(f"{label:52s} notes {len(df):5d}  answered {df['answered'].mean():6.1%}  "
                        f"string+fret {df['both_ok'].mean():6.1%}  string {df['string_ok'].mean():6.1%}  "
                        f"fret {df['fret_ok'].mean():6.1%}  fret±1 {df['fret_pm1'].mean():6.1%}  "
                        f"string+fret±1 {df['both_pm1'].mean():6.1%}")
    pd.DataFrame(rows_out).to_csv(out_dir / "notes_summary.csv", index=False)
    (out_dir / "notes_summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
