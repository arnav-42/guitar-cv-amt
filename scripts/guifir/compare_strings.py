#!/usr/bin/env python3
"""
String detection on GuiFiR-Vid (external data, never used for tuning).

GT strings: the 6 lines from the nut to fret 20, built from GuiFiR's four
fretboard corners (main.video.fretboard.guifir). Every annotated frame: one YOLO mask ->
ROI A (main.video.fretboard.string_detection.roi_a) and ROI B (roi_b), then
  A               ROI A + comb_band_slope_strip_fan
  B               ROI B + lsd_strip_cascade (its own pipeline)
  fuse_arbitrate  ROI A, A and B combined; confident = A and B agree
scored at 21 x positions inside the GT extent and both ROI quads. Pass = 6 lines,
max error <= 1/4 string spacing. Frames of one clip are correlated, so CIs resample clips.

Usage (repo root):
  python -m scripts.guifir.compare_strings --split test [--save-lines]
Outputs: output/guifir/eval/<split>/frames.csv, summary.txt
         (--save-lines: output/guifir/lines_<split>.jsonl, used by eval_notes / eval_av_preview)
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from main.video.fretboard.guifir import ClipReader, gt_geometry, gt_string_lines, load_rows
from main.video.fretboard.string_detection import (comb_band_slope_strip_fan, fuse_arbitrate, lsd_strip_cascade,
                                       roi_a, roi_b)
from scripts.guifir.common import EVAL_DIR, OUT, cached_mask, score_lines, short_path, x_range

RUNS = {"A": ("A", comb_band_slope_strip_fan, False),
        "B": ("B", lsd_strip_cascade, True),
        "fuse_arbitrate": ("A", fuse_arbitrate, True)}


def clip_ci(df, col, n_boot=5000, seed=0):
    """95% CI of the frame mean, resampling clips."""
    groups = [g[col].to_numpy(float) for _, g in df.groupby("clip")]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--limit", type=int, default=0, help="first N frames only (debug)")
    p.add_argument("--n-x", type=int, default=21)
    p.add_argument("--save-lines", action="store_true",
                   help="also write each method's lines (frame coords) to output/guifir/lines_<split>.jsonl")
    args = p.parse_args()
    out_dir = EVAL_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.split)
    rows = rows[:args.limit] if args.limit else rows
    reader, recs, t_start = ClipReader(), [], time.time()
    lines_out = open(OUT / f"lines_{args.split}.jsonl", "w") if args.save_lines else None
    for i, r in enumerate(rows):
        img = reader.read(r["clip"], r["frame_index"])
        if img is None:
            continue
        _, H = gt_geometry(r, img.shape[1])
        gt = gt_string_lines(H)
        mask = cached_mask(f"{r['clip']}_{r['frame_index']:03d}", img)
        views = {}
        if mask.any():
            for name, make in (("A", roi_a), ("B", roi_b)):
                v = make(img, mask)
                rng = x_range(v["quad"], gt) if v is not None else None
                if rng is not None:
                    views[name] = {**v, "x_range": rng}
        lo = max([gt[:, [0, 2]].min()] + [v["x_range"][0] for v in views.values()])
        hi = min([gt[:, [0, 2]].max()] + [v["x_range"][1] for v in views.values()])
        if not views or hi <= lo:
            lo, hi = gt[:, [0, 2]].min(), gt[:, [0, 2]].max()
        xs = np.linspace(lo, hi, args.n_x)
        base = {"clip": r["clip"], "frame": r["frame_index"], "width": img.shape[1], "yolo_found": bool(mask.any())}
        for name, (roi, detect, needs_mask) in RUNS.items():
            rec = {**base, "method": name, "count_ok": False, "max_norm": np.nan,
                   "str_le_0.5": 0, "spacing_px": np.nan, "stage": None}
            view = views.get(roi)
            if view is not None:
                lines = detect(view["image"], view["usable"]) if needs_mask else detect(view["image"])
                frame_lines = view["to_frame"](np.asarray(lines, float).reshape(-1, 4))
                res = score_lines(gt, frame_lines, xs)
                if lines_out is not None:
                    lines_out.write(json.dumps({"clip": r["clip"], "frame": r["frame_index"], "method": name,
                                                "lines": np.round(frame_lines, 2).tolist()}) + "\n")
                rec.update(count_ok=res["count_ok"], spacing_px=res["spacing_px"],
                           stage=getattr(detect, "last_stage", None))
                if res["count_ok"]:
                    per_string = np.abs(res["err_norm"]).max(axis=1)
                    rec.update(max_norm=res["max_norm"], **{"str_le_0.5": int((per_string <= 0.5).sum())})
            recs.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} frames  {time.time() - t_start:.0f} s", flush=True)
    if lines_out is not None:
        lines_out.close()
        print(f"lines -> {short_path(lines_out.name)}")

    df = pd.DataFrame(recs)
    df["passed"] = df["count_ok"] & (df["max_norm"] <= 0.25)
    df["le_0.5"] = df["count_ok"] & (df["max_norm"] <= 0.5)
    df["str_frac"] = df["str_le_0.5"] / 6
    df.to_csv(out_dir / "frames.csv", index=False)

    text = [f"GuiFiR {args.split}: {df['clip'].nunique()} clips, {len(df) // len(RUNS)} frames. "
            "Pass = 6 lines, max error <= 0.25 spacing. CIs resample clips."]
    for name in RUNS:
        d = df[df["method"] == name]
        found = d[d["yolo_found"]]
        ci, ci_s = clip_ci(d, "passed"), clip_ci(d, "str_frac")
        text.append(f"{name:15s} pass {d['passed'].mean():.1%} [{ci[0]:.1%}-{ci[1]:.1%}]  "
                    f"frames<=0.5 {d['le_0.5'].mean():.1%}  strings<=0.5 {d['str_frac'].mean():.1%} "
                    f"[{ci_s[0]:.1%}-{ci_s[1]:.1%}]  no answer {1 - d['count_ok'].mean():.1%}  | board found: "
                    f"pass {found['passed'].mean():.1%}, strings<=0.5 {found['str_frac'].mean():.1%}")
    d = df[(df["method"] == "fuse_arbitrate") & df["count_ok"]]
    for flag, g in d.groupby(d["stage"] == "agree_avg"):
        text.append(f"fuse_arbitrate {'agree' if flag else 'other'}: {len(g)} frames, pass {g['passed'].mean():.1%}, "
                    f"<=0.5 {g['le_0.5'].mean():.1%}")
    a = df[df["method"] == "A"]
    text.append(f"YOLO found the board on {a['yolo_found'].mean():.1%} of frames; "
                f"median GT string spacing {a['spacing_px'].median():.1f} px")
    print("\n".join(text))
    (out_dir / "summary.txt").write_text("\n".join(text) + "\n")
    print(f"Outputs -> {short_path(out_dir)}")


if __name__ == "__main__":
    main()
