#!/usr/bin/env python3
"""
Figures for the GuiFiR benchmark. Reads output/guifir/eval/test/.

  g5_fingertips_on_string.png  hand models: fingertips on the correct string
  g6_metric_schematic.png      what "on the correct string" means
  g7_metric_ladder.png         string detection A under different yardsticks
  g8_notes_video_only.png      note-level, video only (with/without contact calibration)
  g9_av_preview.png            preliminary audio + video
  g4_examples_<res>.png        one frame per hand model -- shows dataset frames: keep local,
                               do not commit (GuiFiR terms forbid redistribution)

Usage: python -m scripts.guifir.plots
"""

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.guifir import ClipReader, gt_geometry, gt_string_lines, load_rows
from scripts.guifir.common import EVAL_DIR, FIG_DIR, PRED_DIR

G_DIR = EVAL_DIR / "test"
OUT = FIG_DIR

# palette (validated for colour-vision deficiency; A = blue, B = orange)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
A_COLOR, B_COLOR = "#2a78d6", "#eb6834"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})   # plain matplotlib look
SURFACE = "white"


def boot(values, groups=None, n=5000, seed=0):
    """95% CI of the mean; resamples groups (clips) when given."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if groups is None:
        means = rng.choice(v, (n, len(v))).mean(axis=1)
    else:
        g = pd.Series(v).groupby(np.asarray(groups)).apply(list).tolist()
        means = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


ON_STRING = [  # (key, label) - all run by us, same evaluation code
    ("hamer", "HaMeR"), ("mediapipe", "MediaPipe"), ("hand_crop", "RTMPose hand crop"),
    ("wilor", "WiLoR"), ("rtmw", "RTMW-x wholebody"), ("vitpose_l", "ViTPose-L wholebody"),
    ("vitpose_b", "ViTPose-B wholebody"),
]


def fig_on_string():
    rows = []
    for key, label in ON_STRING:
        d = pd.read_csv(G_DIR / f"tips_{key}.csv")
        found = d["found"].mean()
        d = d[d["found"]]
        per_tip = [(c, float(v) <= 0.5) for c, s_ in zip(d["clip"], d["across_tips"]) for v in json.loads(s_)]
        clips = np.array([c for c, _ in per_tip])
        hit = np.array([h for _, h in per_tip], float)
        rows.append((label, hit.mean(), boot(hit, clips), found))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (label, v, ci, found) in enumerate(rows):
        best = i == len(rows) - 1
        ax.barh(i, v, color="#4a3aa7" if best else "#c3c2b7", height=0.62)
        ax.plot(ci, [i, i], color=INK, lw=1.3)
        for x in ci:
            ax.plot([x, x], [i - 0.12, i + 0.12], color=INK, lw=1.3)
        note = f"   hand found in only {found:.0%} of frames" if found < 0.9 else ""
        ax.text(ci[1] + 0.012, i, f"{v:.0%}{note}", va="center", fontsize=12, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("Fretting fingertips on the correct string (error across strings ≤ ½ spacing)")
    ax.set_title("Fingertips on the correct string")
    save(fig, "g5_fingertips_on_string.png")


def fig_metric_schematic():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.array([0, 10])
    for k, y in enumerate([1, 2, 3]):          # three strings, 1 spacing apart
        ax.plot(xs, [y, y], color=INK, lw=2.4)
        ax.text(10.2, y, f"string {k + 3}", va="center", fontsize=12, color=INK)
    ax.axhspan(1.5, 2.5, color="#e7f4e7", zorder=0)
    ax.text(0.2, 1.56, "within ½ spacing of string 4: still assigned to string 4", fontsize=11, color="#006300", va="top")
    ax.annotate("", xy=(9.4, 2.0), xytext=(9.4, 2.5), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(9.55, 2.25, "½ spacing", fontsize=10.5, color=INK2, va="center")
    ax.plot([4.0], [2.0], marker="x", color="#d03b3b", ms=14, mew=3, zorder=5)
    ax.text(4.0, 1.78, "GT fingertip", ha="center", fontsize=11, color=INK, va="top")
    ax.plot([5.0], [2.32], "o", color="#4a3aa7", ms=12, mec=SURFACE, mew=2, zorder=5)
    ax.text(5.25, 2.32, "prediction: correct", fontsize=11, color=INK, va="center")
    ax.plot([2.4], [2.72], "o", color=CRITICAL, ms=12, mec=SURFACE, mew=2, zorder=5)
    ax.text(2.65, 2.72, "prediction: nearest string is now 5, wrong", fontsize=11, color=INK, va="center")
    ax.set_xlim(-0.2, 11.6)
    ax.set_ylim(3.4, 0.6)
    ax.axis("off")
    ax.set_title("Fingertip scoring: ½ string spacing")
    save(fig, "g6_metric_schematic.png")


def fig_metric_ladder():
    f = pd.read_csv(G_DIR / "frames.csv")
    f["passed"] = f["count_ok"] & (f["max_norm"] <= 0.25)
    f["str_frac"] = f["str_le_0.5"] / 6
    a = f[f["method"] == "A"]
    fo = a[a["yolo_found"]]
    fu = f[(f["method"] == "fuse_arbitrate") & f["count_ok"]]
    ag = fu[fu["stage"] == "agree_avg"]
    sections = [
        ("Finding the fretboard (YOLO)", [("frames where the fretboard is found", a["yolo_found"].mean(), "frames", AXIS)]),
        ("Lenient: each string within ½ spacing  (good enough to assign fingers)", [
            ("strings, frames with a fretboard", fo["str_frac"].mean(), "strings", A_COLOR),
            ("strings, all frames", a["str_frac"].mean(), "strings", A_COLOR)]),
        ("Strict: all 6 strings within ¼ spacing  (the “pass” in our tables)", [
            ("frames with a fretboard", fo["passed"].mean(), "frames", A_COLOR),
            ("all frames", a["passed"].mean(), "frames", A_COLOR)]),
        (f"Strict, only frames where A and B agree  ({len(ag) / len(a):.0%} of all frames)", [
            ("agreeing frames", ag["passed"].mean(), "frames", GOOD)]),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    y, yt, yl = 0.0, [], []
    for header, bars in sections:
        ax.text(0, y - 0.62, header, fontsize=12.5, color=INK, transform=ax.get_yaxis_transform())
        for label, v, unit, color in bars:
            ax.barh(y, v, color=color, height=0.55)
            ax.text(v + 0.012, y, f"{v:.0%} of {unit}", va="center", fontsize=12, color=INK)
            yt.append(y)
            yl.append(label)
            y += 0.8
        y += 0.75
    ax.set_yticks(yt, yl, fontsize=11.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.6, -1.1)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_title("String detection A: metrics")
    save(fig, "g7_metric_ladder.png")


def fig_notes_video():
    d = pd.read_csv(G_DIR / "notes_summary.csv").set_index("condition")
    groups = [("Ground-truth fingertips\n(label ceiling)", "GT tips + GT grid"),
              ("HaMeR fingertips,\nground-truth strings", "hamer + GT grid"),
              ("HaMeR fingertips,\nour string detection (A)", "hamer + A strings")]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.36
    for i, (label, key) in enumerate(groups):
        before = d.loc[key, "both_pm1"]
        after = d.loc[key + " + contact calibration", "both_pm1"]
        ax.bar(i - w / 2, before, width=w, color=AXIS)
        ax.bar(i + w / 2, after, width=w, color="#4a3aa7" if i else "#898781")
        ax.text(i - w / 2, before + 0.015, f"{before:.0%}", ha="center", fontsize=12, color=INK)
        ax.text(i + w / 2, after + 0.015, f"{after:.0%}", ha="center", fontsize=12, color=INK)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=11.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_ylabel("Pressed notes: string correct, fret within ±1")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=AXIS), plt.Rectangle((0, 0), 1, 1, color="#4a3aa7")]
    ax.legend(handles, ["fingertip as labelled (nail tip)", "+ contact calibration (learned on val)"],
              frameon=False, loc="upper right", fontsize=11)
    ax.set_title("Note-level accuracy, video only")
    save(fig, "g8_notes_video_only.png")


def fig_av_preview():
    import re
    txt = (G_DIR / "av_preview_summary.txt").read_text()
    val = lambda label: float(re.search(re.escape(label) + r"\s+([\d.]+)%", txt).group(1)) / 100
    heard = float(re.search(r"pitch heard by Basic Pitch: ([\d.]+)%", txt).group(1)) / 100
    bars = [("Random candidate", val("random candidate (expected)"), AXIS),
            ("Audio only\n(lowest-fret rule)", val("audio only, lowest fret"), "#86b6ef"),
            ("Audio + video\n(HaMeR + our strings)", val("audio + video (HaMeR, A strings)"), "#4a3aa7")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (label, v, color) in enumerate(bars):
        ax.barh(i, v, color=color, height=0.6)
        ax.text(v + 0.012, i, f"{v:.0%}", va="center", fontsize=13, color=INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("Pressed notes with the exact (string, fret) position")
    ax.set_title("Note position accuracy")
    save(fig, "g9_av_preview.png")


GRID_MODELS = [  # (key, label) in g5's order
    ("hamer", "HaMeR"), ("mediapipe", "MediaPipe"), ("hand_crop", "RTMPose hand crop"),
    ("wilor", "WiLoR"), ("rtmw", "RTMW-x"), ("vitpose_l", "ViTPose-L"), ("vitpose_b", "ViTPose-B"),
]


SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
            (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def _pick_frame(width):
    """Frame of the given width where every model found the hand and HaMeR's
    per-frame across-string error is closest to its overall median (typical, not best)."""
    tabs = {k: pd.read_csv(G_DIR / f"tips_{k}.csv").set_index(["clip", "frame"]) for k, _ in GRID_MODELS}
    ok = None
    for t in tabs.values():
        f = set(t.index[t["found"]])
        ok = f if ok is None else ok & f
    widths = {}
    for line in open(PRED_DIR / "hamer_test.jsonl"):
        d = json.loads(line)
        widths[(d["clip"], d["frame"])] = d["width"]
    h = tabs["hamer"]
    cand = [i for i in ok if widths.get(i) == width]
    med = h.loc[h["found"], "across"].median()
    return min(cand, key=lambda i: abs(h.loc[i, "across"] - med)), tabs


def fig_example_grid(width, name):
    (clip, frame), tabs = _pick_frame(width)
    row = next(r for r in load_rows("test") if r["clip"] == clip and r["frame_index"] == frame)
    img = ClipReader().read(clip, frame)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tips, H = gt_geometry(row, img.shape[1])
    strings = gt_string_lines(H)
    x0, y0, x1, y1 = np.asarray(row["hand_annotation"]["boxes"], float) * img.shape[1] / row["frame_width"]
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.75
    X0, X1 = int(max(cx - half, 0)), int(min(cx + half, img.shape[1]))
    Y0, Y1 = int(max(cy - half * 0.75, 0)), int(min(cy + half * 0.75, img.shape[0]))
    sp = float(np.median(np.linalg.norm(np.diff(strings[:, :2], axis=0), axis=1)))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    panels = [("gt", "Ground truth")] + GRID_MODELS
    for ax, (key, label) in zip(axes.ravel(), panels):
        ax.imshow(rgb, interpolation="nearest")
        for sx1, sy1, sx2, sy2 in strings:
            ax.plot([sx1, sx2], [sy1, sy2], color="white", lw=0.9, alpha=0.85)
        ax.scatter(tips[1:, 0], tips[1:, 1], s=70, marker="x", color="#d03b3b", linewidth=2.2, zorder=6)
        title = label
        if key != "gt":
            preds = {}
            for line in open(PRED_DIR / f"{key}_test.jsonl"):
                d = json.loads(line)
                if d["clip"] == clip and d["frame"] == frame:
                    preds = d
                    break
            hands = [np.asarray(h["keypoints"], float) for h in preds.get("hands", [])]
            k = min(hands, key=lambda p: np.linalg.norm(p[[4, 8, 12, 16, 20]] - tips[:, :2], axis=1).mean())
            for a, b in SKELETON:
                ax.plot(k[[a, b], 0], k[[a, b], 1], color="#86b6ef", lw=1.2, alpha=0.9)
            ax.scatter(k[[8, 12, 16, 20], 0], k[[8, 12, 16, 20], 1], s=60, color="#4a3aa7",
                       edgecolor="white", linewidth=1.2, zorder=7)
            across = json.loads(tabs[key].loc[(clip, frame), "across_tips"])
            hit = sum(a <= 0.5 for a in across)
            title = f"{label}: {hit}/4 on the correct string"
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y1, Y0)
        ax.set_title(title, fontsize=12, loc="left", pad=6)
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="x", ls="", color="#d03b3b", mew=2.2, ms=9, label="GT fingertip"),
               plt.Line2D([], [], marker="o", ls="", color="#4a3aa7", mec="white", ms=9, label="predicted fingertip"),
               plt.Line2D([], [], color="#86b6ef", lw=1.6, label="predicted hand skeleton"),
               plt.Line2D([], [], color="0.6", lw=1.6, label="GT strings")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Hand models on one test frame ({img.shape[1]}×{img.shape[0]})", fontsize=15)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=3.0)
    save(fig, name)


def main():
    import argparse
    p = argparse.ArgumentParser(description="GuiFiR benchmark figures")
    p.add_argument("--examples", action="store_true", help="also draw g4 example frames (local only, do not commit)")
    args = p.parse_args()
    fig_on_string()
    fig_metric_schematic()
    fig_metric_ladder()
    fig_notes_video()
    fig_av_preview()
    if args.examples:
        fig_example_grid(640, "g4_examples_640p.png")
        fig_example_grid(1920, "g4_examples_1080p.png")
    print(f"Figures -> {OUT}")


if __name__ == "__main__":
    main()
