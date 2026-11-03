#!/usr/bin/env python3
"""
Preliminary audio + video fusion on GuiFiR test (no temporal decoding yet).

For each pressed note in the manual tabs whose pitch Basic Pitch heard:
  candidates = every (string, fret) that plays that pitch (standard tuning + capo)
  audio-only = lowest-fret candidate (the usual tab-software default)
  video pick = candidate closest to a HaMeR fingertip, contact-calibrated, in
               (falls back to the audio-only pick when video has no hand / no strings)
               (fret cell + string) units; open-string candidates get a fixed cost 2
Accuracy = picked position equals the tab's (string, fret). Notes the audio
missed count as wrong for every method (reported separately).

Usage: python -m scripts.guifir.eval_av_preview
Writes output/guifir/eval/test/av_preview_summary.txt
"""

import json

import cv2
import numpy as np

from scripts.guifir.eval_audio import HIGH_FIRST, heard, positions
from scripts.guifir.eval_notes import (fret_cell, fretting_tips, learn_offsets, load_lines, load_preds,
                               strings_from_lines)
from pipeline.guifir import GUIFIR_DIR, NOT_PLAYED, gt_geometry, load_rows, to_template
from scripts.guifir.common import EVAL_DIR, OUT

COMPLETE = ("4", "8", "12", "16", "20", "lt", "lb", "rt", "rb")


def main():
    test = [r for r in load_rows("test") if all(k in r["frame_gt_pose"] for k in COMPLETE)]
    val = [r for r in load_rows("val") if all(k in r["frame_gt_pose"] for k in COMPLETE)]
    audio = {json.loads(l)["clip"]: json.loads(l)["notes"]
             for l in open(OUT / "audio_basicpitch_test.jsonl")}
    fps = {c: cv2.VideoCapture(str(GUIFIR_DIR / "raw_clips" / f"{c}.mp4")).get(cv2.CAP_PROP_FPS) or 30.0
           for c in {r["clip"] for r in test}}
    pt, pv = load_preds("hamer", "test"), load_preds("hamer", "val")
    widths = {k: d["width"] for k, d in pt.items() if d["width"]}
    widths_v = {k: d["width"] for k, d in pv.items() if d["width"]}
    tips_t = lambda r, w, g: fretting_tips(r, pt.get((r["clip"], r["frame_index"])), w, g)
    tips_v = lambda r, w, g: fretting_tips(r, pv.get((r["clip"], r["frame_index"])), w, g)
    off = learn_offsets(val, widths_v, tips_v)
    lines = load_lines("test")

    res = {"audio_lowest_fret": [], "video_gt_grid": [], "video_A_strings": [], "chance": []}
    n_total = n_heard = 0
    for r in test:
        w = widths.get((r["clip"], r["frame_index"]))
        if w is None:
            continue
        tips_gt, H = gt_geometry(r, w)
        tips = tips_t(r, w, tips_gt[:, :2])
        t = r["frame_index"] / fps[r["clip"]]
        capo = r["frame_capo"]
        uv = to_template(H, tips) + off if tips is not None else None
        a_lines = lines.get((r["clip"], r["frame_index"]))
        str_A = None
        if uv is not None and a_lines is not None and len(a_lines) == 6:
            xy = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)
            str_A = strings_from_lines(a_lines, xy)
        for i, f in enumerate(r["frame_tab"]):
            if f == NOT_PLAYED or f <= 0:
                continue
            n_total += 1
            pitch = HIGH_FIRST[i] + capo + f
            ok = heard(audio[r["clip"]], t, pitch)
            n_heard += ok
            cands = positions(pitch, capo, HIGH_FIRST)                     # (index, fret above capo)
            truth = (i, f)
            res["chance"].append(ok / max(len(cands), 1))
            res["audio_lowest_fret"].append(ok and min(cands, key=lambda c: c[1]) == truth)
            for key, string_of in (("video_gt_grid", None), ("video_A_strings", str_A)):
                if not ok:
                    res[key].append(False)
                    continue
                if uv is None or (key == "video_A_strings" and string_of is None):   # no video: fall back to audio
                    res[key].append(min(cands, key=lambda c: c[1]) == truth)
                    continue
                cells = fret_cell(uv[:, 0])
                s_tip = np.clip(np.rint(uv[:, 1] * 5), 0, 5).astype(int) if string_of is None else string_of

                def cost(c):
                    idx, fr = c
                    if fr == 0:
                        return 2.0
                    k, n = 5 - idx, fr + capo                               # string row from the top, absolute fret
                    return float(np.min(np.abs(cells - n) + np.abs(s_tip - k)))
                res[key].append(min(cands, key=cost) == truth)
    text = [f"Preliminary audio + video, GuiFiR test. Pressed notes: {n_total}; pitch heard by Basic Pitch: "
            f"{n_heard / n_total:.1%}. Position accuracy over all pressed notes (audio misses count as wrong):"]
    for k, label in (("chance", "random candidate (expected)"), ("audio_lowest_fret", "audio only, lowest fret"),
                     ("video_gt_grid", "audio + video (HaMeR, GT grid)"),
                     ("video_A_strings", "audio + video (HaMeR, A strings)")):
        text.append(f"  {label:36s} {np.mean(res[k]):6.1%}   (of heard notes: {np.sum(res[k]) / max(n_heard, 1):6.1%})")
    out = "\n".join(text)
    print(out)
    (EVAL_DIR / "test" / "av_preview_summary.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
