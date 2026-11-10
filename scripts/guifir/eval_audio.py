#!/usr/bin/env python3
"""
Audio-only check on GuiFiR test: does Basic Pitch hear the notes in the manual tabs,
and how many fretboard positions could play each of them (what video must resolve)?

GT pitch of string i at fret f (capo c, standard tuning) = OPEN[i] + c + f, for every
played string (open strings included). Index 0 is the bottom string in the image;
which end of the tuning that is (high E or low E) is chosen by agreement with the
audio and reported. A note is heard if a Basic Pitch note with the same MIDI pitch
is active within [start - 0.1 s, end + 0.05 s] around the frame time (index / fps).

Usage: python -m scripts.guifir.eval_audio
Needs: extract_audio output. Writes output/guifir/eval/test/audio_summary.txt
"""

import collections
import json

import cv2
import numpy as np

from main.video.fretboard.guifir import GUIFIR_DIR, NOT_PLAYED, load_rows
from scripts.guifir.common import EVAL_DIR, OUT

HIGH_FIRST = [64, 59, 55, 50, 45, 40]   # E4 B3 G3 D3 A2 E2
N_FRETS = 20


def gt_notes(rows, fps, open_pitch):
    for r in rows:
        t = r["frame_index"] / fps[r["clip"]]
        for i, f in enumerate(r["frame_tab"]):
            if f == NOT_PLAYED or f < 0:
                continue
            yield r["clip"], t, i, f, open_pitch[i] + r["frame_capo"] + f, r["frame_capo"]


def heard(notes, t, pitch, mod12=False):
    for s, e, p, _ in notes:
        if s - 0.1 <= t <= e + 0.05 and (p % 12 == pitch % 12 if mod12 else p == pitch):
            return True
    return False


def positions(pitch, capo, open_pitch):
    """Fretboard positions (string, fret above the capo) that produce this pitch."""
    return [(i, pitch - o - capo) for i, o in enumerate(open_pitch)
            if 0 <= pitch - o - capo <= N_FRETS - capo]


def main():
    rows = load_rows("test")
    audio = {}
    for line in open(OUT / "audio_basicpitch_test.jsonl"):
        d = json.loads(line)
        audio[d["clip"]] = d["notes"]
    fps = {c: cv2.VideoCapture(str(GUIFIR_DIR / "raw_clips" / f"{c}.mp4")).get(cv2.CAP_PROP_FPS) or 30.0
           for c in {r["clip"] for r in rows}}

    lines = ["Audio-only (Basic Pitch) on GuiFiR test, manual tabs; standard tuning + capo."]
    best = None
    for name, op in (("index 0 = high E", HIGH_FIRST), ("index 0 = low E", HIGH_FIRST[::-1])):
        notes = list(gt_notes(rows, fps, op))
        hit = np.mean([heard(audio[c], t, p) for c, t, _, _, p, _ in notes])
        lines.append(f"  string order {name:18s}: pitch heard for {hit:.1%} of {len(notes)} tab notes")
        if best is None or hit > best[0]:
            best = (hit, name, op, notes)
    hit, name, op, notes = best
    chroma = np.mean([heard(audio[c], t, p, mod12=True) for c, t, _, _, p, _ in notes])
    pressed = [n for n in notes if n[3] > 0]
    n_pos = [len(positions(p, capo, op)) for _, _, _, _, p, capo in pressed]
    dist = collections.Counter(n_pos)
    lines += [f"Using {name}:",
              f"  exact pitch heard           {hit:.1%}",
              f"  pitch class heard (octave-free) {chroma:.1%}",
              f"  pressed notes: {len(pressed)}; positions that could play the same pitch: "
              f"mean {np.mean(n_pos):.2f}, " + ", ".join(f"{k} pos: {v / len(n_pos):.0%}" for k, v in sorted(dist.items()))]
    text = "\n".join(lines)
    print(text)
    (EVAL_DIR / "test").mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "test" / "audio_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
