#!/usr/bin/env python3
"""
Basic Pitch (Spotify, ICASSP 2022) note transcription of every GuiFiR clip in a
split. Runs in the `audio` conda env (basic-pitch + onnxruntime).

  <audio env python> -m scripts.guifir.extract_audio --split test
Output: output/guifir/audio_basicpitch_<split>.jsonl, one line per clip:
  {"clip", "notes": [[start_s, end_s, midi_pitch, amplitude], ...]}
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

import os

ROOT = Path(__file__).resolve().parents[2]
GUIFIR = Path(os.environ.get("GUIFIR_DIR", ROOT / "data" / "guifir" / "GuiFiR"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", default="test")
    args = p.parse_args()
    ann = {"test": "guifir_test_coco_anno_box.jsonl", "val": "guifir_val_coco_anno_box.jsonl"}[args.split]
    clips = sorted({json.loads(l)["filename"].split("/")[0] for l in open(GUIFIR / "annotations" / ann)})
    out = ROOT / "output" / "guifir" / f"audio_basicpitch_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "w") as f:
        for i, clip in enumerate(clips):
            with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(GUIFIR / "raw_clips" / f"{clip}.mp4"),
                                "-ac", "1", "-ar", "22050", wav.name], check=True)
                _, _, notes = predict(wav.name, ICASSP_2022_MODEL_PATH)
            f.write(json.dumps({"clip": clip, "notes": [[round(float(n[0]), 3), round(float(n[1]), 3), int(n[2]),
                                                          round(float(n[3]), 3)] for n in notes]}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(clips)}  {time.time() - t0:.0f} s", flush=True)
    print(f"{len(clips)} clips -> {out}")


if __name__ == "__main__":
    main()
