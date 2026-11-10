#!/usr/bin/env python3
"""
HaMeR / WiLoR (3D hand mesh models, the GuiFiR paper's strongest baselines) on
GuiFiR frames, top-down on GuiFiR's own left-hand box (same input as hand_crop).
Writes the same JSONL as predict_guifir_tips.py so eval_guifir_tips.py scores it.

Runs in the model's own conda env (third_party/hamer, third_party/WiLoR):
  <hamer env python> -m scripts.guifir.predict_mesh --model hamer
  <wilor env python> -m scripts.guifir.predict_mesh --model wilor

Left hands: right=0 makes the dataset mirror the crop (as in both demos); the
normalized 2D keypoints are un-mirrored (x * (2*right - 1)) and mapped back with
the crop's box center and size. Rendering (pyrender, EGL) is stubbed out.
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]   # repo root (scripts/guifir/ -> repo)
REPOS = {"hamer": ROOT / "third_party" / "hamer", "wilor": ROOT / "third_party" / "WiLoR"}


def _stub_pyrender():
    stub = types.ModuleType("pyrender")

    class _Dummy:                          # accepts any construction / call; rendering is never used
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return _Dummy()

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Dummy()

    def _ga(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (_Dummy,), {})

    stub.__getattr__ = _ga
    sys.modules["pyrender"] = stub


def _numpy_aliases():
    """MANO_RIGHT.pkl unpickles chumpy objects; chumpy imports aliases NumPy 2 removed."""
    for name, value in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("unicode", str), ("str", str)):
        if name not in np.__dict__:
            setattr(np, name, value)


def load(model_name):
    repo = REPOS[model_name]
    sys.path.insert(0, str(repo))
    os.chdir(repo)                         # configs use repo-relative paths (_DATA/, mano_data/)
    _stub_pyrender()
    _numpy_aliases()
    if model_name == "hamer":
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import DEFAULT_CHECKPOINT, load_hamer
        model, cfg = load_hamer(DEFAULT_CHECKPOINT)
    else:
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor
        model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                cfg_path="./pretrained_models/model_config.yaml")
    return model.eval(), cfg, ViTDetDataset


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, choices=list(REPOS))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--boxes", default="gdino", choices=["gdino", "wilor_det"],
                   help="gdino = GuiFiR's left-hand box (Grounding DINO); wilor_det = WiLoR's own hand "
                        "detector, all hands with its left/right flag (the GuiFiR paper's setting for 3D baselines)")
    p.add_argument("--detect-only", action="store_true",
                   help="run WiLoR's detector and cache boxes for --boxes wilor_det (wilor env, needs ultralytics)")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT))
    from main.video.fretboard.guifir import ClipReader, load_rows   # before chdir: paths are absolute
    rows = load_rows(args.split)
    rows = rows[:args.limit] if args.limit else rows
    out = ROOT / "output" / "guifir" / "preds" / f"{args.model}_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    det_path = ROOT / "output" / "guifir" / f"wilor_det_{args.split}.jsonl"
    if args.detect_only:
        from ultralytics import YOLO
        det = YOLO(str(REPOS["wilor"] / "pretrained_models" / "detector.pt"))
        reader = ClipReader()
        with open(det_path, "w") as f:
            for i, r in enumerate(rows):
                img = reader.read(r["clip"], r["frame_index"])
                boxes = []
                if img is not None:
                    for d in det(img, conf=0.3, verbose=False)[0]:
                        b = d.boxes.data.cpu().numpy().squeeze()
                        boxes.append({"box": b[:4].tolist(), "right": float(d.boxes.cls.cpu().item())})
                f.write(json.dumps({"clip": r["clip"], "frame": r["frame_index"], "boxes": boxes}) + "\n")
                if (i + 1) % 200 == 0:
                    print(f"  detect {i + 1}/{len(rows)}", flush=True)
        print(f"detections -> {det_path}")
        return
    dets = {}
    if args.boxes == "wilor_det":
        for line in open(det_path):
            d = json.loads(line)
            dets[(d["clip"], d["frame"])] = d["boxes"]
        out = out.with_name(f"{args.model}_wilordet_{args.split}.jsonl")

    model, cfg, ViTDetDataset = load(args.model)
    dev = torch.device(args.device)
    model.to(dev)
    reader, t0 = ClipReader(), time.time()
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            img = reader.read(r["clip"], r["frame_index"])
            hands = []
            if args.boxes == "wilor_det":
                cand = [(np.asarray(b["box"], float), b["right"]) for b in dets.get((r["clip"], r["frame_index"]), [])]
            else:
                box = (r.get("hand_annotation") or {}).get("boxes")
                cand = [(np.asarray(box, float) * (img.shape[1] / r["frame_width"] if img is not None else 1), 0.0)] if box else []
            for box, right in (cand if img is not None else []):
                ds = ViTDetDataset(cfg, img, box[None], np.array([right]), rescale_factor=2.0)
                item = ds[0]
                batch = torch.utils.data.default_collate([item])
                batch = {k: (v.float() if v.dtype == torch.float64 else v).to(dev) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                with torch.no_grad():
                    o = model(batch)
                kp = o["pred_keypoints_2d"][0].float().cpu().numpy()          # normalized, crop mirrored if left
                kp[:, 0] *= (2 * right - 1)                                 # un-mirror left hands
                xy = kp * float(item["box_size"]) + np.asarray(item["box_center"], float)
                hands.append({"handedness": "Right" if right else "Left", "keypoints": xy.tolist(),
                              "scores": [1.0] * len(xy)})
            f.write(json.dumps({"clip": r["clip"], "frame": r["frame_index"],
                                "width": None if img is None else img.shape[1], "hands": hands}) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
    print(f"{len(rows)} frames -> {out}")


if __name__ == "__main__":
    main()
