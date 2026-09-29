#!/usr/bin/env python3
"""Rectify a playing frame using evidence-constrained fretboard geometry.

python -m pipeline.rectify_atlas --image frame.png --output-dir output/atlas
python -m pipeline.rectify_atlas --image frame.png --mask mask.png --output-dir output/atlas
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from .fretboard_rectify import rectify_fretboard


def predict_proposals(image, weights, confidence=.25, mask_rcnn_weights=None):
    """Separate native-resolution instances; never stretch padded YOLO masks.

    Optional Mask R-CNN proposals provide a second independently trained
    segmentation model. They are scored downstream by actual grid support.
    """
    from ultralytics import YOLO
    if not Path(weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    prediction = YOLO(str(weights)).predict(Image.fromarray(image), conf=confidence,
                                           retina_masks=True, verbose=False)[0]
    proposals = []
    if prediction.masks is not None:
        for i, tensor in enumerate(prediction.masks.data):
            mask = tensor.cpu().numpy()
            if mask.shape != image.shape[:2]:
                raise ValueError("YOLO native mask dimensions do not match input")
            proposals.append((f"yolo_{i}", (mask > .5).astype(np.uint8)*255))
    if mask_rcnn_weights:
        import torch
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        if not Path(mask_rcnn_weights).is_file():
            raise FileNotFoundError(f"Mask R-CNN weights not found: {mask_rcnn_weights}")
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
        model.load_state_dict(torch.load(mask_rcnn_weights, map_location="cpu", weights_only=True))
        model.eval()
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255.
        with torch.inference_mode():
            result = model([tensor])[0]
        for i, (score, label, mask) in enumerate(zip(result["scores"], result["labels"], result["masks"])):
            if float(score) >= confidence and int(label) == 1:
                proposals.append((f"maskrcnn_{i}", (mask[0].numpy()>.5).astype(np.uint8)*255))
    return proposals


def select_atlas(image, proposals, **kwargs):
    """Choose evidence coverage, retaining an explicit list of alternatives."""
    alternatives, best = [], None
    for name, mask in proposals:
        result = rectify_fretboard(image, mask, **kwargs)
        if result is None:
            alternatives.append(dict(proposal=name, status="failed", score=0.))
            continue
        q = result.quality
        score = ((1000 if q["status"] == "grid" else 0)
                 + (100 if q["lattice_accepted"] else 0)
                 + q["observed_frets"] + 2*q["observed_strings"])
        alternatives.append(dict(proposal=name, status=q["status"], score=float(score),
                                 observed_frets=q["observed_frets"], observed_strings=q["observed_strings"]))
        if best is None or score > best[0]:
            best = score, name, mask, result
    if best is None:
        return None, None, alternatives
    _, name, mask, result = best
    result.quality["selected_proposal"] = name
    return result, mask, alternatives


def save_atlas(result, source, mask, output_dir, alternatives=()):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image).save(out/"rectified.png")
    Image.fromarray((result.valid_mask>0).astype(np.uint8)*255).save(out/"valid_mask.png")
    if result.quality["occlusion_known"]:
        Image.fromarray((result.visible_mask>0).astype(np.uint8)*255).save(out/"visible_mask.png")
    Image.fromarray((mask>0).astype(np.uint8)*255).save(out/"proposal_mask.png")
    annotated = source.copy()
    for row in result.source_nodes:
        cv2.polylines(annotated, [np.rint(row).astype(np.int32)], False, (70, 220, 255), 1, cv2.LINE_AA)
    for col in result.source_nodes.transpose(1, 0, 2):
        cv2.polylines(annotated, [np.rint(col).astype(np.int32)], False, (255, 190, 55), 1, cv2.LINE_AA)
    Image.fromarray(annotated).save(out/"source_grid.png")
    overlay = result.image.copy()
    for x in result.target_x:
        cv2.line(overlay, (round(x), 0), (round(x), overlay.shape[0]-1), (255, 190, 55), 1)
    for y in result.target_y:
        cv2.line(overlay, (0, round(y)), (overlay.shape[1]-1, round(y)), (70, 220, 255), 1)
    Image.fromarray(overlay).save(out/"grid_overlay.png")
    report = result.to_dict()
    report["proposals"] = list(alternatives)
    (out/"geometry.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(out/"mapping.npz", source_nodes=result.source_nodes,
                        target_x=result.target_x, target_y=result.target_y)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", help="Optional existing binary mask; bypasses models")
    parser.add_argument("--occlusion-mask", help="Nonzero = occluded pixels, e.g. a hand mask")
    parser.add_argument("--weights", default=str(Path(__file__).with_name("yolo_weights_best.pt")))
    parser.add_argument("--mask-rcnn-weights", default=str(Path(__file__).with_name("model_weights.pt")),
                        help="Second segmentation model (bundled board checkpoint by default)")
    parser.add_argument("--yolo-only", action="store_true", help="Skip the slower Mask R-CNN proposal model")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--strings", type=int, default=6)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--spacing", choices=["uniform", "observed"], default="uniform")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be in [0,1]")
    try:
        image = np.asarray(Image.open(args.image).convert("RGB"))
        occ = None if not args.occlusion_mask else np.asarray(Image.open(args.occlusion_mask).convert("L"))
        proposals = ([("supplied_mask", np.asarray(Image.open(args.mask).convert("L")))] if args.mask
                     else predict_proposals(image, args.weights, args.conf,
                                            None if args.yolo_only else args.mask_rcnn_weights))
        result, mask, alternatives = select_atlas(image, proposals,
                    output_size=(args.width, args.height), num_strings=args.strings,
                    spacing=args.spacing, occlusion_mask=occ)
        if result is None:
            parser.exit(2, "No usable fretboard proposal.\n")
        save_atlas(result, image, mask, args.output_dir, alternatives)
        print(json.dumps(result.quality, indent=2, allow_nan=False))
        print(f"Saved atlas to {Path(args.output_dir).resolve()}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Rectification failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
