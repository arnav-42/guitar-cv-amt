#!/usr/bin/env python3
"""
Run a hand-keypoint model (predict_fingertips.PREDICTORS) on every annotated
GuiFiR frame and store all predicted 21-point hands, frame pixel coordinates.

Usage:
  python -m scripts.guifir.predict_tips --model mediapipe --split test        # needs mediapipe
  python -m scripts.guifir.predict_tips --model hand_crop --split test        # mmpose env (RTMPose hand)
  python -m scripts.guifir.predict_tips --model rtmw --split test             # mmpose env (RTMW-x)
  python -m scripts.guifir.predict_tips --model vitpose_b --split test        # mmpose env + easy_ViTPose
Output: output/guifir/preds/<model>_<split>.jsonl, one line per frame:
  {"clip", "frame", "width", "hands": [{"handedness", "keypoints": [[x, y] * 21], "scores"}]}
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2

import numpy as np

from main.video.fretboard.guifir import ClipReader, load_rows
from scripts.guifir.common import PRED_DIR, THIRD_PARTY, short_path

_MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                 "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_MP_MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
LEFT_HAND = slice(91, 112)     # COCO-WholeBody left-hand keypoints (fretting hand of a right-handed player)


def mediapipe_predictor():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not _MP_MODEL_PATH.exists():
        _MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {_MP_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MP_MODEL_URL, _MP_MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MP_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,  # samples are independent frames
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    def predict(frame_bgr):
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(image)
        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            hands.append({
                "handedness": result.handedness[i][0].category_name,
                "hand_score": float(result.handedness[i][0].score),
                # normalized [0, 1] spans pixel edges; shift by half a pixel to
                # the pixel-center convention (0.08 string spacings at 6px)
                "keypoints": [[lm.x * w - 0.5, lm.y * h - 0.5] for lm in landmarks],
                "scores": [1.0] * len(landmarks),  # MediaPipe has no per-keypoint score
            })
        return hands

    return predict


def _device():
    import torch
    return "cpu"   # mmcv nms has no MPS kernel


def rtmw_predictor():
    """RTMW-x 384x288 wholebody (person detector RTMDet), left hand only."""
    from mmpose.apis import MMPoseInferencer
    inf = MMPoseInferencer(pose2d="rtmw-x_8xb320-270e_cocktail14-384x288", device=_device(), show_progress=False)

    def predict(frame_bgr, row=None):
        res = next(inf(frame_bgr, show=False, return_vis=False))["predictions"][0]
        return [{"handedness": "Left", "person": j,
                 "keypoints": np.asarray(p["keypoints"])[LEFT_HAND].tolist(),
                 "scores": np.asarray(p["keypoint_scores"])[LEFT_HAND].tolist()} for j, p in enumerate(res)]

    return predict


def hand_crop_predictor():
    """RTMPose-m hand (hand5 dataset), top-down on GuiFiR's own left-hand box
    (Grounding DINO, hand_annotation) - the same detector GuiFiR uses."""
    from mmpose.apis import MMPoseInferencer, inference_topdown
    model = MMPoseInferencer(pose2d="rtmpose-m_8xb256-210e_hand5-256x256", device=_device(),
                             show_progress=False).inferencer.model

    def predict(frame_bgr, row=None):
        box = (row.get("hand_annotation") or {}).get("boxes")
        if not box:                          # no left-hand detection for this frame
            return []
        box = np.asarray(box, float) * frame_bgr.shape[1] / row["frame_width"]
        res = inference_topdown(model, frame_bgr, bboxes=box[None], bbox_format="xyxy")
        return [{"handedness": "Left", "keypoints": s.pred_instances.keypoints[0].tolist(),
                 "scores": s.pred_instances.keypoint_scores[0].tolist()} for s in res]

    return predict


def vitpose_wb_predictor(size):
    """ViTPose wholebody (ViTPose++ COCO-WholeBody weights converted by easy_ViTPose,
    third_party/vitpose/vitpose-<size>-wholebody.pth), top-down on RTMDet person boxes
    (the detector RTMW uses). Preprocessing and decoding follow easy_ViTPose's
    VitInference: box +-10 px, pad to 3:4, resize 192x256, ImageNet norm, UDP decoding.
    Only the model, config and decoding modules are imported (its package __init__
    needs ultralytics)."""
    import importlib.util
    import sys

    import cv2
    import torch
    from mmpose.apis import MMPoseInferencer

    import types
    pkg = importlib.util.find_spec("easy_ViTPose").submodule_search_locations[0]
    sys.path.insert(0, pkg)
    sys.modules.setdefault("ffmpeg", types.ModuleType("ffmpeg"))   # only its video writer uses it
    from configs import ViTPose_wholebody as cfg
    from vit_models.model import ViTPose
    from vit_utils.inference import pad_image
    from vit_utils.top_down_eval import keypoints_from_heatmaps

    detector = MMPoseInferencer(pose2d="rtmw-x_8xb320-270e_cocktail14-384x288", device="cpu",
                                show_progress=False).inferencer.detector
    model = ViTPose(getattr(cfg, {"s": "model_small", "b": "model_base", "l": "model_large", "h": "model_huge"}[size]))
    ckpt = torch.load(THIRD_PARTY / "vitpose" / f"vitpose-{size}-wholebody.pth",
                      map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval().to(dev)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    @torch.no_grad()
    def predict(frame_bgr, row=None):
        det = detector(frame_bgr)["predictions"][0]
        rgb = frame_bgr[..., ::-1]
        hands = []
        for j, (box, score, label) in enumerate(zip(det["bboxes"], det["scores"], det["labels"])):
            if label != 0 or score < 0.3:
                continue
            b = np.round(box).astype(int)
            b[[0, 2]] = np.clip(b[[0, 2]] + [-10, 10], 0, rgb.shape[1])
            b[[1, 3]] = np.clip(b[[1, 3]] + [-10, 10], 0, rgb.shape[0])
            crop, (left, top) = pad_image(rgb[b[1]:b[3], b[0]:b[2]], 3 / 4)
            h, w = crop.shape[:2]
            x = (cv2.resize(crop, (192, 256), interpolation=cv2.INTER_LINEAR) / 255 - mean) / std
            heat = model(torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(dev)).cpu().numpy()
            pts, prob = keypoints_from_heatmaps(heatmaps=heat, center=np.array([[w // 2, h // 2]]),
                                                scale=np.array([[w, h]]), unbiased=True, use_udp=True)
            xy = pts[0] + [b[0] - left, b[1] - top]          # pts are (x, y) in the padded crop
            hands.append({"handedness": "Left", "person": j, "keypoints": xy[LEFT_HAND].tolist(),
                          "scores": prob[0, LEFT_HAND, 0].tolist()})
        return hands

    return predict


PREDICTORS = {"mediapipe": mediapipe_predictor, "rtmw": rtmw_predictor, "hand_crop": hand_crop_predictor,
              "vitpose_b": lambda: vitpose_wb_predictor("b"), "vitpose_l": lambda: vitpose_wb_predictor("l")}
ROW_MODELS = ("rtmw", "hand_crop", "vitpose_b", "vitpose_l")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, choices=list(PREDICTORS))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = load_rows(args.split)
    rows = rows[:args.limit] if args.limit else rows
    predict = PREDICTORS[args.model]()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{args.model}_{args.split}.jsonl"
    reader, t0 = ClipReader(), time.time()
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            img = reader.read(r["clip"], r["frame_index"])
            hands = (predict(img, r) if args.model in ROW_MODELS else predict(img)) if img is not None else []
            f.write(json.dumps({"clip": r["clip"], "frame": r["frame_index"],
                                "width": None if img is None else img.shape[1], "hands": hands}) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
    print(f"{len(rows)} frames -> {short_path(out)}")


if __name__ == "__main__":
    main()
