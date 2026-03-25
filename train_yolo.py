#!/usr/bin/env python3
"""
Train YOLOv8 instance segmentation model for fretboard detection.

Usage:
  python train_yolo.py                          # default: yolov8n-seg, 50 epochs
  python train_yolo.py --model yolov8s-seg      # use small model
  python train_yolo.py --epochs 100             # more epochs
"""
import os
import sys
import argparse
import shutil
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.join(SCRIPT_DIR, "dataset_yolo", "dataset.yaml")
OUTPUT_WEIGHTS = os.path.join(SCRIPT_DIR, "yolo_weights_best.pt")
RUNS_DIR = os.path.join(SCRIPT_DIR, "yolo_runs")


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args):
    if not os.path.exists(DATASET_YAML):
        print(f"ERROR: {DATASET_YAML} not found. Run convert_coco_to_yolo.py first.")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Dataset: {DATASET_YAML}\n")

    # Load pretrained YOLO segmentation model
    model = YOLO(f"{args.model}.pt")

    # Train
    results = model.train(
        data=DATASET_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        save=True,
        project=RUNS_DIR,
        name="fretboard_seg",
        exist_ok=True,
        # Augmentation
        mosaic=1.0,
        flipud=0.0,       # no vertical flip (guitars have orientation)
        fliplr=0.5,       # horizontal flip OK
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,      # slight rotation
        translate=0.1,
        scale=0.5,
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Output
        verbose=True,
        plots=True,
    )

    # Copy best weights to project root
    best_path = os.path.join(RUNS_DIR, "fretboard_seg", "weights", "best.pt")
    if os.path.exists(best_path):
        shutil.copy2(best_path, OUTPUT_WEIGHTS)
        print(f"\n✅ Best weights saved to {OUTPUT_WEIGHTS}")
        # Print model size
        size_mb = os.path.getsize(OUTPUT_WEIGHTS) / (1024 * 1024)
        print(f"   Model size: {size_mb:.1f} MB")
    else:
        # Fallback to last weights
        last_path = os.path.join(RUNS_DIR, "fretboard_seg", "weights", "last.pt")
        if os.path.exists(last_path):
            shutil.copy2(last_path, OUTPUT_WEIGHTS)
            print(f"\n⚠️  No best.pt found. Saved last.pt to {OUTPUT_WEIGHTS}")
        else:
            print("\n❌ No weights found after training!")
            sys.exit(1)

    # Validate
    print("\n--- Validation Results ---")
    model_best = YOLO(OUTPUT_WEIGHTS)
    val_results = model_best.val(data=DATASET_YAML, device=device, verbose=True)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO for fretboard segmentation")
    parser.add_argument("--model", default="yolov8n-seg",
                        choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg"],
                        help="YOLO model variant (default: yolov8n-seg)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    args = parser.parse_args()
    train(args)
