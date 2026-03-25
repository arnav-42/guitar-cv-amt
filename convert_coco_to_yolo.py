#!/usr/bin/env python3
"""
Convert COCO-format fretboard dataset to YOLO segmentation format.

Merges all zone categories (Frets + Zone1-12) into a single 'fretboard' mask per image,
matching the Mask R-CNN training approach. Uses mask rasterization + contour extraction
to produce a clean merged polygon.
"""
import json
import os
import sys
import yaml
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_yolo")

# Zone categories to merge as "fretboard" (same as train_mask.py)
ZONE_CAT_IDS = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}


def polygons_to_merged_contour(annotations, W, H):
    """
    Rasterize all zone polygons into a single mask, then extract the
    largest contour as one merged fretboard polygon.
    Returns normalized polygon points or None.
    """
    import cv2

    # Build combined mask
    combined = np.zeros((H, W), dtype=np.uint8)
    for ann in annotations:
        if "segmentation" not in ann or not ann["segmentation"]:
            continue
        for seg in ann["segmentation"]:
            if len(seg) < 6:
                continue
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            pts = pts.astype(np.int32)
            cv2.fillPoly(combined, [pts], 1)

    if combined.sum() < 10:
        return None

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Find largest contour
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    # Simplify contour slightly to reduce point count
    epsilon = 0.001 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    pts = approx.reshape(-1, 2).astype(np.float64)

    # Normalize
    pts[:, 0] /= W
    pts[:, 1] /= H
    return pts


def convert_split(split_name):
    """Convert one COCO split to YOLO-seg format with merged fretboard annotations."""
    src_dir = os.path.join(DATASET_DIR, split_name)
    ann_file = os.path.join(src_dir, "_annotations.coco.json")
    if not os.path.exists(ann_file):
        print(f"  Skipping {split_name}: no annotations file")
        return 0

    with open(ann_file) as f:
        coco = json.load(f)

    # Build image lookup
    img_map = {img["id"]: img for img in coco["images"]}

    # Group zone annotations by image
    ann_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in ZONE_CAT_IDS:
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Create output directories
    img_out = os.path.join(OUTPUT_DIR, split_name, "images")
    lbl_out = os.path.join(OUTPUT_DIR, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    count = 0
    for img_id, img_info in img_map.items():
        fname = img_info["file_name"]
        W, H = img_info["width"], img_info["height"]
        src_img = os.path.join(src_dir, fname)

        if not os.path.exists(src_img):
            continue

        # Symlink image
        dst_img = os.path.join(img_out, fname)
        if os.path.exists(dst_img):
            os.remove(dst_img)
        os.symlink(os.path.abspath(src_img), dst_img)

        # Write YOLO label file
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(lbl_out, base + ".txt")

        anns = ann_by_img.get(img_id, [])
        content = ""
        if anns:
            pts = polygons_to_merged_contour(anns, W, H)
            if pts is not None:
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                content = f"0 {coords}\n"

        with open(label_path, "w") as f:
            f.write(content)
        count += 1

    return count


def main():
    print("Converting COCO → YOLO segmentation format (merged fretboard)")
    print(f"  Source: {DATASET_DIR}")
    print(f"  Output: {OUTPUT_DIR}\n")

    # Clean output directory
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    for split in ["train", "valid", "test"]:
        n = convert_split(split)
        print(f"  {split}: {n} images converted")
        total += n

    # Write dataset.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    config = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "fretboard"},
        "nc": 1,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n  Total: {total} images")
    print(f"  Config: {yaml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
