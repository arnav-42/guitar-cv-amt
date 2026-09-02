# YOLO vs Mask R-CNN: Fretboard Segmentation & Fret Detection Comparison

## Segmentation Quality (Dice Score)

| Model | Dice Score | Std Dev | Model Size |
|-------|-----------|---------|------------|
| **Mask R-CNN** (ResNet-50 FPN) | **0.9071** | ±0.2167 | 176 MB |
| **YOLOv8n-seg** (Nano) | **0.9009** | ±0.1396 | 6.4 MB |

> [!TIP]
> YOLO achieves comparable Dice (0.9009 vs 0.9071) with **27× smaller** model size and **lower variance** (0.14 vs 0.22).

## Fret Detection — Test Split (F1 @ 2.0% threshold)

| Method | Mask R-CNN | YOLO | Δ |
|--------|-----------|------|---|
| A: Arnav Canon. | 0.5737 | **0.5907** | +0.017 |
| **B: PCA + Warp** | 0.7974 | **0.8067** | +0.009 |
| C: PCA (no warp) | 0.7824 | **0.7963** | +0.014 |
| D: Paleari&Huet | 0.5431 | **0.5597** | +0.017 |
| E: Paleari Ref. | 0.5508 | **0.5589** | +0.008 |
| F: Hybrid | 0.5092 | 0.4720 | −0.037 |

## Fret Detection — Test Split (F1 @ 1.0% threshold)

| Method | Mask R-CNN | YOLO | Δ |
|--------|-----------|------|---|
| A: Arnav Canon. | 0.4013 | 0.3947 | −0.007 |
| B: PCA + Warp | 0.6039 | **0.6333** | +0.029 |
| C: PCA (no warp) | 0.5615 | **0.6365** | +0.075 |
| D: Paleari&Huet | 0.4516 | **0.4567** | +0.005 |
| E: Paleari Ref. | 0.4590 | 0.4545 | −0.005 |
| F: Hybrid | 0.4429 | 0.4069 | −0.036 |

## Mean Pixel Error (% of image width, test split)

| Method | Mask R-CNN | YOLO | Better |
|--------|-----------|------|--------|
| A: Arnav Canon. | 1.298 | **1.257** | YOLO |
| B: PCA + Warp | **0.774** | 0.788 | MRCNN |
| C: PCA (no warp) | 0.845 | **0.762** | YOLO |
| D: Paleari&Huet | 0.889 | **0.843** | YOLO |
| E: Paleari Ref. | 0.902 | 0.899 | ~tie |
| F: Hybrid | 0.738 | **0.735** | YOLO |

## Key Findings

1. **YOLO matches or exceeds Mask R-CNN** on most fret detection methods despite being 27× smaller
2. **Best overall method**: **B: PCA + Warp** with YOLO masks achieves F1@2.0%=0.807, F1@1.0%=0.633
3. **YOLO excels at geometric methods** (B, C) — improvements of +1–7.5% F1 at strict thresholds, likely due to lower-variance masks
4. **Mask R-CNN retains edge** with Method F (Hybrid) — the peak-finding approach is sensitive to mask boundary shape
5. **YOLO model stats**: 3.26M params, 11.5 GFLOPs, 6.6ms inference (on MPS), trained in 45 min

## Training Details

- **Architecture**: YOLOv8n-seg (nano variant)
- **Training**: 50 epochs, AdamW optimizer, lr=0.001, batch=8, imgsz=640
- **Augmentation**: mosaic, horizontal flip, HSV jitter, ±5° rotation, 0.5× scale
- **Device**: Apple M2 (MPS), ~45s/epoch
- **Best val Mask mAP50-95**: 0.947 (epoch 50)
- **Weights**: [yolo_weights_best.pt](yolo_weights_best.pt) (6.4 MB)

## Files

- [evaluation_results.csv](evaluation_results.csv) — Full results (24 rows: 6 methods × 2 splits × 2 models)
- [yolo_weights_best.pt](yolo_weights_best.pt) — Trained YOLO model
- [train_yolo.py](train_yolo.py) — Training script
- [convert_coco_to_yolo.py](convert_coco_to_yolo.py) — Dataset converter
- [evaluate_all.py](evaluate_all.py) — Updated evaluation pipeline (supports `--yolo-weights`)
