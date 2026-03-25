#!/usr/bin/env python3
"""
Train Mask R-CNN for fretboard segmentation on the Roboflow COCO dataset.
Combines all zone categories into a single fretboard class (2 classes: bg + fretboard).
"""
import os, sys, time
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_util
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_WEIGHTS = os.path.join(SCRIPT_DIR, "model_weights_new.pt")
NUM_CLASSES = 2  # background + fretboard
BATCH_SIZE = 2
NUM_EPOCHS = 3
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP = 2
LR_GAMMA = 0.1
# All zone categories to merge as "fretboard"
ZONE_CAT_IDS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Resize large images for faster training on CPU
MAX_SIZE = 800


class FretboardDataset(torch.utils.data.Dataset):
    """COCO dataset that merges all zone annotations into one fretboard instance per image."""

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(os.path.join(root, "_annotations.coco.json"))
        # Only keep images that have at least one zone annotation
        all_ids = self.coco.getImgIds()
        self.img_ids = []
        for img_id in all_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=ZONE_CAT_IDS)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Get all zone annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=ZONE_CAT_IDS)
        anns = self.coco.loadAnns(ann_ids)

        # Create individual masks and boxes for each annotation
        masks_list = []
        boxes_list = []
        for ann in anns:
            if "segmentation" not in ann or not ann["segmentation"]:
                continue
            rle = coco_mask_util.frPyObjects(ann["segmentation"], H, W)
            m = coco_mask_util.decode(rle)
            if m.ndim == 3:
                m = m.max(axis=2)
            if m.sum() < 10:
                continue
            # Compute tight bbox from mask
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            if x1 <= x0 or y1 <= y0:
                continue
            masks_list.append(m.astype(np.uint8))
            boxes_list.append([float(x0), float(y0), float(x1), float(y1)])

        if len(masks_list) == 0:
            # Fallback: empty target
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            masks = torch.as_tensor(np.stack(masks_list), dtype=torch.uint8)
            labels = torch.ones(len(masks_list), dtype=torch.int64)  # all class 1 (fretboard)
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([img_id]),
                "area": areas,
                "iscrowd": torch.zeros(len(masks_list), dtype=torch.int64),
            }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


class ResizeAndToTensor:
    """Resize image (preserving aspect ratio) and convert to tensor."""
    def __init__(self, max_size=MAX_SIZE):
        self.max_size = max_size

    def __call__(self, img, target):
        W, H = img.size
        scale = min(self.max_size / max(W, H), 1.0)

        if scale < 1.0:
            new_W, new_H = int(W * scale), int(H * scale)
            img = img.resize((new_W, new_H), Image.BILINEAR)

            # Scale boxes
            if target["boxes"].numel() > 0:
                target["boxes"] = target["boxes"] * scale
                # Scale masks
                masks_np = target["masks"].numpy()
                import cv2
                new_masks = []
                for m in masks_np:
                    new_masks.append(cv2.resize(m, (new_W, new_H), interpolation=cv2.INTER_NEAREST))
                target["masks"] = torch.as_tensor(np.stack(new_masks), dtype=torch.uint8)
                target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * \
                                 (target["boxes"][:, 2] - target["boxes"][:, 0])

        img_tensor = T.ToTensor()(img)
        return img_tensor, target


def build_model(num_classes):
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    in_fm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, num_classes)
    return m


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    print(f"Device: {DEVICE}")
    print(f"Training on: {os.path.join(DATASET_DIR, 'train')}")
    print(f"Validating on: {os.path.join(DATASET_DIR, 'valid')}")
    print(f"Output: {OUTPUT_WEIGHTS}\n")

    # Datasets
    train_ds = FretboardDataset(os.path.join(DATASET_DIR, "train"), transforms=ResizeAndToTensor())
    valid_ds = FretboardDataset(os.path.join(DATASET_DIR, "valid"), transforms=ResizeAndToTensor())
    print(f"Train: {len(train_ds)} images, Valid: {len(valid_ds)} images\n")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Model
    model = build_model(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())
            except Exception as e:
                print(f"  Batch {batch_idx} error: {e}")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        lr_scheduler.step()
        avg_train = train_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        # ── Validate ──
        model.train()  # keep in train mode for loss computation
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                try:
                    loss_dict = model(images, targets)
                    val_loss += sum(l.item() for l in loss_dict.values())
                except:
                    pass
        avg_val = val_loss / max(len(valid_loader), 1)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: train_loss={avg_train:.4f}, "
              f"val_loss={avg_val:.4f}, time={elapsed:.1f}s, lr={optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"  → Saved best model (val_loss={avg_val:.4f})")

    # Always save final
    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print(f"\nTraining complete. Weights saved to {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    train()
