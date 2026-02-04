"""
Lightweight CPU-friendly training script to produce Mask R-CNN weights that are
shape-compatible with `string_detection.py` (num_classes = 3).

Dataset: kagglehub dataset `jacksonlightfoot/guitar-transcription-dataset`
Split:   fretboard_dataset/fretboard_dataset/fretboard_frames_train + COCO labels

Notes:
- The COCO labels only define one foreground category ("fret", id=1).
  We still construct the model with 3 classes (background + 2 foreground slots)
  to match the inference script. The extra class will have no training samples,
  but this keeps tensor shapes aligned.
- Designed for small, quick runs on CPU. Tune epochs/batch_size if you have GPU.
"""

import os
import time
from typing import Any, Dict, List, Tuple

import kagglehub
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

try:
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "pycocotools is required for training. Install with: pip install pycocotools"
    ) from e


# ----------------------------
# Model helpers
# ----------------------------
def get_model_instance_segmentation(num_classes: int) -> torch.nn.Module:
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


# ----------------------------
# COCO -> Torchvision dataset
# ----------------------------
class CocoMaskDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder: str, ann_file: str, transforms=None):
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_folder = img_folder
        self.transforms = transforms

    def __getitem__(self, index: int):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]

        img = Image.open(os.path.join(self.img_folder, path)).convert("RGB")

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[Any] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            # Skip empty segmentation
            if "segmentation" not in ann:
                continue

            # Bounding box: [x, y, width, height] -> [xmin, ymin, xmax, ymax]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            labels.append(ann["category_id"])
            iscrowd.append(ann.get("iscrowd", 0))
            areas.append(ann.get("area", w * h))

            # To binary mask
            rle = mask_utils.frPyObjects(ann["segmentation"], img.height, img.width)
            mask = mask_utils.decode(rle)
            # Some segmentations decode to (H, W, 1)
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


# ----------------------------
# Transforms
# ----------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if torch.rand(1) < self.flip_prob:
            image = F.hflip(image)
            width = image.shape[-1]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


def get_transforms(train: bool = True):
    t = [ToTensor()]
    if train:
        t.append(RandomHorizontalFlip(0.5))
    return Compose(t)


# ----------------------------
# Training loop
# ----------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.33, total_iters=min(5, len(data_loader))
        )

    running_loss = 0.0
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += losses.item()
        if (i + 1) % print_freq == 0:
            avg = running_loss / (i + 1)
            print(f"Epoch {epoch} Iter {i + 1}/{len(data_loader)} - loss: {avg:.4f}")

    elapsed = time.time() - start
    avg_loss = running_loss / max(len(data_loader), 1)
    print(f"Epoch {epoch} done in {elapsed / 60:.1f} min - avg loss: {avg_loss:.4f}")


def main():
    device = torch.device("cpu")
    num_classes = 3  # background + 2 foreground slots (to match inference script)
    epochs = int(os.environ.get("EPOCHS", "2"))
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))  # Windows-safe default

    print("Using device:", device)
    print(
        f"Training for {epochs} epochs, batch_size={batch_size}, num_workers={num_workers}"
    )

    # Download dataset (cached after first run)
    base_path = kagglehub.dataset_download(
        "jacksonlightfoot/guitar-transcription-dataset"
    )
    ds_root = os.path.join(base_path, "fretboard_dataset", "fretboard_dataset")
    img_root = os.path.join(ds_root, "fretboard_frames_train")
    ann_file = os.path.join(ds_root, "fretboard_labels_train_coco.json")

    dataset = CocoMaskDataset(img_root, ann_file, transforms=get_transforms(train=True))
    dataset_test = CocoMaskDataset(
        img_root, ann_file, transforms=get_transforms(train=False)
    )

    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.8 * len(indices))
    dataset = torch.utils.data.Subset(dataset, indices[:split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[split:])

    def collate_fn(batch: List[Tuple[Any, Any]]):
        return tuple(zip(*batch))

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)

        # simple eval: run forward on val split to ensure shape correctness
        model.eval()
        with torch.no_grad():
            images, targets = next(iter(data_loader_test))
            _ = model([img.to(device) for img in images])
        model.train()

    # Save weights where the inference script expects them
    out_path = os.path.join(os.path.dirname(__file__), "model_weights.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}")


if __name__ == "__main__":
    main()
