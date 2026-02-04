import glob
import os
import random

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def draw_strings_via_pca(draw, mask_bool, color=(0, 255, 255), width=2):
    """
    Calculates the orientation of the fretboard mask using PCA and draws
    6 equidistant strings along its length.
    """
    y_coords, x_coords = np.where(mask_bool)

    # If mask is too small, skip
    if len(y_coords) < 50:
        return

    # We stack them so we can do math on the (x,y) points
    pts = np.column_stack([x_coords, y_coords])

    # Mean center the data
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    # Covariance matrix
    cov = np.cov(centered.T)
    # Eigenvalues and Eigenvectors
    vals, vecs = np.linalg.eig(cov)

    # Sort so the largest eigenvalue (longest axis) is first
    sort_idxs = np.argsort(vals)[::-1]
    major_axis = vecs[:, sort_idxs[0]]  # The direction the strings run
    minor_axis = vecs[:, sort_idxs[1]]  # The direction of the width/frets

    # Project all points onto the major axis to find top/bottom of neck
    major_proj = np.dot(centered, major_axis)
    l_min, l_max = np.min(major_proj), np.max(major_proj)

    # Project all points onto minor axis to find width of neck
    minor_proj = np.dot(centered, minor_axis)
    w_min, w_max = np.min(minor_proj), np.max(minor_proj)

    # We want 6 strings across the width (w_min to w_max).
    # We add a small margin (e.g. 10%) so strings aren't exactly on the edge.
    margin = (w_max - w_min) * 0.1

    # Create 6 equidistant offsets along the minor axis
    string_offsets = np.linspace(w_min + margin / 2, w_max - margin / 2, 6)

    for offset in string_offsets:
        # Calculate start point (one end of the neck)
        # Point = Mean + (Length_position * StringDir) + (Width_position * WidthDir)
        p_start = mean + (l_min * major_axis) + (offset * minor_axis)

        # Calculate end point (other end of the neck)
        p_end = mean + (l_max * major_axis) + (offset * minor_axis)

        draw.line(
            [(p_start[0], p_start[1]), (p_end[0], p_end[1])], fill=color, width=width
        )


def run_prediction():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    num_classes = 3
    model_weights_path = "model_weights.pt"

    model = get_model_instance_segmentation(num_classes)
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        print("Error: model_weights.pt not found. Please train the model first.")
        return

    model.to(device)
    model.eval()

    print("\nSetting up test data...")
    try:
        base_path = kagglehub.dataset_download(
            "jacksonlightfoot/guitar-transcription-dataset"
        )
        image_folder_path = os.path.join(
            base_path, "fretboard_dataset", "fretboard_dataset", "fretboard_frames_test"
        )
        image_files = glob.glob(os.path.join(image_folder_path, "*.png")) + glob.glob(
            os.path.join(image_folder_path, "*.jpg")
        )
    except Exception as e:
        print(f"Dataset error: {e}")
        return

    if not image_files:
        print("No images found.")
        return

    sample_images = random.sample(image_files, min(len(image_files), 3))
    transform = T.Compose([T.ToTensor()])
    confidence_threshold = 0.5

    for image_path in sample_images:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)

        with torch.no_grad():
            prediction = model([img_tensor.to(device)])

        # Process results
        pred_scores = prediction[0]["scores"].cpu().numpy()
        pred_masks = prediction[0]["masks"].cpu().numpy()

        # Filter by confidence
        high_conf_indices = [
            i for i, score in enumerate(pred_scores) if score > confidence_threshold
        ]

        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw, "RGBA")

        print(f"Processing {os.path.basename(image_path)}...")

        for i in high_conf_indices:
            raw_mask = pred_masks[i, 0]
            mask_bool = raw_mask > 0.5
            # We pass the invisible boolean mask to the helper function to calculate string positions
            draw_strings_via_pca(draw, mask_bool, color=(0, 255, 255), width=2)

        plt.figure(figsize=(12, 8))
        plt.imshow(img_draw)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    run_prediction()
