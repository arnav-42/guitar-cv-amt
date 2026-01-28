import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import kagglehub
import os
import glob
import random
import matplotlib.pyplot as plt

def get_model_instance_segmentation(num_classes):
    """
    Returns a Mask R-CNN model with a ResNet-50 backbone.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def run_prediction():
    """
    Loads the trained model, runs inference on test images, and visualizes the results.
    """
    # --- 1. Setup ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # We have 2 classes: background and fretboard
    num_classes = 2
    model_weights_path = 'model_weights.pt'

    # --- 2. Load Model ---
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # --- 3. Load Test Data ---
    print("\nSetting up test data paths...")
    base_path = kagglehub.dataset_download("jacksonlightfoot/guitar-transcription-dataset")
    image_folder_path = os.path.join(base_path, 'fretboard_dataset', 'fretboard_dataset', 'fretboard_frames_test')

    if not os.path.exists(image_folder_path):
        print(f"--> ERROR: Folder not found at {image_folder_path}")
        return

    print(f"--> SUCCESS: Found image folder at {image_folder_path}")
    image_files = glob.glob(os.path.join(image_folder_path, '*.png')) + glob.glob(os.path.join(image_folder_path, '*.jpg'))
    print(f"Found {len(image_files)} total test images.")

    # --- 4. Run Inference and Visualize ---
    num_images_to_show = 3
    if len(image_files) < num_images_to_show:
        print("Not enough images found to display.")
        return

    sample_images = random.sample(image_files, num_images_to_show)
    confidence_threshold = 0.5 # Only show predictions with a score > 50%

    # Define the image transformation
    transform = T.Compose([T.ToTensor()])

    print(f"\nRunning predictions with a {confidence_threshold:.0%} confidence threshold...")
    for image_path in sample_images:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)

        with torch.no_grad():
            prediction = model([img_tensor.to(device)])

        # --- 5. Draw Results ---
        # Move prediction results to CPU
        pred_scores = prediction[0]['scores'].cpu().numpy()
        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_masks = prediction[0]['masks'].cpu().numpy()

        # Filter predictions by score
        high_conf_indices = [i for i, score in enumerate(pred_scores) if score > confidence_threshold]

        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw, 'RGBA')

        print(f"\nFound {len(high_conf_indices)} objects in {os.path.basename(image_path)}")

        for i in high_conf_indices:
            box = pred_boxes[i]
            mask = pred_masks[i, 0] # Get the first (and only) channel of the mask

            # Draw the bounding box
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=3)

            # Draw the segmentation mask
            # Create a color for the mask (e.g., green with some transparency)
            mask_color = (0, 255, 0, 100) # RGBA
            mask_array = ((mask > 0.5) * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_array).convert('L')

            # Paste the mask onto the image
            img_draw.paste(mask_color, (0, 0), mask_pil)

        # Display the result using matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(img_draw)
        plt.title(f"Prediction for {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    run_prediction()
