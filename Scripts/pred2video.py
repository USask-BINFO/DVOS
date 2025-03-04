import cv2
import os
import torch
import glob
import numpy as np
import torchvision.transforms as T

from natsort import natsorted

from skimage import io
from torchvision.utils import draw_segmentation_masks
from PIL import Image
from tqdm import tqdm  # Progress bar

# Video settings
output_video_path = "VisualizationData/XMemFeb18_PredictionVideo30fps.mp4"
fps = 30
frame_size = (1024, 1024)

# Get sorted image and mask files
image_files = natsorted(glob.glob("VisualizationData/DJI05-104-Frames/frames/VID00001/*.png"))
mask_files = natsorted(glob.glob("Predictions/Feb18_12.23.02_TrainedOn-PseudoData-FromScratch_s2/VisualizationData/DJI05-104-Frames/Model-35000/VID00001/*.png"))

# Ensure image and mask counts match
assert len(image_files) == len(mask_files), "Mismatch between images and masks!"

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Loop through images and overlay masks
for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
    # Load image
    img = io.imread(img_file)
    mask = io.imread(mask_file).squeeze().astype(np.uint8)
    mask[mask > 0] = 1
    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # Convert to tensor
    img_tensor = torch.tensor(img.transpose(2, 0, 1))  # Shape: (3, H, W)
    mask_tensor = torch.tensor(mask).bool()  # Convert to boolean mask

    # Overlay mask
    overlaid = draw_segmentation_masks(img_tensor, masks=mask_tensor, alpha=0.5, colors=(204, 68, 150))

    # Convert back to NumPy for OpenCV
    overlaid = overlaid.numpy().transpose(1, 2, 0)
    overlaid = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)  # Convert RGB → BGR for OpenCV

    # Write frame to video
    video_writer.write(overlaid)

# Release video writer
video_writer.release()
print(f"Video saved as {output_video_path}")