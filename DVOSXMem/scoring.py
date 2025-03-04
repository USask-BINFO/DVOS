import glob 
import os

import numpy as np
import pandas as pd

from skimage import io
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
from torchvision.utils import draw_segmentation_masks


def calculate_metrics(output: np.ndarray, 
                      target: np.ndarray, 
                      smooth=1.0
) -> Dict[str, float]:
    y_pred = output.astype(np.float32)
    y_true = target.astype(np.float32)

    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    true_segment = np.sum(y_true)
    pred_segment = np.sum(y_pred)

    dice = (2 * intersection + smooth) / (true_segment + pred_segment + smooth)
    iou = (intersection + smooth) / (union + smooth)

    return {'dice': dice, 'iou': iou}

def calculate_and_save_metrics(image_paths: List[str], 
                               groundtruth_paths: List[str], 
                               pred_paths: List[str],
                               out_dir: str,   
                               output_csv: str = "metrics.csv", 
                               color: Tuple = (204, 68, 150), 
                               overlay_interval: int = 1
) -> None:
    os.makedirs(os.path.join(out_dir, "overlaid"), exist_ok=True)

    results = []
    best_sample, worst_sample = None, None
    best_score, worst_score = -1, float('inf')
    
    assert len(image_paths) == len(groundtruth_paths) == len(pred_paths)

    total_dice, total_iou = 0.0, 0.0
    sample_counter = 0
    for img_path, gt_path, pr_path in tqdm(zip(image_paths, groundtruth_paths, pred_paths), desc="Processing", total=len(image_paths)):
        if not os.path.exists(pr_path) or not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue
        
        gt_img = io.imread(img_path)
        gt_msk = io.imread(gt_path).squeeze()
        pr_msk = io.imread(pr_path).squeeze()
        
        gt_msk[gt_msk > 0] = 1
        pr_msk[pr_msk > 0] = 1
        
        score = calculate_metrics(pr_msk, gt_msk)
        total_dice += score['dice']
        total_iou += score['iou']
        sample_counter += 1
        results.append({
            "Image": img_path,
            "Mask": gt_path, 
            "Pred": pr_path,
            "IoU": score['iou'],
            "Dice": score['dice']
        })

        avg_score = (score['iou'] + score['dice']) / 2
        if avg_score > best_score:
            best_score = avg_score
            best_sample = (gt_img, pr_msk)
        if avg_score < worst_score:
            worst_score = avg_score
            worst_sample = (gt_img, pr_msk)

        if sample_counter % overlay_interval == 0: 
            img_tensor = torch.from_numpy(gt_img).permute(2, 0, 1)
            pred_mask_tensor = torch.from_numpy(pr_msk.astype(np.uint8))

            masked_img = draw_segmentation_masks(img_tensor, masks=pred_mask_tensor.bool(), alpha=0.4, colors=color)
            masked_img_np = masked_img.permute(1, 2, 0).byte().numpy()

            io.imsave(os.path.join(out_dir, "overlaid", os.path.basename(img_path)), masked_img_np)
        
    total_dice /= sample_counter
    total_iou /= sample_counter
    print(f"Average Dice is {total_dice:0.4f}; Average IoU is {total_iou:0.4f}.")

    df = pd.DataFrame(results)
    df.loc['Mean'] = df.mean(numeric_only=True)
    df.to_csv(os.path.join(out_dir, output_csv), index=False)
    
    # Save the best sample
    gt_img, pr_msk = best_sample
    img_tensor = torch.from_numpy(gt_img).permute(2, 0, 1)
    pred_mask_tensor = torch.from_numpy(pr_msk.astype(np.uint8))
    masked_img = draw_segmentation_masks(img_tensor, masks=pred_mask_tensor.bool(), alpha=0.4, colors=color)
    masked_img_np = masked_img.permute(1, 2, 0).byte().numpy()
    io.imsave(os.path.join(out_dir, "overlaid", "bset.png"), masked_img_np)
    
    # Save the worst sample
    gt_img, pr_msk = worst_sample
    img_tensor = torch.from_numpy(gt_img).permute(2, 0, 1)
    pred_mask_tensor = torch.from_numpy(pr_msk.astype(np.uint8))
    masked_img = draw_segmentation_masks(img_tensor, masks=pred_mask_tensor.bool(), alpha=0.4, colors=color)
    masked_img_np = masked_img.permute(1, 2, 0).byte().numpy()
    io.imsave(os.path.join(out_dir, "overlaid", "worst.png"), masked_img_np)


if __name__ == "__main__": 
    gt_dir         = "PseudoLabeledData/testing"
    prediction_dir = "Feb20_00.24.14_PseudoData-PretrainedOnSyntheticDataWhichTrainedOnS012_s2/TestData/Pseudo/Model-35000"
    print(f"Scoring the prediction of model {prediction_dir.split('/')[-1]}.")
    pr_dir         = os.path.join("Predictions/", prediction_dir)
    overlay_interval = 100
    
    image_paths = [
        sorted(glob.glob(os.path.join(gt_dir, "frames", item, "*.png")))[-1]
        for item in sorted(os.listdir(os.path.join(gt_dir, "frames")))
    ]
    gt_paths = [
        sorted(glob.glob(os.path.join(gt_dir, "masks", item, "*.png")))[-1]
        for item in sorted(os.listdir(os.path.join(gt_dir, "masks")))
    ]
    pr_paths = [
        sorted(glob.glob(os.path.join(pr_dir, item, "*.png")))[-1]
        for item in sorted(os.listdir(pr_dir))
    ]
    
    calculate_and_save_metrics(
        image_paths, 
        gt_paths, 
        pr_paths, 
        os.path.join("Scoring", prediction_dir), 
        "scores.csv", 
        overlay_interval=overlay_interval
    )