import argparse
import albumentations as A
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from functools import partial
from skimage import io 
from tqdm import tqdm
from typing import Tuple, Optional, Union, Dict, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utilities as utils
from data import get_dataloader
from model import get_model

warnings.filterwarnings("ignore")


def save_predicted_masks(output_dir: str, 
                         inverse_normalize: Callable,
                         upsampler: Callable,
                         predicted_query_masks: torch.Tensor, 
                         sq: List
) -> None:
    vids, fids = sq[0], sq[1]
    
    for i, mask in enumerate(predicted_query_masks):
        vid = vids[i]
        fid = fids[i]
        
        mask = torch.ge(
            torch.sigmoid(mask), 0.5
        ).cpu().squeeze().numpy().astype(np.uint8)
        
        mask = upsampler(image=mask, mask=mask)["mask"]
        mask = mask.astype(np.uint8) * 255

        io.imsave(os.path.join(output_dir, f"{vid}.png"), mask, check_contrast=False)

@torch.no_grad()
def step_prediction(model: nn.Module,
                    ground_query_images: torch.Tensor,
                    ground_query_masks: torch.Tensor,
                    ground_reference_images: torch.Tensor
) -> Tuple[torch.Tensor]:
    model.eval()
    with torch.autocast(device_type=ground_query_images.device.type, dtype=torch.bfloat16):
        _, predicted_query_masks = model(ground_reference_images)
    return predicted_query_masks

def run_pred(step_fn: Callable,  
             loader: DataLoader, 
             device: torch.device, 
             saver: Callable
) -> None:
    loss_recorder = utils.MeanMetric(scalar=False)
    score_recorder = utils.MeanMetric(scalar=False)
    step_index = 1     
    for (sq, sr) in tqdm(loader, total=len(loader), ncols=100, desc="Test"):
        ground_query_images = sq[2].to(device)
        ground_query_masks = sq[3].to(device)
        ground_reference_images = [ref.to(device) for ref in sr[2]]
        
        predicted_query_masks = step_fn(ground_query_images, ground_query_masks, ground_reference_images)
        saver(predicted_query_masks, sq)
    
def prediction_setup(device: torch.device, 
                     configs: Dict                 
) -> None:
    print(f"Prediction process on {device}.")
    
    inverse_normalize = utils.InverseNormalization(
        mean=configs.TrainConfig.TRANSFORM_MEAN,
        std=configs.TrainConfig.TRANSFORM_STD,
        dtype=torch.uint8
    )
    
    loader = get_dataloader(
        metadata_paths=configs.TrainConfig.PRED_METADATA_PATH.SEGMENTATION,
        segmentation=True,
        phase="PRED",
        configs=configs, 
        rank=device
    )

    model = get_model(device, configs)
    
    out_dir = os.path.join(configs.BasicConfig.ROOT_LOG_DIR, configs.BasicConfig.EXPERIMENT_NAME)
    pred_dir = os.path.join(out_dir, configs.BasicConfig.PREDICTION_DIR)
    os.makedirs(pred_dir, exist_ok=True)
    
    run_pred(partial(step_prediction, model), 
             loader, device,
             partial(save_predicted_masks, 
                     pred_dir, 
                     inverse_normalize,
                     A.Resize(512, 512, interpolation=cv2.INTER_NEAREST)
            )
    )      
     
if __name__ == "__main__":        
    parser = argparse.ArgumentParser(
        description='Initializers parameters for running the experiments.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        default='configs/configs.yaml',
                        help='The string path of the config file.'
    )
    args = parser.parse_args()

    configs = utils.ConfigLoader()(config_path=args.config_path)

    # Set the seeds
    utils.set_seeds(
        configs.BasicConfig.SEED,
        configs.BasicConfig.DETERMINISTIC_CUDANN,
        configs.BasicConfig.BENCHMARK_CUDANN, 
        configs.BasicConfig.DETERMINISTIC_ALGORITHM
    )

    configs.TrainConfig.DISTRIBUTED = False
    prediction_setup(
        device=torch.device(f"cuda:{configs.BasicConfig.DEFAULT_DEVICE_IDs[0]}"),
        configs=configs               
    )
