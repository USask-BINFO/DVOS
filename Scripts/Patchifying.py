import os
import cv2
import albumentations as A
import numpy as np 
import pandas as pd
import glob 
import matplotlib.pyplot as plt
import einops
from typing import List 
from skimage import io 
from icecream import ic 

import SimpleITK as sitk


def patchify(image: np.ndarray, 
             mask: np.ndarray,
             patch_size: int
): 
    image_patches = einops.rearrange(
        image, 'b (h p1) (w p2) c -> (h w) b p1 p2 c', p1=patch_size, p2=patch_size
    )
    mask_patches = einops.rearrange(
        mask, 'b (h p1) (w p2) -> (h w) b p1 p2', p1=patch_size, p2=patch_size
    )
    
    return image_patches, mask_patches
    
def patch_raw_images() -> None: 
    metadata = pd.read_csv("data/ChosenLabeledDroneFRames/ValidMetadata.csv")
    groups = metadata.groupby("VID")
                  
    df = {
        "VID": [], 
        "FID": [], 
        "Image": [], 
        "Mask": [], 
        "Label": [], 
    }
    k = 0
    for gid, group in groups: 
        image = np.stack([io.imread(img) for img in group.Image], axis=0)
        mask = np.stack([io.imread(msk).squeeze() for msk in group.Mask], axis=0)
        ic(gid, image.shape)
        image_patches, mask_patches = patchify(image, mask, 1024)
        for ptch_id, (img_ptchs, msk_ptchs) in enumerate(zip(image_patches, mask_patches)):
            labels = [0] * len(img_ptchs)
            labels[-1] = 1
            for j, (img, msk) in enumerate(zip(img_ptchs, msk_ptchs)):  
                img_out_pth = f"data/ChosenLabeledDroneFrames1024Patches/Batch2/{gid}/patch{ptch_id}/img-patch{j:0>4}.png"
                msk_out_pth = f"data/ChosenLabeledDroneFrames1024Patches/Batch2/{gid}/patch{ptch_id}/msk-patch{j:0>4}.png"
                os.makedirs(
                    os.path.dirname(img_out_pth), 
                    exist_ok=True
                )
                io.imsave(img_out_pth, img, check_contrast=False)
                io.imsave(msk_out_pth, msk, check_contrast=False)
                
                df["VID"].append(f"Batch2{gid}-ptch{ptch_id:0>4}")
                df["FID"].append(k)
                df["Image"].append(img_out_pth)
                df["Mask"].append(msk_out_pth)
                df["Label"].append(labels[j])
                k += 1          
    df = pd.DataFrame(df)
    df.to_csv("data/ChosenLabeledDroneFrames1024Patches/Batch2ChosenLabeledDroneFrames1024Patches.csv", index=False)
            
            
if __name__ == "__main__": 
    patch_raw_images()