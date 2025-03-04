import os 
import shutil 

import pandas as pd 


df = pd.read_csv("Main-155-Frames-Validation.csv")
video_id = 0
gt_ids = df[df.Label == 1].FID.values

for gt in gt_ids:
    gt_frame_path = f"SyntheticWheatData/validation/frames/VID-{video_id:0>5}/"
    gt_mask_path = f"SyntheticWheatData/validation/masks/VID-{video_id:0>5}/"
    os.makedirs(gt_frame_path, exist_ok=True)
    os.makedirs(gt_mask_path, exist_ok=True)
    
    for i in range(max(0, gt - 10), gt+1):
        if df[df.FID == i].shape[0] != 1: 
            continue
        shutil.copy(
            os.path.join("../DiffUNet/", df[df.FID == i].Image.item()),
            gt_frame_path
        )  
        shutil.copy(
            os.path.join("../DiffUNet/", df[df.FID == i].Mask.item()),
            gt_mask_path
        )   
    video_id += 1
       