import albumentations as A 
import glob 
import os 
import pandas as pd 

from skimage import io 

import utilities as utils


subset = []

pred_masks_dir = "./Predictions/VideoBasedExperiments/QueryAsInput/Exp01_WholeArchitecture-OnPseudo/Pseudo/"
out_dir = "./Predictions/VideoBasedExperiments/QueryAsInput/Exp01_WholeArchitecture-OnPseudo/Pseudo_overlaid/"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv("../PseudoLabeledData/testing.csv")
print("Images Length: ", df.Label.sum())

tr = A.CenterCrop(512, 512, p=1.0)

for i, row in df.iterrows():
    if (row.Label == 0) or (subset and row.VID not in subset):
        continue
    
    print(row.Image)
    
    gt_image = io.imread(row.Image)
    gt_mask = io.imread(row.Mask)
    pr_mask = io.imread(
        os.path.join(pred_masks_dir, f"{row.VID}.png")
    )    
    
    aug = tr(image=gt_image, mask=gt_mask)
    gt_image = aug["image"]
    gt_mask = aug["mask"]
    
    gt_overlaid = utils.overlay_mask(gt_image, gt_mask)
    pr_overlaid = utils.overlay_mask(gt_image, pr_mask)
    
    io.imsave(os.path.join(out_dir, f"{row.VID}_{row.FID:0>6}_GT.png"), gt_overlaid, check_contrast=False)    
    io.imsave(os.path.join(out_dir, f"{row.VID}_{row.FID:0>6}_PR.png"), pr_overlaid, check_contrast=False)    