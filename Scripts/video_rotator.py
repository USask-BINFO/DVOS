import cv2
import os 
import pandas as pd
import numpy as np
from skimage import io 
import albumentations as A 

from icecream import ic 
from tqdm import tqdm


NUMBER_OF_REFERENCE_FRAMES = 5

df = pd.read_csv("data/simulated/validation/Main-155-Frames-Validation.csv")
ic(df.head())
ic(df.shape)


new_df = {
    "VID": [], 
    "FID": [],
    "Image": [],
    "Mask": [],
    "Label": []
}
output_dir = "data/simulated/validation/rotated/"
os.makedirs(output_dir, exist_ok=True)

fid = 0
for i, row in df.iterrows():
    if row.OriginalLabel == 0: 
        continue
    sample_dir = os.path.join(output_dir, f"Sample-{row.FID:0>6}")
    os.makedirs(sample_dir, exist_ok=True)
    
    with tqdm(total=359) as pbar:
        for deg in range(0, 359): 
            pbar.update(1)
            pbar.set_description(f"Processing {row['VID']}-{row['FID']}-{deg:0>6}")

            deg_dir = os.path.join(sample_dir, f"Deg-{deg:0>3}")
            os.makedirs(deg_dir, exist_ok=True)

            tr = A.Rotate(
                limit=(deg, deg), 
                interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=0, 
                mask_value=0,
                always_apply=True,
                p=1.0
            )
            
            for j in range(max(0, i - NUMBER_OF_REFERENCE_FRAMES), i + 1):
                img = io.imread(df.loc[j, "Image"])
                msk = io.imread(df.loc[j, "Mask"]).squeeze()
                msk[msk > 0] = 255
                msk = msk.astype(np.uint8)
            
                aug_img = tr(image=img, mask=msk)
            
                img_pth = os.path.join(deg_dir, f"IMG-{j:0>3}.png")
                msk_pth = os.path.join(deg_dir, f"MSK-{j:0>3}.png")
            
                io.imsave(img_pth, aug_img["image"].astype(np.uint8), check_contrast=False)
                io.imsave(msk_pth, aug_img["mask"].astype(np.uint8), check_contrast=False)
                
                new_df["VID"].append(f"{row['VID']}-{row['FID']}-{deg:0>6}")
                new_df["FID"].append(fid)
                new_df["Image"].append(img_pth)
                new_df["Mask"].append(msk_pth)
                new_df["Label"].append(int(i == j))

                fid += 1
            
    print(f"Image {row['VID']}-{row['FID']} --> Degree: {deg} saved.")
    print(f"-" * 50)

pd.DataFrame(
    new_df
).to_csv(
    "data/simulated/validation/rotated/Main-155-Frames-Validation-Rotated.csv", 
    index=False
)
    
