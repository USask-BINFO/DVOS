

import glob 
import os 
import shutil


video_id = 0
frame_id = 0
for batch in ["Batch1/", "Batch2/"]:
    for patch in sorted(os.listdir(batch)): 
        for sub in sorted(os.listdir(os.path.join(batch, patch))): 
            os.makedirs(os.path.join("frames", f"VId-{video_id:0>5}"), exist_ok=True)
            os.makedirs(os.path.join("masks",  f"VId-{video_id:0>5}"), exist_ok=True)
            
            images = sorted(glob.glob(os.path.join(batch, patch, sub, "img-*.png")))
            masks = sorted(glob.glob(os.path.join(batch, patch, sub, "msk-*.png")))
            for img_file, msk_file in zip(images, masks): 
                frame_name = f"drone_1024patch_{frame_id:0>5}.png"
                shutil.copy(img_file, os.path.join("frames", f"VId-{video_id:0>5}", frame_name))
                shutil.copy(msk_file, os.path.join("masks", f"VId-{video_id:0>5}", frame_name))
                frame_id += 1
                
            video_id += 1
            
