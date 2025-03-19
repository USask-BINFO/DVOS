import argparse
import os 
import random
import yaml 
from box import Box
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Callable, TypeAlias

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from icecream import ic
from skimage import io
from torchvision.utils import draw_segmentation_masks

from geometry import Coordination, Orientation, Transformations

# Define the type aliases
IMAGE_TYPES: TypeAlias = Union[
    str, np.ndarray, torch.Tensor,
    List[str], List[np.ndarray], List[torch.Tensor],
    Tuple[str], Tuple[np.ndarray], Tuple[torch.Tensor]
]


class VideoSimulation:
    """Simulate the video frames and masks from the given image and mask.
    Args:

    """

    def __init__(self,
                 num_frames: int,
                 positioning: Coordination,
                 orientation: Orientation,
                 transformation: Transformations,
                 group_transformers: Optional[Callable] = None,
                 random_seed: int = 123
    ) -> None:
        self.num_frames = num_frames
        self.positioning = positioning
        self.orientation = orientation
        self.transformation = transformation
        self.group_transformers = group_transformers
        self.random_seed = random_seed

        # Set the random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def __call__(self,
                 image: IMAGE_TYPES,
                 mask: IMAGE_TYPES
    ) -> Tuple[List[Union[np.ndarray, torch.Tensor]],
               List[Union[np.ndarray, torch.Tensor]]]:
        # Convert the image and mask to numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
            if mask.ndim == 3:
                mask = mask.transpose(1, 2, 0)
        # Simulate the video frames and masks
        video_frames, video_masks = self.simulate(image, mask)

        # Apply group-wise transforms
        if self.group_transformers is not None:
            augmentation_data = {
                'image': video_frames[0],
                'mask': video_masks[0]
            }
            augmentation_data.update({
                f"image{i}": item for i, item in enumerate(video_frames)
            })
            augmentation_data.update({
                f"mask{i}": item for i, item in enumerate(video_masks)
            })
            augmented_data = self.group_transformers(**augmentation_data)
            # Recheck the original type of the image and mask

            video_frames = [
                (torch.from_numpy(frame).permute(2, 0, 1)
                 if isinstance(image, torch.Tensor) else frame
                )
                for key, frame in augmented_data.items() if "image" in key
            ]
            video_masks = [
                (torch.from_numpy(mask).permute(2, 0, 1)
                 if isinstance(mask, torch.Tensor) else mask
                )
                for key, mask in augmented_data.items() if "mask" in key
            ]

        # Return the video frames and masks
        return video_frames, video_masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(image_type, mask_type)"

    def simulate(self,
                 image: IMAGE_TYPES,
                 mask: IMAGE_TYPES
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        height, width, _ = image.shape

        video_frames = []
        video_masks = []
        for i in range(self.num_frames):
            self.transformation.update(self.orientation.angle, height, width)
            aug = self.transformation.transformers(image=image, mask=mask)
            img, msk = aug['image'], aug['mask']

            img = img[
                  self.positioning.y - self.positioning.window:
                  self.positioning.y + self.positioning.window,
                  self.positioning.x - self.positioning.window:
                  self.positioning.x + self.positioning.window,
            ]
            msk = msk[
                  self.positioning.y - self.positioning.window:
                  self.positioning.y + self.positioning.window,
                  self.positioning.x - self.positioning.window:
                  self.positioning.x + self.positioning.window
            ]
            # ic(self.positioning.x, self.positioning.y)

            video_frames.append(img)
            video_masks.append(msk)

            self.positioning.update()
            self.positioning.update_step()
            self.orientation.update()

        return video_frames, video_masks


class Simulation:
    def __init__(self,
                 images: IMAGE_TYPES,
                 masks: IMAGE_TYPES,
                 simulator: Callable,
                 image_loader: Callable,
                 mask_loader: Callable,
                 randomize: bool = False,
                 random_seed: int = 123
    ) -> None:
        self.images = images
        self.masks = masks
        self.simulator = simulator
        self.image_loader = image_loader
        self.mask_loader = mask_loader
        self.randomize = randomize
        self.random_seed = random_seed

        if isinstance(images, str):
            self.images = [images]
        if isinstance(masks, str):
            self.masks = [masks]

        # Check the lengths
        if not self.__check_lengths():
            raise ValueError(
                "The number of images and masks must be the same.")

        # Set the random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        # Initialize the indices
        self.__next_index = -1
        self.__rand_index = -1

    def __check_lengths(self) -> bool:
        return len(self.images) == len(self.masks)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self,
                    idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        image, mask = None, None
        if isinstance(self.images[idx], str):
            image = self.image_loader(self.images[idx])
        if isinstance(self.masks[idx], str):
            mask = self.mask_loader(self.masks[idx]).squeeze()
        return image, mask

    def __call__(self,
    ) -> Tuple[Union[None, str],
               List[Union[np.ndarray, torch.Tensor]],
               List[Union[np.ndarray, torch.Tensor]]
    ]:
        if self.randomize is True:
            self.set_random_index()
            index = self.__rand_index
        else:
            self.set_next_index()
            index = self.__next_index
        # Set the sample name or path. If, the images are not strings, then None
        name = None
        if isinstance(self.images[index], str):
            name = self.images[index]
        # Get the image and mask
        image, mask = self.__getitem__(index)
        # Simulate the video frames and masks
        video_frames, video_maks = self.simulator(image, mask)
        # Return the name and the video frames and masks
        return name, video_frames, video_maks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(images, masks, image_loader, mask_loader)"

    def __str__(self) -> str:
        return f"Simulation with {self.__len__()} images"

    def set_next_index(self) -> None:
        self.__next_index += 1

    def reset_index(self) -> None:
        self.__next_index = -1

    def set_random_index(self) -> int:
        self.__rand_index = int(
            random.uniform(
                0,
                self.__len__()
            )
        )

    def reset_random_index(self) -> None:
        self.__rand_index = -1


def preview(images: List[Union[np.ndarray, torch.Tensor]],
            masks: List[Union[np.ndarray, torch.Tensor]],
            frames_per_second: int = 60, 
            out_video_path="simulated_videos/sample.mp4"
) -> None:
    output_video = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        frames_per_second,
        (images[0].shape[1], images[0].shape[0])
    )
    for i, (image, mask) in enumerate(zip(images, masks)):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).squeeze()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).squeeze()

        overlayed_image = draw_segmentation_masks(
            image, mask.type(torch.bool), colors=(204, 68, 150), alpha=0.4
        ).numpy().transpose(1, 2, 0).astype(np.uint8)
        output_video.write(
            cv2.cvtColor(
                overlayed_image,
                cv2.COLOR_RGB2BGR
            )
        )
    output_video.release()


def get_group_transformers(frames: int,
                           height: int = 512,
                           width: int = 512
) -> A.Compose:
    add_targets = {
        f"image{i}": "image"
        for i in range(1, frames)
    }
    add_targets.update({
        f"mask{i}": "mask"
        for i in range(1, frames)
    })
    transoforms = A.Compose(
        transforms=[
            A.Resize(height=height, width=width, p=1.0)
        ],
        additional_targets=add_targets
    )
    return transoforms


if __name__ == "__main__":
    # Input arguments.
    parser = argparse.ArgumentParser(
        description='Initializers parameters for running the experiments.'
    )
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        default='configs/strong.yaml',
                        help='The string path of the config file.'
    )
    args = parser.parse_args()

    configs = Box.from_yaml(filename=args.config_path, Loader=yaml.FullLoader)

    # Set seeds
    random.seed(configs.Basics.SEED)
    np.random.seed(configs.Basics.SEED)
    torch.manual_seed(configs.Basics.SEED)

    group_transformers = get_group_transformers(
        frames=configs.Video.NUM_REQUIRED_FRAMES,
        height=configs.Video.VIDEO_HEIGHT,
        width=configs.Video.VIDEO_WIDTH
    )
    
    positioning = Coordination(
        height=configs.Frame.ORIGINAL_IMAGE_HEIGHT,
        width=configs.Frame.ORIGINAL_IMAGE_WIDTH,
        window_init_range=configs.Frame.WINDOW_INIT_SIZE,
        max_step_size=configs.Frame.MAX_STEP_SIZE,
        allow_updating_coordinates=configs.Frame.ALLOW_UPDATING_COORDINATES,
        allow_updating_step=configs.Frame.ALLOW_UPDATING_STEP,
        random_seed=configs.Basics.SEED
    )

    orientation = Orientation(
        init_range=configs.Frame.ORIENTATION_INIT_RANGE,
        max_adjustment=configs.Frame.ORIENTATION_MAX_ADJUSTMENT,
        allow_updating_angle=configs.Frame.ORIENTATION_ALLOW_UPDATING_ANGLE,
        random_seed=configs.Basics.SEED
    )

    transformation = Transformations(
        use_rotation=configs.Frame.TRANSFORMATION_USE_ROTATION,
        use_elastic=configs.Frame.TRANSFORMATION_USE_ELASTIC,
        use_center_crop=configs.Frame.TRANSFORMATION_USE_CENTER_CROP,
        cc_refinement_upper_bound=configs.Frame.TRANSFORMATION_CC_REFINEMENT_UPPER_BOUND,
        use_resize=configs.Frame.TRANSFORMATION_USE_RESIZE,
        random_seed=configs.Basics.SEED
    )

    simulator = VideoSimulation(
        num_frames=configs.Video.NUM_REQUIRED_FRAMES,
        positioning=positioning,
        orientation=orientation,
        transformation=transformation,
        group_transformers=group_transformers,
        random_seed=configs.Basics.SEED
    )

    image_loader = lambda x: io.imread(x)
    mask_loader = lambda x: io.imread(x).squeeze()
    
    metadata = pd.read_csv(configs.Metadata)

    sim = Simulation(
        metadata.Image, 
        metadata.Mask,
        simulator,
        image_loader,
        mask_loader,
        randomize=False,
        random_seed=configs.Basics.SEED
    )
    
    # Create out video directory. 
    if configs.Basics.CREATE_OVERLAID_VIDEO:
        os.makedirs(configs.Basics.OUT_VIDEO_SIMULATION_DIR, exist_ok=True)
    
    video_df = pd.DataFrame()
    vid = 0
    with tqdm(total=len(sim)) as pbar: 
        for i in range(0, len(sim), configs.Basics.SIMULATE_FROM_EVERY_N_INPUT_PAIRS):
            pbar.update(configs.Basics.SIMULATE_FROM_EVERY_N_INPUT_PAIRS)
            out_frame_dir = os.path.join(
                configs.Basics.OUT_SIMULATION_DIR,
                f"VID_{vid:0>5}",
                "Frames"
            )
            out_mask_dir = os.path.join(
                configs.Basics.OUT_SIMULATION_DIR,
                f"VID_{vid:0>5}",
                "Masks"
            )
            os.makedirs(out_frame_dir, exist_ok=True)
            os.makedirs(out_mask_dir, exist_ok=True)

            name, video_frames, video_masks = sim()
            # Save the video frames and masks.
            frame_paths, mask_paths = [], []
            for j, (frame, mask) in enumerate(zip(video_frames, video_masks)):
                frame_paths.append(
                    os.path.join(
                        out_frame_dir, f"vid-{vid:0>5}_frame-{j:0>5}.png")
                )
                mask_paths.append(
                    os.path.join(out_mask_dir, f"vid-{vid:0>5}_mask-{j:0>5}.png")
                )
                io.imsave(
                    os.path.join(out_frame_dir, f"vid-{vid:0>5}_frame-{j:0>5}.png"),
                    frame
                )
                io.imsave(
                    os.path.join(out_mask_dir, f"vid-{vid:0>5}_mask-{j:0>5}.png"),
                    mask
                )
            # Save the metadata.
            frames_length = len(frame_paths)
            video_df = pd.concat([
                video_df,
                pd.DataFrame({
                    "Name": [name] * frames_length,
                    "GID": [f"VID_{vid:0>5}"] * frames_length,
                    "IID": np.arange(frames_length).astype(np.int32),
                    "Image": frame_paths,
                    "Mask": mask_paths,
                    "Label": [1] * frames_length
                })
            ], ignore_index=True, axis=0)
            
            if configs.Basics.CREATE_OVERLAID_VIDEO is True: 
                preview(video_frames, video_masks, 
                        frames_per_second=configs.Basics.FRAMES_PER_SECOND, 
                        out_video_path=os.path.join(
                            configs.Basics.OUT_VIDEO_SIMULATION_DIR, 
                            f"VID_{vid:0>5}.mp4"
                        )
                )
            vid += 1
            
            if i >= 2: 
                break

    video_df.to_csv(configs.Basics.OUT_METADATA_PATH, index=False)
