from typing import Tuple, List, Optional, Union, Dict, Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform


class MaskSafeRandomCrop(DualTransform):
    def __init__(self, 
                 height: int, 
                 width: int, 
                 p: float = 1.0
    ) -> None:
        super(MaskSafeRandomCrop, self).__init__(p)
        self.height = height
        self.width = width

    def apply(self, 
              img: np.ndarray, 
              x_min: int=0, 
              y_min: int=0, 
              **params
    ) -> np.ndarray:
        return img[
            y_min: y_min + self.height, 
            x_min: x_min + self.width
        ]

    def apply_to_mask(self, 
                      msk: np.ndarray, 
                      x_min: int=0, 
                      y_min: int=0, 
                      **params
    ) -> np.ndarray:
        return msk[
            y_min: y_min + self.height, 
            x_min: x_min + self.width
        ]

    def get_params_dependent_on_targets(self, 
                                        params: Dict[str, Any]
    ) -> Dict[str, int]:
        mask = params["mask"]
        h, w = mask.shape[:2]

        assert self.height <= h and self.width <= w, "The crop size is greater than the image size"

        # Compute the bounding box for all objects in the mask
        non_zero_indices = np.argwhere(mask > 0)
        if non_zero_indices.size == 0: 
            y_min = np.random.randint(0, max(1, h - self.height))
            x_min = np.random.randint(0, max(1, w - self.width))
            return {"x_min": x_min, "y_min": y_min}
        
        y_min, x_min = non_zero_indices.min(axis=0)
        y_max, x_max = non_zero_indices.max(axis=0)
        
        crop_height = max(self.height, y_max - y_min)
        crop_width  = max(self.width,  x_max - x_min)

        # Expand bounding box to fit the crop size while ensuring it stays within the image
        y_extension = max(0, crop_height - (y_max - y_min))
        x_extension = max(0, crop_width  - (x_max - x_min))

        # Randomly shift the crop within allowable limits
        y_criteria = y_min > 0 and y_extension > 0
        x_criteria = x_min > 0 and x_extension > 0
        crop_y_min = np.random.randint(max(0, y_min - y_extension), y_min) if y_criteria else y_min
        crop_x_min = np.random.randint(max(0, x_min - x_extension), x_min) if x_criteria else x_min

        return {"x_min": crop_x_min, "y_min": crop_y_min}

    @property
    def targets_as_params(self):
        return ["image", "mask"]
    

def get_pixel_level_transformations() -> Dict: 
    return {
        'TRAIN': [
            A.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5, hue=0.2, p=0.3),
            A.OneOf([
                A.ToGray(p=1.0),
                A.ToSepia(p=1.0),
                A.FancyPCA(alpha=0.2, p=1.0),
                A.Posterize(num_bits=4, p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)
            ], p=0.5),
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1,
                                           brightness_by_max=True, p=1.0),
                A.CLAHE(clip_limit=(1.0, 5.0), tile_grid_size=(8, 8), p=1.0)
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Blur(blur_limit=(3, 7), p=1.0),
                A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode='fast',
                            p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=(3, 7), p=1.0)
            ], p=0.3)
        ]
    }

def get_main_spatial_transformations(height, width, mean, std) -> Dict: 
    HEIGHT, WIDTH, MEAN, STD = height, width, mean, std
    return {
        'TRAIN': [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=20, 
                                    interpolation=1, border_mode=0, value=0,
                                    mask_value=None, approximate=False, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1,
                                border_mode=0, value=0, mask_value=None, p=0.5),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0)
            ], p=0.3),
            A.Rotate(limit=(-30, 30), interpolation=1, border_mode=0,
                    value=0, mask_value=0, p=0.5),
            A.OneOf([
                MaskSafeRandomCrop(256, 256, p=1.0),
                MaskSafeRandomCrop(256, 384, p=1.0),
                MaskSafeRandomCrop(384, 384, p=1.0),
                MaskSafeRandomCrop(384, 512, p=1.0),
                MaskSafeRandomCrop(512, 384, p=1.0),
                MaskSafeRandomCrop(512, 512, p=1.0),
                MaskSafeRandomCrop(640, 640, p=1.0),
                MaskSafeRandomCrop(640, 768, p=1.0),
                MaskSafeRandomCrop(768, 640, p=1.0),
                MaskSafeRandomCrop(768, 768, p=1.0),
            ], p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ],
        'VALID': [
            A.CenterCrop(height=512, width=512, p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ],
        'TEST': [
            A.CenterCrop(height=512, width=512, p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ],
        'PREDICT': [
            A.CenterCrop(height=512, width=512, p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ]
    }

def get_smalldata_spatial_transformations(height, width, mean, std) -> Dict: 
    HEIGHT, WIDTH, MEAN, STD = height, width, mean, std
    return {
        'TRAIN': [
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=20, 
                                    interpolation=1, border_mode=0, value=0,
                                    mask_value=None, approximate=False, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1,
                                 border_mode=0, value=0, mask_value=None, p=0.5),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0)
            ], p=0.5),
            A.Rotate(limit=(-30, 30), interpolation=1, border_mode=0,
                    value=0, mask_value=0, p=0.5),
            A.OneOf([
                A.RandomCrop(384, 384, p=1.0),
                A.RandomCrop(400, 400, p=1.0),
                A.RandomCrop(450, 450, p=1.0),
                A.RandomCrop(500, 500, p=1.0),
                A.RandomCrop(512, 512, p=1.0),
                A.RandomCrop(600, 600, p=1.0),
                A.RandomCrop(700, 700, p=1.0),
                A.RandomCrop(750, 750, p=1.0)
            ], p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ],
        'VALID': [
            A.RandomCrop(512, 512, p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ],
        'TEST': [
            A.RandomCrop(512, 512, p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ],
        'PREDICT': [
            A.RandomCrop(height=512, width=512, p=1.0),
            A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
            A.Normalize(mean=MEAN, std=STD, p=1.0),
            ToTensorV2(p=1.0)
        ]
    }


class ColorTransformations:
    def __init__(self, 
                 transforms: List
    ) -> None:
        self.transforms = A.Compose(
            transforms=transforms,
            p=1.0
        )
        

    def __call__(self,
                 images: List[np.ndarray]
    ) -> Tuple[List, Optional[List]]:    
        return [
            self.transforms(image=img)['image']
            for img in images[:-1]
        ] + images[-1:]


class SpatialTransformations:
    def __init__(self,
                 num_extra_images: int = 0,
                 include_mask: bool = True,
                 transforms: Optional[List] = None
    ) -> None:
        self.num_extra_images = num_extra_images
        self.include_mask = include_mask
         
        self.additional_targets = {
            f'image{i}': 'image'
            for i in range(self.num_extra_images)
        }
        if self.include_mask:
            for i in range(self.num_extra_images):
                self.additional_targets[f'mask{i}'] = 'mask'

        self.transforms = A.Compose(
            transforms=transforms,
            additional_targets=self.additional_targets,
            is_check_shapes=False,
            p=1.0
        )        

    def __call__(self,
                 images: List[np.ndarray],
                 masks: Union[List, None] = None,
    ) -> Tuple[List, Optional[List]]:        
        tr_images = {'image': images[0]}
        for i, img in enumerate(images[1:]):
            tr_images[f'image{i}'] = img
        tr_masks = None
        if self.include_mask:
            tr_masks = {'mask': masks[0]}
            for i, mask in enumerate(masks[1:]):
                tr_masks[f'mask{i}'] = mask
        if tr_masks is not None:
            aug = self.transforms(**tr_images, **tr_masks)
        else: 
            aug = self.transforms(**tr_images)
            
        tr_images = []
        tr_masks = []
        for item in aug.keys():
            if item.startswith('image'):
                tr_images.append(aug[item])
            if item.startswith('mask'):
                tr_masks.append(aug[item])
        if not self.include_mask or len(tr_masks) == 0:
            tr_masks = None
        return tr_images, tr_masks
