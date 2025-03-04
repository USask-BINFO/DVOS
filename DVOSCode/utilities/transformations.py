from typing import Union, Tuple, Callable

import albumentations as A
import numpy as np
import torch
import torchvision
import importlib

from albumentations.pytorch import ToTensorV2

from .utils import AttrDict

DataType = Union[np.ndarray, torch.Tensor, None]
TrfmsType = Union[A.core.serialization.SerializableMeta, torchvision.transforms.transforms.Compose, Callable, None]


class Transformations:
    """A flexible utility class for defining data transformations.
        This class is based on the Albumentations.
    Args:
        confs: a dictionary of configurations.
        transforms: a composition of transformations functions, could be applied to the input image or both image and mask.
            For example, in case of albumentations, the transformations would be aplpied to image and mask,
            while the torchvision transforms would be applied to image only. Therefore, there is a need for mask transformations.
        mask_transforms: a composition of transformations functions, could be applied to the input mask only.
    Returns:
         augmented image and mask.
    """
    def __init__(self,
                 transforms: TrfmsType = None,
                 mask_transforms: TrfmsType = None
    ) -> None:
        self.transforms = transforms
        self.mask_transforms = mask_transforms

    def __apply_albumentations(self,
                               image: DataType,
                               mask: DataType = None
    ) -> Tuple[DataType, DataType]:
        aug = self.transforms(image=image, mask=mask)
        image, mask = aug['image'], mask if not mask else aug['mask']
        return image, mask

    def __apply_torch_transforms(self,
                                image: DataType,
                                mask: DataType = None
    ) -> Tuple[DataType, DataType]:
        image = self.transforms(image)
        if mask is not None:
            mask = self.mask_transforms(mask)
        return image, mask

    def __call__(self,
                 image: DataType,
                 mask: DataType = None
    ) -> Tuple[DataType, DataType]:
        if self.transforms is None and self.mask_transforms is None:
            return image, mask
        elif isinstance(self.transforms, A.core.serialization.SerializableMeta):
            return self.__apply_albumentations(image, mask)
        elif (isinstance(self.transforms, (torchvision.transforms.transforms.Compose, Callable)) or
              isinstance(self.mask_transforms, (torchvision.transforms.transforms.Compose, Callable))
        ):
            return self.__apply_torch_transforms(image, mask)
        else:
            raise ValueError(f"Unsupported transformations type: {type(self.transforms)}")



def get_transformations(configs: AttrDict
) -> AttrDict['str', Transformations]:
    """A utility function for defining data transformations.
        The use should change the transformations based on the dataset.
    Args:
        configs: a dictionary of configurations.
            The configuration file contains a key that references the path to the augmentation module, as well as three
            additional keys that point to dictionaries of transformations for the training, validation, and testing sets.

    Returns: a dictionary of transformations.
    """
    return AttrDict(records={
        phase: Transformations(transforms=trfms['IMAGE'], mask_transforms=trfms['MASK'])
        for phase, trfms in importlib.import_module(configs.TRANSFORMATIONS_PATH).TRANSFORMS.items()
    })