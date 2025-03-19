"""Wheat Head Segmentation dataset loader.
It uses generator to create and return a new simulated image and its contour.
"""
import random
from typing import Callable, Union, Tuple, List

import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd
from skimage import io

# Define Constants.
MASKMINPIXELS = 400


def image_loader(path: str,
                 reader: Union[Callable, None] = None,
                 reader_params: Tuple = ()
) -> np.ndarray:
    if reader is None:
        if path.endswith('.nrrd'):
            image = sitk.GetArrayFromImage(sitk.ReadImage(path))
            image = np.squeeze(image)
        else:
            image = io.imread(path)
    else:
        image = reader(path, *reader_params)
    return image

def crop(image: np.ndarray,
         mask: np.ndarray,
         tol: int=0
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Crop the regions that contain no pixes of a object."""
    new_mask = image > tol
    # Coordinates of non-black pixels.
    coords = np.argwhere(new_mask[:, :, 0])
    if 0 in coords.shape:  # If there is no object.
        return image, mask, False
    # Bounding box of non-black pixels.
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    # Get the contents of the bounding box.
    image = image[r0:r1, c0:c1, :]
    mask = mask[r0:r1, c0:c1]
    return image, mask, True


class BackgroundDataset:
    """Load extracted background paths, load, augment, and return images.
    Args:
        metadata_paths (str): The `.csv` path of the metadata for loading
            background images.
        transformer (Callable): A list of transformations to be applied to
        the background images. Default is `None`.
    """
    def __init__(self,
                 metadata_paths: List[str],
                 transformer: Union[Callable, None] = None
    ) -> None:
        self.metadata_paths = metadata_paths

        self.background_dataframe = pd.DataFrame()
        for path in self.metadata_paths:
            self.background_dataframe = pd.concat([
                self.background_dataframe, 
                pd.read_csv(path).sort_values(by="Image").reset_index(drop=True)
            ])
        self.background_dataframe.reset_index(drop=True, inplace=True)
        self.transformer = transformer

    def __len__(self):
        return self.background_dataframe.shape[0]

    def __getitem__(self, item: int
    ) -> np.ndarray:
        path = self.background_dataframe.loc[item, 'Image']
        image = image_loader(path)
        if self.transformer is not None:
            image = self.transformer(image=image)['image']
        return image

    def get_batch(self, batch_size: int) -> List[np.ndarray]:
        random_indices = np.random.randint(0, self.__len__(), batch_size)
        batch = []
        for item in random_indices:
            batch.append(self.__getitem__(item))
        return batch


class RealObjectDataset:
    """Load extracted foreground objects pathes, and return the transformed
        objects.
    Args:
        metadata_paths (str): The `.csv` path of the metadata for loading
            wheat head objects.
        scale_trfms (Callable): The image scaler transformer. Default is `None`.
        transformer (Callable): A list of transformations to be applied to
            the real wheat head objects. Default is `None`.
    """
    def __init__(self,
                 metadata_paths: List[str],
                 transformer: Union[Callable, None] = None
    ) -> None:
        self.metadata_paths = metadata_paths
        self.real_dataframe = pd.DataFrame()
        for path in self.metadata_paths:
            self.real_dataframe = pd.concat([
                self.real_dataframe, pd.read_csv(path)
            ])
        self.real_dataframe.reset_index(drop=True, inplace=True)
        self.transformer = transformer

    def __len__(self):
        return self.real_dataframe.shape[0]

    def __getitem__(self, item: int):
        path = self.real_dataframe.loc[item, 'Image']
        transparent_object = image_loader(path, reader=cv2.imread,
                                          reader_params=(cv2.IMREAD_UNCHANGED,))
        image = transparent_object[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = transparent_object[:, :, 3] // 255
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)
        if self.transformer is not None:
            augmented = self.transformer(image=image, mask=mask)
            aug_image, aug_mask, flag = crop(
                image=augmented['image'],
                mask=augmented['mask'],
                tol=0
            )
            if flag is True and np.sum(aug_mask) > MASKMINPIXELS:
                image, mask = aug_image, aug_mask
        return image, mask

    def get_batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        random_indices = np.random.randint(0, self.__len__(), batch_size)
        batch = []
        for item in random_indices:
            batch.append(
                self.__getitem__(item)
            )
        return batch


class FakePatchDataset:
    """Load extracted fake objects, and return the transformed
        objects.
    Args:
        metadata_paths (str): The `.csv` path of the metadata for loading
            fake head objects.
        transformer (Callable): A list of transformations to be applied to
            the real wheat head objects. Default is `None`.
    """
    def __init__(self,
                 metadata_paths: str,
                 transformer: Callable = None
    ) -> None:
        self.metadata_paths = metadata_paths
        self.fake_dataframe = pd.DataFrame()
        for path in self.metadata_paths:
            self.fake_dataframe = pd.concat([
                self.fake_dataframe, pd.read_csv(path)
            ])
        self.fake_dataframe.reset_index(drop=True, inplace=True)
        self.transformer = transformer

    def __len__(self):
        return self.fake_dataframe.shape[0]

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self.fake_dataframe.loc[item, 'Image']
        transparent_patch = image_loader(path, reader=cv2.imread,
                                         reader_params=(cv2.IMREAD_UNCHANGED,))
        image = transparent_patch[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = transparent_patch[:, :, 3] // 255
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)
        if self.transformer is not None:
            augmented = self.transformer(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            if np.sum(aug_mask) > MASKMINPIXELS:
                image, mask = aug_image, aug_mask
        return image, mask

    def get_batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        random_indices = np.random.randint(0, self.__len__(), batch_size)
        batch = []
        for item in random_indices:
            batch.append(
                self.__getitem__(item)
            )
        return batch
