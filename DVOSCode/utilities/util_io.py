import copy
from typing import Dict, Optional, Callable, List, Union, Any

import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from box import Box
from omegaconf import OmegaConf
from skimage import io
from torch.utils.data import Sampler
from tqdm import tqdm

from .converters import ChannelBroadcasting, DtypeCoversion
from .converters import IntensityScaler
from .utils import AttrDict


# Global colors
COLOR = "\033[38;2;199;90;147m"
PROGRESS_BAR_COLOR = "\033[38;2;91;169;102m"
RESET = "\033[38;2;0;0;0m"

# Global configs. 
plt.rcParams["savefig.bbox"] = "tight"


class ImageLoader:
    """Loads image from file
    Args:
        image_path: path to image file
        clip: whether to clip the intensity of the image
        clip_min: minimum value of the intensity
        clip_max: maximum value of the intensity
        scaler: whether to scale the intensity of the image
        scaler_func: function to scale the intensity of the image. Options: `MinMaxScaler`, `StandardScaler`, `Scale`.
            Default is 'min_max_scaler'.
        scaler_args: arguments for the scaler function, only for scale function. It includes `min_intensity` and `max_intensity`.
        broadcast_channels_num: number of channels to broadcast the image to. Default is None. If None, no broadcasting is performed.
        dtype: data type of the image. Default is None. If None, no conversion is performed.
    """
    def __init__(self,
                 clip: bool = False,
                 clip_min: Optional[float] = None,
                 clip_max: Optional[float] = None,
                 scaler: bool = False,
                 scaler_func: Optional[Callable] = 'MinMaxScaler',
                 scaler_args: Optional[Dict] = None,
                 broadcast_channels_num: Optional[int] = None,
                 dtype: Optional[Union[str, type, torch.dtype]] = None
    ) -> None:
        self.clip = clip
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.scaler = scaler
        self.scaler_func = scaler_func
        self.scaler_args = scaler_args
        self.broadcast_channels_num = broadcast_channels_num
        self.dtype = dtype

        if self.clip is True:
            assert self.clip_min is not None, 'clip_min must be specified!'
            assert self.clip_max is not None, 'clip_max must be specified!'
            assert self.clip_min < self.clip_max, 'clip_min must be less than clip_max!'
        if self.scaler is True and self.scaler_func == 'Scale':
            assert self.scaler_args is not None, 'scaler_args must be specified!'
            assert 'min_intensity' in self.scaler_args, 'min_intensity must be specified!'
            assert 'max_intensity' in self.scaler_args, 'max_intensity must be specified!'
            assert self.scaler_args['min_intensity'] < self.scaler_args['max_intensity'], 'min_intensity must be less than max_intensity!'
        self.intensity_scaler = IntensityScaler()
        self.channel_broadcaster = None
        if self.broadcast_channels_num is not None: 
            self.channel_broadcaster = ChannelBroadcasting(num_channels=self.broadcast_channels_num)
        self.dtype_converter = None if self.dtype is None else DtypeCoversion(self.dtype) 

    @staticmethod
    def load_npy(path: str
    ) -> np.ndarray:
        """Load numpy files. The path must end in `.npy`.
        Args:
            path (str): Path to the numpy file.
        Returns:
            np.ndarray: The loaded numpy array.
        """
        assert path.endswith('.npy'), 'Only `npy` files are allowed!'
        image = np.load(path)
        return image

    @staticmethod
    def load_png(path: str
    ) -> np.ndarray:
        """Load an image file in png format. The path must end in `.png`.
        Args:
            path (str): Path to the numpy file.
        Returns:
            np.ndarray: The loaded numpy array.
        """
        assert path.endswith('.png'), 'Only `npy` files are allowed!'
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    @staticmethod
    def load_nrrd(path: str
    ) -> np.ndarray:
        """Load an image file in nrrd format. The path must end in `.nrrd`.
        Args:
            path (str): Path to the numpy file.
        Returns:
            np.ndarray: The loaded numpy array.
        """
        assert path.endswith('.nrrd'), 'Only `nrrd` files are allowed!'
        image = sitk.GetArrayFromImage(sitk.ReadImage(path)).squeeze()
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        return image
    
    @staticmethod
    def load_dicom(path: Union[str, List[str]]
    ) -> np.ndarray:
        """Load an image or a list of image files in dicom format. 
        The pathes must end in `.dcm`.
        Args: 
            path (str or list): Path to the dicom file/files.
        Returns:
            np.ndarray: The loaded numpy array.
        """
        if isinstance(path, str):
            image = sitk.GetArrayFromImage(
                sitk.ReadImage(path)
            ).squeeze()
        if isinstance(path, List):
            image = [
                sitk.GetArrayFromImage(
                    sitk.ReadImage(mask)
                ).squeeze()
                for pth in path
            ]
            image = np.stack(image, axis=0)
        return image

    def __call__(self, 
                 image_path: Union[str, List[str]]
    ) -> None:
        image = None
        if image_path.endswith('.npy'):
            image = ImageLoader.load_npy(image_path)
        elif image_path.endswith('.png'):
            image = ImageLoader.load_png(image_path)
        elif image_path.endswith('.nrrd'):
            image = ImageLoader.load_nrrd(image_path)
        elif ((isinstance(image_path, str) and image_path.endswith('.dcm')) or
              (isinstance(image_path, List) and image_path[0].endswith('.dcm'))
        ):
            image = ImageLoader.load_dicom(image_path)
        else:
            image = io.imread(image_path)
        # Remove Dimensions of size 1.
        if self.clip is True:
            image = self.intensity_scaler.clip_intensity(image, self.clip_min, self.clip_max)
        if self.scaler is True:
            if self.scaler_func == 'MinMaxScaler':
                image = self.intensity_scaler.min_max_scaler(image)
            elif self.scaler_func == 'StandardScaler':
                image = self.intensity_scaler.standard_scaler(image)
            elif self.scaler_func == 'Scale':
                image = self.intensity_scaler.scale(image, **self.scaler_args)
            else:
                raise ValueError('Invalid scaler function!')
        # Broadcast the image to 3 channels if required, and change the output dtype as well.
        if self.channel_broadcaster is not None:
            image = self.channel_broadcaster(image=image)
        if self.dtype_converter is not None:
            image = self.dtype_converter(data=image)
        return image


class MaskLoader:
    def __init__(self,
                 dtype: Optional[str] = None,
                 binary: bool = False
    ) -> None:
        self.binary = binary
        self.dtype = dtype
        self.dtype_converter = None if self.dtype is None else DtypeCoversion(self.dtype, is_mask=True)

    def __call__(self,
                 mask_path: Union[str, List[str]]
    ) -> np.ndarray:
        if mask_path.endswith('.npy'):
            mask = ImageLoader.load_npy(mask_path)
        elif mask_path.endswith('.nrrd'):
            mask = ImageLoader.load_nrrd(mask_path)
        elif ((isinstance(mask_path, str) and mask_path.endswith('.dcm')) or
              (isinstance(mask_path, List) and mask_path[0].endswith('.dcm'))
        ):
            mask = ImageLoader.load_dicom(mask_path)
        else:
            mask = io.imread(mask_path)
        if self.binary is True:
            mask[mask > 0] = 1            
        mask = mask.squeeze()
        if mask.ndim == 3: 
            mask = np.sum(mask, axis=2)
            mask[mask > 0] = 1
        if self.dtype_converter is not None:
            mask = self.dtype_converter(data=mask)
        return mask


class VideoLoader:
    pass


class Metadata:
    """Create a customized Pandas format dataframe to manage metadata.
    Args: 
        meta: either a panda dataframe or a List of paths to csv files. 
        num_clusters: number of cluster, when it is needed to split the large-scale metadata 
            into partitions. Default to 1. 
    """
    def __init__(self, 
                 meta: Union[List[str], pd.DataFrame], 
                 group_column_name: str, 
                 num_clusters: Optional[int] = 1 
    ) -> None:
        self.group_column_name = group_column_name
        self.num_clusters = num_clusters
        self.current_cluster = 0
        
        df = meta
        if isinstance(meta, List):  
            images = []
            for pth in meta:
                image = pd.read_csv(pth)
                images.append(image)
            df = pd.concat(images)
            df.reset_index(drop=True, inplace=True)
        self.columns = df.columns
        self.df = df
        self.df_dict = df.to_dict(orient='index')       

    def __call__(self):
        return self.df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.df_dict[item]
    
    def get_indices(self, 
                    column_name: str = 'Label', 
                    unique_id: Any = 1
    ) -> pd.Index: 
        return self.df[
            self.df[column_name] == unique_id
        ].index

    def get_group(self,
                  group_id: Union[int, str]
    ) -> pd.DataFrame:
        group = self.df[self.df[self.group_column_name] == group_id]
        group.reset_index(drop=True, inplace=True)
        return group
    
    def get_cluster(self, 
                    group: pd.DataFrame
    ) -> pd.DataFrame: 
        group = group.reset_index(drop=True, inplace=False)
        cluster_indices = np.linspace(
            0, group.shape[0], self.num_clusters + 1
        ).astype(np.int64)
        return group.loc[
            cluster_indices[
                self.current_cluster
            ]: cluster_indices[
                self.current_cluster + 1
            ]
        ]
    
    def get_next_clusters(self):
        clusters = self.df.groupby(
            self.group_column_name,
            group_keys=False
        ).apply(self.get_cluster)
        clusters.reset_index(drop=True, inplace=True)
        self.current_cluster = (self.current_cluster + 1) % self.num_clusters
        return Metadata(
            meta=clusters, 
            group_column_name=self.group_column_name, 
            num_clusters=1
        )


class DynamicRandomSampler(Sampler):
    """

    Args:
        metadata: A metadata object containing the metadata for a dataset
        categories: Array-like of categories for each sample

    """
    def __init__(self, metadata: Metadata, category: str):
        self.metadata = metadata
        self.categories = np.array(copy.deepcopy(metadata.df[category]))
        unique_categories = np.unique(metadata.df.loc[:, category])
        self.category_probability = {c: 1 for c in unique_categories}
        self.sampels = np.random.permutation(len(metadata))

    def __iter__(self):
        return iter(copy.deepcopy(self.sampels))

    def set_probability(self, category_probability:dict):
        prob = np.random.uniform(low=0, high=1, size=len(self.categories))
        selected = [True if p <= category_probability[c] else False
                    for c, p in zip(self.categories, prob)]
        sampels = np.arange(len(self.metadata))
        self.sampels = sampels[selected]
        np.random.shuffle(self.sampels)

    def __len__(self) -> int:
        return len(self.sampels)


class ConfigLoader:
    """Load configuration either from a `YAML` or `JSON` file.
        Args:
            config_path (str): The path to the config file.
        Returns:
            configs (Dict): a dictionary contains all the configuration parameters.
    """
    def load_yaml(self,
                  config_path: str
    ) -> Union[Dict, AttrDict, Box, OmegaConf]:
        configs = OmegaConf.load(config_path)
        return configs

    def load_json(self,
                  config_path: str
    ) -> Union[Dict, AttrDict, Box, OmegaConf]:
        configs = OmegaConf.load(config_path)
        return configs

    def __call__(self,
                 config_path: str
    ) -> Union[Dict, AttrDict, Box]:
        if config_path.endswith('.json'):
            configs = self.load_json(config_path)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            configs = self.load_yaml(config_path)
        else:
            raise ValueError('Only `Json` or `Yaml` configs are acceptable.')
        
        return configs


def get_progress_bar(desc: str, 
                     total: int
) -> tqdm:
    """get progress bar"""
    return tqdm(desc=COLOR + desc + RESET, leave=True, total=total, ascii=True, 
                bar_format="{l_bar}%s{bar}%s | {n_fmt}/{total_fmt} - [{elapsed}, {rate_fmt}] - [{postfix}]" % (PROGRESS_BAR_COLOR, RESET)
    )
