import random
from box import Box 
from icecream import ic 
from skimage import io
from typing import Callable, List, Dict, Union, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import utilities as utils
from transformations import ColorTransformations, SpatialTransformations
from transformations import get_pixel_level_transformations
from transformations import get_main_spatial_transformations
from transformations import get_smalldata_spatial_transformations
from utilities import AttrDict


class VideoMetadata: 
    """Create a metadata of CSV files, with the video and frame ids, 
        and constant Image, Mask, and Label columns. 
    """
    def __init__(self,
                 file_paths: Union[List[str], Tuple[str]],
                 VID: str, 
                 FID: str, 
                 query_identifier: int = 1, 
                 dataset_size: int = 0,
                 preload: bool = False, 
                 image_loader: Callable = lambda item: io.imread(item).astype(np.uint8),
                 mask_loader: Callable = lambda item: io.imread(item).squeeze().astype(np.uint8)
    ) -> None:
        self.file_paths = file_paths
        self.VID = VID
        self.FID = FID
        self.query_identifier = query_identifier
        self.dataset_size = dataset_size
        self.preload = preload
        self.image_loader = image_loader
        self.mask_loader = mask_loader
        
        self.metadatas = pd.DataFrame()
        for pth in self.file_paths:
            self.metadatas = pd.concat([
                self.metadatas, 
                pd.read_csv(pth)
            ], axis=0)
        self.metadatas.reset_index(drop=True, inplace=True)
        
        self.query_indices = self.metadatas[
            self.metadatas.Label == query_identifier
        ].index.tolist()
        
        if self.preload is True:
            self.load_data()
        
    def __len__(self): 
        return max(len(self.query_indices), self.dataset_size)
        
    def __getitem__(self, 
                    item: int
    ) -> Dict[str, Any]: 
        item = self.query_indices[
            item % len(self.query_indices)
        ]
        return self.metadatas.iloc[item][
            [self.VID, self.FID]
        ].values
    
    def load_data(self) -> None: 
        self.preloaded_data = Box({})
        for _, row in self.metadatas.iterrows(): 
            if row[self.VID] in self.preloaded_data.keys(): 
                self.preloaded_data[row[self.VID]].update({
                    row[self.FID]: {
                        "Image": self.image_loader(row.Image),
                        "Mask": None if row.Mask is None else self.mask_loader(row.Mask)
                    }
                })
            else: 
                self.preloaded_data[row[self.VID]] = {
                    row[self.FID]: {
                        "Image": self.image_loader(row.Image),
                        "Mask": None if row.Mask is None else self.mask_loader(row.Mask)
                    }
                }
    
    def _get_records(self, 
                     vid: Any
    ) -> pd.DataFrame: 
        return self.metadatas[
            self.metadatas[self.VID] == vid
        ]
    
    def _get_frame(self, 
                   vid: Any, 
                   fid: Any
    ) -> str: 
        if self.preload is True: 
            return self.preloaded_data[vid][fid].Image.copy()
        else: 
            return self.image_loader(
                self.metadatas[
                    (self.metadatas[self.VID] == vid) & (self.metadatas[self.FID] == fid)
                ].Image.item()
            )
    
    def _get_mask(self, 
                  vid: Any,
                  fid: Any
    ) -> str: 
        if self.preload is True: 
            return self.preloaded_data[vid][fid].Mask.copy()
        else: 
            mask = self.metadatas[
                (self.metadatas[self.VID] == vid) & (self.metadatas[self.FID] == fid)
            ].Mask.item()
            return None if mask is None else self.mask_loader(mask)

    def _get_fids(self, 
                  qvid: Any, 
                  qfid: Any, 
                  size: int, 
                  interval: int
    ) -> List: 
        fids = []
        records = self._get_records(qvid)
        for i in range(size + 1): 
            f_id = qfid - (i * interval)
            if f_id in records[self.FID].values: 
                fids.append(f_id.item())
            else: 
                fids.append(
                    records[self.FID].min().item()
                )
        return list(reversed(fids))
        
        
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 metadata: VideoMetadata,
                 references_size: int = 1,
                 interval_size: int = 1,
                 random_interval: bool = False,
                 pixel_level_transformer: Union[Callable, None] = None,
                 spatial_level_transformer: Union[Callable, None] = None,
                 is_vos: bool = True,
                 query_as_input: bool = True
    ) -> None:
        super(Dataset, self).__init__()
        self.metadata = metadata
        self.references_size = references_size
        self.interval_size = interval_size
        self.random_interval = random_interval
        self.pixel_level_transformer = pixel_level_transformer
        self.spatial_level_transformer = spatial_level_transformer
        self.is_vos = is_vos
        self.query_as_input = query_as_input

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,
                    item: int
    ) -> Tuple[List[Any], List[Any]]:
        vid, fid = self.metadata[item]
        
        vids, fids = [vid], [fid]
        if self.is_vos or not self.query_as_input:
            chosen_interval = self.interval_size
            if self.random_interval: 
                chosen_interval = random.choice(range(1, self.interval_size + 1))
            vids = [vid] * (self.references_size + 1)
            fids = self.metadata._get_fids(
                vid, fid, self.references_size, chosen_interval
            )
            
        if not self.is_vos: 
            # Use query as input for the image-based task.
            # OR, Randomly choose a reference frame as input.
            vids = [vid, vid]
            fids = [fid, fid] if self.query_as_input else [random.choice(fids[:-1]), fid]
        
        images = []
        contours = []
        for fid in fids: 
            images.append(
                self.metadata._get_frame(vid, fid)
            )
            contours.append(
                self.metadata._get_mask(vid, fid) 
            )

        if self.pixel_level_transformer is not None:
            images = self.pixel_level_transformer(images)
        if self.spatial_level_transformer is not None: 
            images, contours = self.spatial_level_transformer(images, contours)
            
        return vids, fids, images, contours

    @staticmethod
    def collate(batch):
        vids, fids, images, contours = list(zip(*batch))
        
        # Query data
        query_vids = [item[-1] for item in vids]
        query_fids = [item[-1] for item in fids]
        query_images = torch.stack([item[-1] for item in images])
        query_contours = None
        if contours[0][0] is not None:
            query_contours = torch.stack([item[-1] for item in contours]).unsqueeze(dim=1)
        
        # Reference Data
        reference_vids = [item[:-1] for item in vids]
        reference_fids = [item[:-1] for item in fids]
        reference_images = [
            torch.stack([item[j] for item in images])
            for j in range(len(images[0]) - 1)  # Exclude the query image
        ]
        reference_contours = None
        if contours[0][0] is not None:
            reference_contours = [
                torch.stack([item[j] for item in contours]).unsqueeze(dim=1)
                for j in range(len(contours[0]) - 1)  # Exclude the query contour
            ]
        
        return (
            (query_vids, query_fids, query_images, query_contours), 
            (reference_vids, reference_fids, reference_images, reference_contours)
        )


def get_dataloader(metadata_paths: List,
                   segmentation: False,
                   phase: str = 'TRAIN',
                   configs: Union[Dict, AttrDict, DictConfig] = None, 
                   rank: Union[int, str] = torch.device(0)
) -> DataLoader:
    height = configs.TrainConfig.IMG_SHAPE[-2]
    width = configs.TrainConfig.IMG_SHAPE[-1]
    mean = configs.TrainConfig.TRANSFORM_MEAN
    std = configs.TrainConfig.TRANSFORM_STD
    
    pixel_level_transformer = None
    if (configs.TrainConfig.APPLY_COLOR_AUGMENTATION and 
        segmentation == True and 
        phase == 'TRAIN'
    ):
        pixel_level_transformer = ColorTransformations(
            transforms=get_pixel_level_transformations()[phase]
        )
    spatial_trfms_list = (
        get_main_spatial_transformations(height, width, mean, std)[phase] 
        if configs.TrainConfig.UsingSmallData is False 
        else get_smalldata_spatial_transformations(height, width, mean, std)[phase]
    )
    spatial_level_transformer = SpatialTransformations(
        num_extra_images=(
            configs.TrainConfig.NUM_REFERENCES 
            if (configs.ModelConfig.IS_VIDEO_TASK or 
                not configs.ModelConfig.USING_QUERY_AS_INPUT
            ) else 1
        ),
        include_mask=(segmentation == True),
        transforms=spatial_trfms_list
    )
    
    metadata = VideoMetadata(
        file_paths=metadata_paths,
        VID=configs.TrainConfig.VID, 
        FID=configs.TrainConfig.FID, 
        query_identifier=configs.TrainConfig.QUERY_IDENTIFIER, 
        dataset_size=(configs.TrainConfig.UsingSamllSize 
                    if configs.TrainConfig.UsingSmallData is True else 0
        ),
        preload=configs.TrainConfig.UsingSmallData, 
        image_loader = utils.ImageLoader(
            dtype=np.uint8,
            clip=False,
            scaler=False
        ),
        mask_loader = utils.MaskLoader(
            dtype=np.uint8,
            binary=True
        )
    )
    
    dataset = VideoDataset(
        metadata=metadata,
        references_size=configs.TrainConfig.NUM_REFERENCES,
        interval_size=configs.TrainConfig.REFERENCES_INTERVAL,
        random_interval=((phase in ["TRAIN", "VALID"]) and 
                         configs.TrainConfig.RANDOM_INTERVAL),
        pixel_level_transformer=pixel_level_transformer,
        spatial_level_transformer=spatial_level_transformer, 
        is_vos=configs.ModelConfig.IS_VIDEO_TASK, 
        query_as_input=configs.ModelConfig.USING_QUERY_AS_INPUT
    )
    
    loader_generator = torch.Generator().manual_seed(configs.BasicConfig.SEED)
    loader_shuffle = (phase == "TRAIN" and configs.TrainConfig.SHUFFLE and not configs.TrainConfig.DISTRIBUTED)
    sampler_shuffle = (phase == "TRAIN" and configs.TrainConfig.SHUFFLE)
    
    dataloader = DataLoader(
        dataset,
        batch_size=configs.TrainConfig.BATCH_SIZE,
        pin_memory=configs.TrainConfig.PIN_MEMORY,
        num_workers=configs.TrainConfig.NUM_WORKERS,
        prefetch_factor=configs.TrainConfig.PREFETCH_FACTOR,
        shuffle=loader_shuffle,
        drop_last=(phase in ["TRAIN", "VALID"]),
        collate_fn=VideoDataset.collate,
        sampler=(DistributedSampler(dataset, num_replicas=len(configs.BasicConfig.DEFAULT_DEVICE_IDs), rank=rank, shuffle=sampler_shuffle) 
                 if configs.TrainConfig.DISTRIBUTED else None),
        worker_init_fn=utils.set_dataloader_workers_seeds,
        generator=loader_generator
    )
    return dataloader


if __name__ == "__main__": 
    from omegaconf import OmegaConf
    configs = OmegaConf.load("configs/trove_configs.yaml")
    configs.TrainConfig.DISTRIBUTED = False
    loader = get_dataloader(
        metadata_paths=configs.TrainConfig.TEST_METADATA_PATH.SEGMENTATION,
        segmentation=True,
        phase='TEST',
        configs=configs, 
        rank=torch.device("cpu")
    )
    _ = next(iter(loader))