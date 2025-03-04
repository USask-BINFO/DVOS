import math

from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import wandb
from torch import nn


def init_module_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
           torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
           torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class Swish(nn.Module):
    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        return inputs * torch.sigmoid(inputs)


class AttrDict(Dict):
    """AttrDict class to handle both index and attribute access to a dictionary.
    Args:
        records: dictionary of records.
    Returns:
        None
    """
    def __init__(self,
                 records: Dict
    ) -> None:
        super().__init__()
        self.records = records
        for key, value in self.records.items():
            if isinstance(value, Dict):
                setattr(self, key, AttrDict(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, name):
        return self.records[name]

    def __setitem__(self,
                    key: str,
                    value: Union[str, int, float, Dict, List, Tuple]
    ) -> None:
        setattr(self, key, value)
        self.records[key] = value

    def keys(self):
        return self.records.keys()

    def values(self):
        return self.records.values()

    def items(self):
        return self.records.items()

    def update(self,
               other: Dict
    ) -> None:
        self.records.update(other)
        for key, value in other.items():
            setattr(self, key, value)

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def __getattr__(self,
                    name: str
    ) -> Union[str, int, float, Dict, List, Tuple]:
        if name in self.records:
            return self.records[name]
        raise AttributeError(f"Attribute {name} not found")

    def __repr__(self
                 ) -> str:
        return f"{self.__class__.__name__}(records: Dict={self.records})"


class WeightAndBiases:
    """Weight and biases logger.
    Args:
        project_name: name of the project
        experiment_name: name of the experiment
        api_key: API key for weight and biases
    Returns:
        None
    """
    def __init__(self,
                 project_name: str,
                 experiment_name: str,
                 entity: str = 'default',
                 configs: Union[AttrDict, Dict] = None
    ) -> None:
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.entity = entity
        self.configs = configs

        # Initialize wandb.
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            entity=self.entity,
            config=configs
        )
        self.logger = wandb

    def watch(self,
              model
    ) -> None:
        self.logger.watch(model)

    def log(self,
            message: Union[str, Dict]
    ) -> None:
        if isinstance(message, str):
            message = {'message': message}
        self.logger.log(message)

    def log_image(self,
                  images: Optional[Union[torch.Tensor, np.ndarray]],
                  predictions: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  ground_truth: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  class_labels: Optional[Dict] = None
    ) -> None:
        """Log images with or without ground truth to weight and biases."""
        if class_labels is None:
            class_labels = {0: 'Background', 1: 'Foreground'}

        if isinstance(images, torch.Tensor):
            images = images.permute(0, 2, 3, 1).numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.permute(0, 2, 3, 1).numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.permute(0, 2, 3, 1).numpy()

        predictions = [
            self.logger.Image(
                images[i],
                masks={
                    'predictions': {
                        'mask_data': np.zeros(images[i].shape[0:-1], dtype=np.uint8) if predictions is None else
                        predictions[i].squeeze(),
                        'class_labels': {0: 'Background'} if predictions is None else class_labels
                    },
                    'ground_truth': {
                        'mask_data': np.zeros(images[i].shape[0:-1], dtype=np.uint8) if ground_truth is None else
                        ground_truth[i].squeeze(),
                        'class_labels': {0: 'Background'} if ground_truth is None else class_labels
                    }
                }
            )
            for i in range(images.shape[0])
        ]
        self.logger.log({
            'predictions': predictions
        })
        
        
ACTIVATION = Swish