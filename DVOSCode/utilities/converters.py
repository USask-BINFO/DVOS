import warnings
from typing import Union, Optional, List, Tuple

import cv2
import numpy as np
import torch


class DtypeCoversion:
    """Convert the data type of the images.
        If the data max is 1 and output dtype is int it will be converted to 255.
    Args:
        dtype: data type of the image.
    Returns:
        image: image with the same shape and size but different data type.
    """
    def __init__(self,
                 dtype: Optional[Union[str, np.dtype, torch.dtype]] = None,
                 is_mask: bool = False
    ) -> None:
        self.dtype = dtype
        self.is_mask = is_mask

    def __numpy_dtype(self,
                      data: np.ndarray
    ) -> np.ndarray:
        if (self.is_mask is False and (data.min() == 0.0 and data.max() == 1.0) and
            ((isinstance(self.dtype, str) and self.dtype in ['uint8', 'uint16']) or
             (isinstance(self.dtype, type) and self.dtype in [np.uint8, np.int8, 
                                                              np.int16, np.uint16, 
                                                              np.int32])
            )
        ):
            data = data * 255.0
        data = data.astype(self.dtype)
        return data

    def __tensor_dtype(self,
                       data: torch.Tensor
    ) -> torch.Tensor:
        if (self.is_mask is False and (data.min() == 0.0 and data.max() == 1.0) and
            ((isinstance(self.dtype, str) and self.dtype in ['uint8', 'uint16']) or
             (isinstance(self.dtype, torch.dtype) and self.dtype in [torch.uint8, torch.int8, 
                                                                     torch.short, torch.int])
            )
        ):
            data = data * 255.0
        data = data.type(self.dtype)
        return data

    def __call__(self,
                 data: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        # Check if the dtype is not None. If None, then return the images without any conversion.
        if self.dtype is None:
            return data
        # Check if the images are in the correct format.
        if isinstance(data, torch.Tensor):
            data = self.__tensor_dtype(data)
        elif isinstance(data, np.ndarray):
            data = self.__numpy_dtype(data)
        else:
            raise ValueError('Images are not in the correct format. Expected format: torch.Tensor or np.ndarray')
        return data


class ChannelBroadcasting:
    """Broadcast a 2d image to 3d image with multiple channels.
    Args:
        num_channels: number of channels to broadcast to.
    """
    def __init__(self,
                num_channels: int = 3
    ) -> None:
        self.num_channels = num_channels

    def __numpy_broadcasting(self,
                             image: np.ndarray
    ) -> np.ndarray:
        """Broadcast a 2d image to 3d image with multiple channels.
        Args:
            image: A 2d image of shape (H, W) and in numpy format.
        Returns:
            image: broadcasted image of shape (H, W, C) and in numpy format.
        """
        image = image.squeeze()
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.broadcast_to(image, (image.shape[0], image.shape[1], self.num_channels)).astype(np.uint8)
        else:
            warnings.warn('Image is not in the correct format. Expected format: 2d or 3d with 1 channel', category=Warning)
        return image

    def __tensor_broadcasting(self,
                              image: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast a 2d image to 3d image with multiple channels.
        Args:
            image: A 2d image of shape (H, W) and in torch format.
        Returns:
            image: broadcasted image of shape (H, W, C) and in torch format.
        """
        image = image.squeeze()
        if image.ndim == 2:
            image = image.expand(image.shape[0], image.shape[1], self.num_channels)
        else:
            raise Warning('Image is not in the correct format. Expected format: 2d or 3d with 1 channel')
        return image

    def __call__(self,
                 image: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.num_channels is None or self.num_channels == 0:
            warnings.warn('Number of channels is not specified. Expected number of channels > 0', category=Warning)
            return image
        if isinstance(image, torch.Tensor):
            image = self.__tensor_broadcasting(image)
        elif isinstance(image, np.ndarray):
            image = self.__numpy_broadcasting(image)
        else:
            raise ValueError('Image is not in the correct format. Expected format: torch.Tensor or np.ndarray')

        return image


class InverseNormalization:
    """Inverse transform data to its original form.

    Args:
        mean (Union[Tuple, List]): the same mean values used for
            transforming data, now used to inverse the transform.
        std (Union[Tuple, List]): the same std values used for
            transforming data, now used to inverse the transform.
    Returns:
        torch.Tensor: Inversed data would be the same size and shape as the inpu.
    """
    def __init__(self,
                mean: Union[Tuple, List],
                std: Union[Tuple, List],
                dtype: Optional[Union[str, np.dtype, torch.dtype]] = None
    ) -> None:
        self.mean = mean
        self.std = std
        self.dtype = dtype
        self.dtype_converter = DtypeCoversion(dtype=self.dtype)

    def __numpy_inverse(self,
                        images: np.ndarray
    ) -> np.ndarray:
        assert isinstance(images, np.ndarray), "images must be in numpy.ndarray type"
        mean = np.array(self.mean).reshape(1, 1, 3)
        std = np.array(self.std).reshape(1, 1, 3)

        images = (images * std) + mean
        images = (images - images.min()) / (images.max() - images.min())

        return images

    def __tensor_inverse(self,
                         images: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(images, torch.Tensor), "images must be in torch.Tensor type"
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)

        images = (images * std) + mean
        images = (images - images.min()) / (images.max() - images.min())

        return images

    def __call__(self,
                 images: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(images, torch.Tensor):
            images = self.__tensor_inverse(images)
        elif isinstance(images, np.ndarray):
            images = self.__numpy_inverse(images)
        else:
            raise ValueError('Image is not in the correct format. Expected format: torch.Tensor or np.ndarray')

        # Convert the data type of the images.
        images = self.dtype_converter(images)

        return images


class Seg2RGBMap:
    """Map the segmentation mask to RGB color.
    Args:
        nc: number of classes including background.
        label_colors: list of RGB colors for each class. If the nc is set, the label_colors should be provided as well.
    Returns:
        mask: RGB image with the same shape and size as the input.
    """
    def __init__(self,
                 nc: Optional[int] = None,
                 label_colors: Union[List, Tuple, type(None)] = None
    ) -> None:
        self.nc = nc
        self.label_colors = np.array(label_colors)
        if self.label_colors is None:
            self.label_colors = np.array([
                (0, 0, 0),       # 0=Unlabeled
                (193, 92, 165),  # Single Class code
            ])
        if self.nc == None:
            self.nc = len(self.label_colors)
        assert len(self.label_colors) == self.nc, "The number of colors must be the same as the number of classes."

    def __map_numpy(self,
                    mask: np.ndarray
    ) -> np.ndarray:
        mask = mask.squeeze()
        if mask.ndim == 3:
            channel_dim = 0 if mask.shape[0] == 3 else -1
            mask = np.argmax(mask, axis=channel_dim)
        else:
            channel_dim = -1

        r = np.zeros_like(mask, dtype=np.uint8)
        g = np.zeros_like(mask, dtype=np.uint8)
        b = np.zeros_like(mask, dtype=np.uint8)

        for i, l in enumerate(range(self.nc)):
            idx = mask == l
            r[idx] = self.label_colors[i, 0]
            g[idx] = self.label_colors[i, 1]
            b[idx] = self.label_colors[i, 2]
        rgb = np.stack([r, g, b], axis=channel_dim)

        return rgb

    def __map_tensor(self,
                     mask: torch.Tensor
    ) -> torch.Tensor:
        mask = mask.detach().cpu().squeeze()
        if mask.ndim == 3:
            mask = torch.argmax(mask, axis=0)

        r = torch.zeros_like(mask).type(torch.uint8)
        g = torch.zeros_like(mask).type(torch.uint8)
        b = torch.zeros_like(mask).type(torch.uint8)

        for i, l in enumerate(range(self.nc)):
            idx = mask == l
            r[idx] = self.label_colors[i, 0]
            g[idx] = self.label_colors[i, 1]
            b[idx] = self.label_colors[i, 2]
        rgb = torch.stack([r, g, b], axis=0)

        return rgb

    def __call__(self,
                 mask: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(mask, torch.Tensor):
            mask = self.__map_tensor(mask)
        elif isinstance(mask, np.ndarray):
            mask = self.__map_numpy(mask)
        else:
            raise ValueError('Mask is not in the correct format. Expected format: torch.Tensor or np.ndarray')
        return mask


class ColorSpaceConverter(object):
    """Converts image color space to another format
    Args:
        source_format (str): source image format. Supported formats are `RGB`, `BGR`, `HSV`, `HLS`, `LAB`, `GRAY`.
            Default is `RGB`.
        destination_format (str): destination image format. Supported formats are `RGB`, `BGR`, `HSV`, `HLS`, `LAB`, `GRAY`.
    """
    def __init__(self,
                source_format: str = 'RGB',
                destination_format: str = 'HSV'
    ) -> None:
        self.source_format = source_format
        self.destination_format = destination_format

    def __check(self,
                image: np.ndarray
    ) -> None:
        """Checks if image is in the correct format
        Args:
            image: image to check
        Returns:
            True if image is in the correct format, False otherwise
        """
        if self.source_format in ['RGB', 'BGR', 'HSV', 'LAB', 'HLS']:
            return image.ndim == 3 and image.shape[2] == 3
        elif self.source_format == 'GRAY':
            return image.ndim == 2
        else:
            raise ValueError('Unknown format: {}'.format(self.source_format))

    def __tobgr(self,
                image: np.ndarray
    ) -> np.ndarray:
        """Converts image to BGR format
        Args:
            image: image to convert
        Returns:
            image in BGR format
        """
        if self.__check(image):
            if self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if self.source_format == 'GRAY':
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def __torgb(self,
                image: np.ndarray
    ) -> np.ndarray:
        """Converts image to RGB format
        Args:
            image: image to convert
        Returns:
            image in RGB format
        """
        if self.__check(image):
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.source_format == 'GRAY':
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def __togray(self,
                 image: np.ndarray
    ) -> np.ndarray:
        """Converts image to GRAY format
        Args:
            image: image to convert
        Returns:
            image in GRAY format
        """
        if self.__check(image):
            if self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.Color_BGR2GRAY)
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def __tohsv(self,
                image: np.ndarray
    ) -> np.ndarray:
        """Converts image to HSV format
        Args:
            image: image to convert
        Returns:
            image in HSV format
        """
        if self.__check(image):
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            else:
                raise ValueError('Only RGB or BGR source color spaces are allowed for '
                                 'converting to HSV color space. Source format is {}'.format(self.source_format))
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def __tolab(self,
                image: np.ndarray
    ) -> np.ndarray:
        """Converts image to LAB format
        Args:
            image: image to convert
        Returns:
            image in LAB format
        """
        if self.__check(image):
            if format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            else:
                raise ValueError('Only RGB or BGR source color spaces are allowed for '
                                 'converting to LAB color space. Source format is {}'.format(self.source_format))
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def __tohls(self,
                image: np.ndarray
    ) -> np.ndarray:
        """Converts image to HLS format
        Args:
            image: image to convert
        Returns:
            image in HLS format
        """
        if self.__check(image):
            if self.source_format == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif self.source_format == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            else:
                raise ValueError('Only RGB or BGR source color spaces are allowed for '
                                 'converting to LAB color space. Source format is {}'.format(self.source_format))
        else:
            raise ValueError('Image is not in the correct format. Expected format: {}'.format(self.source_format))
        return image

    def __call__(self,
                 image: np.ndarray
    ) -> np.ndarray:
        match self.destination_format:
            case 'RGB':
                return self.__torgb(image)
            case 'BGR':
                return self.__tobgr(image)
            case 'GRAY':
                return self.__togray(image)
            case 'HSV':
                return self.__tohsv(image)
            case 'LAB':
                return self.__tolab(image)
            case 'HLS':
                return self.__tohls(image)
            case default:
                return image


class IntensityScaler(object):
    """Converts image intensity to another intensity
    The options are as follows:
        - min_max_scaler: scales the image intensity to [0, 1] range
        - standard_scaler: scales the image intensity to zero mean and unit variance
        - scale: scales the image intensity to the specified range
        - clip_intensity: clips the image intensity to the specified range
    Args:
        method: method to use for intensity scaling
        min_intensity: minimum intensity value
        max_intensity: maximum intensity value
    Returns:
        image with scaled intensity
    """

    def clip_intensity(self,
                       image: np.ndarray,
                       min_intensity: Union[int, float],
                       max_intensity: Union[int, float],
    ) -> np.ndarray:
        """Clips image intensity to the specified range
        Args:
            image: image to clip
            min_intensity: minimum intensity value
            max_intensity: maximum intensity value
        Returns:
            image with clipped intensity
        """
        image = np.clip(image, min_intensity, max_intensity)
        image = image.astype(np.float32)
        return image

    def min_max_scaler(self,
                       image: np.ndarray
    ) -> np.ndarray:
        """Scale the input image to [0, 1] range
        Args:
            image: image to convert
        Returns:
            scaled image in [0, 1] range
        """
        image = (image - image.min()) / (image.max() - image.min())
        image = self.clip_intensity(image, min_intensity=0.0, max_intensity=1.0)
        return image

    def standard_scaler(self,
                        image: np.ndarray
    ) -> np.ndarray:
        """Scale the input image to mean=0 and std=1
        Args:
            image: image to convert
        Returns:
            scaled image with mean=0 and std=1
        """
        image = (image - image.mean()) / image.std()
        image = self.clip_intensity(image, min_intensity=0.0, max_intensity=1.0)
        return image

    def scale(self,
              image: np.ndarray,
              min_intensity: Union[int, float],
              max_intensity: Union[int, float]
    ) -> np.ndarray:
        """Scale the input image to the specified range
        Args:
            image: image to convert
            min_intensity: minimum intensity value
            max_intensity: maximum intensity value
        Returns:
            scaled image
        """
        image = self.min_max_scaler(image)
        image = image * (max_intensity - min_intensity) + min_intensity
        image = self.clip_intensity(image, min_intensity=min_intensity, max_intensity=max_intensity)
        return image