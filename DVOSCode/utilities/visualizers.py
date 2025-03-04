from typing import Union, Tuple, List, Optional

import cv2
import numpy as np
import torch
from IPython.display import display, HTML
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, draw_segmentation_masks


# Default color
PINK_COLOR = (204,68,150)

def frames2video(images: List,
                 path: str,
                 frame_rate: int = 25
) -> None:
    """Convert a list of images to a video, and save it in the provided path.
    Args:
        images (List): list of images to convert to video.
        path (str): path to save the video.
    """
    width = images[0].shape[1]
    height = images[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, frame_rate, (width, height))

    for image in images:
        video.write(image)

    video.release()

def draw_segmentation_map(images: Union[List, torch.Tensor],
                          masks: Union[List, torch.Tensor],
                          nrows: Optional[int]=None,
                          colors: Union[List, Tuple]=PINK_COLOR,
                          out_path: str=None, 
                          visualize: bool=False
) -> None:
    """Draw overlayed segmentation map on images.
    Args:
        images (List, torch.Tensor): images to draw segmentation map on.
        masks (List, torch.Tensor, optional): masks to overlay on the images. Defaults to None.
        nrows (_type_, optional): number of images per row for the grid. Defaults to None.
        colors (tuple, optional): colors for the segmnetation maps. Defaults to (193,92,165).
        out_path (_type_, optional): path to save overlaid images. Defaults to None.
        visualize (bool, optional): whether visualize the map or not. Defaults to True.
    Returns:
        None
    """
    assert isinstance(images, (List, torch.Tensor)), "images must be a torch.Tensor"
    assert isinstance(masks, (List, torch.Tensor)), "masks must be a torch.Tensor"
    if nrows is None:
        nrows = int(np.sqrt(len(images)))
    images_with_masks = [
        draw_segmentation_masks(
            img.squeeze(), 
            masks=msk.type(torch.bool).squeeze(), 
            colors=colors, alpha=0.4
        )
        for img, msk in zip(images, masks)
    ]
    grid = make_grid(images_with_masks, nrow=nrows, padding=5, pad_value=255)
    grid = ToPILImage()(grid)
    if out_path is not None:
        grid.save(out_path)
    if visualize is True:
        grid.show()
        
def draw_grid_map(images: Union[List, torch.Tensor],
                  nrows: Optional[int]=None,
                  out_path: str=None, 
                  visualize: bool=False,
) -> None:
    """Draw a grid of input images, save or visualize.
    Args:
        images (torch.Tensor): images to draw the grid map.
        nrows (_type_, optional): number of images per each row for the grid. Defaults to None.
        colors (tuple, optional): colors for the segmnetation maps. Defaults to (193,92,165).
        out_path (_type_, optional): path to save overlaid images. Defaults to None.
        visualize (bool, optional): whether visualize the map or not. Defaults to True.
    Returns:
        None
    """
    assert isinstance(images, (List, torch.Tensor)), "images must be a torch.Tensor"
    if nrows is None:
        nrows = int(np.sqrt(len(images)))
    grid = make_grid([img for img in images], nrow=nrows, padding=5, pad_value=255)
    grid = ToPILImage()(grid)
    if out_path is not None:
        grid.save(out_path)
    if visualize is True:
        grid.show()

def overlay_mask(image: Union[np.ndarray, torch.Tensor], 
                 mask: Union[np.ndarray, torch.Tensor], 
                 color: Union[List, Tuple]=PINK_COLOR, 
                 alpha: float=0.4
) -> Union[np.ndarray, torch.Tensor]:
    """
    Overlays a binary segmentation mask on an image using draw_segmentation_masks.
    
    Args:
        image (numpy.ndarray or torch.Tensor): Image (H, W, 3) or (3, H, W) in [0, 255] (uint8) or [0,1] (float).
        mask (numpy.ndarray or torch.Tensor): Binary mask (H, W), values in {0,1}.
        color (tuple): RGB color for the mask overlay (default: red).
        alpha (float): Transparency factor for blending (0 = transparent, 1 = solid).
    
    Returns:
        numpy.ndarray or torch.Tensor: The image with mask overlay, in the same type as input.
    """
    is_numpy = isinstance(image, np.ndarray)
    
    if is_numpy:
        image_dtype = image.dtype
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
    
    if mask.dtype != torch.bool:
        mask = mask.byte().bool()

    overlayed_image = draw_segmentation_masks(image, mask.squeeze(), colors=[color], alpha=alpha)

    if is_numpy:
        overlayed_image = overlayed_image.permute(1, 2, 0).numpy().astype(image_dtype)

    return overlayed_image
