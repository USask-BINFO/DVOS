import math
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.utils import make_grid
import einops


class Patching:
    """Patchify image into smaller tiles at their height and width dimensions.
    Args:
        patch_height: The height of the patch.
        patch_width: The width of the patch.
        patch_depth: The depth of the patch.
    Returns:
        The list of patches for both images and masks.
    """
    def __init__(self,
                 patch_height: int = None,
                 patch_width: int = None,
                 patch_depth: int = None
    ) -> None:
        assert patch_height is not None or patch_width is not None, \
            "height_ptch_size and width_ptch_size must be specified."

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_depth = patch_depth

    def __unfold_numpy(self,
                       images: np.ndarray,
                       masks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Patchify image into smaller tiles at their height and width dimensions.
        Args:
            images: 3D or 4D tensor of shape (batch_size, height, width, channels)
                or (height, width, channels).
            masks: 2D or 3D tensor of shape (batch_size, height, width) or
                (height, width).
        Returns:
            The list of patches, concatenated into a single numpy array,
                for both images and masks in the follwing dimensions:
                (batch_size, height, width, channels). If the input images
                contain only one image, the output would be still in the
                above dimensions.
        """
        if images.ndim == 2:
            # 2D single-channel images (height, width)
            images = np.expand_dims(images, axis=(0, -1))
        if images.ndim == 3:
            if images.shape[0] == 3:
                # First channel to last channel (height, width, 3)
                images = images.transpose(1, 2, 0)
                images = np.expand_dims(images, axis=0)
            elif images.shape[-1] != 3:
                # 2D to 3D single-channel 3D images (batch_size, height, width, 1)
                images = np.expand_dims(images, axis=-1)
            else:
                # 3D multi-channel images (height, width, channels)
                images = np.expand_dims(images, axis=0)
        elif images.ndim == 4 and images.shape[1] == 3:
            # First channel to last channel (batch_size, height, width, 3)
            images = images.transpose(0, 2, 3, 1)

        if masks is not None:
            if masks.ndim == 2:
                # 2D single-channel masks (height, width)
                masks = np.expand_dims(masks, axis=(0, -1))
            if masks.ndim == 3:
                if masks.shape[0] == 1:
                    # First channel to last channel (height, width, 1)
                    masks = masks.transpose(1, 2, 0)
                    masks = np.expand_dims(masks, axis=0)
                elif masks.shape[-1] != 1:
                    # 2D to 3D single-channel 3D masks (batch_size, height, width, 1)
                    masks = np.expand_dims(masks, axis=-1)
                else:
                    # 3D multi-channel masks (height, width, channels)
                    masks = np.expand_dims(masks, axis=0)
            elif masks.ndim == 4 and masks.shape[1] == 1:
                # First channel to last channel (batch_size, height, width, 1)
                masks = masks.transpose(0, 2, 3, 1)

        images_patches = []
        masks_patches = []
        for k in range(images.shape[0]):
            image_k_patches = []
            mask_k_patches = []
            for i in range(0, images[k].shape[0],
                           self.patch_height):
                for j in range(0, images[k].shape[1],
                               self.patch_width):
                    if (i + self.patch_height > images[k].shape[0] or
                        j + self.patch_width > images[k].shape[1]
                    ):
                        continue
                    image_k_patches.append(
                        images[k][i:i + self.patch_height,
                                  j:j + self.patch_width,
                                  :]
                    )
                    if masks is not None:
                        mask_k_patches.append(
                            masks[k][i:i + self.patch_height,
                                     j:j + self.patch_width,
                                     :]
                        )
            images_patches.append(np.array(image_k_patches))
            if masks is not None:
                masks_patches.append(np.array(mask_k_patches))

        images_patches = np.stack(images_patches, axis=0)
        if len(masks_patches) > 0:
            masks_patches = np.stack(masks_patches, axis=0)
        else:
            masks_patches = None

        return images_patches, masks_patches

    def __unfold_tensor(self,
                        images: torch.Tensor,
                        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Patchify image into smaller tiles at their height and width dimensions.
        Args:
            images: 3D or 4D tensor of shape (batch_size, channels, height, width) or
                (channels, height, width).
            masks: 2D or 3D tensor of shape (batch_size, height, width) or
                (height, width).
        Returns:
            A list of patches, concatenated into a single tensor for both
                images and masks with the following dimensions:
                (batch_size, channels, height, width). If the input images contain
                only one image, the output would be still in the above
                dimensions.
        """
        # Fix the dimensions of the input images
        if images.ndim == 2:
            images = images.unsqueeze(0)
            self.patch_depth = 1
        if images.ndim == 3:
            if images.shape[0] == 3 == self.patch_depth:
                images = images.unsqueeze(0)
            elif images.shape[-1] == 3:
                images = images.permute(2, 0, 1)
                images = images.unsqueeze(0)
            else:
                images = images.unsqueeze(1)
        elif images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        if masks is not None:
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if masks.ndim == 3:
                if masks.shape[0] == 1:
                    masks = masks.unsqueeze(0)
                elif masks.shape[-1] == 1:
                    masks = masks.permute(2, 0, 1)
                    masks = masks.unsqueeze(0)
                else:
                    masks = masks.unsqueeze(1)
            elif masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks.permute(0, 3, 1, 2)

        images_patches = []
        for img in images:
            images_patches.append(
                img.unfold(
                0, self.patch_depth, self.patch_depth
                ).unfold(
                    1, self.patch_height, self.patch_height
                ).unfold(
                    2, self.patch_width, self.patch_width
                ).reshape(
                    -1, self.patch_depth,
                    self.patch_height, self.patch_width
                )
            )
        if masks is not None:
            masks_patches = []
            for msk in masks:
                masks_patches.append(
                    msk.unfold(
                        0, 1, 1
                    ).unfold(
                        1, self.patch_height, self.patch_height
                    ).unfold(
                        2, self.patch_width, self.patch_width
                    ).reshape(
                        -1, 1, self.patch_height, self.patch_width
                    )
                )
        else:
            masks_patches = None

        images_patches = torch.stack(images_patches, dim=0)
        if masks_patches is not None:
            masks_patches = torch.stack(masks_patches, dim=0)

        return images_patches, masks_patches

    def __call__(self,
                 images: Union[np.ndarray, torch.Tensor],
                 masks: Union[np.ndarray, torch.Tensor] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        if isinstance(images, np.ndarray):
            image_patches, mask_patches = self.__unfold_numpy(images, masks)
        elif isinstance(images, torch.Tensor):
            image_patches, mask_patches = self.__unfold_tensor(images, masks)
        else:
            raise TypeError("image must be np.ndarray or torch.Tensor.")

        image_patches = image_patches.squeeze()
        if mask_patches is not None:
            mask_patches = mask_patches.squeeze()
        return image_patches, mask_patches


class UnPatching:
    """Fold the patches into the original image.
    Args:
        num_rows: Number of rows in the grid of patches. If None, the number of
            rows is automatically calculated for a square grid.
    """
    def __init__(self,
                 num_rows: Optional[int] = None,
                 num_cols: Optional[int] = None
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.set_num_rows = (self.num_rows is None)

    def __fold_numpy(self,
                     images: Union[np.ndarray, List[np.ndarray]],
                     masks: Union[np.ndarray, List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Fold the patches into the original image.
        This function uses PIL image to create grids of patches and then
            folds them into the original image.
        Args:
            images: 4D tensor of shape (batch_size, height, width, channels) or
                (height, width, channels).
            masks: 3D tensor of shape (batch_size, height, width) or
                (height, width).
        Returns:
            A list of patches, concatenated into a single tensor for both
                images and masks with the following dimensions:
                (batch_size, channels, height, width). If the input images contain
        """
        assert isinstance(images, (List, np.ndarray)), \
            "images must be List[np.ndarray] or np.ndarray"
        assert isinstance(masks, (List, np.ndarray, type(None))), \
            "masks must be List[np.ndarray] or np.ndarray or None"

        images = np.array(images)
        if images.ndim == 3: # a series of 2D images
            images = np.expand_dims(images, axis=-1)
        if images.ndim == 4:
            if images.shape[1] in [1, 3]:
                images = np.transpose(images, (0, 2, 3, 1))
                images = np.expand_dims(images, axis=0)
            elif images.shape[-1] not in [1, 3]:
                images = np.expand_dims(images, axis=-1)
            else:
                images = np.expand_dims(images, axis=0)
        if images.ndim == 5 and images.shape[2] in [1, 3]:
            images = np.transpose(images, (0, 1, 3, 4, 2))
        if masks is not None:
            masks = np.array(masks)
            if masks.ndim == 3:
                masks = np.expand_dims(masks, axis=-1)
            if masks.ndim == 4:
                if masks.shape[1] == 1:
                    masks = np.transpose(masks, (0, 2, 3, 1))
                    masks = np.expand_dims(masks, axis=0)
                elif masks.shape[-1] != 1:
                    masks = np.expand_dims(masks, axis=-1)
                else:
                    masks = np.expand_dims(masks, axis=0)
            if masks.ndim == 5 and masks.shape[2] == 1:
                masks = np.transpose(masks, (0, 1, 3, 4, 2))

        # Set number of rows if not set after the fixed dimensions.
        if self.set_num_rows is True:
            total_num_patches = images[0].shape[0]
            self.num_rows = math.ceil(np.sqrt(total_num_patches))

        image_grids = []
        for k in range(len(images)):
            patches = images[k]
            grid_height = len(patches) // self.num_rows
            grid_width = len(patches) // grid_height
            img_height, img_width = patches.shape[1], patches.shape[2]
            grid = PILImage.new(
                'RGB' if patches.shape[-1] == 3 else 'L',
                (grid_width * img_width, grid_height * img_height)
            )
            # Loop through the patches and paste them onto the grid
            for i, img in enumerate(patches):
                img = (
                    ((img - img.min()) / (img.max() - img.min())) * 255.
                ).astype(np.uint8)
                if img.shape[-1] == 1:
                    img = np.squeeze(img, axis=-1)
                img = PILImage.fromarray(img)
                r = (i // grid_width) * img_height
                c = (i % grid_width) * img_width
                grid.paste(img, (c, r))
            image_grids.append(np.array(grid))
        images = np.stack(image_grids, axis=0)

        if masks is not None:
            mask_grids = []
            for k in range(len(masks)):
                patches = masks[k]
                grid_width = len(patches) // self.num_rows
                grid_height = len(patches) // grid_width
                msk_height, msk_widht = patches.shape[1], patches.shape[2]
                grid = PILImage.new(
                    'L',
                    (grid_width * msk_widht, grid_height * msk_height)
                )
                # Loop through the patches and paste them onto the grid
                for i, msk in enumerate(patches):
                    if msk.ndim == 3:
                        msk = msk.squeeze(-1)
                    msk = PILImage.fromarray(msk.astype(np.uint8))
                    c = (i % grid_width) * msk_widht
                    r = (i // grid_width) * msk_height
                    grid.paste(msk, (c, r))
                mask_grids.append(np.array(grid))
            masks = np.stack(mask_grids, axis=0)

        return images, masks

    def __fold_tensor(self,
                      images: Union[torch.Tensor, List[torch.Tensor]],
                      masks: Union[torch.Tensor, List[torch.Tensor], None] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fold the patches into the original image.
        This function uses torchvision make grids to create grids of
        patches and then fold them into the original image.
        Args:
            images: 4D tensor of shape (batch_size, channels, height, width) or
                (channels, height, width).
            masks: 3D tensor of shape (batch_size, height, width) or
                (height, width).
        Returns:
            A list of patches, concatenated into a single tensor for both
                images and masks with the following dimensions:
                (batch_size, channels, height, width). If the input images contain
        """
        assert isinstance(images, (List, torch.Tensor)), \
            "images must be List[torch.Tensor] or torch.Tensor"
        assert isinstance(masks, (List, torch.Tensor, type(None))), \
            "masks must be List[torch.Tensor] or torch.Tensor or None"

        if isinstance(images, List):
            images = torch.Tensor(images)
        if isinstance(masks, List):
            masks = torch.Tensor(masks)

        if images.ndim == 3: # a series of 2D images with no channel
            images = images.unsqueeze(dim=1)
        if masks is not None and masks.ndim == 3:
            masks = masks.unsqueeze(dim=1)
        if images.ndim == 4: # A series of 2D images with proper channels
            if images.shape[-1] in [1, 3]:
                images = images.permute(0, 3, 1, 2)
                images = images.unsqueeze(dim=0)
            elif images.shape[-3] not in [1, 3]:
                images = images.unsqueeze(dim=-3)
            else:
                images = images.unsqueeze(dim=0)
        if images.ndim == 5 and images.shape[-1] in [1, 3]:
            images = images.permute(0, 1, 4, 2, 3)
        if masks is not None:
            if masks.ndim == 4:
                if masks.shape[-1] == 1:
                    masks = masks.permute(0, 3, 1, 2)
                    masks = masks.unsqueeze(dim=0)
                elif masks.shape[-3] != 1:
                    masks = masks.unsqueeze(dim=-3)
                else:
                    masks = masks.unsqueeze(dim=0)
            if masks.ndim == 5 and masks.shape[-1] == 1:
                masks = masks.permute(0, 1, 4, 2, 3)
        # Set number of rows if not set after the fixed dimensions.
        if self.set_num_rows is True:
            total_num_patches = images[0].shape[0]
            self.num_rows = math.ceil(np.sqrt(total_num_patches))

        image_grids = []
        for patches in images:
            if patches.shape[1] == 3:
                image_grids.append(
                    make_grid(patches, nrow=self.num_rows,
                              padding=0, pad_value=0)
                )
            else:
                image_grids.append(
                    make_grid(patches, nrow=self.num_rows,
                              padding=0, pad_value=0)[0].unsqueeze(dim=0)
                )
        images = torch.stack(image_grids, dim=0)
        if masks is not None:
            masks = torch.stack([
                make_grid(patches, nrow=self.num_rows,
                          padding=0, pad_value=0)[0].unsqueeze(dim=0)
                for patches in masks
            ], dim=0)

        return images, masks

    def __call__(self,
             images: Union[np.ndarray, List[np.ndarray], torch.Tensor],
             masks:  Union[np.ndarray, List[np.ndarray],
                           torch.Tensor, List[torch.Tensor], None] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor],
               Union[np.ndarray, torch.Tensor, None]]:
        assert images.ndim <= 5, "images must be 2D, 3D, 4D or 5D."
        assert masks is None or masks.ndim <= 5, "masks must be 2D, 3D, 4D or 5D."

        if isinstance(images, np.ndarray):
            unpatched_images, unpatched_masks =  self.__fold_numpy(images, masks)
        elif isinstance(images, torch.Tensor):
            unpatched_images, unpatched_masks = self.__fold_tensor(images,  masks)
        else:
            raise TypeError("image must be np.ndarray or torch.Tensor.")

        unpatched_images = unpatched_images.squeeze()
        if unpatched_masks is not None:
            unpatched_masks = unpatched_masks.squeeze()
        return unpatched_images, unpatched_masks


class EinPatching:
    """Patchify input pairs of images/masks into smaller tiles at their height
    and width dimensions.
    Args:
        patch_height: The height of the patch.
        patch_width: The width of the patch.
        patch_depth: The depth of the patch.
    Returns:
        The list of patches for both images and masks.
    """
    def __init__(self,
                 patch_height: int = None,
                 patch_width: int = None,
    ) -> None:
        assert patch_height is not None or patch_width is not None, \
            "height_ptch_size and width_ptch_size must be specified."

        self.patch_height = patch_height
        self.patch_width = patch_width

    def __fix_dimension(self,
                        images: Union[np.ndarray, torch.Tensor],
                        masks: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if isinstance(images, torch.Tensor):
            images = images.numpy()
        if masks is not None and isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        if images.ndim == 2:
            # 2D single-channel images (height, width)
            images = np.expand_dims(images, axis=(0, -1))
        if images.ndim == 3:
            if images.shape[0] == 3:
                # First channel to last channel (height, width, 3)
                images = images.transpose(1, 2, 0)
                images = np.expand_dims(images, axis=0)
            elif images.shape[-1] != 3:
                # 2D to 3D single-channel 3D images (batch_size, height, width, 1)
                images = np.expand_dims(images, axis=-1)
            else:
                # 3D multi-channel images (height, width, channels)
                images = np.expand_dims(images, axis=0)
        elif images.ndim == 4 and images.shape[1] == 3:
            images = images.transpose(0, 2, 3, 1)

        if masks is not None:
            if masks.ndim == 2:
                # 2D single-channel masks (height, width)
                masks = np.expand_dims(masks, axis=(0, -1))
            if masks.ndim == 3:
                if masks.shape[0] == 1:
                    # First channel to last channel (height, width, 1)
                    masks = masks.transpose(1, 2, 0)
                    masks = np.expand_dims(masks, axis=0)
                elif masks.shape[-1] != 1:
                    # 2D to 3D single-channel 3D masks (batch_size, height, width, 1)
                    masks = np.expand_dims(masks, axis=-1)
                else:
                    # 3D multi-channel masks (height, width, channels)
                    masks = np.expand_dims(masks, axis=0)
            elif masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.transpose(0, 2, 3, 1)
        return images, masks

    def __patchify(self,
                   images: np.ndarray,
                   masks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Patchify image into smaller tiles at their height and width dimensions.
        Args:
            images: 3D or 4D tensor of shape (batch_size, height, width, channels)
                or (height, width, channels).
            masks: 2D or 3D tensor of shape (batch_size, height, width) or
                (height, width).
        Returns:
            The list of patches, concatenated into a single numpy array,
                for both images and masks in the follwing dimensions:
                `(batch_size, height, width, patch_height, patch_width, channels)`.
                If the input images contain only one image, the output would
                be still in the above dimensions.
        """
        images, masks = self.__fix_dimension(images, masks)

        images_patches = einops.rearrange(
            images,
            'b (h ph) (w pw) c -> b (h w) ph pw c',
            ph=self.patch_height,
            pw=self.patch_width
        )
        masks_patches = None
        if masks is not None:
            masks_patches = einops.rearrange(
                masks,
                'b (h ph) (w pw) c -> b (h w) ph pw c',
                ph=self.patch_height,
                pw=self.patch_width
            )
        return images_patches, masks_patches

    def __call__(self,
                 images: Union[np.ndarray, torch.Tensor],
                 masks: Union[np.ndarray, torch.Tensor] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        assert isinstance(images, (np.ndarray, torch.Tensor))
        assert masks is None or isinstance(masks, (np.ndarray, torch.Tensor))

        is_torch = isinstance(images, torch.Tensor)
        images_patches, masks_patches = self.__patchify(images, masks)

        if is_torch:
            images_patches = torch.from_numpy(
                einops.rearrange(
                    images_patches, 'b p h w c -> b p c h w'
                )
            )
        images_patches = images_patches.squeeze()

        if masks is not None:
            if is_torch:
                masks_patches = torch.from_numpy(
                    einops.rearrange(
                        masks_patches, 'b p h w c -> b p c h w'
                    )
                )
            masks_patches = masks_patches.squeeze()
        return images_patches, masks_patches


class EinUnPatching:
    """Fold the patches into the original image.
    Args:
        num_rows: Number of rows in the grid of patches. If None, the number of
            rows is automatically calculated for a square grid.
        num_cols: Number of columns in the grid of patches. If None, the number
            of columns is automatically calculated for a square grid.
    """
    def __init__(self,
                 num_rows: Optional[int] = None,
                 num_cols: Optional[int] = None
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.set_grid = (self.num_rows is None or self.num_cols is None)

    def __set_grid_dims(self,
                        total_num_patches: int
    ) -> None:
        # Set number of rows if not set after the fixed dimensions.
        if self.set_grid is True:
            if self.num_rows is None and self.num_cols is None:
                self.num_rows = math.ceil(np.sqrt(total_num_patches))
                self.num_cols = total_num_patches // self.num_rows
            elif self.num_rows is None:
                self.num_rows = total_num_patches // self.num_cols
            elif self.num_cols is None:
                self.num_cols = total_num_patches // self.num_rows

    def __fix_dimension(self,
                        images: Union[np.ndarray, torch.Tensor],
                        masks: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        if isinstance(images, torch.Tensor):
            images = images.numpy()
        if masks is not None and isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        images = np.array(images)
        if images.ndim == 3:  # a series of 2D images
            images = np.expand_dims(images, axis=-1)
        if images.ndim == 4:
            if images.shape[1] in [1, 3]:
                images = np.transpose(images, (0, 2, 3, 1))
                images = np.expand_dims(images, axis=0)
            elif images.shape[-1] not in [1, 3]:
                images = np.expand_dims(images, axis=-1)
            else:
                images = np.expand_dims(images, axis=0)
        if images.ndim == 5 and images.shape[2] in [1, 3]:
            images = np.transpose(images, (0, 1, 3, 4, 2))
        if masks is not None:
            masks = np.array(masks)
            if masks.ndim == 3:
                masks = np.expand_dims(masks, axis=-1)
            if masks.ndim == 4:
                if masks.shape[2] == 1:
                    masks = np.transpose(masks, (0, 2, 3, 1))
                    masks = np.expand_dims(masks, axis=0)
                elif masks.shape[-1] != 1:
                    masks = np.expand_dims(masks, axis=-1)
                else:
                    masks = np.expand_dims(masks, axis=0)
            if masks.ndim == 5 and masks.shape[2] == 1:
                masks = np.transpose(masks, (0, 1, 3, 4, 2))
        return images, masks

    def __unpatchify(self,
                     images: Union[np.ndarray, List[np.ndarray]],
                     masks: Union[np.ndarray, List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Fold the patches into the original image.
        This function uses PIL image to create grids of patches and then
            folds them into the original image.
        Args:
            images: 4D tensor of shape (batch_size, height, width, channels) or
                (height, width, channels).
            masks: 3D tensor of shape (batch_size, height, width) or
                (height, width).
        Returns:
            A list of patches, concatenated into a single tensor for both
                images and masks with the following dimensions:
                (batch_size, channels, height, width). If the input images contain
        """
        # Set number of rows and columns
        self.__set_grid_dims(total_num_patches=images[0].shape[0])
        images, masks = self.__fix_dimension(images, masks)

        images = einops.rearrange(
            images,
            'b (h w) ph pw c -> b (h ph) (w pw) c',
            h=self.num_rows,
            w=self.num_cols
        )
        if masks is not None:
            masks = einops.rearrange(
                masks,
                'b (h w) ph pw c -> b (h ph) (w pw) c',
                h=self.num_rows,
                w=self.num_cols
            )
        return images, masks

    def __call__(self,
             images: Union[np.ndarray, List[np.ndarray], torch.Tensor],
             masks:  Union[np.ndarray, List[np.ndarray],
                           torch.Tensor, List[torch.Tensor], None] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor],
               Union[np.ndarray, torch.Tensor, None]]:
        assert images.ndim <= 5, "images must be 2D, 3D, 4D or 5D."
        assert masks is None or masks.ndim <= 5, "masks must be 2D, 3D, 4D or 5D."

        is_torch = isinstance(images, torch.Tensor)

        unpached_images, unpached_masks = self.__unpatchify(images, masks)

        if is_torch:
            unpached_images = torch.from_numpy(
                einops.rearrange(unpached_images, 'b h w c -> b c h w')
            )
        unpached_images = unpached_images.squeeze()
        if unpached_masks is not None:
            if is_torch:
                unpached_masks = torch.from_numpy(
                    einops.rearrange(unpached_masks, 'b h w c -> b c h w')
                )
            unpached_masks = unpached_masks.squeeze()
        return unpached_images, unpached_masks
