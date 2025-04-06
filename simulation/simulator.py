import argparse
import os
import random
import warnings
from math import ceil, floor
from typing import List, Tuple, Union, Optional, Callable, Type, Dict, Any, Set

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from box import Box
from icecream import ic
from skimage import io
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm

from data import BackgroundDataset, RealObjectDataset, FakePatchDataset

warnings.filterwarnings("ignore")


class WheatHeadNode:
    """A node that contains the data of a wheat head.
    Args:
        position: The coordinate of the wheat head (y, x) or (row, column).
        data: The data of the wheat head.
        annotation: The annotation of the wheat head.
        annot_id: The id of the annotation.
        transformer: A callable that transforms the data and annotation of the wheat head.
    """
    def __init__(self,
                 position: np.ndarray,
                 data: Union[torch.Tensor, np.ndarray],
                 annotation: Union[torch.Tensor, np.ndarray],
                 annot_id: int = 1,
                 trfms_confs: Dict[str, Any] = {},
                 is_fake: bool = False
    ) -> None:
        self.position = position
        self.data = data
        self.annotation = annotation
        self.annot_id = annot_id
        self.trfms_confs = trfms_confs
        self.is_fake = is_fake

        self.direction = np.random.choice([-1, 1], size=2, replace=True)

        self.prev = None
        self.next = None

        self.aug_data = None
        self.aug_annotation = None

        self.rotation_angle = None
        self.rotation_transformer = None
        self.transformer = None

        self.nudge_rotation_transformer()
        self.set_transformer()

    def custom_rotation(self,
                        image: np.ndarray,
                        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotates the wheat head and its annotation.
        """
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        new_image = image.rotate(
            self.rotation_angle, resample=Image.BICUBIC, expand=True
        )
        new_mask = mask.rotate(
            self.rotation_angle, resample=Image.BICUBIC, expand=True
        )

        new_image = np.array(new_image)
        new_mask = np.array(new_mask)

        return {'image': new_image, 'mask': new_mask}

    def nudge_rotation_transformer(self) -> None:
        """Randomly nudges the rotation angle of the wheat head.
            Should be called before setting the rotation transformer,
            and after each time the rotation transformer is used.
        """
        assert (
            0 <= self.trfms_confs.ROTATE_RANGE[0] <
            self.trfms_confs.ROTATE_RANGE[1] <= 360
        ),  'Invalid rotation range, must be 0 <= min < max <= 360'
        if self.rotation_angle is None:
            self.rotation_angle = (
                    random.choice([-1, 1]) *
                    random.randint(*self.trfms_confs.ROTATE_RANGE)
            )
        elif self.rotation_angle >= 0:
            self.rotation_angle += random.randint(*self.trfms_confs.ROTATE_RANGE)
        else:
            self.rotation_angle -= random.randint(*self.trfms_confs.ROTATE_RANGE)

        if self.rotation_angle >= 359:
            self.rotation_angle = 1
        elif self.rotation_angle <= -359:
            self.rotation_angle = -1

        self.rotation_transformer = self.custom_rotation

    def get_rotation_angle(self) -> int:
        return self.rotation_angle

    def get_rotation_transformer(self) -> Callable:
        return self.rotation_transformer

    def set_transformer(self) -> Callable:
        """Sets the transformer for the wheat head.
        """
        trfms = [
            A.HorizontalFlip(p=self.trfms_confs.HORIZONAL_FLIP_PROB),
            A.VerticalFlip(p=self.trfms_confs.VERTICAL_FLIP_PROB),
            A.ElasticTransform(
                alpha=random.uniform(*self.trfms_confs.ELASTIC_ALPHA),
                sigma=random.randint(*self.trfms_confs.ELASTIC_SIGMA),
                alpha_affine=random.randint(
                    *self.trfms_confs.ELASTIC_ALPHA_AFFINE
                ),
                interpolation=1, border_mode=cv2.BORDER_CONSTANT,
                value=0, mask_value=0,
                p=self.trfms_confs.ELASTIC_PROB
            )
        ]
        if self.is_fake is True:
            trfms.extend([
                A.RandomBrightnessContrast(p=0.2),
                A.ToGray(p=0.2),
                A.ToSepia(p=0.2),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                              hue=0.2, p=0.2)
            ])
        self.transformer = A.Compose(trfms, p=1.0)

    def get_direction(self) -> np.ndarray:
        return self.direction

    def set_direction(self, direction: np.ndarray) -> None:
        self.direction = direction

    def get_position(self) -> Tuple[int, int]:
        """Returns the coordinate of the wheat head, np.array([y, x])
        """
        return self.position

    def set_position(self, position: np.ndarray) -> None:
        self.position = position
    def get_shape(self) -> np.ndarray:
        """Returns the shape, height and width, of the wheat head.
        """
        return np.array(self.data.shape[:2])

    def get_data(self) -> np.ndarray:
        return self.data

    def get_aug_data(self) -> np.ndarray:
        return self.aug_data

    def get_annotation(self) -> np.ndarray:
        return self.annotation

    def get_aug_annotation(self) -> np.ndarray:
        return self.aug_annotation

    def get_annotation_id(self) -> int:
        return self.annot_id

    def set_prev(self, prev: Type["WheatHeadNode"]) -> None:
        self.prev = prev

    def set_next(self, next: Type["WheatHeadNode"]) -> None:
        self.next = next

    def set_data(self, data: np.ndarray) -> None:
        self.data = data

    def set_aug_data(self, aug_data: np.ndarray) -> None:
        self.aug_data = aug_data

    def set_annotation(self, annotation: np.ndarray) -> None:
        self.annotation = annotation

    def set_aug_annotation(self, aug_annotation: np.ndarray) -> None:
        self.aug_annotation = aug_annotation

    def set_coordinate(self, position: np.ndarray) -> None:
        self.position = position

    def get_prev(self) -> Type["WheatHeadNode"]:
        return self.prev

    def get_next(self) -> Type["WheatHeadNode"]:
        return self.next

    def __repr__(self) -> str:
        return f"HeadNode(position={self.position}, data={self.data}, " \
               f"annotation={self.annotation}, annot_id={self.annot_id}, " \
               f"transformer={self.transformer})"


class WheatHeadChain:
    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def __len__(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.get_next()
        return count

    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.get_next()

    def __getitem__(self, item: int) -> WheatHeadNode:
        current = self.head
        count = 0
        while current:
            if count == item:
                return current
            count += 1
            current = current.get_next()
        return current

    def get_head(self) -> WheatHeadNode:
        return self.head

    def get_tail(self) -> WheatHeadNode:
        return self.tail

    def append(self, node: WheatHeadNode) -> None:
        if self.tail is None:
            self.tail = node
        else:
            current = self.get_tail()
            current.set_next(node)
            node.set_prev(current)
            self.tail = node
        if self.head is None:
            self.head = node

    def prepend(self, node: WheatHeadNode) -> None:
        if self.head is None:
            self.head = node
        else:
            current = self.get_head()
            current.set_prev(node)
            node.set_next(current)
            self.head = node
        if self.tail is None:
            self.tail = node

    def remove(self, node: WheatHeadNode) -> None:
        prev_node = node.get_prev()
        next_node = node.get_next()
        if prev_node is not None:
            prev_node.set_next(next_node)
        else:
            self.head = next_node
        if next_node is not None:
            next_node.set_prev(prev_node)
        else:
            self.tail = prev_node
        del node

    def get_positions(self) -> List[Tuple[int, int]]:
        """Returns a list of numpy arrays of leangth of 2 containing the
        coordinates of the wheat heads. [(y1, x1), (y2, x2), ..., (yn, xn)]
        """
        node = self.head
        positions = []
        while node:
            positions.append(node.get_position())
            node = node.get_next()
        return positions

    def get_unique_ids(self) -> Set[int]:
        """Returns a list of unique annotation ids.
        """
        node = self.head
        ids = []
        while node:
            ids.append(node.get_annotation_id())
            node = node.get_next()
        return set(ids)


class FrameSimulation:
    def __init__(self,
                 background: np.ndarray,
                 foregrounds: WheatHeadChain = WheatHeadChain(),
                 node_movement_range: Tuple[int, int] = (0, 5),
                 frame_movement_range: Tuple[int, int] = (0, 5),
                 recovery_variance: int = 0,
                 allow_leave_frame: bool = False,
                 recovery_area_expansion: int = 1,
                 change_direction_p: float = 0.1,
                 trfms_confs: Dict[str, Any] = {}
    ) -> None:
        self.original_background = background.copy()
        self.background = background
        self.foregrounds: WheatHeadChain = foregrounds
        self.node_movement_range = node_movement_range
        self.frame_movement_range = frame_movement_range
        self.recovery_variance = recovery_variance
        self.allow_leave_frame = allow_leave_frame
        self.recovery_area_expansion = recovery_area_expansion
        self.change_direction_p = change_direction_p
        self.trfms_confs = trfms_confs

        self.height, self.width = self.background.shape[:2]
        self.contour = np.zeros((self.height, self.width), dtype=np.uint8)

        self.direction = np.random.choice([-1, 1], size=2)

        # The number of removed objects due to the movement of the frame.
        self.removed_nodes = 0

    def __len__(self):
        return len(self.foregrounds)

    def __getitem__(self, item: int) -> WheatHeadNode:
        return self.foregrounds[item]

    def is_in_frame(self, position: np.ndarray) -> bool:
        """Returns True if the position is within the frame, False otherwise.
        """
        return 0 <= position[0] < self.height and 0 <= position[1] < self.width

    def get_background(self) -> Union[torch.Tensor, np.ndarray]:
        return self.background

    def set_background(self, background: np.ndarray) -> None:
        self.background = background

    def get_contour(self) -> np.ndarray:
        return self.contour

    def set_contour(self, contour: np.ndarray) -> None:
        self.contour = contour

    def get_nodes(self) -> WheatHeadChain:
        return self.foregrounds

    def set_nodes(self, foregrounds: WheatHeadChain) -> None:
        self.foregrounds = foregrounds

    def get_node(self, index: int) -> WheatHeadNode:
        return self.__getitem__(index)

    def get_nodes_positions(self) -> WheatHeadChain:
        return self.foregrounds.get_positions()

    def get_node_position(self, index: int) -> Tuple[int, int]:
        return self.foregrounds.get_positions()[index]

    def set_dtypes(self,
                   background_dtype: np.dtype,
                   contour_dtype: np.dtype
    ) -> None:
        self.background = self.background.astype(background_dtype)
        self.contour = self.contour.astype(contour_dtype)

    def find_best_fit(self,
                      annotation: np.ndarray,
                      h_range: Tuple[int, int] = None,
                      w_range: Tuple[int, int] = None
    ) -> Tuple[int, int]:
        """Finds the best fit for the annotation in the frame.
        """
        if h_range is None:
            h_range = (0, self.height - 1)
        if w_range is None:
            w_range = (0, self.width - 1)
        h, w = annotation.shape[:2]
        num_tries = 0
        best_fit = float("inf")
        best_y, best_x = None, None
        while num_tries <= 50 and best_fit >= 0.4:
            y = random.randint(*h_range)
            x = random.randint(*w_range)
            annot = annotation[
                floor(h / 2) - min(y, floor(h / 2)):
                floor(h / 2) + min(self.height - y, ceil(h / 2)),
                floor(w / 2) - min(x, floor(w / 2)):
                floor(w / 2) + min(self.width - x, ceil(w / 2))
            ]
            coverage = np.sum(
                np.multiply(
                    self.contour[max(0, y - floor(h / 2)):
                                 min(self.height, y + ceil(h / 2)),
                                 max(0, x - floor(w / 2)):
                                 min(self.width, x + ceil(w / 2))],
                    annot
                ) > 0
            ) / np.sum(annot)
            if coverage <= best_fit:
                best_fit = coverage
                best_y, best_x = y, x
            num_tries += 1
        if best_y is None or best_x is None:
            best_y = random.randint(*h_range)
            best_x = random.randint(*w_range)
        return best_y, best_x

    def overlay(self,
                position: np.ndarray,
                node_data: np.ndarray,
                node_mask: np.ndarray,
                node_unique_id: int
    ) -> None:
        y, x = position[:2]
        fh, fw = node_data.shape[:2]
        bh, bw = self.background.shape[:2]

        blr = max(0, y - floor(fh / 2))
        bur = min(y + ceil(fh / 2), bh)
        blc = max(0, x - floor(fw / 2))
        buc = min(x + ceil(fw / 2), bw)

        flr = floor(fh / 2) - floor((bur - blr) / 2)
        fur = floor(fh / 2) + ceil((bur - blr) / 2)
        flc = floor(fw / 2) - floor((buc - blc) / 2)
        fuc = floor(fw / 2) + ceil((buc - blc) / 2)

        mask = node_mask[flr: fur, flc: fuc]

        self.background[blr: bur, blc: buc] = cv2.multiply(
            np.stack((1 - mask,) * 3, axis=-1),
            self.background[blr: bur, blc: buc]
        )
        self.background[blr: bur, blc: buc] = cv2.add(
            node_data[flr: fur, flc: fuc],
            self.background[blr: bur, blc: buc]
        )

        # Contour update
        if node_unique_id != 0:   # if the object is not a fake object
            self.contour[blr: bur, blc: buc] = cv2.multiply(
                1 - mask,
                self.contour[blr: bur, blc: buc]
            )
            self.contour[blr: bur, blc: buc] = cv2.add(
                node_mask[flr: fur, flc: fuc] * node_unique_id,
                self.contour[blr: bur, blc: buc]
            )

    def set_foreground_node(self,
                            node_data: np.ndarray,
                            node_mask: np.ndarray,
                            is_fake_object: bool = False
    ) -> None:
        """Set the foregrounds when it is the initial frame of the video.
        """
        # apply transformation
        original_data = node_data.copy()
        original_mask = node_mask.copy()
        node_mask[original_mask > 0] = 1
        original_mask = original_mask.astype(np.uint8)

        # create a new node
        node_id = 0 if is_fake_object else (len(self.foregrounds) + 1)
        node = WheatHeadNode(
            np.array([0, 0]), original_data, original_mask, node_id,
            self.trfms_confs, is_fake_object
        )

        aug = node.rotation_transformer(image=node_data, mask=node_mask)
        aug = node.transformer(image=aug['image'], mask=aug['mask'])
        node_data, node_mask = aug["image"], aug["mask"]

        # if the object is a fake object, then set the non-masked area to 0
            # and that is because of the applied color transformations.
        if node.is_fake is True:
            node_data[node_mask == 0] = 0

        # Prepare the rotation transformer for future use.
        node.nudge_rotation_transformer()

        # find the right position
        y, x = self.find_best_fit(node_mask)
        position = np.array([y, x])

        # Update the node with the new position and add it to the frame list.
        node.set_position(position)
        self.foregrounds.append(node)

        # overlay the wheat head on top of the background
        self.overlay(position, node_data, node_mask, node_id)

    def nudge(self,
              node: WheatHeadNode
    ) -> None:
        """Adjust the coordinate of the wheat head within the original image
            coordinates.
        """
        ns = np.array(node.get_shape())
        bs = np.array(self.contour.shape[:2])

        step = np.random.randint(*self.node_movement_range, 2)
        node.position += node.direction * step
        if random.random() < self.change_direction_p:
            self.direction *= -1

        if (self.allow_leave_frame is False and
            (np.any(node.position < (ns // 2) + 1) or
             np.any(node.position > bs - (ns // 2) - 1))
        ):
            node.position = np.minimum(
                np.maximum(
                    (ns // 2) + 1,
                    node.position
                ),
                bs - (ns // 2) - 1
            )
            node.direction *= -1

    def nudge_nodes(self) -> None:
        """Move the foreground given from the previous frame to a new
            position in the new current frame.
        """
        node = self.foregrounds.get_head()
        while node is not None:
            node_data = node.get_data().copy()
            node_mask = node.get_annotation().copy()
            node_mask[node_mask > 0] = 1
            node_mask = node_mask.astype(np.uint8)

            aug = node.rotation_transformer(image=node_data,
                                            mask=node_mask)
            aug = node.transformer(image=aug['image'], mask=aug['mask'])
            node_data, node_mask = aug["image"], aug["mask"]

            # if the object is a fake object, then set the non-masked area to 0
                # and that is because of the applied color transformations.
            if node.is_fake is True:
                node_data[node_mask == 0] = 0

            # Update the node's rotation transformer
            node.nudge_rotation_transformer()

            if node_mask.sum() <= 0.05 * node.get_annotation().size:
                next_node = node.get_next()
                self.foregrounds.remove(node)
                self.removed_nodes += 1
                node = next_node
                continue

            self.nudge(node)
            position = node.get_position()

            # if random.random() < 0.05: 
            #     node.set_direction(node.direction * -1)

            next_node = node.get_next()
            if self.is_in_frame(position):
                self.overlay(position, node_data, node_mask, node.annot_id)
                node.set_aug_data(node_data)
                node.set_aug_annotation(node_mask)
            else:
                self.foregrounds.remove(node)
                self.removed_nodes += 1
            node = next_node
        self.removed_nodes = 0

    def move_frame(self) -> np.ndarray:
        """Move the whole frame in random directions.
        """
        self.set_background(self.original_background)
        self.set_contour(np.zeros_like(self.contour, dtype=np.uint8))

        if random.random() < self.change_direction_p:
            self.direction *= -1

        movement = np.random.randint(*self.frame_movement_range, 2)
        movement *= self.direction
        node = self.foregrounds.get_head()
        while node is not None:
            # overlay the wheat head on top of the background
            node.position += movement

            next_node = node.get_next()
            if self.is_in_frame(node.get_position()):
                self.overlay(node.get_position(), node.get_aug_data(),
                             node.get_aug_annotation(), node.annot_id
                )
                node.set_aug_data(None)
                node.set_aug_annotation(None)
            else:
                self.foregrounds.remove(node)
                self.removed_nodes += 1
            node = next_node

        return movement

    def recover_lost_nodes(self,
                           movement: np.ndarray,
                           real_recovery_batch: List[np.ndarray],
                           fake_recovery_batch: List[np.ndarray]
    ) -> None:
        """Recover the number of the removed nodes by adding new ones
        """
        num_fakes = random.randint(
            max(0, self.removed_nodes - self.recovery_variance),
            self.removed_nodes
        )
        num_reals = random.randint(
            max(0, self.removed_nodes - self.recovery_variance),
            self.removed_nodes
        )

        nodes2add = [
            fake_recovery_batch[idx]
            for idx in random.sample(range(len(fake_recovery_batch)), num_fakes)
        ] + [
            real_recovery_batch[idx]
            for idx in random.sample(range(len(real_recovery_batch)), num_reals)
        ]
        h_range = (
            0 if movement[0] >= 0
              else self.height + self.recovery_area_expansion * movement[0] - 1,
            self.recovery_area_expansion * movement[0] if movement[0] >= 0
                                                       else self.height
        )
        w_range = (
            0 if movement[1] >= 0
            else self.width + self.recovery_area_expansion * movement[1] - 1,
            self.recovery_area_expansion * movement[1] if movement[1] >= 0
                                                       else self.width
        )

        # unique_ids: Set = self.foregrounds.get_unique_ids()
        # unique_ids: List = list(
        #     set(range(max(unique_ids) + self.removed_nodes)) - unique_ids
        # )
        # unique_ids = sorted(unique_ids)[:num_reals]
        # try:
        #     assert len(unique_ids) >= num_reals
        # except:
        #     ic(unique_ids, num_reals)
        #     exit(0)

        # Add the new nodes to the frame
        for count, obj in enumerate(nodes2add):
            node_data, node_mask = obj[0].copy(), obj[1].copy()
            node_mask[node_mask > 0] = 1
            node_mask = node_mask.astype(np.uint8)

            if count < num_fakes:
                node_unique_id = 0
            else:
                node_unique_id = 1    # unique_ids.pop(0)

            node = WheatHeadNode(
                np.array([0, 0]), node_data, node_mask, node_unique_id,
                self.trfms_confs, is_fake=(count < num_fakes)
            )

            aug = node.rotation_transformer(image=node_data, mask=node_mask)
            aug = node.transformer(image=aug['image'], mask=aug['mask'])
            node_data, node_mask = aug["image"], aug["mask"]

            # if the object is a fake object, then set the non-masked area to 0
            # and that is because of the applied color transformations.
            if node.is_fake is True:
                node_data[node_mask == 0] = 0

            # Prepare the rotation transformer for future use.
            node.nudge_rotation_transformer()

            position = None
            if (count <= num_fakes // 2 or
                num_fakes < count <= num_fakes + num_reals // 2
            ):  # add verticall
                position = np.array(
                    self.find_best_fit(node_mask, h_range, (0, self.width))
                )
            elif (count <= num_fakes or
                  count > num_fakes + num_reals // 2
            ):  # add horizontally
                position = np.array(
                    self.find_best_fit(node_mask, (0, self.height), w_range)
                )

            # Update the node's position and add it to the frame's list of nodes
            node.set_position(position)
            self.foregrounds.append(node)

            self.overlay(position, node_data, node_mask, node_unique_id)

        self.removed_nodes = 0


class VideoSimulation:
    def __init__(self,
                 back_loader: Callable,
                 real_loader: Callable,
                 fake_loader: Callable,
                 video_height: int = 1024,
                 video_width: int = 1024,
                 num_required_frames: int = 600,
                 num_reals_per_frame: Tuple[int, int] = (10, 50),
                 num_fakes_per_frame: Tuple[int, int] = (10, 50),
                 frame_movement_range: Tuple[int, int] = [0, 0],
                 node_movement_range: Tuple[int, int] = [0, 0],
                 recovery_variance: int = 3,
                 trfms_confs: Dict[str, Any] = {},
                 allow_leave_frame: bool = False,
                 recovery_area_expansion: int = 1,
                 change_direction_p: float = 0.1
    ) -> None:
        self.back_loader = back_loader
        self.real_loader = real_loader
        self.fake_loader = fake_loader
        self.video_height = video_height
        self.video_width = video_width
        self.num_required_frames = num_required_frames
        self.num_reals_per_frame = num_reals_per_frame
        self.num_fakes_per_frame = num_fakes_per_frame
        self.frame_movement_range = frame_movement_range
        self.node_movement_range = node_movement_range
        self.recovery_variance = recovery_variance
        self.trfms_confs = trfms_confs
        self.allow_leave_frame = allow_leave_frame
        self.recovery_area_expansion = recovery_area_expansion
        self.change_direction_p = change_direction_p

        self.video: List[FrameSimulation] = []

    def __len__(self) -> int:
        return self.num_required_frames

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.video[index].background, self.video[index].contour

    def get_image(self, index: int) -> np.ndarray:
        return self.video[index].background

    def get_mask(self, index: int) -> np.ndarray:
        return self.video[index].contour

    def __call__(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        image_frames = []
        mask_frames = []
        init_idx = random.randint(
            0, len(self.back_loader) - self.num_required_frames
        )
        reference_frame_simulator = None
        row_crop_lower = random.randint(
            0, self.back_loader[0].shape[0] - self.video_height
        )
        col_crop_lower = random.randint(
            0, self.back_loader[0].shape[1] - self.video_width
        )
        with tqdm(total=self.num_required_frames) as pbar:
            for idx in range(init_idx, init_idx + self.num_required_frames):
                pbar.update(1)
                background = self.back_loader[idx]
                background = background[
                    row_crop_lower:row_crop_lower + self.video_height,
                    col_crop_lower:col_crop_lower + self.video_width,
                    :
                ]

                fake_batch_size = random.randint(*self.num_fakes_per_frame)
                real_batch_size = random.randint(*self.num_reals_per_frame)

                fake_batch = self.fake_loader.get_batch(fake_batch_size)
                real_batch = self.real_loader.get_batch(real_batch_size)

                frame_simulator = FrameSimulation(
                    background=background,
                    foregrounds=WheatHeadChain(),
                    frame_movement_range=self.frame_movement_range,
                    node_movement_range=self.node_movement_range,
                    recovery_variance=self.recovery_variance,
                    allow_leave_frame=self.allow_leave_frame,
                    recovery_area_expansion=self.recovery_area_expansion,
                    change_direction_p=self.change_direction_p,
                    trfms_confs=self.trfms_confs
                )

                if idx == init_idx:
                    for (fobj, fmsk) in fake_batch:
                        frame_simulator.set_foreground_node(
                            fobj, fmsk, is_fake_object=True
                        )
                    frame_simulator.set_contour(
                        np.zeros_like(frame_simulator.get_contour(), dtype=np.uint8)
                    )
                    for (robj, rmsk) in real_batch:
                        frame_simulator.set_foreground_node(
                            robj, rmsk, is_fake_object=False
                        )
                else:
                    frame_simulator.set_nodes(
                        reference_frame_simulator.get_nodes()
                    )
                    frame_simulator.nudge_nodes()
                    if frame_simulator.frame_movement_range[1] > 0:
                        movement = frame_simulator.move_frame()
                    if frame_simulator.removed_nodes > 0:
                        fake_batch = self.fake_loader.get_batch(
                            frame_simulator.removed_nodes +
                            self.recovery_variance
                        )
                        real_batch = self.real_loader.get_batch(
                            frame_simulator.removed_nodes +
                            self.recovery_variance
                        )
                        frame_simulator.recover_lost_nodes(
                            movement, real_batch, fake_batch
                        )
                image = frame_simulator.get_background()
                mask = frame_simulator.get_contour()

                image_frames.append(image)
                mask_frames.append(mask)
                reference_frame_simulator = frame_simulator

        return image_frames, mask_frames


def preview(images: List[Union[np.ndarray, torch.Tensor]],
            masks: List[Union[np.ndarray, torch.Tensor]],
            frames_per_second: int = 60,
            output_file: Optional[str] = None
) -> None:
    video = cv2.VideoWriter(
        output_file if output_file else 'output.mp4',
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
        video.write(
            cv2.cvtColor(
                overlayed_image,
                cv2.COLOR_RGB2BGR
            )
        )
    video.release()


if __name__ == "__main__":
    ic.enable()

    # Input arguments.
    parser = argparse.ArgumentParser(
        description='Initializers parameters for running the experiments.'
    )
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        default='configs/weak.yaml',
                        help='The string path of the config file.'
    )
    args = parser.parse_args()

    configs = Box.from_yaml(filename=args.config_path, Loader=yaml.FullLoader)

    # Set seeds
    random.seed(configs.Basics.SEED)
    np.random.seed(configs.Basics.SEED)
    torch.manual_seed(configs.Basics.SEED)

    video_df = pd.DataFrame()
    vid = 0
    for bid, metadata_path in enumerate(configs.BackgroundMetadata):
        metadata_path = [metadata_path]
        for _ in range(configs.Video.VIDEO_PER_BACKGROUND):
            back_transformer = None
            back_loader = BackgroundDataset(
                metadata_paths=metadata_path,
                transformer=back_transformer
            )

            real_transformer = None
            real_loader = RealObjectDataset(
                metadata_paths=configs.RealObjectMetadata,
                transformer=real_transformer
            )

            fake_transformer = None
            fake_loader = FakePatchDataset(
                metadata_paths=configs.FakePatchMetadata,
                transformer=fake_transformer
            )

            video_simulator = VideoSimulation(
                back_loader=back_loader,
                real_loader=real_loader,
                fake_loader=fake_loader,
                video_height=configs.Video.VIDEO_HEIGHT,
                video_width=configs.Video.VIDEO_WIDTH,
                num_required_frames=configs.Video.NUM_REQUIRED_FRAMES,
                num_reals_per_frame=configs.Frame.NUM_REALS_PER_FRAME,
                num_fakes_per_frame=configs.Frame.NUM_FAKES_PER_FRAME,
                frame_movement_range=configs.Frame.FRAME_MOVEMENT_RANGE,
                node_movement_range=configs.Frame.NODE_MOVEMENT_RANGE,
                recovery_variance=configs.Frame.RECOVERY_VARIANCE,
                trfms_confs=configs.Transformations,
                allow_leave_frame=configs.Frame.ALLOW_LEAVE_FRAME,
                recovery_area_expansion=configs.Frame.RECOVERY_AREA_EXPANSION,
                change_direction_p=configs.Frame.CHANGE_DIRECTION_PROBABILITY
            )
            frames, masks = video_simulator()

            frame_paths = []
            mask_paths = []
            # Save the frames and masks.
            os.makedirs(os.path.join(configs.Basics.OUT_SIMULATION_DIR,
                                     f"VID_{vid:0>5}",
                                     "Frames"),
                        exist_ok=True
            )
            os.makedirs(os.path.join(configs.Basics.OUT_SIMULATION_DIR,
                                     f"VID_{vid:0>5}",
                                     "Masks"),
                        exist_ok=True
            )
            os.makedirs(configs.Basics.VIDEOS_OUT_DIR, exist_ok=True)
            
            for i, (frame, mask) in enumerate(zip(frames, masks)):
                out_frame_path = os.path.join(
                    configs.Basics.OUT_SIMULATION_DIR, f"VID_{vid:0>5}",
                    f"Frames/vid-{vid:0>5}_frame-{i:0>5}.png"
                )
                frame_paths.append(out_frame_path)

                out_mask_path = os.path.join(
                    configs.Basics.OUT_SIMULATION_DIR, f"VID_{vid:0>5}",
                    f"Masks/vid-{vid:0>5}_mask-{i:0>5}.png"
                )
                mask_paths.append(out_mask_path)

                io.imsave(out_frame_path, frame)
                io.imsave(out_mask_path, mask, check_contrast=False)

            frames_length = len(frame_paths)
            video_df = pd.concat([
                video_df,
                pd.DataFrame({
                    'BID': [f"Back_{bid:0>3}"] * frames_length,
                    'VID': [f"{configs.Basics.VIDEOS_IDS_PREFIX}VID_{vid:0>5}"] * frames_length,
                    'FID': np.arange(frames_length).astype(np.int32),
                    'Image': frame_paths,
                    'Mask': mask_paths,
                    'Label': [1] * frames_length
                })
            ], ignore_index=True, axis=0)

            # Visualize the frames and masks.
            if configs.Basics.CREATE_OVERLAID_VIDEO is True:
                preview(
                    frames, masks,
                    frames_per_second=configs.Basics.FRAMES_PER_SECOND,
                    output_file=os.path.join(
                        configs.Basics.VIDEOS_OUT_DIR,
                        f"BID-{bid}_VID-{vid}.mp4"
                    )
                )
            
            vid += 1
            
    video_df = video_df.astype({
        "FID": np.int32, 
        "Label": np.uint8
    })
    video_df["FID"] = np.arange(len(video_df)).astype(np.int32)
    video_df.to_csv(configs.Basics.OUT_METADATA_PATH, index=False)