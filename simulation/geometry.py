import random
from typing import Tuple, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from icecream import ic
from weaksimulationpipe import WheatHeadNode


class Coordination:
    """Coordination class to store x and y coordination.
    y: represents the number of rows or height.
    x: represents the number of columns or width.
    window: represents the window radius with the center of the coordination.
    """
    def __init__(self,
                 height: int,
                 width: int,
                 window_init_range: Tuple[int, int] = [512, 512],
                 max_step_size: int = 10,
                 allow_updating_coordinates: bool = False,
                 allow_updating_step: bool = False,
                 random_seed: Optional[int] = None
    ) -> None:
        self.height = height
        self.width = width
        self.window_size_range = window_init_range
        self.max_step_size = max_step_size
        self.allow_updating_coordinates = allow_updating_coordinates
        self.allow_updating_step = allow_updating_step
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        self.__init_coordinates()

    def __init_coordinates(self) -> None:
        self.window = random.randint(
            self.window_size_range[0],
            self.window_size_range[1]
        ) // 2
        self.x = random.randint(self.window, self.width - self.window - 1)
        self.y = random.randint(self.window, self.height - self.window - 1)
        self.x_direction = random.choice([-1, 1])
        self.y_direction = random.choice([-1, 1])
        self.step = random.randint(1, self.max_step_size)

    def update(self) -> None:
        if self.allow_updating_coordinates:
            direction_changed: bool = False
            # Update the position
            if ((self.y + self.step * self.y_direction) - self.window > 0 and
                (self.y + self.step * self.y_direction) + self.window < self.height
            ):
                self.y += self.step * self.y_direction
            else:
                self.y_direction *= -1
                direction_changed = True

            if ((self.x + self.step * self.x_direction) - self.window > 0 and
                (self.x + self.step * self.x_direction) + self.window < self.width
            ):
                self.x += self.step * self.x_direction
            else:
                self.x_direction *= -1
                direction_changed = True

            # Update the direction
            if not direction_changed and random.random() < 0.05:
                if random.random() <= 0.5:
                    self.y_direction *= -1
                else:
                    self.x_direction *= -1

    def update_step(self) -> None:
        if self.allow_updating_step:
            if self.step >= self.max_step_size or random.random() < 0.1:
                self.step = random.randint(1, self.max_step_size)
            elif random.random() <= 0.5:
                self.step = max(0, self.step + random.choice([-1, 0, 1]))

    def reset(self) -> None:
        """Reset the coordination."""
        self.__init_coordinates()


class Orientation:
    def __init__(self,
                 init_range: Tuple[int, int] = (0, 360),
                 max_adjustment: int = 5,
                 allow_updating_angle: bool = False,
                 random_seed: Optional[int] = None
    ) -> None:
        self.init_range = init_range
        self.max_adjustment = max_adjustment
        self.allow_updating_angle = allow_updating_angle

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.__init_angle()
        self.__init_direction()

    def __init_angle(self) -> None:
        self.angle = random.randint(
            self.init_range[0],
            self.init_range[1]
        )

    def __init_direction(self) -> None:
        self.direction = random.choice([-1, 1])

    def update(self) -> None:
        if self.allow_updating_angle:
            if random.random() < 0.05:
                self.direction *= -1

            if random.random() < 0.1:
                interval = random.randint(0, self.max_adjustment)
            elif random.random() < 0.0:
                interval = 0
            else:
                interval = 1
            self.angle += self.direction * interval

            if self.angle <= self.init_range[0] or self.angle >= self.init_range[1]:
                self.angle = min(
                    max(self.init_range[0], self.angle),
                    self.init_range[1]
                )
                self.direction *= -1


class Transformations:
    def __init__(self,
                 use_rotation: bool = False,
                 use_elastic: bool = False,
                 use_center_crop: bool = False,
                 cc_refinement_upper_bound: int = 10,
                 use_resize: bool = False,
                 random_seed: Optional[int] = None
    ) -> None:
        self.use_rotation = use_rotation
        self.use_elastic = use_elastic
        self.use_center_crop = use_center_crop
        self.cc_refinement_upper_bound = cc_refinement_upper_bound
        self.cc_adjustment = None
        self.cc_direction = None
        self.use_resize = use_resize
        self.random_seed = random_seed

        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        self.__reset()
        self.transformers = None

    def __reset(self) -> None:
        self.trfms_list = []

    def __init_rotation(self,
                        angle: int
    ) -> None:
        self.trfms_list.append(
            A.Rotate(
                limit=(angle, angle), interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101, value=0, mask_value=0,
                p=0.8
            )
        )

    def __init_elastic(self) -> None:
        self.trfms_list.append(
            A.ElasticTransform(
                alpha=2, sigma=10, alpha_affine=20,
                interpolation=1, border_mode=3, value=0,
                approximate=False, p=0.5
            )
        )

    def __init_center_crop(self,
                           height: int,
                           width: int
    ) -> None:
        if self.cc_adjustment is None or random.random() < 0.01:
            self.cc_adjustment = random.randint(
                0, self.cc_refinement_upper_bound
            )
            self.cc_direction = random.choice([-1, 1])
        else:
            self.cc_adjustment += self.cc_direction * random.randint(0, 1)
        if self.cc_adjustment < 0 or self.cc_adjustment > height:
            self.cc_adjustment = min(max(0, self.cc_adjustment), height)
            self.cc_direction *= -1
        if self.cc_adjustment < 0 or self.cc_adjustment > width:
            self.cc_adjustment = min(max(0, self.cc_adjustment), width)
            self.cc_direction *= -1

        self.trfms_list.append(
            A.CenterCrop(
                height=height - self.cc_adjustment,
                width=width - self.cc_adjustment,
                p=1.0
            )
        )

    def __init_resize_to_original(self,
                                  height: int,
                                  width: int
    ) -> None:
        self.trfms_list.append(
            A.Resize(height=height, width=width, p=1.0)
        )

    def update(self,
               angle: int,
               height: int,
               width: int
    ) -> None:
        self.__reset()
        if self.use_rotation:
            self.__init_rotation(angle)
        if self.use_elastic:
            self.__init_elastic()
        if self.use_center_crop:
            self.__init_center_crop(height, width)
        if self.use_resize:
            self.__init_resize_to_original(height, width)

        self.transformers = A.Compose(self.trfms_list)