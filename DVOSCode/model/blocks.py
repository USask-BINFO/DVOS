import random

from icecream import ic
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn

from .utils import ACTIVATION


class ResnetBlock(nn.Module):
    """Resnet block with group norm and predefined activation
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        num_groups: number of groups for GroupNorm
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int = 8
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6),
            ACTIVATION(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6),
            ACTIVATION(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            )

    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        outputs = inputs
        outputs = self.block(outputs)
        outputs = outputs + self.shortcut(inputs)
        return outputs


class ConvDownSample(nn.Module):
    """Down sampling block with 2D convolution.
    Args:
        channels: number of input and output channels
        kernel_size: kernel size for 2D convolution
        stride: stride for 2D convolution
        padding: padding for 2D convolution, 
        num_groups: The GroupNorm parameter.
    """
    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[str, int, Tuple[int, int]] = 1, 
                 num_groups: int = 8
    ) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6),
            ACTIVATION(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False
            )
        )
        
    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.downsample(inputs)


class PoolDownSample(nn.Module):
    """Down sampling block with max-pooling.
    Args:
        channels: number of input and output channels
        kernel_size: kernel size for 2D convolution
        stride: stride for 2D convolution
        padding: padding for 2D convolution
    """
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = 2,
                 stride:  Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 0, 
                 dilation: Union[int, Tuple[int, int]] = 1,
                 return_indices: bool = False
    ) -> None:
        super().__init__()
        self.downsample = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, 
            padding=padding, dilation=dilation, 
            return_indices=return_indices
        )

    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.downsample(inputs)


class ConvUpSample(nn.Module):
    def __init__(self,
                 channels: int,
                 scale_factor: int = 2,
                 mode: str = 'nearest',
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 1, 
                 num_groups: int = 8
    ) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6),
            ACTIVATION(),
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(channels, channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
        )

    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.upsample(inputs)


class TransConvUpSample(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[str, int, Tuple[int, int]] = 1,
                 output_padding: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            channels, channels, kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=False
        )

    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.upsample(inputs)


class UniformUpsampling(nn.Module):
    """ an upsamling module maps all the input tensors to the same output shape. 
    Args: 
        out_dim (int, sequence): the size of the desiered output shape. 
        upsampling_method (string): The upsampling algorithm. Options are 
            [`nearest`, `linear`, `bilinear`, `bicubic`, `trilinear`]. 
            Default set to `nearest`. 
    """
    def __init__(self, 
                   out_dim: Union[int, List[int], Tuple[int]], 
                   upsampling_method: 'str' = 'bilinear'
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.upsampling_method = upsampling_method
        
        self.upsample = nn.Upsample(
            size=self.out_dim, 
            mode=self.upsampling_method
        )
       
    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.upsample(inputs)
   
   
class SpatioTemporalExcitationAttention(nn.Module):
    """Spatiotemporal excitation block.
    Args:
        channels: number of base channels
        kernel_size: kernel size for 2D convolution
    """
    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 num_groups: Optional[int]=None
    ) -> None:
        super().__init__()
        self.norm_act = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6),
            ACTIVATION(),
        )
        # Spatial depth-wise convolution with one 1x1 convolution. (SE-Net)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size,
                      dilation=1, padding=1, groups=num_groups, bias=False
            ),  # depth-wise convolution
            nn.Conv2d(channels, channels, kernel_size=1,
                      bias=False
            ),  # point-wise convolution
            nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6),
            ACTIVATION(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size,
                      dilation=2, padding=(2 * (kernel_size - 1)) // 2
            ),  # Dilation convolution
            nn.Conv2d(channels, channels, kernel_size=1,
                      dilation=1, padding=0, groups=num_groups, bias=False
            ),  # 1x1 convolution
            nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6),
            nn.Sigmoid()
        )
        self.temporal = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(channels, channels // 2, bias=False),
            nn.LayerNorm(channels // 2),
            ACTIVATION(),
            nn.Linear(channels // 2, channels, bias=False),
            nn.LayerNorm(channels),
            nn.Sigmoid()
        )

    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        inputs = self.norm_act(inputs)
        spatial_attention = self.spatial(inputs)
        temporal_attention = self.temporal(inputs)
        temporal_attention = temporal_attention.view(
            temporal_attention.shape[0], temporal_attention.shape[1], 1, 1
        )
        return (temporal_attention * spatial_attention) * inputs
    

class DiffusionLayer(nn.Module): 
    """Diffusion sampling class.
    Args: 
        steps (int): Number of diffusion timesteps, default to 1000.
        beta_dist_a (float): The Beta distribution parameter `a`. 
        beta_dist_b (float): the Beta distribution parameter `b`. 
        p (float): The probability of adding or not adding any noise. 
        device (str): Device to use, default to 'cpu'.  
    """
    def __init__(self,
                 steps: int = 1000,
                 beta_dist_a: float = 1.0, 
                 beta_dist_b: float = 1.0, 
                 device: str = 'cpu'
    ):
        super().__init__()
        
        self.steps = steps
        self.beta_dist_a = beta_dist_a
        self.beta_dist_b = beta_dist_b
        self.device = device
        
        self.beta_distribution = torch.distributions.Beta(
            self.beta_dist_a, self.beta_dist_b
        )

        self.inititialize()

    def inititialize(self):
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alhpa = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self
    ) -> torch.Tensor:
        scale = 1000 / self.steps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.steps,
            dtype=torch.float32,
            device=self.device
        )

    def get_alpha_cumulative(self, 
                             ts: torch.Tensor
    ) -> torch.Tensor:
        """Get the value at index position "ts" in "element" and
            reshape it to have the same dimension as a batch of images (B, C, H, W)
        """
        return self.sqrt_alpha_cumulative.gather(
            -1, ts
        ).view(-1, 1, 1, 1)
        
    def get_sqrt_one_minus_alpha_cumulative(self,
                                            ts: torch.Tensor 
    ) -> torch.Tensor: 
        """Get the value at index position "ts" in "element" and
            reshape it to have the same dimension as a batch of images (B, C, H, W)
        """
        return self.sqrt_one_minus_alpha_cumulative.gather(
            -1, ts
        ).view(-1, 1, 1, 1)
    
    def generate_ts(self, 
                    size: int
    ) -> torch.Tensor:
        if isinstance(size, (int, float)): 
            size = (int(size), )
        ts = self.beta_distribution.sample(size) 
        ts = ts * self.steps
        return ts.long()
    
    def forward(self, 
                inputs: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.randn_like(inputs, device=inputs.device)
        ts = self.generate_ts(inputs.shape[0])
        mean = self.get_alpha_cumulative(
            ts.to(self.sqrt_alpha_cumulative.device)
        ).to(inputs.device) * inputs
        std = self.get_sqrt_one_minus_alpha_cumulative(
            ts.to(self.sqrt_one_minus_alpha_cumulative.device)
        ).to(inputs.device)
        outputs = mean + std * eps
        return outputs


class SkipDiffusionAttention(nn.Module): 
    """A diffusion-based layer added to skip connection, 
        along with a attention module to reduce dimentionality and 
        capture the spatio-temporal features.
    Args: 
        channels (int): The number of input channels (skip connection out channels). 
        channel_reduction_factor: out_channels // in_channels. 
        kernel_size: the size of convolution kernels in the attention module. 
    """
    def __init__(self, 
                 channels: int, 
                 channel_reduction_factor: int,
                 kernel_size: int = 3, 
                 diff_steps: int = 1000, 
                 diff_dist_a: float = 1.0, 
                 diff_dist_b: float = 1.0, 
                 dropout_p: float = 0.0, 
                 apply_diffuser: bool = False,
                 device: str = 'cpu' 
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.apply_diffuser = apply_diffuser
        
        self.attention = SpatioTemporalExcitationAttention(
            channels=channels,
            kernel_size=kernel_size, 
            num_groups=channel_reduction_factor
        )
        self.reduction = nn.AvgPool3d(
            kernel_size=(channel_reduction_factor, 1, 1),
            stride=(channel_reduction_factor, 1, 1)
        )
        self.diffuser = DiffusionLayer(
            diff_steps, diff_dist_a, diff_dist_b, device
        )
        self.dropout = nn.Dropout2d(
            p=dropout_p
        )
        
    def forward(self, 
                inputs: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.attention(inputs)
        outputs = self.reduction(outputs)
        if self.apply_diffuser: 
            outputs = self.diffuser(outputs)
        outputs = self.dropout(outputs)
        return outputs
