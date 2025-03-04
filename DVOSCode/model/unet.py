from icecream import ic 
from box import Box
from typing import Tuple, Union, List, Dict

import torch
import torch.nn as nn

from .utils import AttrDict, ACTIVATION, init_module_weights
from . import blocks as BLCS
from . import utils

from torch.nn.parallel import DistributedDataParallel as DDP


class Encoder(nn.Module):
    """Encoder module of the unet.
    Args: 
        init_channels (int): Number of input channels.
        latent_channels (int): Number of channels in the latent space.
        num_res_blocks (int): Number of residual blocks in the model.
        base_channels (int): Number of base channels to build the model based on it. 
        base_channels_multiplier (Union[Tuple, List]): Multiplier for the base channels.
        num_groups (int): Number of groups in the group normalization.
    """
    def __init__(self,
                 init_channels: int = 3,
                 latent_channels: int = 8,
                 num_res_blocks: int = 2,
                 base_channels: int = 8,
                 base_channels_multiplier: Union[Tuple, List] = (1, 2, 4, 8),
                 num_groups: int = 8
    ) -> None:
        super().__init__()
        
        self.input_channels = init_channels
        
        self.backbone = nn.Sequential(
            nn.Conv2d(init_channels, min(num_groups, base_channels),
                      kernel_size=3, stride=1, padding=1
            ),
            BLCS.ResnetBlock(
                in_channels=min(num_groups, base_channels),
                out_channels=base_channels,
                num_groups=num_groups
            ),
            BLCS.ResnetBlock(
                in_channels=base_channels,
                out_channels=base_channels,
                num_groups=num_groups
            )
        )
        # Defining the encoder blocks.
        current_chans = [base_channels]
        in_chans = base_channels
        self.encoder = nn.ModuleList()
        for multiplier in base_channels_multiplier:
            out_chans = base_channels * multiplier
            for _ in range(num_res_blocks):
                self.encoder.append(
                    BLCS.ResnetBlock(
                        in_channels=in_chans,
                        out_channels=out_chans,
                        num_groups=num_groups
                    )
                )
                in_chans = out_chans
            current_chans.append(in_chans)
            self.encoder.append(
                BLCS.ConvDownSample(
                    channels=in_chans, kernel_size=3, stride=2, padding=1, num_groups=num_groups
                )
            )
        # Defining the bottleneck.
        self.bottleneck = BLCS.ResnetBlock(
            in_channels=in_chans,
            out_channels=latent_channels,
            num_groups=num_groups
        )
        
        self.output_channles = latent_channels

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self,
                inputs: torch.Tensor
    ) -> torch.Tensor:
        feature_maps = []
        outputs = self.backbone(inputs)
        for layer in self.encoder:
            if isinstance(layer, (BLCS.ConvDownSample, BLCS.PoolDownSample)):
                feature_maps.append(outputs)
            outputs = layer(outputs)
        outputs = self.bottleneck(outputs)
        return outputs, feature_maps


class Decoder(nn.Module):
    """Decoder module of the unet.
    Args:
        latent_channels (int): Number of channels in the latent space.
        base_channels (int): Number of base channels to build the model based on it. 
        base_channels_multiplier (Union[Tuple, List]): Multiplier for the base channels.
        num_res_blocks (int): Number of residual blocks in the model.
        num_groups (int): Number of groups in the group normalization.
        num_reference_frames (int): Number of reference frames.
        diffusion_params (Dict): Dictionary containing the diffusion parameters.
        skip_dropout_ratio (float): Dropout ratio for the skip connections.
        device (str): Device to run the model.
        is_video_task (boolean): whether it is image- or video-based task. 
    Returns:
        torch.Tensor: The output of the model.
    """
    def __init__(self,
                 latent_channels: int = 512,
                 base_channels: int = 8,
                 base_channels_multiplier: Union[Tuple, List] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 num_groups: int = 8,  
                 num_reference_frames: int = 4,
                 diffusion_params: Dict = {},
                 skip_dropout_ratio: float = 0.0, 
                 device: str = 'cpu', 
                 is_video_task: bool = True
    ):
        super().__init__()
        
        ts_distro_params = [
            (1.0, 40.0)  # Diffusion is close to zero and will not be applied for the latent. 
        ] + [
            (
                diffusion_params.INIT_ALPHA + t, 
                diffusion_params.INIT_BETA - (diffusion_params.BETA_REDUCTION_COEFF * t)
            ) 
            for t in range(len(base_channels_multiplier))
        ]
        dist_a, dist_b = ts_distro_params.pop(0)
        self.bottleneck_diff_atten = BLCS.SkipDiffusionAttention(
            channels=latent_channels * num_reference_frames, 
            channel_reduction_factor=num_reference_frames,
            kernel_size=3, 
            diff_steps=diffusion_params.TIMESTEPS, 
            diff_dist_a=dist_a, 
            diff_dist_b=dist_b, 
            apply_diffuser=False,                # No diffusion at the latent space 
            dropout_p=0.0,             # No dropout at the latent space
            device=device 
        ) if is_video_task else nn.Identity()
        self.bottleneck = BLCS.ResnetBlock(
            in_channels=latent_channels,
            out_channels=base_channels * base_channels_multiplier[-1],
            num_groups=num_groups
        )
        self.decoder = nn.ModuleList()
        in_chans = base_channels * base_channels_multiplier[-1]
        for multiplier in reversed(base_channels_multiplier):
            out_chans = base_channels * multiplier
            dist_a, dist_b = ts_distro_params.pop(0)
            self.decoder.append(  # Attention 
                BLCS.SkipDiffusionAttention(
                    channels=out_chans * num_reference_frames, 
                    channel_reduction_factor=num_reference_frames,
                    kernel_size=3, 
                    diff_steps=diffusion_params.TIMESTEPS, 
                    diff_dist_a=dist_a, 
                    diff_dist_b=dist_b, 
                    apply_diffuser=diffusion_params.APPLY,
                    dropout_p=skip_dropout_ratio, 
                    device=device 
                ) if is_video_task else nn.Identity()
            )
            self.decoder.append(  # Upsample
                BLCS.ConvUpSample(channels=in_chans, num_groups=num_groups)
            )
            for _ in range(num_res_blocks):  # Residual blocks.
                self.decoder.append(
                    BLCS.ResnetBlock(
                        in_channels=in_chans + out_chans,
                        out_channels=out_chans,
                        num_groups=num_groups
                    )
                )
                in_chans = out_chans
        
        # Defining the decoder output channels as a class attribute.
        self.output_channels = out_chans

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)
    
    def forward(self,
                latent: torch.Tensor,
                feature_maps: List[torch.Tensor]
    ) -> torch.Tensor:
        out =  self.bottleneck(
            self.bottleneck_diff_atten(latent)
        )
        skip = None
        for layer in self.decoder:
            if isinstance(layer, (nn.Identity, BLCS.SkipDiffusionAttention)):
                skip = layer(feature_maps.pop())
            if isinstance(layer, (BLCS.ConvUpSample, BLCS.TransConvUpSample)):
                out = layer(out)
            if isinstance(layer, BLCS.ResnetBlock):
                out = torch.cat([out, skip], dim=1)
                out = layer(out)
        return out


class Heads(nn.Module): 
    """Segmentation and Reconstruction head modules of the architecture.
    Args:
        init_channels (int): Number of input channels.
        middle_channels (int): Number of channels in the middle of the model.
        num_groups (int): Number of groups in the group normalization.
    """
    def __init__(self, 
                 init_channels: int, 
                 middle_channels: int,
                 num_groups: int = 8
    ) -> None:
        super().__init__()
        self.input_channels = init_channels
        
        self.rec_head = nn.Sequential(
            BLCS.ResnetBlock(
                in_channels=init_channels,
                out_channels=init_channels,
                num_groups=num_groups
            ),
            BLCS.ResnetBlock(
                in_channels=init_channels,
                out_channels=middle_channels,
                num_groups=num_groups
            ),
            BLCS.ResnetBlock(
                in_channels=middle_channels,
                out_channels=middle_channels//2,
                num_groups=num_groups
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=middle_channels//2),
            BLCS.ACTIVATION(),
            nn.Conv2d(
                in_channels=middle_channels//2,
                out_channels=3,
                kernel_size=3,
                padding=1
            )
        )
        self.seg_head =  nn.Sequential(
            BLCS.ResnetBlock(
                in_channels=init_channels,
                out_channels=init_channels,
                num_groups=num_groups
            ),
            BLCS.ResnetBlock(
                in_channels=init_channels,
                out_channels=middle_channels,
                num_groups=num_groups
            ),
            BLCS.ResnetBlock(
                in_channels=middle_channels,
                out_channels=middle_channels//2,
                num_groups=num_groups
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=middle_channels//2),
            BLCS.ACTIVATION(),
            nn.Conv2d(
                in_channels=middle_channels//2,
                out_channels=1,
                kernel_size=3,
                padding=1
            )
        )
        
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)
    
    def forward(self,
                inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rec_outputs = self.rec_head(inputs)
        seg_outputs = self.seg_head(inputs)
        return rec_outputs, seg_outputs 
    

class UNet(nn.Module):
    """The unified model of the architecture.
    Args:
        encoder (Encoder): The encoder module of the architecture.
        decoder (Decoder): The decoder module of the architecture.
        heads (Heads): The heads module of the architecture.
    """
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 heads: Heads            
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads = heads
          
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)      

    def forward(self,
                inputs: List[torch.Tensor]
    ) -> torch.Tensor:
        latents, skips = [], []
        for sample in inputs:
            latent, skip = self.encoder(sample)
            latents.append(latent)
            skips.append(skip)
        latents = torch.cat(latents, dim=1)
        skips = [
            torch.cat([
                skips[i][j]
                for i in range(len(skips))
            ], dim=1)
            for j in range(len(skips[0]))
        ]
        
        decoder_out = self.decoder(latents, skips)
        
        rec_out, seg_out = self.heads(decoder_out)
        return rec_out, seg_out


def get_model(rank: int,
              configs: Union[Dict, AttrDict, Box],
              wandb_logger: utils.WeightAndBiases = None
) -> nn.Module:
    device = torch.device(f"cuda:{rank}") if isinstance(rank, int) else torch.device(rank)
            
    # ------------------- #
    encoder = Encoder(
        init_channels=configs.ModelConfig.INIT_CHANNELS,
        latent_channels=configs.ModelConfig.LATENT_CHANNELS,
        base_channels=configs.ModelConfig.BASE_CHANNELS,
        base_channels_multiplier=configs.ModelConfig.BASE_CHANNELS_MULTIPLIER,
        num_res_blocks=configs.ModelConfig.NUM_RES_BLOCKS,
        num_groups=configs.ModelConfig.NUM_GROUPS
    )
    encoder.unfreeze()
    if configs.ModelConfig.FREEZE_ENCODER is True:
        print("freeze the encoder".upper())
        encoder.freeze()
    
    # ------------------- #
    decoder = Decoder(
        latent_channels=configs.ModelConfig.LATENT_CHANNELS,
        base_channels=configs.ModelConfig.BASE_CHANNELS,
        base_channels_multiplier=configs.ModelConfig.BASE_CHANNELS_MULTIPLIER,
        num_res_blocks=configs.ModelConfig.NUM_RES_BLOCKS,
        num_groups=configs.ModelConfig.NUM_GROUPS, 
        num_reference_frames=configs.TrainConfig.NUM_REFERENCES,
        diffusion_params=configs.TrainConfig.Diffusion,
        skip_dropout_ratio=configs.ModelConfig.SKIP_DROPOUT_RATIO, 
        device=device, 
        is_video_task=configs.ModelConfig.IS_VIDEO_TASK, 
    )
    if configs.ModelConfig.FREEZE_DECODER is True:
        print("freeze the decoder".upper())
        decoder.freeze()
    
    # ------------------- #
    heads = Heads(
        init_channels=decoder.output_channels,
        middle_channels=configs.ModelConfig.Heads_MIDDLE_CHANNELS, 
        num_groups=configs.ModelConfig.NUM_GROUPS
    )
    if configs.ModelConfig.FREEZE_Heads is True:
        print("freeze the heads".upper())
        heads.freeze()
    if configs.ModelConfig.FREEZE_REC_HEAD is True:
        print("freeze the reconstruction head".upper())
        for param in heads.rec_head.parameters():
            param.requires_grad_(False)
    if configs.ModelConfig.FREEZE_SEG_HEAD is True:
        print("freeze the segmentation head".upper())
        for param in heads.seg_head.parameters():
            param.requires_grad_(False)
    # ------------------- #
    model = UNet(
        encoder=encoder,
        decoder=decoder,
        heads=heads
    )
    
    # ------------------- #
    if (configs.ModelConfig.PRETRAINED_MODEL_PATH is None and 
        configs.ModelConfig.INIT_MODEL is True
    ): 
        print("initialize the model".upper())
        model.apply(init_module_weights)
    
    # ------------------- #
    if configs.ModelConfig.PRETRAINED_MODEL_PATH is not None:
        checkpoint = torch.load(
            configs.ModelConfig.PRETRAINED_MODEL_PATH,
            map_location=torch.device('cpu')
        )
        model.load_state_dict(checkpoint)
        print("=" * 50)
        print(
            "using pretrained model loaded from {}".format(
                configs.ModelConfig.PRETRAINED_MODEL_PATH
            ).upper()
        )
        print("=" * 50)
    
    # ------------------- #
    model = model.to(rank)   
    if configs.TrainConfig.DISTRIBUTED:
        model = DDP(model, device_ids=[rank])
                
    # ------------------- #
    if wandb_logger is not None:
        wandb_logger.watch(model)
    
    
    return model
