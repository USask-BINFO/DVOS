import gc
import os
from collections import Counter
from enum import Enum
from icecream import ic
from typing import Dict, Union, List, Tuple, Optional, Literal
from omegaconf.dictconfig import DictConfig

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torchmetrics
from pytorch_msssim import MS_SSIM
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .utils import AttrDict


class ClassificationLoss(torch.nn.Module):
    """Calculate loss by combining different losses for a batch of data.
    Args:
        task (str): the task of the classification mode.
            `binary` or `multi` mode is used to preprocess the model output for
            caluclating the loss.
    Return:
        loss (Tensor): the torch.Tensor calculated loss.
    """

    def __init__(self,
                 task: str = 'multiclass'
    ) -> None:
        super(ClassificationLoss, self).__init__()

        self.task = task
        if self.task == 'multiclass':
            self.ce = torch.nn.CrossEntropyLoss()
        elif self.task == 'binary':
            self.ce = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Only `binary` or `multiclass` classification task modes are supported.')

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = preds.squeeze()
        if self.task == 'binary':
            labels = labels.float()
        loss = self.ce(preds, labels)
        return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self,
                 tau: float = 0.1
    ) -> None:
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self,
                pos_features: torch.Tensor,
                neg_features: torch.Tensor
    ) -> torch.Tensor:
        pos_features = torch.div(pos_features, torch.norm(pos_features, dim=1, keepdim=True))
        neg_features = torch.div(neg_features, torch.norm(neg_features, dim=1, keepdim=True))
        cat12_feat = torch.cat([pos_features, neg_features], dim=0)
        cat21_feat = torch.cat([neg_features, pos_features], dim=0)
        similarities = torch.exp(torch.mm(cat12_feat, cat12_feat.t()) / self.tau)
        rows_sum = torch.sum(similarities, dim=1)
        diag = torch.diag(similarities)
        numerators = torch.exp(
                torch.div(
                    torch.nn.CosineSimilarity()(cat12_feat, cat21_feat),
                    self.tau
                )
        )
        denominators = rows_sum - diag
        contrastive_loss = torch.mean(
                    -1 * torch.log(
                            torch.div(numerators, denominators)
                    )
        )
        return contrastive_loss


class TverskyLoss(torch.nn.Module):
    def __init__(self, 
                 sigmoid: bool = True, 
                 threshold: float = 0.5,
                 alpha: float = 0.5, 
                 beta: float = 0.5, 
                 smooth: float = 1e-12
    ) -> None:
        """
        Args:
            sigmoid (bool): Whether to apply sigmoid activation to the output.
            alpha (float): Weighting factor for false positives.
            beta (float): Weighting factor for false negatives.
            smooth (float): Smoothing constant to avoid division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, 
                output: torch.Tensor, 
                target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky loss between y_pred and y_true.

        Args:
            output (torch.Tensor): Predicted probabilities (after sigmoid for binary tasks).
            target (torch.Tensor): Ground truth binary mask (same size as y_pred).
        
        Returns:
            torch.Tensor: Computed Tversky Loss.
        """
        if self.sigmoid:
            y_pred = torch.ge(
                torch.sigmoid(output), 
                self.threshold
            ).float()
        else:
            y_pred = output
        y_true = target.float()
        
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        TP = (y_true * y_pred).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky_index


class FocalLoss(torch.nn.Module):
    def __init__(self, 
                 sigmoid: bool = True, 
                 gamma: float = 0.1
    ) -> None:
        """
        Args:
            sigmoid (bool): Whether to apply sigmoid activation to the output.
            gamma (float): Focusing parameter for hard examples.When gamma is 0, it is equivalent to the binary cross-entropy loss.
        """
        super(FocalLoss, self).__init__()
        self.sigmoid = sigmoid
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, 
                output: torch.Tensor, 
                target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss between y_pred and y_true.

        Args:
            output (torch.Tensor): Predicted probabilities (after sigmoid for binary tasks).
            target (torch.Tensor): Ground truth binary mask (same size as y_pred).
        
        Returns:
            torch.Tensor: Computed Focal Loss.
        """      
        if self.sigmoid:
            y_pred = torch.sigmoid(output)
        else:
            y_pred = output
        y_true = target.float()
        
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        bce_loss = self.ce(y_pred, y_true)
        focal_loss = torch.pow(1 - y_pred, self.gamma) * bce_loss
        
        return focal_loss.mean()
    

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, 
                 sigmoid: bool = True, 
                 threshold: float = 0.5,
                 alpha: float = 0.5, 
                 beta: float = 0.5, 
                 gamma: float = 1.0, 
                 smooth: float = 1e-12
    ) -> None:
        """
        Args:
            sigmoid (bool): Whether to apply sigmoid activation to the output.
            alpha (float):  Weighting factor for false positives.
            beta (float):   Weighting factor for false negatives.
            gamma (float):  Focusing parameter for hard examples.
            smooth (float): Smoothing constant to avoid division by zero.
        """
        super(FocalTverskyLoss, self).__init__()
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, 
                output: torch.Tensor, 
                target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Tversky loss between y_pred and y_true.

        Args:
            output (torch.Tensor): Predicted probabilities (after sigmoid for binary tasks, or softmax for multi-class).
            target (torch.Tensor): Ground truth binary mask (same size as y_pred).
        
        Returns:
            torch.Tensor: Computed Focal Tversky Loss.
        """
        if self.sigmoid:
            y_pred = torch.ge(
                torch.sigmoid(output), 
                self.threshold
            ).float()
        else:
            y_pred = output
        y_true = target.float()
        
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        TP = (y_true * y_pred).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky_loss = (1 - tversky_index) ** self.gamma
        
        return focal_tversky_loss
    

class IoULoss(nn.Module):
    def __init__(self, 
                 sigmoid: bool = True, 
                 threshold: float = 0.5,
                 smooth: float = 1e-6
    ) -> None:
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(IoULoss, self).__init__()
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, 
                output: torch.Tensor,
                target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU Loss between y_pred and y_true.

        Args:
            output (torch.Tensor): Predicted probabilities (after sigmoid for binary tasks).
            target (torch.Tensor): Ground truth binary mask (same size as y_pred).
        
        Returns:
            torch.Tensor: Computed IoU Loss.
        """
        if output.shape != target.shape:
            raise ValueError("output and targets should have same shapes.")
        if len(output.shape) != 4:
            raise ValueError("targets and output shape should be of length 4.")
        if self.sigmoid:
            y_pred = torch.ge(
                torch.sigmoid(output), 
                self.threshold
            ).float()
        else:
            y_pred = output.float()
        y_true = target.float()
        
        n_len = len(y_pred.shape)
        reduce_axis = list(range(1, n_len))
        
        intersection = torch.sum(
            torch.logical_and(y_true, y_pred), dim=reduce_axis
        )
        union = torch.sum(
            torch.logical_or(y_true, y_pred), dim=reduce_axis
        )
        IoU = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - IoU.mean()

 
class DiceLoss(torch.nn.Module):
    def __init__(self, 
                 sigmoid: bool = True,
                 threshold: float = 0.5,
                 smooth: float = 1e-6
    ) -> None:
        super(DiceLoss, self).__init__()
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.smooth = smooth
        
    def forward(self, 
                output: torch.Tensor,
                target: torch.Tensor
    ) -> torch.Tensor:
        if output.shape != target.shape:
            raise ValueError("output and targets should have same shapes.")
        if len(output.shape) != 4:
            raise ValueError("targets and output shape should be of length 4.")
        if self.sigmoid:
            y_pred = torch.ge(
                torch.sigmoid(output), 
                self.threshold
            ).float()
        else:
            y_pred = output.float()
        y_true = target.float()
        
        n_len = len(y_pred.shape)
        reduce_axis = list(range(1, n_len))
        
        intersection = torch.sum(
            torch.logical_and(y_true, y_pred), dim=reduce_axis
        )
        true_segment = torch.sum(y_true, dim=reduce_axis)
        pred_segment = torch.sum(y_pred, dim=reduce_axis)
        dice = (2.0 * intersection + self.smooth) / (true_segment + pred_segment + self.smooth)
        
        return 1 - dice.mean()
    

class SegmentationLossFunctions:
    def __init__(self, 
                 sigmoid: bool = True,
                 threshold: float = 0.5,
                 smooth: float = 1e-12, 
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 0.1
    ) -> None:
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(sigmoid=sigmoid, threshold=threshold, smooth=smooth)
        self.iou = IoULoss(sigmoid=sigmoid, threshold=threshold, smooth=smooth)
        self.tversky = TverskyLoss(sigmoid=sigmoid, threshold=threshold, smooth=smooth)
        self.focal = FocalLoss(sigmoid=sigmoid, gamma=gamma)
        self.ftversky = FocalTverskyLoss(sigmoid=sigmoid, threshold=threshold, alpha=alpha, 
                                         beta=beta, gamma=gamma, smooth=smooth)
        
    def __getitem__(self,
                    item: str
    ) -> torch.nn.Module:
        match item.upper():
            case 'BCE':
                return self.bce
            case 'DICE':
                return self.dice
            case 'IOU':
                return self.iou
            case 'TV':
                return self.tversky
            case 'FOCAL':
                return self.focal
            case 'FTVERS':
                return self.ftversky
            case default:
                raise ValueError(f'Loss function `{item}` is not supported.')


class SegmentationLoss(torch.nn.Module):
    def __init__(self,
                 loss_names: Union[Tuple[str], List[str]] = ['BCE', 'DICE', 'IOU', 'TV', 'FOCAL', 'FTVERS'],
                 loss_weights: Union[Tuple[float], List[float]] = None,
                 sigmoid: bool = True,
                 threshold: float = 0.5,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 1.0,
                 smooth: float = 1e-12,
                 device: torch.device = torch.device('cpu')
    ) -> None:
        super(SegmentationLoss, self).__init__()
        self.loss_names = loss_names
        self.device = device
        self.loss_functions = SegmentationLossFunctions(
            sigmoid=sigmoid,
            threshold=threshold,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth
        )
        if loss_weights is None:
            loss_weights = [1.0 for _ in loss_names]
        self.loss_weights = loss_weights
        if len(loss_names) != len(loss_weights):
            raise ValueError("loss_names and loss_weights should have same length.")
        
    def forward(self, 
                output: torch.Tensor, 
                target: torch.Tensor
    ) -> torch.Tensor:
        target = target.float()
        loss = self.loss_weights[0] * self.loss_functions[self.loss_names[0]](output, target)
        for name, weight in zip(self.loss_names[1:], self.loss_weights[1:]):
            loss += weight * self.loss_functions[name](output, target)
        return loss
    
    def __repr__(self):
        return f"{self.__class__.__name__}(loss_names={self.loss_names}, device={self.device})"


class PerceptualLoss(torch.nn.Module):
    def __init__(self, 
                 device: str = 'cpu'
    ) -> None: 
        super().__init__()
        self.per_net = models.vgg19(weights="DEFAULT").features
        self.per_net = self.per_net.to(device)
        self.per_net.requires_grad = False
        self.per_net.eval()
        self.perceptual_loss_func = torch.nn.MSELoss(reduction="mean")

    def forward(self, 
                prediction: torch.Tensor, 
                groundtruth: torch.Tensor
    ) -> torch.Tensor:
        prediction_features = self.per_net(prediction)
        with torch.no_grad():
            groundtruth_features = self.per_net(groundtruth)
        perceptual_loss = self.perceptual_loss_func(
            prediction_features, 
            groundtruth_features
        )
        return perceptual_loss
    

class GradientDifferenceLoss(torch.nn.Module):
    def __init__(self, 
                 alpha: int = 1,
                 reduction: str = "SUM"
    ) -> None:
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def _compute_gradients(self, 
                           images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute gradients along x and y directions for images of size (B, C, H, W)
        grad_y = images.diff(dim=2)
        grad_x = images.diff(dim=3)
        
        # Pad to maintain same dimensions
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0)
        
        return grad_y, grad_x

    def forward(self, 
                prediction: torch.Tensor, 
                groundtruth: torch.Tensor
    ) -> torch.Tensor:  
        grad_y_prd, grad_x_prd = self._compute_gradients(prediction)
        grad_y_grd, grad_x_grd = self._compute_gradients(groundtruth)
        
        grad_diff_y = torch.abs(grad_y_prd - grad_y_grd).pow(self.alpha)
        grad_diff_x = torch.abs(grad_x_prd - grad_x_grd).pow(self.alpha)
        
        if self.reduction == "SUM":
            loss = grad_diff_y.sum() + grad_diff_x.sum()
        elif self.reduction == "MEAN":
            total_pixels = prediction.numel()
            loss = (grad_diff_y.sum() / total_pixels) + (grad_diff_x.sum() / total_pixels)
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")
        
        return loss
    

class ReconstructionLossFunctions:
    """A class to handle the reconstruction loss functions.
    Args:
        device: The device to use for the loss functions.
    """
    def __init__(self,
                 alpha: int = 1,
                 reduction: str = "SUM",
                 device: torch.device = torch.device('cpu')
    ) -> None:
        self.device = device
        self.mse = torchmetrics.regression.MeanSquaredError().to(self.device)
        self.mae = torchmetrics.regression.MeanAbsoluteError().to(self.device)
        self.ssim = MS_SSIM(size_average=True, channel=3).to(self.device)
        self.perceptual = PerceptualLoss(device=self.device).to(self.device)
        self.gdl = GradientDifferenceLoss(alpha=alpha, reduction=reduction).to(self.device)
        
    def __getitem__(self,
                 loss_name: str
    ) -> torch.nn.Module:
        match loss_name.upper():
            case 'MSE':
                return self.mse
            case 'MAE':
                return self.mae
            case 'SSIM':
                return self.ssim
            case 'PERCEPTUAL':
                return self.perceptual
            case 'GDL':
                return self.gdl
            case default:
                raise ValueError(f'Loss {loss_name} not implemented.')


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, 
                 loss_names: Union[Tuple[str], List[str]] = ["MSE", "MAE", "SSIM", "PERCEPTUAL", "GDL"],
                 loss_weights: Union[Tuple[float], List[float]] = None,
                 alpha: int = 1,
                 reduction: str = "SUM",
                 device: torch.device = torch.device('cpu')
    ) -> None:
        super().__init__()
        self.loss_names = loss_names  
        if loss_weights is None:
            loss_weights = [1.0 for _ in loss_names]
        self.loss_weights = loss_weights
        self.device = device
        self.loss_functions = ReconstructionLossFunctions(alpha=alpha, reduction=reduction, device=self.device)
        
    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor
    ) -> torch.Tensor:
        loss_value = self.loss_weights[0] * self.loss_functions[self.loss_names[0]](output, target)
        for name, weight in zip(self.loss_names[1:], self.loss_weights[1:]):
            loss_value += weight * self.loss_functions[name](output, target)
        return loss_value
    
    def __repr__(self):
        return f"{self.__class__.__name__}(loss_names={self.loss_names}, device={self.device})"


class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, 
                 batch_size, 
                 lambda_coeff=5e-3
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff 
        
    def off_diagonal(self, 
                     z: torch.Tensor
    ) -> torch.Tensor:
        n, m = z.shape
        assert n == m
        return z.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, 
                z1: torch.Tensor, 
                z2: torch.Tensor
    ) -> torch.Tensor:
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


class AdversarialLoss(torch.nn.Module): 
    def __init__(self,
                 rank: Union[str, int, torch.device],
                 configs: Union[Dict, AttrDict, DictConfig],
                 task: str = None, 
                 pretraind_path: str = None
    ) -> None: 
        super().__init__()
        self.rank = rank 
        self.task = task 
        self.pretraind_path = pretraind_path
        
        self.gen_loss_fn = {
            'RECONSTRUCTION': ReconstructionLoss(
                loss_names=configs.TrainConfig.Losses.IMAGE_LOSSES,
                device=torch.device(rank)
            ),
            'SEGMENTATION': SegmentationLoss(
                loss_names=configs.TrainConfig.Losses.MASK_LOSSES,
                device=torch.device(rank)
            )
        }.get(self.task, None)
        
        self.lambda_adv = torch.nn.Parameter(
            torch.tensor(0.5)
        )
        self.load_pretrained_weights()
        
    def load_pretrained_weights(self
    ) -> None:
        if self.pretraind_path is not None:
            pretraind_checkpoints = torch.load(
                self.pretraind_path, 
                map_location=torch.device('cpu')
            )
            if 'LossFunction' in pretraind_checkpoints.keys(): 
                self.load_state_dict(pretraind_checkpoints['LossFunction'], strict=False)
        self = self.to(self.rank)
        
    def gan_discriminator(self, 
                          adv_preds: torch.Tensor,
                          adv_grounds: torch.Tensor          
    ) -> torch.Tensor: 
        adv_preds = adv_preds.view(-1)
        fake = adv_preds[adv_grounds == 0.0]
        real = adv_preds[adv_grounds == 1.0]
        ic(fake.shape, real.shape)
        ic(
            (fake.mean() - real.mean())
        )
        return fake.mean() - real.mean()
    
    def gan_generator(self, 
                      adv_preds: torch.Tensor
    ) -> torch.Tensor: 
        adv_preds = adv_preds.view(-1)
        return -1 * adv_preds.mean()
           
    def forward(self, 
                preds: torch.Tensor=None, 
                grounds: torch.Tensor=None,  
                adv_preds: torch.Tensor = None, 
                adv_grounds: torch.Tensor = None, 
                only_discriminator: bool = False
    ) -> torch.Tensor: 
        ## Discriminator Loss 
        if only_discriminator: 
            return self.gan_discriminator(adv_preds, adv_grounds)
        else: 
            # Generator Loss
            if self.task == 'SEGMENTATION':
                return self.gen_loss_fn(preds, grounds)
            gen_total_loss = self.gen_loss_fn(preds, grounds)
            if adv_preds is not None: 
                gen_total_loss += self.lambda_adv * self.gan_generator(adv_preds)
            return gen_total_loss