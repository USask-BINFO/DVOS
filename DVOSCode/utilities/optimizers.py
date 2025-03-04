import math 
import torch

from typing import Union, Dict, Tuple
from omegaconf.dictconfig import DictConfig

from .utils import AttrDict


class LARS(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr: float = 0.001, 
                 weight_decay: float = 0.0, 
                 momentum: float = 0.9, 
                 eta: float = 0.001,
                 weight_decay_filter: bool = False, 
                 lars_adaptation_filter: bool = False, 
    ) -> None:
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            eta=eta, weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter
        )
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, 
                              p: torch.Tensor
    ) -> bool:
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if (not g['weight_decay_filter'] or 
                    not self.exclude_bias_and_norm(p)
                ):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if (not g['lars_adaptation_filter'] or 
                    not self.exclude_bias_and_norm(p)
                ):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.,
                        torch.where(
                            update_norm > 0,
                            (g['eta'] * param_norm / update_norm), 
                            one
                        ), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class LARSScheduler: 
    def __init__(self,
                 optimizer: LARS,  
                 batch_size: int, 
                 total_epochs: int, 
                 loader_length: int,
                 learning_rate_weights: float, 
                 learning_rate_biases: float
    ) -> None:
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.loader_length = loader_length
        self.learning_rate_weights = learning_rate_weights
        self.learning_rate_biases = learning_rate_biases
    
    def step(self, 
             step: int
    ) -> None:
        max_steps = self.total_epochs * self.loader_length
        warmup_steps = 10 * self.loader_length
        base_lr = self.batch_size / 256
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        self.optimizer.param_groups[0]["lr"] = lr * self.learning_rate_weights
        self.optimizer.param_groups[1]["lr"] = lr * self.learning_rate_biases
        

def init_lars_optimizer(model: torch.nn.Module, 
                        loader_length: int, 
                        configs: Union[Dict, AttrDict, DictConfig]
) -> Tuple[LARS, LARSScheduler]:
    """Initializes the optimizer and learning rate scheduler"""
    # initialize the learning rate scheduler
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{"params": param_weights}, {"params": param_biases}]

    optimizer = LARS(
        parameters, 
        **configs.TrainConfig.Optimizer.PARAMS
    )
    scheduler = LARSScheduler(
        optimizer=optimizer, 
        batch_size=configs.TrainConfig.BATCH_SIZE, 
        total_epochs=configs.TrainConfig.NUM_EPOCHS, 
        loader_length=loader_length,
        learning_rate_weights=configs.TrainConfig.Optimizer.LR_WEIGHTS, 
        learning_rate_biases=configs.TrainConfig.Optimizer.LR_BIASES
    )
    return optimizer, scheduler