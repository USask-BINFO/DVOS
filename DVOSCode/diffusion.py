import einops
import random
from typing import Tuple

import torch


class SimpleDiffusion:
    """Diffusion sampling class.
    Args: 
        num_diffusion_timesteps (int): Number of diffusion timesteps, default to 1000.
        patch_size (int, float): the squared dimension of patches to be masked or noisy patch_size. 
        mp (float): masking probability. 
        dp (float): floating probability. 
        device (str): Device to use, default to 'cpu'.  
    """
    def __init__(self,
                 num_steps: int = 1000,
                 time_steps: int = 1000,
                 beta_range: Tuple = (1e-4, 2e-2),
                 patch_size: int = 32,
                 mp: float = 0.0,
                 dp: float = 0.0, 
                 apply: bool = True,
                 device: str = 'cpu'
    ):
        self.num_steps = num_steps
        self.time_steps = time_steps
        self.beta_range = beta_range
        self.patch_size = patch_size
        self.mp = mp
        self.dp = dp
        self.apply = apply
        self.device = device

        self.inititialize()

    def inititialize(self
    ) -> None:
        self.beta = torch.linspace(
            self.beta_range[0], self.beta_range[1], self.num_steps,
            dtype=torch.float32, device=self.device
        )
        self.alpha = 1 - self.beta
        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alhpa = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
        
    @staticmethod
    def get_tensor_element(element: torch.Tensor, 
                           t: torch.Tensor
    ) -> torch.Tensor:
        """Get the value at index position "t" in "element" and
            reshape it to have the same dimension as a batch of images (B, C, H, W)
        """
        ele = element.gather(-1, t)
        return ele.reshape(-1, 1, 1, 1)
            
    def diffforward(self, 
                    x0: torch.Tensor, 
                    ts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x0)                      # noise
        mean = SimpleDiffusion.get_tensor_element(
            self.sqrt_alpha_cumulative, t=ts
        ) * x0                                          # image
        std_dev = SimpleDiffusion.get_tensor_element(
            self.sqrt_one_minus_alpha_cumulative, t=ts
        )                                               # Noise scaled
        sample = mean + std_dev * eps                   # scaled image * scaled noise
        return sample, eps
    
    def diffuse(self,
               images: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        diffused_images = images.clone()
        if self.apply is True:
            batches, _, rows, cols = diffused_images.shape
            ts = torch.randint(
                low=1, high=self.time_steps,
                size=(batches,),
                device=self.device
            )
            noise, _ = self.diffforward(
                diffused_images.clone(), ts
            )
            for b in range(batches):
                for i in range(0, rows, self.patch_size):
                    for j in range(0, cols, self.patch_size):
                        if random.random() < self.mp: 
                            diffused_images[b, :, i:i+self.patch_size, j:j+self.patch_size] = 0.0
                        elif random.random() < self.dp: 
                            diffused_images[b, :, i:i+self.patch_size, j:j+self.patch_size] = noise[
                                b, :, i:i+self.patch_size, j:j+self.patch_size
                            ]
        return diffused_images
