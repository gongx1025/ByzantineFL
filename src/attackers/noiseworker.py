from typing import Optional
import torch

from src.worker import ByzantineWorker


class NoiseWorker(ByzantineWorker):
    def __init__(self, mean: Optional[float] = 0.0, std: Optional[float] = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_mean = mean
        self._noise_std = std

    def get_gradient(self):
        return self._gradient

    def omniscient_callback(self):
        self._gradient = super().get_gradient() + torch.normal(self._noise_mean,
                                                               self._noise_std,
                                                               size=super().get_gradient().shape
                                                               ).to('cuda')
