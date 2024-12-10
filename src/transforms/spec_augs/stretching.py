import random
import torchaudio
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, fixed_rate=1, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.TimeStretch(fixed_rate=fixed_rate, *args, **kwargs)

    def forward(self, data: Tensor):
        self._aug.fixed_rate = random.uniform(0.7, 1.5)
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)