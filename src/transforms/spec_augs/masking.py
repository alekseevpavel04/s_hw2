import torch
import torchaudio.transforms
from torch import Tensor, nn

class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param, p):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.p = p

    def __call__(self, data: Tensor):
        if torch.rand(1) < self.p:
            return self._aug(data)
        return data



class TimeMasking(nn.Module):
    def __init__(self, time_mask_param, p):
        super().__init__()
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param)
        self.p = p

    def __call__(self, data: Tensor):
        if torch.rand(1) < self.p:
            return self._aug(data)
        return data

