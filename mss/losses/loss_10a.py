from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Loss10a(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, target: Tensor) -> torch.float:

        window_size = 2048
        hop_length = 480

        n_fft = window_size
        hop_length = 480
        out = self.stft(output, n_fft, hop_length)
        tar = self.stft(target, n_fft, hop_length)

        loss1 = (out - tar).abs().mean()
        loss1 *= 100

        loss2 = torch.log10((out - tar).abs() + 1e-8).mean()
        loss2 *= 0.1

        # print(loss1, loss2)

        loss = loss1 + loss2

        return loss
    
    def stft(self, x: Tensor, n_fft: int, hop_length: int) -> Tensor:
        B, C = x.shape[0 : 2]
        x = rearrange(x, 'b c l -> (b c) l')
        x = torch.stft(
            input=x, 
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=x.device),
            normalized=True,
            onesided=True,
            return_complex=True
        )
        x = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)
        return x

    def enframe(self, x, window_size, hop_length):

        N = window_size
        x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, l)
        x = x.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
        return x