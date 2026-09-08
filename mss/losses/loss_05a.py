from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Loss05a(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, target: Tensor) -> torch.float:

        n_fft = 2048
        hop_length = 480
        out = self.stft(output, n_fft, hop_length)
        tar = self.stft(target, n_fft, hop_length)

        # mu_law
        out = self.convert(out, 255)
        tar = self.convert(tar, 255)
        
        # import matplotlib.pyplot as plt
        # plt.matshow(tar.abs()[0, 0].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")

        loss = F.l1_loss(out, tar)
        loss *= 10
        # print(loss)
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

    def mu_law(self, x, mu=255):
        return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / math.log1p(mu)

    def convert(self, x, mu=255, eps=1e-8):
        mag = torch.abs(x)
        mag2 = self.mu_law(mag, mu)
        phase_unit = x / mag.clamp_min(eps)
        return mag2 * phase_unit