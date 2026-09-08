from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


'''
class Loss04a(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, target: Tensor) -> torch.float:

        n_fft = 2048
        hop_length = 480
        out = self.stft(output, n_fft, hop_length)
        tar = self.stft(target, n_fft, hop_length)

        # import matplotlib.pyplot as plt
        # plt.matshow(tar.abs()[0, 0].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")

        scale = max(10, tar.abs().max())
        out /= scale
        tar /= scale

        out = self.to_log_complex(out)
        tar = self.to_log_complex(tar)

        # plt.figure()
        # plt.matshow(-tar.abs()[1, 0].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz2.pdf")

        loss = F.l1_loss(out, tar)
        loss *= 0.01
        print(loss)
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

    def to_log_complex(self, x):
        phase = torch.angle(x)
        mag = torch.clamp(torch.abs(x), 1e-10)
        mag = torch.clamp(20 * torch.log10(mag), -100)
        # return mag
        return torch.polar(-mag, phase)
'''