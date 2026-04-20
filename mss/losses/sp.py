from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F


class L1Sp(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        self.window_sizes = [2048]

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.

        for window_size in self.window_sizes:
            output_stft = self.stft(output, window_size, window_size // 4)
            target_stft = self.stft(target, window_size, window_size // 4)
            loss += F.l1_loss(output_stft, target_stft)
            
        loss /= 2
        
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
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)
        return x



class L1SpMulti(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        self.window_sizes = [256, 512, 1024, 2048, 4096]

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.

        for window_size in self.window_sizes:
            output_stft = self.stft(output, window_size, window_size // 4)
            target_stft = self.stft(target, window_size, window_size // 4)
            loss += F.l1_loss(output_stft, target_stft)
            
        loss /= 2
        loss /= len(self.window_sizes)
        
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
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)
        return x


class L1LogSp(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        self.window_sizes = [2048]

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.

        for window_size in self.window_sizes:
            
            output_stft = self.stft(output, window_size, window_size // 4)
            target_stft = self.stft(target, window_size, window_size // 4)

            log_loss = (log_complex(output_stft) - log_complex(target_stft)).abs().mean()
            log_loss /= 10

            loss = loss + F.l1_loss(output_stft, target_stft) + log_loss
            
        loss /= 2
        
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
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)
        return x


def log_complex(x):
    return torch.clamp(x.abs(), 1e-6).log10()
    # x.real
    # x.real.sign() * torch.log10(torch.clamp(x.real.sign() * x.real, 1e-6))