from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F


class MultiResolutionSTFTLoss2(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        self.window_sizes = [4096, 2048, 1024, 512, 256]

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.

        for window_size in self.window_sizes:
            output_stft = self.stft(output, window_size, window_size // 4)
            target_stft = self.stft(target, window_size, window_size // 4)
            loss += F.l1_loss(output_stft, target_stft)
            
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


class WavMultiResolutionSTFTLoss2(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        self.window_sizes = [4096, 2048, 1024, 512, 256]

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        
        loss = F.l1_loss(output, target) * 2
        
        for window_size in self.window_sizes:
            output_stft = self.stft(output, window_size, window_size // 4)
            target_stft = self.stft(target, window_size, window_size // 4)
            loss += F.l1_loss(output_stft, target_stft)
            
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


class MultiResolutionSTFTLoss2Hop147(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        self.window_sizes = [4096, 2048, 1024, 512, 256]

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.

        for window_size in self.window_sizes:
            output_stft = self.stft(output, window_size, 147)
            target_stft = self.stft(target, window_size, 147)
            loss += F.l1_loss(output_stft, target_stft)
            
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


class MultiResolutionSTFTLoss2Scale(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        self.window_sizes = [4096, 2048, 1024, 512, 256]

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.

        for window_size in self.window_sizes:
            output_stft = self.stft(output, window_size, window_size // 4)
            target_stft = self.stft(target, window_size, window_size // 4)
            loss += F.l1_loss(output_stft, target_stft)
            
        loss *= 100

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