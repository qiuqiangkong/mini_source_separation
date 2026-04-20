from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F

from mss.models2.dsp3.banks import erb_linear_banks_overlap
from mss.models2.dsp3.subband_fast_overlap import SubbandFilterOverlap


class SubbandLoss(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(
        self,
        window_sizes: List[int] = [4096, 2048, 1024, 512, 256],
        hop_size=147,
        stft_n_fft=2048,
        normalized=False,
        window_fn=torch.hann_window,
    ) -> None:
        super(L1SubbandLoss, self).__init__()

        self.window_sizes = window_sizes
        self.hop_size = hop_size
        self.stft_n_fft = stft_n_fft
        self.normalized = normalized
        self.window_fn = window_fn
  
        self.multi_stft_kwargs = dict(
            hop_length = hop_size,
            normalized = normalized,
        )

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = 0.

        for window_size in self.window_sizes:
            stft_kwargs = dict(
                n_fft = max(window_size, self.stft_n_fft),
                win_length = window_size,
                return_complex = True,
                window = self.window_fn(window_size).to(output.device),
                **self.multi_stft_kwargs
            )
            
            output_Y = torch.stft(rearrange(output, "... s t -> (... s) t"), **stft_kwargs)
            target_Y = torch.stft(rearrange(target, "... s t -> (... s) t"), **stft_kwargs)
            
            loss = loss + F.l1_loss(output_Y, target_Y)
        
        return loss


class L1SubbandSp(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self):
        super().__init__()

        sample_rate = 48000
        n_bands = 111
        self.n_fft = 32
        self.hop_length = 8
        self.patch_size_t = 4
        max_bandwidth = 390
        factor = sample_rate // 800
        chunk_size = 16

        banks = erb_linear_banks_overlap(sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilterOverlap(sample_rate, banks, factor, chunk_size=chunk_size)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:

        output_sb = self.sb_filter.analysis(output)  # (b, c, k, l')
        output_stft = self.stft(output_sb, self.n_fft, self.hop_length)  # (b, c, k, t, f)

        target_sb = self.sb_filter.analysis(target)  # (b, c, k, l')
        target_stft = self.stft(target_sb, self.n_fft, self.hop_length)  # (b, c, k, t, f)

        loss = F.l1_loss(output_stft, target_stft)    
        loss /= 2
        
        return loss

    def stft(self, x: Tensor, n_fft: int, hop_length: int) -> Tensor:
        B, C = x.shape[0 : 2]
        x = rearrange(x, 'b c k l -> (b c k) l')
        x = torch.stft(
            input=x, 
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c k) f t -> b c k t f', b=B, c=C)
        return x