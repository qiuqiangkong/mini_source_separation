from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F

from mss.models2.dsp3.banks import erb_linear_banks_overlap

from mss.models2.dsp3.banks import mel_linear_banks, erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter


class L1Subband(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self) -> None:
        super(L1Subband, self).__init__()

        sample_rate = 48000
        n_bands = 112
        self.n_fft = 16
        self.hop_length = 4
        self.patch_size_t = 4
        max_bandwidth = 390
        factor = sample_rate // 400
        chunk_size = 16

        banks = erb_linear_banks(sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilter(sample_rate, banks, factor, chunk_size=chunk_size)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(4, 1, sharex=True)
        # tmp = self.sb_filter.analysis(target).real.cpu().numpy()
        # axs[0].plot(tmp[0, 0, 10, :])
        # axs[1].plot(tmp[0, 0, 11, :])
        # axs[2].plot(tmp[0, 0, 39, :])
        # axs[3].plot(tmp[0, 0, 40, :])
        # plt.savefig("_zz.pdf")
        loss = (self.sb_filter.analysis(output) - self.sb_filter.analysis(target)).abs().mean()
        
        return loss


class L1SubbandStft(nn.Module):
    r"""Multi-resolution STFT loss."""

    def __init__(self) -> None:
        super(L1SubbandStft, self).__init__()

        sample_rate = 48000
        n_bands = 112
        self.n_fft = 16
        self.hop_length = 4
        self.patch_size_t = 4
        max_bandwidth = 390
        factor = sample_rate // 400
        chunk_size = 16

        banks = erb_linear_banks(sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilter(sample_rate, banks, factor, chunk_size=chunk_size)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:

        output = self.sb_filter.analysis(output)
        target = self.sb_filter.analysis(target)

        output = self.stft(output, self.n_fft, self.n_fft // 4)
        target = self.stft(target, self.n_fft, self.n_fft // 4)

        loss = (output - target).abs().mean()
        
        return loss

    def stft(self, x: Tensor, n_fft: int, hop_length: int) -> Tensor:
        B, C, K = x.shape[0 : 3]
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