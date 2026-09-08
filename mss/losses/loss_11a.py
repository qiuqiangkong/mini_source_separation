from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from mss.models2.dsp3.banks import hz_to_erb_ex, erb_to_hz_ex, hz_to_erb, erb_to_hz, hz_to_mel, mel_to_hz


class Loss11a(nn.Module):
    def __init__(self):
        super().__init__()

        self.window_size = 2048
        self.hop_length = 480

        a = 21.4
        b = 0.0001
        sr = 48000

        for window_size in [self.window_size]:
            n_bands = window_size // 4 + 1
            freqs = np.linspace(0, hz_to_erb_ex(sr / 2, a, b), n_bands + 1)
            freqs = erb_to_hz_ex(freqs, a, b)
            fb = triangular_filterbank(
                freqs=freqs,
                sr=sr,
                n_fft=window_size,
                norm=False,
            )
            self.register_buffer(f"fb_{window_size}", Tensor(fb))

    def forward(self, output: Tensor, target: Tensor) -> torch.float:

        window_size = 2048
        hop_length = 480

        n_fft = window_size
        hop_length = self.hop_length
        out = self.stft(output, n_fft, hop_length)
        tar = self.stft(target, n_fft, hop_length)
        loss1 = (out - tar).abs().mean()
        loss1 *= 100

        # from IPython import embed; embed(using=False); os._exit(0)

        # out = rearrange(out, 'b c t f -> (b c t) f').abs() ** 2
        # tar = rearrange(tar, 'b f t -> b t f').abs() ** 2
        out = (out.abs() ** 2) @ getattr(self, f"fb_{window_size}").T
        tar = (tar.abs() ** 2) @ getattr(self, f"fb_{window_size}").T
        out = torch.clamp(out, min=1e-10)
        tar = torch.clamp(tar, min=1e-10)
        out = torch.clamp(torch.log10(out), -6)
        tar = torch.clamp(torch.log10(tar), -6)
        loss2 = (out - tar).abs().mean()
        # loss2 *= 0.1

        # plt.matshow(tar.cpu().numpy()[2, 0].T, origin='lower', aspect='auto', extent=[0,50,0,10], cmap='jet')
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0)

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


def triangular_filterbank(
    freqs: list[float] | np.ndarray,
    sr: int,
    n_fft: int,
    norm: bool = False,
) -> np.ndarray:
    """
    Build a triangular filterbank.

    Args:
        freqs:
            Frequency nodes in Hz, shape: (n_filters + 2,).
            Example: [0, 100, 200, 400, 800, 16000]
            Each filter uses three adjacent points:
                left = freqs[i]
                center = freqs[i + 1]
                right = freqs[i + 2]

        sr:
            Sampling rate.

        n_fft:
            FFT size.

        norm:
            If True, normalize each filter to unit sum.

    Returns:
        fb:
            Filterbank matrix, shape: (n_filters, n_fft // 2 + 1).
    """

    freqs = np.asarray(freqs, dtype=np.float32)

    if freqs[0] < 0:
        raise ValueError("freqs must be non-negative.")

    if freqs[-1] > sr / 2:
        raise ValueError("freqs[-1] must be <= Nyquist frequency.")

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    n_filters = len(freqs) - 2
    n_bins = len(fft_freqs)

    fb = np.zeros((n_filters, n_bins), dtype=np.float32)

    for i in range(n_filters):
        left = freqs[i]
        center = freqs[i + 1]
        right = freqs[i + 2]

        # Rising slope: left -> center
        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        fb[i, left_mask] = (
            (fft_freqs[left_mask] - left) / (center - left)
        )

        # Falling slope: center -> right
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        fb[i, right_mask] = (
            (right - fft_freqs[right_mask]) / (right - center)
        )

        if norm:
            s = fb[i].sum()
            if s > 0:
                fb[i] /= s

    return fb