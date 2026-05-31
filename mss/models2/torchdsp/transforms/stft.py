from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mss.models2.torchdsp.functional.overlap_add import overlap_add


class STFTLearnable(nn.Module):
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        n_fractions=1, 
        learnable=True
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        N = n_fft
        r = n_fractions
        k = torch.arange(0, N * r)  # (f,)
        n = torch.arange(0, N)  # (n,)
        kn = torch.outer(k, n)  # (f, n)
        w = torch.exp(-1.j * 2 * math.pi / (N * r) * kn) / math.sqrt(N * r)  # (f, n)

        if learnable:
            self.w = nn.Parameter(w)
        else:
            self.register_buffer("w", w)

    def analysis(self, x: Tensor) -> Tensor:
        r"""STFT.

        b: batch_size
        l: audio_samples
        t: n_frames
        n: frame_samples
        f: freq_bins

        Args:
            x: (b, l)

        Returns:
            out: (b, t, f)
        """
        N = self.n_fft
        x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, l)
        x = x.unfold(dimension=-1, size=N, step=self.hop_length).contiguous()  # (b, t, n)
        x *= self.window  # (b, t, n)
        x = x.to(self.w.dtype)  # (b, t, n)
        out = x @ self.w.T  # (b, t, f)
        return out

    def synthesis(self, x: Tensor, length: int) -> Tensor:
        r"""Inverse STFT.

        b: batch_size
        t: n_frames
        f: freq_bins
        n: frame_samples
        l: audio_samples

        Args:
            x: (b, t, f)

        Returns:
            x: (b, l)
        """
        x = x @ self.w.conj()  # (b, t, n)

        # Overlap add
        out = overlap_add(x=x, hop_length=self.hop_length, window=self.window)  # (b, l)
        out = out[..., self.n_fft // 2 :]  # (b, l)
        
        if length is not None:
            out = out[..., 0 : length]  # (b, l)

        return out


if __name__ == "__main__":
    r"""
    b: batch_size
    l: audio_samples
    t: frames_num
    f: freq_bins
    """

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_fft = 2048
    hop_length = 480

    stft = STFTLearnable(n_fft, hop_length)

    # Data
    L = 48000
    x = torch.randn(4, L)  # (b, l)

    # Analysis & synthesis
    h = stft.analysis(x)  # (b, t, f)
    x_hat = stft.synthesis(h, L)  # (b, l)
    
    print(f"x (B, L): {x.shape}")
    print(f"h (B, T, K): {h.shape}")
    print(f"x_hat (B, L): {x_hat.shape}")
    print("Error: {}".format((x - x_hat).abs().mean()))