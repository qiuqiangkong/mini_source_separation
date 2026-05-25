from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import torch.nn as nn
import math
import numpy as np
from mss.utils import fast_sdr

from .overlap_add import overlap_add

from einops import rearrange
from mss.models2.dsp3.banks import erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter


class LowbandFractionalSTFT(nn.Module):
    r"""Apply fractional STFT to low-band-limited signals."""

    def __init__(
        self, 
        sr: float, 
        half_bandwidths: list[float], 
        n_fft: int, 
        hop_length: int,
        alpha: float = 0.3
    ):
        r"""
        m: n_bands
        f: freq_bins
        n: frame_samples
        """
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        N = n_fft
        M = len(half_bandwidths)
        r = (sr / 2) / Tensor(half_bandwidths)  # (m,)
        r = torch.clamp(torch.floor(r * alpha), 1.)  # (m,), fractional factors
        m = 1. / r  # (m,)

        k = torch.arange(0, N // 2 + 1)  # (f,)
        n = torch.arange(0, N)  # (n,)
        mkn = torch.einsum('m,k,n->mkn', m, k, n)  # (m, f/2+1, n)

        mkn0 = mkn[:, 0:1, :]
        mkn1 = mkn[:, 1:-1, :]
        mkn2 = mkn[:, -1:, :]
        mkn3 = - torch.flip(mkn1, dims=[1])
        mkn = torch.cat([mkn0, mkn1, mkn2, mkn3], dim=1)  # (m, f, n)
        
        w = torch.exp(-1.j * 2 * math.pi / N * mkn) / math.sqrt(N)  # (m, f, n)
        w /= torch.sqrt(r[:, None, None])

        self.register_buffer("w", w)  # (m, f, n)

    def analysis(self, x: Tensor) -> Tensor:
        r"""
        b: batch_size
        m: n_bands
        l: audio_samples
        t: n_frames
        n: frame_samples
        f: freq_bins

        Args:
            x: (b, m, l)

        Returns:
            out: (b, m, t, f)
        """
        N = self.n_fft
        x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, m, l)
        x = x.unfold(dimension=-1, size=N, step=self.hop_length).contiguous()  # (b, m, t, n)
        x *= self.window
        out = torch.einsum('bmtn,mfn->bmtf', x, self.w)  # (b, m, t, f)
        return out

    def synthesis(self, x: Tensor, length: int | None) -> Tensor:
        r"""
        b: batch_size
        m: n_bands
        t: n_frames
        f: freq_bins
        l: audio_samples
        n: frame_samples

        Args:
            x: (b, m, t, f)

        Returns:
            x: (b, m, l)
        """
        x = torch.einsum('bmtf,mfn->bmtn', x, self.w.conj())  # (b, m, t, n)

        # Overlap add
        B = x.shape[0]
        x = rearrange(x, 'b m t n -> (b m) t n')  # (b*m, t, n)
        out = overlap_add(x=x, hop_length=self.hop_length, window=self.window)  # (b*m, l)
        out = rearrange(out, '(b m) l -> b m l', b=B)
        out = out[..., self.n_fft // 2 :]  # (b, m, l)
        
        if length is not None:
            out = out[..., 0 : length]  # (b, m, l)

        return out


if __name__ == "__main__":

    sr = 48000
    n_bands = 112
    max_bandwidth = 390
    factor = sr // 400
    chunk_size = 16
    device = "cuda"

    n_fft = 16
    hop_length = 4

    # Subband
    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)
    
    # Lowband Fractional STFT
    half_bandwidths = [(bank[1] - bank[0]) / 2 for bank in banks]
    stft = LowbandFractionalSTFT(400, half_bandwidths, n_fft, hop_length).to(device)

    # Data
    rs = np.random.RandomState(1234)
    B = 4
    C = 2
    L = sr * 2
    audio = rs.uniform(low=-1, high=1, size=(B, C, L))  # (b, c, l)
    audio = Tensor(audio).to(device)  # (c, l)

    # Subband analysis
    x = sb_filter.analysis(audio)  # (b, c, m, l)

    # STFT analysis
    tmp = rearrange(x, 'b c m l -> (b c) m l')
    tmp = stft.analysis(tmp)
    h = rearrange(tmp, '(b c) m t f -> b c m t f', b=B)

    # STFT synthesis
    tmp = rearrange(h, 'b c m t f -> (b c) m t f')
    tmp = stft.synthesis(tmp, x.shape[-1])
    x_hat = rearrange(tmp, '(b c) m l -> b c m l', b=B)
    
    # Subband synthesis
    pred_audio = sb_filter.synthesis(x_hat)

    # SDR
    sdr = fast_sdr(audio.cpu().numpy(), pred_audio.cpu().numpy())
    print((x_hat - x).abs().mean(), sdr)