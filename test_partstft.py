import torch
import os
from pathlib import Path
import h5py
import json
import math
import re
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from mss.utils import fast_sdr

from einops import rearrange
from mss.models2.dsp3.banks import erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter


def add():

    sr = 48000
    n_bands = 112
    max_bandwidth = 390
    factor = sr // 400
    chunk_size = 16
    device = "cuda"

    n_fft = 16
    hop_length = 4

    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)
    # print(banks)

    rs = np.random.RandomState(1234)
    audio = rs.uniform(low=-1, high=1, size=(4, 2, sr * 2))
    audio = Tensor(audio).to(device)  # (c, l)
            
    # Analysis
    x = sb_filter.analysis(audio)  # (b, c, k, l)
    pred_audio = sb_filter.synthesis(x)
    sdr = fast_sdr(audio.cpu().numpy(), pred_audio.cpu().numpy())
    print(sdr)
    B, C, K, L = x.shape

    feat1 = stft1(x, n_fft, hop_length)  # (b, c, k, t, f)
    y1 = istft1(feat1, n_fft, hop_length)
    pred_audio = sb_filter.synthesis(y1)
    sdr = fast_sdr(audio.cpu().numpy(), pred_audio.cpu().numpy())
    print((y1 - x).abs().mean(), sdr)

    # stft
    half_bandwidths = [[(bank[1] - bank[0]) / 2] for bank in banks]
    stft = Stft(400, half_bandwidths, n_fft, hop_length).to(device)
    
    tmp = rearrange(x, 'b c k l -> (b c) k l')
    tmp = stft.analysis(tmp)
    feat2 = rearrange(tmp, '(b c) k t f -> b c k t f', b=B)

    tmp = rearrange(feat2, 'b c k t f -> (b c) k t f')
    tmp = stft.synthesis(tmp, L)
    y2 = rearrange(tmp, '(b c) k l -> b c k l', b=B)
    pred_audio = sb_filter.synthesis(y2)
    print((feat1 - feat2).abs().mean())
    sdr = fast_sdr(audio.cpu().numpy(), pred_audio.cpu().numpy())
    print((y2 - x).abs().mean(), sdr)

    # part stft
    half_bandwidths = [(bank[1] - bank[0]) / 2 for bank in banks]
    part_stft = LowbandFractionalSTFT(400, half_bandwidths, n_fft, hop_length).to(device)
    
    tmp = rearrange(x, 'b c k l -> (b c) k l')
    tmp = part_stft.analysis(tmp)
    feat3 = rearrange(tmp, '(b c) k t f -> b c k t f', b=B)

    tmp = rearrange(feat3, 'b c k t f -> (b c) k t f')
    tmp = part_stft.synthesis(tmp, L)
    y3 = rearrange(tmp, '(b c) k l -> b c k l', b=B)
    pred_audio = sb_filter.synthesis(y3)
    sdr = fast_sdr(audio.cpu().numpy(), pred_audio.cpu().numpy())
    print((y3 - x).abs().mean(), sdr)
    
    from IPython import embed; embed(using=False); os._exit(0)

def stft1(x, n_fft, hop_length):
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

def istft1(x, n_fft, hop_length):
    B, C = x.shape[0 : 2]
    x = rearrange(x, 'b c k t f -> (b c k) f t')
    x = torch.istft(
        input=x, 
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=x.device),
        normalized=True,
        onesided=False,
        return_complex=True
    )
    x = rearrange(x, '(b c k) l -> b c k l', b=B, c=C)
    return x


class Stft(nn.Module):
    def __init__(self, sr: float, half_bandwidths: list[float], n_fft: int, hop_length: int):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        N = n_fft
        k = torch.arange(0, N)  # (n,)
        n = torch.arange(0, N)  # (n,)
        kn = torch.outer(k, n)  # (k, n)
        w = torch.exp(-1.j * 2 * math.pi / N * kn) / math.sqrt(N)  # (k, n)
        self.register_buffer("w", w)

    def analysis(self, x: Tensor) -> Tensor:
        r"""
        b: batch_size
        k: n_bands
        l: audio_samples
        t: n_frames
        n: frame_samples

        Args:
            x: (b, k, l)

        Returns:
            out: (b, k, t, n)
        """
        N = self.n_fft
        x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, k, l)
        x = x.unfold(dimension=-1, size=N, step=self.hop_length).contiguous()  # (b, k, t, n)
        # x.mul_(self.window)  # (b, t, n)
        x *= self.window
        out = x @ self.w.T
        return out

    def synthesis(self, x: Tensor, length: int) -> Tensor:
        r"""

        Args:
            x: (b, k, t, n)

        Returns:
            x: (b, k, l)
        """
        x = x @ self.w.conj()

        # Overlap add
        B = x.shape[0]
        x = rearrange(x, 'b k t n -> (b k) t n')
        out = overlap_add(x=x, hop_length=self.hop_length, window=self.window)  # (b, l)
        out = rearrange(out, '(b k) l -> b k l', b=B)
        out = out[..., self.n_fft // 2 :]  # (b, k, l)
        
        if length is not None:
            out = out[..., 0 : length]  # (b, k, l)

        return out

class LowbandFractionalSTFT(nn.Module):
    r"""Apply fractional STFT to low-band-limited signals."""

    def __init__(
        self, 
        sr: float, 
        half_bandwidths: list[float], 
        n_fft: int, 
        hop_length: int
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
        r = torch.clamp(torch.floor(r * 0.5), 1.)  # (m,), fractional factors
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

    def synthesis(self, x: Tensor, length: int) -> Tensor:
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


def overlap_add(x: Tensor, hop_length: int, window: Tensor | None) -> Tensor:
    r"""Overlap-add.

    b: batch_size
    t: n_frames
    n: frame_samples
    l: audio_samples

    Args:
        x: (b, t, n)

    Returns:
        x: (b, l)
    """

    n_frames, frame_length = x.shape[-2:]  # (t, n)
    L = frame_length + (n_frames - 1) * hop_length
    
    # Overlap-add
    x = F.fold(
        input=rearrange(x, 'b t n -> b n t'),  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, 1, 1, l)
    out = x.squeeze(dim=[1, 2])  # (b, l)

    # Divide overlap-add window
    if window is not None:
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, n_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        ).squeeze(dim=(0, 1, 2))  # (l,)

        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out


'''
def overlap_add(x: Tensor, hop_length: int, window: Tensor | None) -> Tensor:
    r"""Overlap-add.

    b: batch_size
    t: n_frames
    n: frame_samples
    l: audio_samples

    Args:
        x: (b, t, n)

    Returns:
        x: (b, l)
    """

    n_frames, frame_length = x.shape[-2:]  # (t, n)
    L = frame_length + (n_frames - 1) * hop_length
    
    # Overlap-add
    x = F.fold(
        input=rearrange(x, 'b t n -> b n t'),  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, 1, 1, l)
    out = x.squeeze(dim=[1, 2])  # (b, l)

    # Divide overlap-add window
    if window is not None:
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, n_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        )  # (1, 1, 1, L)
        win_norm = win_norm.squeeze(dim=[0, 1, 2])  # (l,)
        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out
'''


if __name__ == '__main__':

    add()