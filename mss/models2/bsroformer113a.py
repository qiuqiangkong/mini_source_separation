from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models2.bandsplit42a import BandSplit
from mss.models2.dsp3.banks import mel_linear_banks, erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter
from mss.models.attention import Block
from mss.models.fourier import Fourier
from mss.models.rope import RoPE
from mss.utils import fast_sdr


class BSRoformer113a(Fourier):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        patch_size=[4, 4],
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__()
        
        self.ac = audio_channels
        # self.patch_size = patch_size

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=band_dim
        )

        # Transformer blocks
        self.patch1 = Patch(band_dim * audio_channels, 96, (1, 1))
        self.enc1 = BSBlock(dim=96, n_heads=3, n_layers=1, rope_len=rope_len)

        self.patch2 = Patch(96, 384, (4, 1))
        self.enc2 = BSBlock(dim=384, n_heads=12, n_layers=3, rope_len=rope_len)

        self.dec2 = BSBlock(dim=384, n_heads=12, n_layers=3, rope_len=rope_len)
        self.unpatch2 = UnPatch(384, 96, (4, 1))

        self.cat1 = Cat(192, 96)
        self.dec1 = BSBlock(dim=96, n_heads=3, n_layers=1, rope_len=rope_len)
        self.unpatch1 = UnPatch(96, band_dim * audio_channels, (1, 1))


    def forward(self, audio: Tensor) -> Tensor:
        r"""Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        k: n_bands
        l'
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, l)

        Returns:
            out: (b, c, l)
        """
        # --- 1. Encode ---
        # 1.1 Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)
        T0 = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)

        # 1.3 Convert STFT to mel scale
        x = self.bandsplit.transform(x)  # shape: (b, c, t, f, o)
        x = rearrange(x, 'b c t k d -> b (c d) t k')
        T = x.shape[2]

        # 1.2 Pad stft
        x = self.pad_tensor(x, 4)  # x: (b, d, t, f)
        
        # Enc
        x1 = self.patch1(x)
        x1 = self.enc1(x1)

        x2 = self.patch2(x1)
        x2 = self.enc2(x2)

        # Dec
        y2 = x2
        y2 = self.dec2(y2)
        y1 = self.unpatch2(y2)

        y0 = self.cat1(x1, y1)
        y0 = self.dec1(y0)
        y0 = self.unpatch1(y0)

        x = y0[:, :, 0 : T, :]
        x = rearrange(x, 'b (c d) t k -> b c t k d', c=2)
        x = self.bandsplit.inverse_transform(x)  # shape: (b, c, t, f, k)
        mask = torch.view_as_complex(x.contiguous())
        
        sep_stft = complex_sp * mask

        # Subband synthesis
        out = self.istft(sep_stft)  # (b, c, k, t, f)
     
        return out

    def pad_tensor(self, x: Tensor, patch_size_t) -> Tensor:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t, f)
        
        Returns:
            out: E.g., (b, c, t f)
        """

        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % patch_size_t  # Equals to p - (T % p)
        x = F.pad(x, pad=(0, 0, 0, pad_t))
        return x

    def check_sdr(self, audio, complex_sp) -> None:
        y = self.istft(complex_sp)  # (b, c, k, t, f)
        y = self.sb_filter.synthesis(y)
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"SDR: {sdr:.2f} dB")


class Patch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        return self.conv(x)


class UnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        return self.conv(x)


class Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

    def __call__(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class BSBlock(nn.Module):
    def __init__(self, dim, n_heads, n_layers, rope_len):
        super().__init__()

        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.k_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

    def forward(self, x):

        B = x.shape[0]

        # --- 2. Transformer along time and frequency axes ---
        for t_block, k_block in zip(self.t_blocks, self.k_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = k_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        return x