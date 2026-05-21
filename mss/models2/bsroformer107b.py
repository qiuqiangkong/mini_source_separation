from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mss.models2.dsp3.banks import mel_linear_banks, erb_linear_banks
from mss.models2.dsp3.subband_fast import SubbandFilter
from mss.models.attention import Block
from mss.models.rope import RoPE
from mss.utils import fast_sdr


class BSRoformer107b(nn.Module):
    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_layers=12,
        n_heads=12,
        dim=768,
        rope_len=8192,
        **kwargs
    ) -> None:

        super().__init__()
        
        n_bands = 112
        self.n_fft = 16
        self.hop_length = 4
        self.patch_size_t = 4
        max_bandwidth = 390
        factor = sample_rate // 400
        chunk_size = 16

        # Subband filter
        
        banks = erb_linear_banks(sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilter(sample_rate, banks, factor, chunk_size=chunk_size)
        
        # Patch
        in_channels = audio_channels * 2
        
        # Transformer blocks
        n_hid = 6
        self.patch1 = FreqPatch(in_channels, n_hid, (1, 1))
        self.enc1 = FreqBlock(dim=n_hid, n_heads=1, n_layers=1, rope_len=rope_len)

        self.patch2 = Patch(n_hid*16, 96, (1, 1))
        self.enc2 = BSBlock(dim=96, n_heads=3, n_layers=1, rope_len=rope_len)

        self.patch3 = Patch(96, 384, (4, 1))
        self.enc3 = BSBlock(dim=384, n_heads=12, n_layers=3, rope_len=rope_len)

        self.dec3 = BSBlock(dim=384, n_heads=12, n_layers=3, rope_len=rope_len)
        self.unpatch3 = UnPatch(384, 96, (4, 1))

        self.cat2 = Cat(192, 96)
        self.dec2 = BSBlock(dim=96, n_heads=3, n_layers=1, rope_len=rope_len)
        self.unpatch2 = UnPatch(96, n_hid*16, (1, 1))

        self.cat1 = Cat(192, 96)
        self.dec1 = FreqBlock(dim=n_hid, n_heads=1, n_layers=1, rope_len=rope_len)
        self.unpatch1 = FreqUnPatch(n_hid, in_channels, (1, 1))


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
        # from IPython import embed; embed(using=False); os._exit(0)

        # Subband Analysis
        x = self.sb_filter.analysis(audio)  # (b, c, k, l')

        # if False:  # For debug. Analysis-synthesis SDR should over 30 dB.
        #     self.check_sdr(audio, complex_sp)
        #     os._exit(0)

        complex_sp = self.stft(x)  # (b, c, k, t, f)
        B, C, K, T, F_ = complex_sp.shape

        x0 = rearrange(torch.view_as_real(complex_sp), 'b c k t f x -> b (c x) k t f')
        x0 = self.pad_tensor(x0, self.patch_size_t)  # (b, c, k, t, f)

        # Enc
        x1 = self.patch1(x0)
        x1 = self.enc1(x1)
        x1 = rearrange(x1, 'b d k t f -> b (d f) k t')
        
        x2 = self.patch2(x1)
        x2 = self.enc2(x2)

        x3 = self.patch3(x2)
        x3 = self.enc3(x3)

        # Dec
        y3 = x3
        y3 = self.dec3(y3)
        y2 = self.unpatch3(y3)

        y1 = self.cat2(x2, y2)
        y1 = self.dec2(y1)
        y1 = self.unpatch2(y1)

        y0 = self.cat1(x1, y1)
        y0 = rearrange(y0, 'b (d f) k t -> b d k t f', f=F_)
        y0 = self.dec1(y0)
        y0 = self.unpatch1(y0)

        # Postprocess
        x = y0[:, :, :, 0 : T, :]
        x = rearrange(x, 'b (c x) k t f -> b c k t f x', c=C, x=2)
        mask = torch.view_as_complex(x.contiguous())
        sep_stft = complex_sp * mask

        # Subband synthesis
        x = self.istft(sep_stft)  # (b, c, k, t, f)
        out = self.sb_filter.synthesis(x)
     
        return out

    def stft(self, x):
        B, C = x.shape[0 : 2]
        x = rearrange(x, 'b c k l -> (b c k) l')
        x = torch.stft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c k) f t -> b c k t f', b=B, c=C)
        return x

    def istft(self, x):
        B, C = x.shape[0 : 2]
        x = rearrange(x, 'b c k t f -> (b c k) f t')
        x = torch.istft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            normalized=True,
            onesided=False,
            return_complex=True
        )
        x = rearrange(x, '(b c k) l -> b c k l', b=B, c=C)
        return x

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


class FreqPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        T = x.shape[3]
        x = rearrange(x, 'b c k t f -> b c k (t f)')
        x = self.conv(x)
        x = rearrange(x, 'b c k (t f) -> b c k t f', t=T)
        return x


class FreqUnPatch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def __call__(self, x):
        T = x.shape[3]
        x = rearrange(x, 'b c k t f -> b c k (t f)')
        x = self.conv(x)
        x = rearrange(x, 'b c k (t f) -> b c k t f', t=T)
        return x


class FreqBlock(nn.Module):
    def __init__(self, dim, n_heads, n_layers, rope_len):
        super().__init__()

        self.f_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

    def forward(self, x):

        B = x.shape[0]
        K = x.shape[2]

        # --- 2. Transformer along time and frequency axes ---
        for f_block in self.f_blocks:

            x = rearrange(x, 'b d k t f -> (b k t) f d')
            x = f_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)
            x = rearrange(x, '(b k t) f d -> b d k t f', b=B, k=K)  # shape: (b, d, t, f)

        return x