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


class BSRoformer97a(nn.Module):
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
        
        n_bands = 64
        self.T1 = 32
        max_bandwidth = 800
        factor = sample_rate // max_bandwidth
        chunk_size = 16

        # Subband filter        
        banks = erb_linear_banks(sr=sample_rate, n_bands=n_bands, max_bandwidth=max_bandwidth)
        self.sb_filter = SubbandFilter(sample_rate, banks, factor, chunk_size=chunk_size)

        # Patch
        in_channels = audio_channels * 2
        self.pre_conv = nn.Conv1d(audio_channels * 2, dim, kernel_size=self.T1, stride=self.T1)
        self.post_conv = nn.ConvTranspose1d(dim, audio_channels * 2, kernel_size=self.T1, stride=self.T1)

        # RoPE
        self.rope = RoPE(head_dim=dim // n_heads, max_len=rope_len)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.k_blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))

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

        # Subband Analysis
        x = self.sb_filter.analysis(audio)  # (b, c, k, l')
        
        if False:  # For debug. Analysis-synthesis SDR should over 30 dB.
            self.check_sdr(audio, x)
            os._exit(0)

        # Patchify
        B, C, K, L = x.shape
        x = rearrange(torch.view_as_real(x), 'b c k l x -> (b k) (c x) l')
        x = self.pre_conv(x)
        x = rearrange(x, '(b k) d t -> b d t k', b=B)

        # --- 2. Transformer along time and frequency axes ---
        for t_block, k_block in zip(self.t_blocks, self.k_blocks):

            x = rearrange(x, 'b d t k -> (b k) t d')
            x = t_block(x, rope=self.rope, pos=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b k) t d -> (b t) k d', b=B)
            x = k_block(x, rope=self.rope, pos=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) k d -> b d t k', b=B)  # shape: (b, d, t, f)

        x = rearrange(x, 'b d t k -> (b k) d t')
        x = self.post_conv(x)
        x = rearrange(x, '(b k) (c x) l -> b c k l x', b=B, x=2)
        x = torch.view_as_complex(x.contiguous())

        # Subband synthesis
        out = self.sb_filter.synthesis(x)
     
        return out