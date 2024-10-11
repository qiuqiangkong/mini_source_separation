import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
import torchaudio
from einops import rearrange
import numpy as np
from rotary_embedding_torch import RotaryEmbedding

from models.fourier import Fourier
from models.bs_roformer import FREQ_NUM_PER_BANDS, BandSplit, BandCombine


class EncDec(Fourier):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        time_stacks: int = 4,
        dim: int = 384
    ):
        super().__init__(n_fft, hop_length)

        self.dim = dim

        self.cmplx_num = 2
        self.audio_channels = 2
        self.time_stacks = time_stacks

        band_input_dims = self.cmplx_num * self.audio_channels * self.time_stacks * np.array(FREQ_NUM_PER_BANDS)
        band_input_dims = list(band_input_dims)

        self.band_split = BandSplit(
            band_input_dims=band_input_dims,
            dim=dim,
        )

        self.band_combine = BandCombine(
            dim=dim,
            band_output_dims=band_input_dims
        )
        
    def forward(self, mixture):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)

        Constants:
            b: batch_size
            c: channels_num=2
            T: time_steps
            F: n_fft // 2 + 1
            m: time_stacks
            t: time_bins
            f: freq_bins
            z: complex_num=2
        """

        # Complex spectrum.
        complex_sp = self.stft(mixture)
        # shape: (b, c, T, F)

        batch_size = complex_sp.shape[0]
        time_steps = complex_sp.shape[2]

        x = self.process_image(complex_sp)

        x = torch.view_as_real(x)
        # shape: (b, c, T, F, z)

        x = rearrange(x, 'b c (t m) F z -> b t (m F c z)', m=self.time_stacks)
        # shape: (b, t, m*F*c*z)

        x = self.band_split(x)
        # shape: (b, t, f, d)

        x = self.band_combine(x)
        # shape: (b, t, m*F*c*z)

        x = rearrange(x, 'b t (m F c z) -> b c (t m) F z', m=self.time_stacks, c=self.audio_channels, z=self.cmplx_num)
        # (b, c, T, F, z)
        
        x = torch.view_as_complex(x)
        # (b, c, T, F)

        sep_stft = self.unprocess_image(x, time_steps)

        output = self.istft(sep_stft)
        # (b, c, samples_num)

        return output

    def process_image(self, x):

        B, C, T, Freq = x.shape

        pad_len = (
            int(np.ceil(T / self.time_stacks)) * self.time_stacks
            - T
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        return x

    def unprocess_image(self, x, time_steps):

        output = x[:, :, 0 : time_steps, :]

        return output

