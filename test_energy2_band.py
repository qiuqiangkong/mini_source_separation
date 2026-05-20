import math
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import firwin
from torch import Tensor

from mss.utils import fast_sdr

from mss.models2.dsp3.convolve import fftconvolve


class SubbandFilter(nn.Module):
    r"""Save memory version. Split signal into subbands."""

    def __init__(
        self, 
        sr: int, 
        banks: list[tuple[int, int]],
        bandpass_filter_len = 48001,
    ):
        r"""
        k: n_bands
        m: filter_len
        """

        super().__init__()

        self.sr = sr
        self.banks = banks
        self.window_type = "hamming"
        self.bandpass_filter_len = bandpass_filter_len

        # Bandpass filter
        n_banks = len(banks)
        # N = self.bandpass_filter_len - 1
        N = self.bandpass_filter_len
        w = torch.zeros((n_banks, self.bandpass_filter_len))  # (k, n)
        for i in range(n_banks):
            # if i == 0:
            if banks[i][0] == 0:
                # w[i, 1 : ] = self.lowpass(banks[i][1], N)
                w[i] = self.lowpass(banks[i][1], N)
            # elif i == n_banks - 1:
            elif banks[i][1] == self.sr / 2:
                # w[i, 1 :] = self.highpass(banks[i][0], N)
                w[i] = self.highpass(banks[i][0], N)
            else:
                # w[i, 1 :] = self.bandpass(banks[i][0], banks[i][1], N)
                w[i] = self.bandpass(banks[i][0], banks[i][1], N)
        self.register_buffer("w", w)  # (k, n)

    def analysis(self, x: Tensor) -> Tensor:
        r"""Split signal into subbands.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, l)

        Returns:
            out: (b, c, k, l)
        """ 

        B, C, L = x.shape

        # Split into bands
        x = rearrange(x, 'b c l -> (b c) 1 l')
        x = fftconvolve(x, self.w[:, None, :], use_complex_fft=False)  # (b, k, l)
        out = rearrange(x, '(b c) k l -> b c k l', b=B)

        return out
    

    def synthesis(self, x: Tensor) -> Tensor:
        r"""Sum subband signals into original signal.

        b: batch_size
        c: audio_channels
        l: audio_samples
        k: n_bands

        Args:
            x: (b, c, k, l)

        Returns:
            out: (b, c, l)
        """
        return x.sum(2)

    def lowpass(self, f: float, n: int) -> Tensor:
        h = firwin(
            numtaps=n, 
            cutoff=f / (self.sr / 2), 
            pass_zero="lowpass",
            window=self.window_type
        )
        # from IPython import embed; embed(using=False); os._exit(0)
        # h = firwin(numtaps=5, cutoff=f / (self.sr / 2), pass_zero="lowpass",window=self.window_type)
        return torch.from_numpy(h)  # (n,)

    def bandpass(self, f1: float, f2: float, n: int) -> Tensor:
        h = firwin(
            numtaps=n, 
            cutoff=[f1 / (self.sr / 2), f2 / (self.sr / 2)], 
            pass_zero="bandpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)

    def highpass(self, f: float, n: int) -> Tensor:
        h = firwin(
            numtaps=n, 
            cutoff=f / (self.sr / 2), 
            pass_zero="highpass",
            window=self.window_type
        )
        return torch.from_numpy(h)  # (n,)