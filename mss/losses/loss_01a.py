from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Loss01a(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, target: Tensor) -> torch.float:
        loss = F.l1_loss(output, target)
        loss *= 50
        # print(loss)
        return loss


# from mss.models2.dsp3.banks import hz_to_erb_ex, erb_to_hz_ex, hz_to_erb, erb_to_hz, hz_to_mel, mel_to_hz


# class MultiResolutionSTFTLogMelLoss(nn.Module):
#     r"""Multi-resolution STFT loss."""

#     def __init__(
#         self,
#         window_sizes: List[int] = [4096, 2048, 1024, 512, 256],
#         hop_size=147,
#         stft_n_fft=2048,
#         normalized=False,
#         window_fn=torch.hann_window,
#     ) -> None:
#         super(MultiResolutionSTFTLogMelLoss, self).__init__()

#         self.window_sizes = window_sizes
#         self.hop_size = hop_size
#         self.stft_n_fft = stft_n_fft
#         self.normalized = normalized
#         self.window_fn = window_fn
  
#         self.multi_stft_kwargs = dict(
#             hop_length = hop_size,
#             normalized = normalized,
#         )

#         a = 21.4
#         b = 0.0001
#         sr = 48000

#         for window_size in self.window_sizes:
#             n_bands = window_size // 4 + 1
#             freqs = np.linspace(0, hz_to_erb_ex(sr / 2, a, b), n_bands + 1)
#             freqs = erb_to_hz_ex(freqs, a, b)
#             fb = triangular_filterbank(
#                 freqs=freqs,
#                 sr=sr,
#                 n_fft=window_size,
#                 norm=False,
#             )
#             self.register_buffer(f"fb_{window_size}", Tensor(fb))

#     def forward(self, output: Tensor, target: Tensor) -> Tensor:
#         loss1 = 0.
#         loss2 = 0.

#         for window_size in self.window_sizes:
#             stft_kwargs = dict(
#                 n_fft = max(window_size, self.stft_n_fft),
#                 win_length = window_size,
#                 return_complex = True,
#                 window = self.window_fn(window_size).to(output.device),
#                 **self.multi_stft_kwargs
#             )
            
#             output_Y = torch.stft(rearrange(output, "... s t -> (... s) t"), **stft_kwargs)
#             target_Y = torch.stft(rearrange(target, "... s t -> (... s) t"), **stft_kwargs)
            
#             loss1 = loss1 + F.l1_loss(output_Y, target_Y)
        
#         for window_size in [256]:
#             stft_kwargs = dict(
#                 n_fft = window_size,
#                 win_length = window_size,
#                 return_complex = True,
#                 window = self.window_fn(window_size).to(output.device),
#                 **self.multi_stft_kwargs
#             )
            
#             out = torch.stft(rearrange(output, "... s t -> (... s) t"), **stft_kwargs)  # (b, f, t)
#             tar = torch.stft(rearrange(target, "... s t -> (... s) t"), **stft_kwargs)  # (b, f, t)
            
#             out = rearrange(out, 'b f t -> b t f').abs() ** 2
#             tar = rearrange(tar, 'b f t -> b t f').abs() ** 2
#             out = out @ getattr(self, f"fb_{window_size}").T
#             tar = tar @ getattr(self, f"fb_{window_size}").T
#             out = torch.clamp(out, min=1e-10)
#             tar = torch.clamp(tar, min=1e-10)
#             out = torch.clamp(10. * torch.log10(out), -60)
#             tar = torch.clamp(10. * torch.log10(tar), -60)

#             # from IPython import embed; embed(using=False); os._exit(0)
#             # import matplotlib.pyplot as plt
#             # plt.figure()
#             # plt.matshow(tar[6].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
#             # plt.savefig("_zz.pdf")

#             # import soundfile
#             # soundfile.write(file="_zz.wav", data=target[3].cpu().numpy().T, samplerate=48000)
            

#             loss2 = loss2 + F.l1_loss(out, tar)

#         loss2 *= 0.1
#         loss =  loss1 + loss2

#         return loss



# def triangular_filterbank(
#     freqs: list[float] | np.ndarray,
#     sr: int,
#     n_fft: int,
#     norm: bool = False,
# ) -> np.ndarray:
#     """
#     Build a triangular filterbank.

#     Args:
#         freqs:
#             Frequency nodes in Hz, shape: (n_filters + 2,).
#             Example: [0, 100, 200, 400, 800, 16000]
#             Each filter uses three adjacent points:
#                 left = freqs[i]
#                 center = freqs[i + 1]
#                 right = freqs[i + 2]

#         sr:
#             Sampling rate.

#         n_fft:
#             FFT size.

#         norm:
#             If True, normalize each filter to unit sum.

#     Returns:
#         fb:
#             Filterbank matrix, shape: (n_filters, n_fft // 2 + 1).
#     """

#     freqs = np.asarray(freqs, dtype=np.float32)

#     if freqs[0] < 0:
#         raise ValueError("freqs must be non-negative.")

#     if freqs[-1] > sr / 2:
#         raise ValueError("freqs[-1] must be <= Nyquist frequency.")

#     fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

#     n_filters = len(freqs) - 2
#     n_bins = len(fft_freqs)

#     fb = np.zeros((n_filters, n_bins), dtype=np.float32)

#     for i in range(n_filters):
#         left = freqs[i]
#         center = freqs[i + 1]
#         right = freqs[i + 2]

#         # Rising slope: left -> center
#         left_mask = (fft_freqs >= left) & (fft_freqs <= center)
#         fb[i, left_mask] = (
#             (fft_freqs[left_mask] - left) / (center - left)
#         )

#         # Falling slope: center -> right
#         right_mask = (fft_freqs >= center) & (fft_freqs <= right)
#         fb[i, right_mask] = (
#             (right - fft_freqs[right_mask]) / (right - center)
#         )

#         if norm:
#             s = fb[i].sum()
#             if s > 0:
#                 fb[i] /= s

#     return fb