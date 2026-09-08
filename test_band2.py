import numpy as np
from torch import Tensor
import time

from mss.models2.dsp3.banks import mel_linear_banks_triangle
from mss.models2.dsp3.subband_fast_triangle import SubbandFilter
from mss.utils import fast_sdr


def add():
    sr = 48000
    n_bins = 64

    if n_bins == 32:
        n_bands = 28
        max_half_bandwidth = 1590
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (1600 * 2)  # Can be smaller but not larger!
        device = "cuda"

    elif n_bins == 64:
        n_bands = 58
        max_half_bandwidth = 790
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (800 * 2)  # Can be smaller but not larger!
        device = "cuda"

    elif n_bins == 128:
        n_bands = 118
        max_half_bandwidth = 390
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (400 * 2)  # Can be smaller but not larger!
        device = "cuda"

    elif n_bins == 256:
        n_bands = 235
        max_half_bandwidth = 190
        chunk_size = 16  # Try to tune this to balance RAM and computation speed
        factor = sr // (200 * 2)  # Can be smaller but not larger!
        device = "cuda"

    # Melbanks
    banks = mel_linear_banks_triangle(sr, n_bands, max_half_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size).to(device)
    print(len(banks))

    for _ in range(5):

        # Audio
        rs = np.random.RandomState(1234)
        audio = rs.uniform(low=-1, high=1, size=(4, 2, sr * 2))
        audio = Tensor(audio).to(device)  # (c, l)
                
        # Analysis
        t0 = time.time()
        x = sb_filter.analysis(audio)  # (b, c, k, l)
        y = sb_filter.synthesis(x)
        
        # Print
        t1 = time.time() - t0
        sdr = fast_sdr(audio.cpu().numpy(), y.cpu().numpy())
        print(f"time: {t1:.4f} s, latent: {x.shape}, SDR: {sdr:.2f} dB")


if __name__ == '__main__':
    add()