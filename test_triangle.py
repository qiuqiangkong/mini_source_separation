import librosa
import numpy as np
import matplotlib.pyplot as plt
from mss.models2.dsp3.banks import hz_to_erb_ex, erb_to_hz_ex, hz_to_erb, erb_to_hz, hz_to_mel, mel_to_hz



def triangular_filterbank(
    freqs: list[float] | np.ndarray,
    sr: int,
    n_fft: int,
    norm: bool = False,
) -> np.ndarray:
    """
    Build a triangular filterbank.

    Args:
        freqs:
            Frequency nodes in Hz, shape: (n_filters + 2,).
            Example: [0, 100, 200, 400, 800, 16000]
            Each filter uses three adjacent points:
                left = freqs[i]
                center = freqs[i + 1]
                right = freqs[i + 2]

        sr:
            Sampling rate.

        n_fft:
            FFT size.

        norm:
            If True, normalize each filter to unit sum.

    Returns:
        fb:
            Filterbank matrix, shape: (n_filters, n_fft // 2 + 1).
    """

    freqs = np.asarray(freqs, dtype=np.float32)

    if freqs[0] < 0:
        raise ValueError("freqs must be non-negative.")

    if freqs[-1] > sr / 2:
        raise ValueError("freqs[-1] must be <= Nyquist frequency.")

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    n_filters = len(freqs) - 2
    n_bins = len(fft_freqs)

    fb = np.zeros((n_filters, n_bins), dtype=np.float32)

    for i in range(n_filters):
        left = freqs[i]
        center = freqs[i + 1]
        right = freqs[i + 2]

        # Rising slope: left -> center
        left_mask = (fft_freqs >= left) & (fft_freqs <= center)
        fb[i, left_mask] = (
            (fft_freqs[left_mask] - left) / (center - left)
        )

        # Falling slope: center -> right
        right_mask = (fft_freqs >= center) & (fft_freqs <= right)
        fb[i, right_mask] = (
            (right - fft_freqs[right_mask]) / (right - center)
        )

        if norm:
            s = fb[i].sum()
            if s > 0:
                fb[i] /= s

    return fb


def add():
    sr = 48000
    n_fft = 2048
    n_bands = 512
    # n_fft = 256
    n_bands = 64

    # freqs = [
    #     0,
    #     50,
    #     100,
    #     200,
    #     400,
    #     800,
    #     1600,
    #     3200,
    #     6400,
    #     16000,
    # ]
    

    # freqs = np.linspace(0, hz_to_erb(sr / 2), n_bands + 1)
    # freqs = erb_to_hz(freqs)

    # freqs = np.linspace(0, hz_to_mel(sr / 2), n_bands + 1)
    # freqs = mel_to_hz(freqs)

    a = 21.4
    b = 0.0001
    freqs = np.linspace(0, hz_to_erb_ex(sr / 2, a, b), n_bands + 1)
    freqs = erb_to_hz_ex(freqs, a, b)

    fb = triangular_filterbank(
        freqs=freqs,
        sr=sr,
        n_fft=n_fft,
        norm=False,
    )

    print(fb.shape)
    # (8, 513)

    plt.matshow(fb, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")


    #
    sr = 48000
    path = "assets/music_10s.wav"
    audio, fs = librosa.load(path=path, sr=sr, mono=True)

    X = librosa.core.stft(y=audio, n_fft=n_fft, hop_length=480, window='hann', center=True)
    X = (np.abs(X) ** 2).T
    X = X @ fb.T

    plt.matshow(np.log(X).T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz2.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add2():
    sr = 48000
    n_bands = 512
    a = 21.4
    b = 0.0001
    freqs = np.linspace(0, hz_to_erb_ex(sr / 2, a, b), n_bands + 1)
    from IPython import embed; embed(using=False); os._exit(0)
    freqs = erb_to_hz_ex(freqs, a, b)

if __name__ == '__main__':
    add()