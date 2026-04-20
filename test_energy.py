from pathlib import Path
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch import Tensor
import math

from mss.models2.dsp.subband import SubbandFilter, SubbandResampler
from mss.models.fourier import Fourier


def add():

    root = "/datasets/musdb18hq"
    split = "train"
    sr = 48000

    audios_dir = Path(root, split)
    list_names = sorted(os.listdir(audios_dir))

    audio_paths = [Path(audios_dir, name, "mixture.wav") for name in list_names]

    n_fft = 2048
    n_bands = 256

    if False:
        xs = []
        for n, audio_path in enumerate(audio_paths):
            print(n)
            audio, fs = librosa.load(path=audio_path, sr=sr, mono=True) # (channels, audio_samples)
            x = librosa.core.stft(y=audio, n_fft=n_fft, hop_length=480, window='hann', center=True)
            x = np.mean(np.abs(x), axis=-1)
            xs.append(x)
            
            if n == 100:
                break

        xs = np.stack(xs, axis=0)
        pickle.dump(xs, open("_zz.pkl", "wb"))
    else:
        xs = pickle.load(open("_zz.pkl", "rb"))

    # W, _ = init_melbanks(sr, n_fft, n_bands)  # (n_bands, n_fft/2+1) 
    # W, _ = init_melbanks_no_overlap(sr, n_fft, n_bands)  # (n_bands, n_fft/2+1) 
    # W, _ = init_uniform_bands(sr, n_fft, n_bands)  # (n_bands, n_fft/2+1) 
    # W = np.eye(n_fft//2+1)

    # W, _ = init_kqq_bands(xs, sr, n_fft, n_bands)
    W, _ = init_kqq_bands2(xs, sr, n_fft, n_bands)

    y = W @ xs.T
    y = np.mean(y, axis=-1)
    plt.stem(y)
    plt.savefig("_zz.pdf")

    plt.matshow(W, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz2.pdf")

    out_path = "_W_kqq_bands_22b.pkl"
    pickle.dump(W, open(out_path, "wb"))
    print(f"Write out to {out_path}")
    
    from IPython import embed; embed(using=False); os._exit(0)

    # audio_paths[stem] = str(Path(audios_dir, audio_names[stem], "{}.wav".format(stem)))


def init_melbanks(sr, n_fft, n_bands):
    r"""Initialize mel bins from librosa.

    f: stft bins
    k: mel bins

    Args:
        None

    Returns:
        melbanks: (k, f)
        ola_window: (f,)
    """

    melbanks = librosa.filters.mel(
        sr=sr, 
        n_fft=n_fft, 
        n_mels=n_bands - 2, 
        norm=None
    )  # shape: (k, f)

    F = n_fft // 2 + 1
    
    # The zeroth bank, e.g., [1., 0.66, 0.32, 0, ..., 0.]
    melbank_0 = np.zeros(F)
    idx = np.argmax(melbanks[0])
    melbank_0[0 : idx] = 1. - melbanks[0, 0 : idx]  # (f,)

    # The last bank, e.g., [0., ..., 0., 0.18, 0.87, 1.]
    melbank_last = np.zeros(F)
    idx = np.argmax(melbanks[-1])
    melbank_last[idx :] = 1. - melbanks[-1, idx :]  # (f,)

    # Concatenate
    melbanks = np.concatenate(
        [melbank_0[None, :], melbanks, melbank_last[None, :]], axis=0
    )  # (n_mels, f)

    # Calculate overlap-add window
    ola_window = np.sum(melbanks, axis=0)  # overlap add window
    assert ola_window.max() >= 0.5

    return melbanks, ola_window



def init_melbanks_no_overlap(sr, n_fft, n_bands):
    r"""Initialize mel bins from librosa.

    f: stft bins
    k: mel bins

    Args:
        None

    Returns:
        melbanks: (k, f)
        ola_window: (f,)
    """

    # f = self.sr / self.n_fft * np.arange(self.n_fft // 2 + 1)
    # mel = hz_to_mel(f)
    N = n_fft // 2 + 1
    mel = hz_to_mel(sr / 2)
    f0 = mel_to_hz(mel / (n_bands - 1) * np.arange(n_bands))

    f = sr / n_fft * np.arange(n_fft // 2 + 1)
    # np.abs((f - f0))

    melbanks = np.zeros((n_bands, N))
    for j in range(N):
        i = np.argmin(np.abs(f[j] - f0))
        melbanks[i, j] = 1

    for i in range(1, n_bands):
        if melbanks[i].sum() == 0:
            melbanks[i] = melbanks[i - 1]

    # Calculate overlap-add window
    ola_window = np.sum(melbanks, axis=0)  # overlap add window
    assert ola_window.max() >= 0.5

    return melbanks, ola_window


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1) 


def init_uniform_bands(sr, n_fft, n_bands):
    r"""Initialize mel bins from librosa.

    f: stft bins
    k: mel bins

    Args:
        None

    Returns:
        melbanks: (k, f)
        ola_window: (f,)
    """

    melbanks = np.zeros((n_bands, n_fft // 2 + 1))

    for i in range(n_bands):
        melbanks[i, i * 4 : (i + 1) * 4] = 1
    melbanks[-1, -1] = 1

    # Calculate overlap-add window
    ola_window = np.sum(melbanks, axis=0)  # overlap add window
    assert ola_window.max() >= 0.5

    return melbanks, ola_window


def init_kqq_bands(xs, sr, n_fft, n_bands):

    eng_array = np.mean(xs, axis=0)
    unit_energy = np.sum(eng_array)/64

    banks = np.zeros((n_bands, n_fft // 2 + 1))
    j = 0
    total_energy = 0.
    for i in range(n_bands):
        while total_energy < (i + 1) * unit_energy:
            banks[i, j] = 1
            j += 1
            if j < 1025:
                total_energy += eng_array[j]
            else:
                break

    for i in range(1, n_bands):
        if banks[i].sum(axis=-1) == 0:
            banks[i] = banks[i - 1]

    # Calculate overlap-add window
    ola_window = np.sum(banks, axis=0)  # overlap add window
    assert ola_window.max() >= 0.5

    return banks, ola_window


def init_kqq_bands2(xs, sr, n_fft, n_bands):

    eng_array = np.mean(xs, axis=0)  # (N,)
    sum_array = np.cumsum(eng_array)  # (N,)
    N = len(eng_array)
    dE = np.sum(eng_array) / n_bands
    df = sr / n_fft

    fs = [0]

    for i in range(n_bands):
        for j in range(N - 1):
            if sum_array[j] < (dE * i) <= sum_array[j + 1]:
                f = j + (dE * i - sum_array[j]) / (sum_array[j + 1] - sum_array[j])
                f *= df
                fs.append(f)
    fs.append(sr / 2)
    mel_f = np.array(fs)

    # 
    df = sr / n_fft
    linear_f = np.arange(N) * df

    banks = np.zeros((n_bands, N))

    i = 0
    for j in range(N):
        if 0 <= linear_f[j] <= mel_f[i]:
            banks[i, j] = 1
        elif mel_f[i] < linear_f[j] <= mel_f[i + 1]:
            banks[i, j] = (mel_f[i + 1] - linear_f[j]) / (mel_f[i + 1] - mel_f[i])

    for i in range(1, n_bands - 1):
        for j in range(N):
            if mel_f[i - 1] < linear_f[j] <= mel_f[i]:
                banks[i, j] = (linear_f[j] - mel_f[i - 1]) / (mel_f[i] - mel_f[i - 1])
            elif mel_f[i] < linear_f[j] <= mel_f[i + 1]:
                banks[i, j] = (mel_f[i + 1] - linear_f[j]) / (mel_f[i + 1] - mel_f[i])

    i = n_bands - 1
    for j in range(N):
        if mel_f[i - 1] < linear_f[j] <= mel_f[i]:
            banks[i, j] = (linear_f[j] - mel_f[i - 1]) / (mel_f[i] - mel_f[i - 1])
        elif mel_f[i] < linear_f[j] <= sr / 2:
            banks[i, j] = 1

    for i in range(1, n_bands):
        if np.sum(banks[i]) == 0:
            banks[i] = banks[i - 1]

    # import matplotlib.pyplot as plt
    # plt.matshow(banks, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)

    # Calculate overlap-add window
    ola_window = np.sum(banks, axis=0)  # overlap add window
    assert ola_window.max() >= 0.5

    return banks, ola_window


'''
def compute_entropy(x):
    """x: (t, f)"""
    eps = 1e-8
    x = x / (x.sum(-1) + eps)[:, None]
    entropy = -(x * (x + eps).log2()).sum(dim=-1).mean()
    return entropy
'''

def compute_entropy(x):
    """x: (t, f)"""
    eps = 1e-8
    x = x / (x.sum(-1) + eps)[:, None]
    entropy = -(x * (x + eps).log2()).sum(dim=-1).mean()
    return entropy - math.log2(x.shape[-1])


'''
def compute_entropy(x):
    """x: (t, f)"""
    eps = 1e-8
    # from IPython import embed; embed(using=False); os._exit(0)
    # x = x / (x.sum() + eps)
    entropy = -(x * (x + eps).log2()).sum()
    return entropy
'''

def add2():

    audio_path = "assets/music_10s.wav"
    sr = 48000
    # banks = [[0., 8000.], [8000., 24000.]]
    banks = [[0., 8000.], [8000., 24000.]]
    filter_len = 10001
    device = "cuda"
    eps = 1e-8

    sb_filter = SubbandFilter(sr, banks, filter_len).to(device)
    fourier = Fourier(n_fft=2048, hop_length=480, return_complex=True, normalized=True).to(device)

    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    audio = np.random.randn(audio.shape[-1])

    audio = Tensor(audio[None, None, :]).to(device)

    subbands = sb_filter.analysis(audio)  # (b, c, k, l)

    x = fourier.stft(subbands[:, :, 0, :])  # (b, c, t, f)
    x = x.abs()[0, 0]  # (t, f)
    entropy = compute_entropy(x)# - math.log2(x.shape[0])
    print(entropy)

    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    audio_path = "assets/music_10s.wav"
    sr = 48000
    device = "cuda"
    eps = 1e-8
    fourier = Fourier(n_fft=2048, hop_length=480, return_complex=True, normalized=True).to(device)

    for _ in range(20):
        audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
        audio = np.random.randn(audio.shape[-1])

        audio = Tensor(audio[None, None, :]).to(device)
        x = fourier.stft(audio)  # (b, c, t, f)
        x = x.abs()[0, 0]  # (t, f)
        entropy = compute_entropy(x) - math.log2(x.shape[0])

        print(entropy)

    from IPython import embed; embed(using=False); os._exit(0)


def binary_search(x, left, right):
    
    mid = (left + right) // 2

    if mid == left or mid == right:
        return mid

    entropy1 = compute_entropy(x[:, left : mid])
    entropy2 = compute_entropy(x[:, mid : right])

    if entropy1 > entropy2:
        return binary_search(x, left, mid)
    else:
        return binary_search(x, mid, right)


def search_all(x, layers):

    mid = binary_search(x, 0, x.shape[-1])

    binary_search(x[:, 0 : mid], 0, mid)
    binary_search(x[:, mid :], 0, mid)

def add4():

    audio_path = "assets/music_10s.wav"
    sr = 48000
    device = "cuda"
    eps = 1e-8
    fourier = Fourier(n_fft=2048, hop_length=480, return_complex=True, normalized=True).to(device)

    for _ in range(20):
        audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
        audio = np.random.randn(audio.shape[-1])

        audio = Tensor(audio[None, None, :]).to(device)
        x = fourier.stft(audio)  # (b, c, t, f)
        x = x.abs()[0, 0]  # (t, f)

        mid = binary_search(x, 0, x.shape[-1])



        print(mid)

        from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    add4()