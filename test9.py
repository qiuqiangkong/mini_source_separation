import numpy as np
import math
import matplotlib.pyplot as plt
from mss.models.fourier import Fourier
from torch import Tensor
import torch
import librosa
from pathlib import Path
import soundfile
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time
import torchaudio
from torch import einsum


def add():

    sr = 48000
    n_fft = 2048
    hop_length = 480

    t = np.arange(sr)
    x = np.sin(2 * math.pi * 392. * t / sr)
    plt.plot(x[0:1000])
    plt.savefig("_zz.pdf")
    # asdf


    # tmp = librosa.util.frame(x, frame_length=2048, hop_length=480)
    # fig, axs = plt.subplots(4, 1, sharex=True)
    # axs[0].plot(tmp[:, 0])
    # axs[1].plot(tmp[:, 1])
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)

    model = Fourier(n_fft=n_fft, hop_length=hop_length, return_complex=True, normalized=True)
    x = Tensor(x[None, None, :])
    X = model.stft(x)

    X[0, 0, :, 19].abs()
    X[0, 0, :, 19].angle()
    X[0, 0, :, 19]

    T = X.shape[2]
    t = torch.exp(- 1.j * 2 * math.pi / n_fft * hop_length * torch.arange(T))
    f = torch.arange(n_fft // 2 + 1)
    W = torch.outer(t, f)

    Y = X * W[None, None, :, :]
    Y[0, 0, :, 19].angle()

    plt.matshow(np.abs(X[0, 0].numpy().T), origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")

    a1 = np.exp(1.j * 2 * math.pi / n_fft * hop_length * 19)
    X[0, 0, 10, 19] * a1
    X[0, 0, 11, 19]

    X[0, 0, :, 19].angle()

    from IPython import embed; embed(using=False); os._exit(0)

    
def add2():

    sr = 48000
    n_fft = 2048
    hop_length = 480

    t = np.arange(sr)
    # f1 = 392.
    f1 = sr / n_fft * 20
    # f2 = 847.
    f2 = sr / n_fft * 21
    f3 = 8645.
    x = np.sin(2 * math.pi * f1 * t / sr) + np.sin(2 * math.pi * f2 * t / sr) + np.sin(2 * math.pi * f3 * t / sr)
    plt.plot(x[0:1000])
    plt.savefig("_zz.pdf")

    X = torch.stft(Tensor(x), n_fft=n_fft, hop_length=hop_length, center=False, normalized=True, return_complex=True).T

    2 * math.pi / n_fft * hop_length * 17
    2 * math.pi / n_fft * hop_length * 17 % (2 * math.pi)
    a1 = np.exp(1.j * 2 * math.pi / n_fft * hop_length * 17)
    X[0:30, 17]
    X[10, 17] * a1
    X[11, 17]

    X[1:30, 17].angle() - X[0:29, 17].angle()


    T = X.shape[0]
    t = - 2 * math.pi / n_fft * hop_length * torch.arange(T)
    f = torch.arange(n_fft // 2 + 1)
    W = torch.exp(1.j * torch.outer(t, f))

    # b1 = X.angle() + W.angle()
    b1 = X * W

    X[1:30, 17].angle()
    W[1:30, 17].angle()

    # X[1:30, 17].angle() - W[1:30, 17].angle()
    b1[1:30, 17].angle()

    fig, axs = plt.subplots(1,2, sharex=True)
    # axs[0].matshow(b1.abs().numpy().T, origin='lower', aspect='auto', cmap='jet')
    # axs[1].matshow(c1.angle().numpy().T, origin='lower', aspect='auto', cmap='jet', vmin=-5, vmax=5)
    axs[0].matshow((torch.cos(X.angle()) * X.abs()).numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet', vmin=-4, vmax=4)
    axs[1].matshow((torch.cos(b1.angle()) * X.abs()).numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet', vmin=-4, vmax=4)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# DFT class
class DFT:
  def __init__(self, N):
    self.W = np.zeros((N, N), dtype=np.complex64)

    for n in range(N):
      for k in range(N):
        self.W[n, k] = np.exp(-1.j * (2 * math.pi / N) * k * n)

  def transform(self, x):
    output = x @ self.W
    return output


# Short-time Fourier transform function.
def stft(audio, n_fft, hop_length):

  frames = librosa.util.frame(audio, frame_length=n_fft, hop_length=hop_length).T
  # (frames_num, n_fft)

  print("frames:", frames.shape)

  stft_matrix = np.zeros_like(frames, dtype=np.complex64)

  dft = DFT(N=n_fft)

  for frame_index, frame in enumerate(frames):
    stft_matrix[frame_index] = dft.transform(frame)

  return stft_matrix


def add3():

    sr = 48000
    n_fft = 2048
    hop_length = 480

    t = np.arange(10000)
    x = np.sin(2 * math.pi * 392. * t / sr)
    # plt.plot(x[0:1000])
    # plt.savefig("_zz.pdf")

    # X = torch.stft(Tensor(x), n_fft=n_fft, hop_length=hop_length, center=False, normalized=True, return_complex=True).T
    X = stft(x, n_fft, hop_length)

    2 * math.pi / n_fft * hop_length * 16
    2 * math.pi / n_fft * hop_length * 16 % (2 * math.pi)
    a1 = np.exp(-1.j * 2 * math.pi / n_fft * hop_length * 16)
    X[0:30, 16]
    X[10, 16] * a1
    X[11, 16]

    np.angle(X[1:16, 16]) - np.angle(X[0:15, 16])

    from IPython import embed; embed(using=False); os._exit(0)


def add4():
    sr = 48000
    n_fft = 2048
    hop_length = 480
    mono = True
    stems = ["vocals", "bass", "drums", "other"]
    audio_dir = "/datasets/musdb18hq/test/Al James - Schoolboy Facination"

    data = {}

    for stem in stems:
        audio_path = Path(audio_dir, f"{stem}.wav")
        audio, fs = librosa.load(path=audio_path, sr=sr, mono=mono)
        data[stem] = audio[sr * 30 : sr * 31]
    
    mixture = 0
    for stem in stems:
        mixture += data[stem]

    # soundfile.write(file="_zz1.wav", data=data["vocals"], samplerate=sr)
    # soundfile.write(file="_zz2.wav", data=mixture, samplerate=sr)

    model = Fourier(n_fft=n_fft, hop_length=hop_length, return_complex=True, normalized=True)

    mix_stft = model.stft(Tensor(mixture[None, None, :]))
    vocals_stft = model.stft(Tensor(data["vocals"][None, None, :]))

    X = mix_stft[0, 0]
    Y = vocals_stft[0, 0]

    W = torch.exp(- 1.j * X.angle())
    X *= W
    Y *= W

    fig, axs = plt.subplots(2,2, sharex=True)
    axs[0, 0].matshow(X.abs().numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet')
    axs[0, 1].matshow(Y.abs().numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet')
    # axs[1, 0].matshow((X.angle()).numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet', vmin=-4, vmax=4)
    # axs[1, 1].matshow((Y.angle()).numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet', vmin=-4, vmax=4)
    axs[1, 0].matshow((X.abs() * X.angle()).numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet', vmin=-2, vmax=2)
    axs[1, 1].matshow((Y.abs() * Y.angle()).numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet', vmin=-2, vmax=2)
    # axs[1].matshow((torch.cos(b1.angle()) * X.abs()).numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet', vmin=-4, vmax=4)
    plt.savefig("_zz.pdf")


    from IPython import embed; embed(using=False); os._exit(0)


# DFT
def add6():

    sr = 48000
    N = 2048

    # W = np.exp(-1.j * 2 * math.pi / N)
    n = torch.arange(0, N)
    k = torch.arange(0, N)
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k, n)) 

    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    frames = librosa.util.frame(audio, frame_length=N, hop_length=N).T  # (t, n)
    frames = torch.from_numpy(frames.astype(np.complex64))
    Y = frames @ W.T

    inv_W = torch.linalg.inv(W.T @ W) @ W.T
    Z = Y @ inv_W.T
    print((frames - Z).abs().max())

    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[0].matshow(Y.abs().numpy()[:, 0:100].T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add7():

    sr = 48000
    # N = 2048
    N = 16

    # f = Tensor(librosa.mel_frequencies(n_mels=N, fmin=0.0, fmax=sr/2))
    f = Tensor(librosa.mel_frequencies(n_mels=100, fmin=0, fmax=sr))
    
    n = torch.arange(0, N)
    k = f * N / sr
    # k = torch.arange(N)
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k, n))

    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    frames = librosa.util.frame(audio, frame_length=N, hop_length=N).T  # (t, n)
    frames = torch.from_numpy(frames.astype(np.complex64))
    Y = frames @ W.T

    lam = 0.
    inv_W = torch.linalg.pinv(W.T @ W + lam * torch.eye(N, dtype=torch.complex64)) @ W.T
    Z = Y @ inv_W.T
    print((frames - Z).abs().max(), (frames - Z).abs().mean())

    # Z = torch.linalg.lstsq(W, Y.T).solution.T
    # print((frames - Z).abs().mean())

    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[0].matshow(Y.abs().numpy()[:, :].T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


class DFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        N = n_fft
        n = torch.arange(0, N)
        k = torch.arange(0, N)
        W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k, n))
        W_inv = torch.linalg.pinv(W.T @ W) @ W.T
        self.register_buffer("W", W)
        self.register_buffer("W_inv", W_inv)

    def encode(self, x):
        return x.to(torch.complex64) @ self.W.T

    def decode(self, x):
        return x.to(torch.complex64) @ self.W_inv.T


class DFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        N = n_fft
        n = torch.arange(0, N)
        k = torch.arange(0, N)
        W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k, n)) / math.sqrt(N)
        W_inv = torch.linalg.pinv(W.T @ W) @ W.T
        self.register_buffer("W", W)
        self.register_buffer("W_inv", W_inv)

    def encode(self, x):
        return x.to(torch.complex64) @ self.W.T

    def decode(self, x):
        return x.to(torch.complex64) @ self.W_inv.T


class DFTLearnableDecoder(nn.Module):
    def __init__(self, n_fft, hop_length, requires_grad=True):
        super().__init__()
        N = n_fft
        n = torch.arange(0, N)
        k = torch.arange(0, N)
        W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k, n)) / math.sqrt(N)
        W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k, n)) / math.sqrt(N)
        self.register_buffer("W", W)
        self.register_buffer("W_dec", W_dec)
        self.beta = nn.Parameter(torch.ones(N), requires_grad=requires_grad)

    def encode(self, x):
        return x.to(torch.complex64) @ self.W.T

    def decode(self, x):
        return (x * self.beta).to(torch.complex64) @ self.W_dec.T


# Test DFT
def add8():

    n_fft = 16
    hop_length = 16
    model = DFT(n_fft, hop_length)

    x = torch.randn(4, 2, 16)
    y = model.encode(x)
    x_hat = model.decode(y)
    print((x - x_hat).abs().max())
    from IPython import embed; embed(using=False); os._exit(0)


# Test Learnable DFT
def add9():
    
    n_fft = 16
    hop_length = 16
    model = DFTLearnableDecoder(n_fft, hop_length)

    x = torch.randn(4, 2, 16)
    y = model.encode(x)
    x_hat = model.decode(y)
    print((x - x_hat).abs().max())
    from IPython import embed; embed(using=False); os._exit(0)


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

        N = n_fft
        n = torch.arange(0, N)
        k = torch.arange(0, N)
        W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k, n)) / math.sqrt(N)
        W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k, n)) / math.sqrt(N)

        self.register_buffer("W_enc", W_enc)
        self.register_buffer("W_dec", W_dec)

    def encode(self, x: Tensor) -> Tensor:
        r"""

        b: batch_size
        c: num_channels
        l: segment_samples
        t: num_frames
        f: freq_bins
        n: frame_samples

        Args:
            x: (b, c, l)

        Returns: 
            out: (b, c, t, f)
        """

        x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        x = x.unfold(dimension=-1, size=self.n_fft, step=self.hop_length).contiguous()  # (b, t, n)
        x *= self.window
        out = x.to(torch.complex64) @ self.W_enc.T
        out = out[..., 0 : self.n_fft // 2 + 1]
        return out

    def decode(self, x: Tensor, length: int | None) -> Tensor:
        r"""

        b: batch_size
        c: num_channels
        l: segment_samples
        t: num_frames
        f: freq_bins

        Args:
            x: (b, c, t, f)

        Returns:
            out: (b, c, l)
        """

        # Inverse transform
        x_flip = torch.flip(x[..., 1 : -1], dims=[-1]).conj()
        x = torch.cat([x, x_flip], dim=-1)
        x = x @ self.W_dec.T

        # Overlap-add
        out = fold(
            x=rearrange(x, 'b c t n -> (b c) n t'), 
            hop_length=self.hop_length, 
            window=self.window
        )  # (b*c, l)
        out = rearrange(out, '(b c) l -> b c l', b=x.shape[0])
        
        # Remove padding
        out = out[..., self.n_fft // 2 :]
        
        if length is not None:
            out = out[..., 0 : length]

        return out
        


def fold(x: Tensor, hop_length: int, window: Tensor | None):
    r"""

    b: batch_size
    t: num_frames
    n: frame_samples
    l: segment_samples

    Args:
        x: (b, n, t)

    Returns:
        x: (b, l)
    """

    frame_length, num_frames = x.shape[-2:]  # (t, n)
    L = frame_length + (num_frames - 1) * hop_length

    # Overlap-add
    x = F.fold(
        input=x,  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, c, 1, l)
    out = x.squeeze(dim=[1, 2])  # (b, l)

    # Divide overlap-add window
    if window is not None:
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, num_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        )  # (1, 1, 1, L)
        win_norm = win_norm.squeeze(dim=[0, 1, 2])  # (l,)

        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out


# Test torch.stft
def add10():
    
    sr = 48000
    L = 48000
    C = 2
    n_fft = 2048
    hop_length = 480
    window = torch.hann_window(n_fft)
    normalized = True
    return_complex = True

    # x = torch.randn(8, L)
    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    audio = np.stack([audio, audio], axis=0)
    x = Tensor(audio[None, :, 0 : sr])

    y = torch.stft(
        input=rearrange(x, 'b c l -> (b c) l'), 
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        normalized=normalized,
        return_complex=return_complex
    )  # (b*c, f, t)
    y = rearrange(y, '(b c) f t -> b c t f', c=C)

    x_hat = torch.istft(
        input=rearrange(y, 'b c t f -> (b c) f t'), 
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        normalized=normalized,
    )  # (b*c, l)
    x_hat = rearrange(x_hat, '(b c) l -> b c l', c=C)

    print((x - x_hat).abs().max())
    
    #
    model = STFT(n_fft, hop_length)
    y2 = model.encode(x)
    print((y - y2).abs().max())


    x2_hat = model.decode(y2, x.shape[-1])
    print((x - x2_hat).abs().max())



    # fig, axs = plt.subplots(1,2, sharex=True)
    # axs[0].matshow(y.abs().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
    # axs[1].matshow(y2.abs().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")


    from IPython import embed; embed(using=False); os._exit(0)


# Test ERB
def add11():

    # for f0 in [0, 100, 1000, 10000, 24000]:
    #     erb = hz_to_erb(f0)
    #     f1 = erb_to_hz(erb)
    #     print(erb, f1)

    tmp = []
    for erb in np.linspace(0, hz_to_erb(24000), num=64*100):
        f1 = erb_to_hz(erb)
        print(erb, f1)
        tmp.append(f1)

    tmp = np.array(tmp)
    print(tmp[1:] - tmp[0:-1])
    # from IPython import embed; embed(using=False); os._exit(0)


# Test Mel
def add12():
    # for f0 in [0, 100, 1000, 10000, 24000]:
    #     erb = hz_to_mel(f0)
    #     f1 = mel_to_hz(erb)
    #     print(erb, f1)

    # asdf

    tmp = []
    for mel in np.linspace(0, hz_to_mel(24000), num=64*100):
        f1 = mel_to_hz(mel)
        print(mel, f1)
        tmp.append(f1)

    tmp = np.array(tmp)
    print(tmp[1:] - tmp[0:-1])
    # from IPython import embed; embed(using=False); os._exit(0)


# Test coeff
def add13():
    path = "checkpoints/train2_recon/recon_01b/step=10000_ema.pth"
    ckpt = torch.load(path)
    a1 = ckpt["model.beta_dec"].data.cpu().numpy()

    plt.stem(a1)
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def add14():

    sr = 48000
    # N = 2048
    N = 16

    # f = Tensor(librosa.mel_frequencies(n_mels=N, fmin=0.0, fmax=sr/2))
    # f = Tensor(librosa.mel_frequencies(n_mels=100, fmin=0, fmax=sr))

    sr = 48000
    n_bands = 64 * 100
    mel = np.linspace(0, hz_to_mel(sr / 2), num=n_bands)
    f = Tensor(mel_to_hz(mel))

    n = torch.arange(0, N)
    k = f * N / sr
    # k = torch.arange(N)
    
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k, n))
    lam = 0.
    W_dec = torch.linalg.pinv(W.T @ W + lam * torch.eye(N, dtype=torch.complex64)) @ W.T

    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    frames = librosa.util.frame(audio, frame_length=N, hop_length=N).T  # (t, n)
    frames = torch.from_numpy(frames.astype(np.complex64))
    Y = frames @ W_enc.T

    # 
    x = Y
    x_flip = torch.flip(x[..., 1 : -1], dims=[-1]).conj()
    x = torch.cat([x, x_flip], dim=-1)
    x = (x * self.beta_dec) @ self.W_dec.T

    from IPython import embed; embed(using=False); os._exit(0)

    # IE
    
    Z = Y @ inv_W.T
    print((frames - Z).abs().max(), (frames - Z).abs().mean())




    # Z = torch.linalg.lstsq(W, Y.T).solution.T
    # print((frames - Z).abs().mean())

    # fig, axs = plt.subplots(1, 2, sharex=True)
    # axs[0].matshow(Y.abs().numpy()[:, :].T, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


# pinv
def add15():

    sr = 48000
    N = 2048

    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    frames = librosa.util.frame(audio, frame_length=N, hop_length=N).T  # (t, n)
    frames = torch.from_numpy(frames.astype(np.complex64))

    n = torch.arange(0, N)
    # k1 = f * N / sr
    k1 = torch.arange(0, N // 2 + 1)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)
    
    # Forward
    Y = frames @ W_enc.T

    # Inverse
    lam = 0.
    W = W_enc
    W_dec = torch.linalg.pinv(W.T @ W + lam * torch.eye(N, dtype=torch.complex64)) @ W.T

    Z = Y @ W_dec.T

    print((frames - Z).abs().max())

    from IPython import embed; embed(using=False); os._exit(0)


def add15b():

    sr = 48000
    N = 2048

    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    frames = librosa.util.frame(audio, frame_length=N, hop_length=N).T  # (t, n)
    frames = torch.from_numpy(frames.astype(np.complex64))

    n = torch.arange(0, N)

    f = Tensor(librosa.mel_frequencies(n_mels=1000, fmin=0, fmax=sr // 2))
    k1 = f * N / sr

    # k1 = torch.arange(0, N // 2 + 1)

    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)
    
    # Forward
    Y = frames @ W_enc.T

    # Inverse
    lam = 0.
    W = W_enc
    W_dec = torch.linalg.pinv(W)

    Z = Y @ W_dec.T

    print((frames - Z).abs().max(), (frames - Z).abs().mean())

    from IPython import embed; embed(using=False); os._exit(0)


# Fractional FFT and inverse
def add15c():

    sr = 48000
    N = 2048

    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    frames = librosa.util.frame(audio, frame_length=N, hop_length=N).T  # (t, n)
    frames = torch.from_numpy(frames.astype(np.complex64))

    n = torch.arange(0, N)

    n_bands = 6400
    mel = np.linspace(0, hz_to_mel(sr / 2), num=n_bands)
    f = Tensor(mel_to_hz(mel))
    k1 = f * N / sr

    # k1 = torch.arange(0, N // 2 + 1)

    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)
    
    # Forward
    Y = frames @ W_enc.T

    # Inverse
    lam = 0.
    W = W_enc
    # W_dec = torch.linalg.pinv(W.T @ W + lam * torch.eye(N, dtype=torch.complex64)) @ W.T
    W_dec = torch.linalg.pinv(W)

    Z = Y @ W_dec.T

    print((frames - Z).abs().max(), (frames - Z).abs().mean())
    print(W.abs().mean())


    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(W_dec[:, 0].abs())
    axs[1].plot(W_dec[:, 10].abs())
    axs[2].plot(W_dec[:, 100].abs())
    axs[3].plot(W_dec[:, 1000].abs())
    axs[4].plot(W_dec[:, 10000].abs())
    plt.savefig("_zz.pdf")

    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(W_dec[:, 0].real)
    axs[1].plot(W_dec[:, 10].real)
    axs[2].plot(W_dec[:, 100].real)
    axs[3].plot(W_dec[:, 1000].real)
    axs[4].plot(W_dec[:, 10000].real)
    plt.savefig("_zz.pdf")
    # axs[0].matshow(Y.abs().numpy()[:, :].T, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add16():

    from mss.models2.gabor_transform import GaborTransform
    from mss.models.fourier import Fourier

    sr = 48000
    device = "cuda"
    
    x = torch.randn((4, 2, sr * 2), device=device)

    gabor = GaborTransform(
        n_ffts=[2048],
        hop_lengths=[480],
        r=1,
    ).to(device)

    y = gabor.encode(x)
    x_hat = gabor.decode(y, x.shape[-1])
    print("Error: {}".format((x - x_hat).abs().mean()))

    fourier = Fourier(n_fft=2048, hop_length=480).to(device)
    y2 = fourier.stft(x)
    y2 - y[0]
    from IPython import embed; embed(using=False); os._exit(0)


def add17():

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    a1 = nn.Parameter(torch.zeros((10)))
    nn.init.uniform_(a1, -1, 1)

    a2 = nn.Parameter(torch.zeros((5)))
    nn.init.uniform_(a2, -1, 1)
    print(a1)
    print(a2)


def add18():

    from mss.augmentations.gain import RandomGain
    from mss.augmentations.eq import RandomEQ
    from mss.augmentations.pitch import RandomPitch
    from mss.augmentations.resample_stretch import RandomResampleStretch
    from mss.augmentations.resample import RandomResample

    x = np.random.uniform(low=-1, high=1, size=(2, 96000))
    # x = x[None, :]

    gain = RandomGain()
    eq = RandomEQ(sr=48000)
    pitch = RandomPitch(sr=48000)
    resample = RandomResample(sr=48000)
    rs = RandomResampleStretch(sr=48000)

    for _ in range(10):
        t1 = time.time()
        x = gain(x)
        x = eq(x)
        # x = pitch(x)
        # x = resample(x)
        print(time.time() - t1)

    from IPython import embed; embed(using=False); os._exit(0)


def add19():

    x = np.random.uniform(low=-1, high=1, size=(960000,))

    if True:
        for _ in range(100):
            t1 = time.time()
            librosa.resample(y=x, orig_sr=48000, target_sr=47990, res_type="soxr_hq")
            print(time.time() - t1)

    else:
        x = Tensor(x)
        for _ in range(100):
            t1 = time.time()
            torchaudio.functional.resample(x, orig_freq=48000, new_freq=47990, resampling_method="sinc_interp_kaiser")
            print(time.time() - t1)


def add20():
    x = np.random.uniform(low=-1, high=1, size=(960000,))

    if True:
        for _ in range(100):
            t1 = time.time()
            librosa.effects.pitch_shift(x, sr=48000, n_steps=1.23)
            print(time.time() - t1)


def add20c():

    x = np.random.uniform(low=-1, high=1, size=(960000,))

    if False:
        for _ in range(100):
            t1 = time.time()
            librosa.core.stft(y=x, n_fft=2048, hop_length=512, window='hann', center=True)
            print(time.time() - t1)

    else:
        x = Tensor(x)
        for _ in range(100):
            t1 = time.time()
            torch.stft(
                input=x[None, :],  # (b*c, l)
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048),
                normalized=True,
                return_complex=True
            )  # (b*c, f, t)
            print(time.time() - t1)
    

def add21():

    sr = 48000
    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=False)

    from mss.augmentations.numpy.gain import RandomGain
    from mss.augmentations.numpy.resample import RandomResample
    from mss.augmentations.numpy.pitch import RandomPitch
    from mss.augmentations.torch.eq import RandomEQ

    for _ in range(100):
        t1 = time.time()
        # random_gain = RandomGain()
        # y = random_gain(audio)

        # random_resample = RandomResample(sr=sr)
        # y = random_resample(audio)

        # random_pitch = RandomPitch(sr=sr)
        # y = random_pitch(audio)

        random_eq = RandomEQ()
        y = random_eq(Tensor(audio))
        # y = random_eq(Tensor(audio)[None, :, :])[0]

        print(time.time() - t1)

    soundfile.write(file="_zz.wav", data=audio.T, samplerate=sr)
    soundfile.write(file="_zz2.wav", data=y.T, samplerate=sr)
    # y / audio
    from IPython import embed; embed(using=False); os._exit(0)


def add22():
    # x = np.arange(10)
    f = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    x = np.log10(f)
    y = np.ones(10)
    plt.scatter(x, y, s=4)
    plt.savefig("_zz.pdf")


def add23():

    import pywt
    import numpy as np
    import matplotlib.pyplot as plt

    # 示例信号
    t = np.linspace(0, 1, 1024)
    x = np.sin(50*2*np.pi*t) + np.sin(120*2*np.pi*t)  # 两个频率叠加
    x += 0.2 * np.random.randn(len(t))  # 加一点噪声

    # 选择 Daubechies 小波
    wavelet = 'db4'  # db2 或 db4

    # 分解层数
    level = 4

    # DWT 分解
    coeffs = pywt.wavedec(x, wavelet, level=level)

    # coeffs = [cA4, cD4, cD3, cD2, cD1]
    cA = coeffs[0]  # 最低频逼近
    cDs = coeffs[1:]  # 各层细节

    # 重构信号
    x_rec = pywt.waverec(coeffs, wavelet)

    # 绘图
    plt.figure(figsize=(12,6))

    plt.subplot(2,1,1)
    plt.plot(t, x)
    plt.title("Original Signal")

    plt.subplot(2,1,2)
    plt.plot(t, x_rec)
    plt.title(f"Reconstructed Signal using {wavelet}")

    plt.tight_layout()
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add24():

    m1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
    m2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2, bias=False, output_padding=1)
    with torch.no_grad():
        m2.weight[0, 0] = torch.ones(5)

    x = torch.zeros((4, 1, 10))
    m1(x)

    y = torch.Tensor([0, 2, 4, 6]).float()[None, None, :]
    y2 = m2(y)

    from IPython import embed; embed(using=False); os._exit(0)

def add25():

    from cqt_pytorch import CQT

    transform = CQT(
        num_octaves = 8,
        num_bins_per_octave = 64,
        sample_rate = 48000,
        block_length = 2 ** 18
        # block_length = None
    )

    # (Random) audio waveform tensor x
    x = torch.randn(1, 2, 2**18) # [1, 1, 262144] = [batch_size, channels, timesteps]
    z = transform.encode(x) # [1, 2, 512, 2839] = [batch_size, channels, frequencies, time]
    y = transform.decode(z) # [1, 1, 262144]

    (x - y).abs().mean()

    from IPython import embed; embed(using=False); os._exit(0)



def add26():
    from cqt_pytorch import CQT

    sr = 44100
    transform = CQT(
        num_octaves = 8,
        num_bins_per_octave = 64,
        sample_rate = sr,
        # block_length = 2 ** 18
        # block_length = None
    )

    # (Random) audio waveform tensor x
    # x = torch.randn(1, 2, 2**18) # [1, 1, 262144] = [batch_size, channels, timesteps]
    x, fs = librosa.load(path="./assets/music_10s.wav", sr=sr, mono=False)
    
    x = torch.from_numpy(x)[None, :, 0 : sr*10]

    z = transform.encode(x) # [1, 2, 512, 2839] = [batch_size, channels, frequencies, time]
    y = transform.decode(z) # [1, 1, 262144]

    (x - y).abs().mean()

    soundfile.write(file="_zz.wav", data=y.data.numpy()[0].T, samplerate=sr)

    import matplotlib.pyplot as plt
    plt.matshow(np.abs(z).data.numpy()[0, 0], origin='lower', aspect='auto', cmap='jet')   
    plt.savefig("_zz.pdf")

    from mss.utils import fast_sdr
    fast_sdr(x.numpy(), y.numpy())
    from IPython import embed; embed(using=False); os._exit(0)


# 4-point DFT
def add27():

    N = 4
    x = np.arange(N)
    tmp = np.exp(-1.j * 2 * math.pi / N)
    w = tmp ** (np.arange(N)[None, :] * np.arange(N)[:, None]) / np.sqrt(N)

    y = x @ w
    z = y @ np.conj(w).T
    print(np.mean(np.abs(x - z)))

    from IPython import embed; embed(using=False); os._exit(0)
    

# 7-point DFt
def add27b():

    np.set_printoptions(precision=2, suppress=True)
    N = 4
    K = 7
    x = np.arange(N)
    tmp = np.exp(-1.j * 2 * math.pi / K)
    w = tmp ** (np.arange(K)[None, :] * np.arange(4)[:, None]) / np.sqrt(K)  # (N, K)

    y = x @ w
    iw = np.conj(w).T
    z = y @ iw
    print(np.mean(np.abs(x - z)))

    print(np.linalg.pinv(w))
    print(iw)

    from IPython import embed; embed(using=False); os._exit(0)


def add28a():

    np.set_printoptions(precision=2, suppress=True)

    N = 4
    tmp = 2 * math.pi * np.array([0, 1/4, 2/4, 3/4])
    tmp = np.arange(4)[:, None] * tmp[None, :]
    w = np.exp(-1.j * tmp) / 2

    x = np.arange(N)
    y = x @ w
    iw = np.conj(w).T
    z = y @ iw
    print(np.mean(np.abs(x - z)))

    print(np.linalg.pinv(w))
    print(iw)

    from IPython import embed; embed(using=False); os._exit(0)


def add28b():

    np.set_printoptions(precision=2, suppress=True)

    N = 4
    tmp = 2 * math.pi * np.array([0, 2/8, 3/8, 4/8, 5/8, 6/8])
    tmp = np.arange(4)[:, None] * tmp[None, :]
    w = np.exp(-1.j * tmp) / 2

    x = np.arange(N)
    y = x @ w
    iw = np.conj(w).T
    z = y @ iw
    print(np.mean(np.abs(x - z)))

    print(np.linalg.pinv(w))
    print(iw)

    from IPython import embed; embed(using=False); os._exit(0)


# FIR filter with window method
def add29():

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import firwin, lfilter, freqz
    from mss.utils import fast_sdr

    # -----------------------
    # 参数设置
    # -----------------------
    fs = 1000   # 采样率
    numtaps = 101
    window_type = 'hamming'

    # 非均匀子带频率边界
    bands = [
        (50, 150),   # 子带 1
        (150, 300),  # 子带 2
        (300, 450)   # 子带 3
    ]

    # -----------------------
    # 生成测试信号
    # -----------------------
    t = np.arange(0, 1.0, 1/fs)
    x = np.sin(2*np.pi*75*t) + np.sin(2*np.pi*200*t) + np.sin(2*np.pi*400*t)

    # -----------------------
    # 分析：每个子带滤波
    # -----------------------
    y_subbands = []
    for f_low, f_high in bands:
        b = firwin(numtaps, [f_low/(fs/2), f_high/(fs/2)], pass_zero=False, window=window_type)
        # y = lfilter(b, 1.0, x)
        y = np.convolve(x, b, mode='same')
        y_subbands.append(y)

    # -----------------------
    # 合成：直接相加近似重构
    # -----------------------
    y_recon = sum(y_subbands)

    x - y_recon

    plt.plot(x, c='r')
    plt.plot(y_recon, c='b')
    plt.savefig("_zz.pdf")

    sdr = fast_sdr(x, y_recon)
    print(sdr)

    from IPython import embed; embed(using=False); os._exit(0)


# FIR filter with window method
def add30():

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import firwin, lfilter, freqz
    from mss.utils import fast_sdr

    fs = 48000
    numtaps = 101
    window_type = 'hamming'

    bands = [
        (0, 8000),   # 子带 1
        (8000, 24000),  # 子带 2
    ]

    x, _ = librosa.load(path="../mss2/assets/music_10s.wav", sr=fs, mono=True)

    y_subbands = []
    for f_low, f_high in bands:
        if f_low == 0:
            b = firwin(numtaps, f_high/(fs/2), pass_zero=True, window=window_type)
        elif f_high == fs // 2:
            b = firwin(numtaps, f_low/(fs/2), pass_zero=False, window=window_type)
        else:
            b = firwin(numtaps, [f_low/(fs/2), f_high/(fs/2)], pass_zero=False, window=window_type)
        y = np.convolve(x, b, mode='same')
        y_subbands.append(y)

    y_recon = sum(y_subbands)

    sdr = fast_sdr(x, y_recon)
    print(sdr)

    from IPython import embed; embed(using=False); os._exit(0)


def add30b():

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import firwin, lfilter, freqz
    from mss.utils import fast_sdr
    from scipy.signal import fftconvolve

    fs = 48000
    numtaps = 10001
    window_type = 'hamming'

    
    mel_f = librosa.mel_frequencies(n_mels=65, fmin=0, fmax=fs//2)

    bands = []
    for i in range(len(mel_f) - 1):
        bands.append([mel_f[i], mel_f[i + 1]])
    
    x, _ = librosa.load(path="../mss2/assets/music_10s.wav", sr=fs, mono=True)

    y_subbands = []
    for f_low, f_high in bands:
        if f_low == 0:
            b = firwin(numtaps, f_high/(fs/2), pass_zero=True, window=window_type)
        elif math.isclose(f_high, fs / 2):
            b = firwin(numtaps, f_low/(fs/2), pass_zero=False, window=window_type)
        else:
            b = firwin(numtaps, [f_low/(fs/2), f_high/(fs/2)], pass_zero=False, window=window_type)
        # y = np.convolve(x, b, mode='same')
        y = fftconvolve(x, b, mode='same')
        y_subbands.append(y)

    y_recon = sum(y_subbands)

    sdr = fast_sdr(x, y_recon)
    print(sdr)

    for i in range(len(y_subbands)):
        soundfile.write(file=f"_tmp/{i}.wav", data=y_subbands[i], samplerate=fs)

    from IPython import embed; embed(using=False); os._exit(0)



def add31():

    from scipy.signal import fftconvolve
    x = np.random.uniform(size=(1000,))
    h = np.random.uniform(size=(99,))

    y = fftconvolve(x, h, mode='same')
    y2 = torch_fftconvolve(Tensor(x), Tensor(h)).numpy()
    y3 = batch_fftconvolve(Tensor(x)[None, None, :], Tensor(h[None, None, :]), mode='same').numpy()[0, 0]

    np.mean(np.abs(y-y2))
    np.mean(np.abs(y-y3))

    from IPython import embed; embed(using=False); os._exit(0)

def batch_fftconvolve(x, h, mode='same'):
    """
    Args:
        x: (b, i, l)
        h: (o, i, l)

    Outputs:
        out: (b, o, l)
    """
    L1 = x.shape[-1]
    L2 = h.shape[-1]
    L = L1 + L2 - 1  # full convolution length

    # pad both sequences to length L
    X = torch.fft.rfft(torch.nn.functional.pad(x, (0, L - L1)))
    H = torch.fft.rfft(torch.nn.functional.pad(h, (0, L - L2)))

    # multiply in frequency domain and IFFT
    Y = torch.einsum('bil,oil->bol', X, H)  # (b, o, l)
    y = torch.fft.irfft(Y, n=L)

    start = (L2 - 1) // 2
    end = start + L1
    return y[:, :, start:end]


def torch_fftconvolve(x, h, mode='same'):
    """
    x, h: 1D torch tensors
    mode: 'full', 'same', 'valid'
    """
    N = x.shape[0]
    K = h.shape[0]
    L = N + K - 1  # full convolution length

    # pad both sequences to length L
    X = torch.fft.rfft(torch.nn.functional.pad(x, (0, L - N)))
    H = torch.fft.rfft(torch.nn.functional.pad(h, (0, L - K)))

    # multiply in frequency domain and IFFT
    y = torch.fft.irfft(X * H, n=L)

    start = (K - 1) // 2
    end = start + N
    return y[start:end]


def add32():

    x = torch.ones((4, 4800))
    n_fft = 128
    hop_length = 48

    X = torch.stft(
        input=x, 
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft),
        normalized=True,
        return_complex=True,
        onesided=False
    )  # (k*b*c, f, t)

    # x = F.pad(x, (0, 0, n_fft//2-1, 0))  # (k*b*c, f, t)

    y = torch.istft(
        input=X, 
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft),
        normalized=True,
        return_complex=True,
        onesided=False,
    )  # (b*c, l)

    plt.matshow(np.abs(X[0].data.cpu().numpy()), origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")

    x - y

    from IPython import embed; embed(using=False); os._exit(0)


def add33():
    from mss.models2.dsp.banks import mel_linear_banks
    sr = 48000
    n_bands = 64
    max_bandwidth = 800

    banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    from IPython import embed; embed(using=False); os._exit(0)



def hz_to_erb(f):
    return 21.4 * np.log10(1 + 0.00437 * f)

def erb_to_hz(erb):
    return 1 / 0.00437 * (10 ** (erb / 21.4) - 1)


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def add34():
    from mss.models2.dsp.banks import linear_banks, mel_linear_banks2
    
    sr = 48000
    n_bands = 64
    max_bandwidth = 800

    banks = mel_linear_banks2(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    from IPython import embed; embed(using=False); os._exit(0)


def add35():
    from mss.models2.dsp.banks import erb_linear_banks
    
    sr = 48000
    n_bands = 64
    max_bandwidth = 800

    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    print(bw)

    boarders = [0, 25, 50, 100, 200, 400, 800]
    out = [[] for _ in  range(len(boarders) - 1)]
    
    for i in range(len(bw)):
        for j in range(len(boarders) - 1):
            if boarders[j] <= bw[i] < boarders[j + 1]:
                out[j].append(i)

    from IPython import embed; embed(using=False); os._exit(0)



def add36():
    from mss.models2.dsp3.banks import erb_linear_banks
    from mss.models2.dsp3.subband_fast import SubbandFilter
    from mss.utils import fast_sdr

    # sr = 48000
    # n_bands = 228
    # max_bandwidth = 200
    # banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    # bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    # print(bw)
    # print(len(bw))

    sr = 48000
    n_bands = 223
    max_bandwidth = 190
    chunk_size = 16  # Try to tune this to balance RAM and computation speed
    factor = sr // (max_bandwidth + 10)
    device = "cuda"

    # Melbanks
    # banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size, bandpass_filter_len=24000, upsample_filter_len=12000).to(device)

    bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    print(bw)
    print(len(bw))

    asdf
    
    for _ in range(2000):

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


def add36b():
    from mss.models2.dsp3.banks import erb_linear_banks
    from mss.models2.dsp3.subband_fast import SubbandFilter
    from mss.utils import fast_sdr

    # sr = 48000
    # n_bands = 228
    # max_bandwidth = 200
    # banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    # bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    # print(bw)
    # print(len(bw))

    sr = 48000
    n_bands = 112
    max_bandwidth = 390
    chunk_size = 16  # Try to tune this to balance RAM and computation speed
    factor = sr // 400
    device = "cuda"

    # Melbanks
    # banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilter(sr, banks, factor, chunk_size=chunk_size, bandpass_filter_len=24000, upsample_filter_len=12000).to(device)

    bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    print(bw)
    print(len(bw))

    for _ in range(2000):

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


def add36c():
    from mss.models2.dsp3.banks import erb_linear_banks, erb_linear_banks_overlap
    from mss.models2.dsp3.subband_fast import SubbandFilter
    from mss.models2.dsp3.subband_fast_overlap import SubbandFilterOverlap
    from mss.utils import fast_sdr

    # sr = 48000
    # n_bands = 228
    # max_bandwidth = 200
    # banks = erb_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    # bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    # print(bw)
    # print(len(bw))

    sr = 48000
    n_bands = 111
    max_bandwidth = 390
    chunk_size = 16  # Try to tune this to balance RAM and computation speed
    factor = sr // 800
    device = "cuda"

    # from IPython import embed; embed(using=False); os._exit(0)

    # Melbanks
    # banks = mel_linear_banks(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    banks = erb_linear_banks_overlap(sr=sr, n_bands=n_bands, max_bandwidth=max_bandwidth)
    sb_filter = SubbandFilterOverlap(sr, banks, factor, chunk_size=chunk_size, bandpass_filter_len=24000, upsample_filter_len=12000).to(device)

    bw = [bank[1] - bank[0] for bank in banks]  # (k,)
    print(bw)
    print(len(bw))

    for _ in range(2000):

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
    add36c()