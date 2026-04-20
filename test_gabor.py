import math
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# FFT inverse
def add():

    device = "cuda"
    n_fft = 2048

    N = n_fft
    n = torch.arange(0, N)
    window = torch.hann_window(n_fft)

    # k1 = torch.arange(0, N // 2 + 1)
    k1 = torch.arange(0, N // 2 + 1e-6, 1/4)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    # W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k1, n)) / math.sqrt(N)  # (N/2+1, N)
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)  # (N, N)
    # W *= window
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W.T
    z = y @ W_dec.T
    print((x - z).abs().mean(), (x - z).abs().max())

    # Plot
    fig, axs = plt.subplots(5, 1, sharex=True)
    # axs[0].plot(W[0].real)
    # axs[1].plot(W[10].real)
    # axs[2].plot(W[100].real)
    # axs[3].plot(W[1000].real)
    axs[0].plot(W_dec[:, 0].real)
    axs[1].plot(W_dec[:, 10].real)
    axs[2].plot(W_dec[:, 100].real)
    axs[3].plot(W_dec[:, 1000].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)



# Gabor no overlap
def add2():

    device = "cuda"
    n_fft = 2048

    N = n_fft
    n = torch.arange(0, N)
    # window = torch.hann_window(n_fft)

    k1 = torch.arange(0, N // 2 + 1)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)  # (N, N)

    a1 = []
    n_layers = 4
    for l in range(n_layers):
        n = N // (2 ** l)
        i = 0
        while i < N:
            tmp = torch.zeros(N)
            tmp[i : i + n] = 1
            i += n
            a1.append(tmp)

    a1 = torch.stack(a1, dim=0)  # (2**l-1, N)

    W_new = W[:, :, None] * a1.T[None, :, :]  # (K, N, A)
    W_new = rearrange(W_new, 'k n a -> (k a) n')


    # plt.matshow(a1.data.numpy(), origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)
    
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W_new.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W_new.T
    z = y @ W_dec.T
    print((x - z).abs().mean(), (x - z).abs().max())

    # Plot
    fig, axs = plt.subplots(5, 1, sharex=True)
    # axs[0].plot(W[0].real)
    # axs[1].plot(W[10].real)
    # axs[2].plot(W[100].real)
    # axs[3].plot(W[1000].real)
    axs[0].plot(W_dec[:, 0].real)
    axs[1].plot(W_dec[:, 10].real)
    axs[2].plot(W_dec[:, 11].real)
    axs[3].plot(W_dec[:, 1000].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# Gabor has overlap
def add3():

    device = "cuda"
    n_fft = 2048

    N = n_fft
    n = torch.arange(0, N)
    # window = torch.hann_window(n_fft)

    k1 = torch.arange(0, N // 2 + 1)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)  # (N, N)

    a1 = []
    n_layers = 4
    for l in range(n_layers):
        n = N // (2 ** l)
        i = 0
        while i + n <= N:
            tmp = torch.zeros(N)
            tmp[i : i + n] = 1
            i += n // 2
            a1.append(tmp)

    a1 = torch.stack(a1, dim=0)  # (2**l-1, N)

    W_new = W[:, :, None] * a1.T[None, :, :]  # (K, N, A)
    W_new = rearrange(W_new, 'k n a -> (k a) n')


    # plt.matshow(a1.data.numpy(), origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)
    
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W_new.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W_new.T
    z = y @ W_dec.T
    print((x - z).abs().mean(), (x - z).abs().max())

    # Plot
    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(W_dec[:, 0].real)
    axs[1].plot(W_dec[:, 10].real)
    axs[2].plot(W_dec[:, 11].real)
    axs[3].plot(W_dec[:, 1000].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# Gabor has overlap
def add3b():

    device = "cuda"
    n_fft = 2048

    N = n_fft
    n = torch.arange(0, N)
    # window = torch.hann_window(n_fft)

    k1 = torch.arange(0, N // 2 + 1)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)  # (N, N)

    a1 = []
    n_layers = 4
    for l in range(n_layers):
        n = N // (2 ** l)

        tmp = torch.zeros(N)
        tmp[0 : n // 2] = 1
        a1.append(tmp)

        i = 0
        while i + n <= N:
            tmp = torch.zeros(N)
            tmp[i : i + n] = 1
            i += n // 2
            a1.append(tmp)

        tmp = torch.zeros(N)
        tmp[-n // 2 :] = 1
        a1.append(tmp)

    a1 = torch.stack(a1, dim=0)  # (2**l-1, N)

    W_new = W[:, :, None] * a1.T[None, :, :]  # (K, N, A)
    W_new = rearrange(W_new, 'k n a -> (k a) n')


    # plt.matshow(a1.data.numpy(), origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)
    
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W_new.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W_new.T
    z = y @ W_dec.T
    print((x - z).abs().mean(), (x - z).abs().max())

    # Plot
    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(W_dec[:, 0].real)
    axs[1].plot(W_dec[:, 10].real)
    axs[2].plot(W_dec[:, 11].real)
    axs[3].plot(W_dec[:, 1000].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# Fractional + Gabor
def add4():

    device = "cuda"
    n_fft = 2048

    N = n_fft
    n = torch.arange(0, N)
    # window = torch.hann_window(n_fft)

    # k1 = torch.arange(0, N // 2 + 1)
    k1 = torch.arange(0, N // 2 + 1e-6, 1/4)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N)  # (N, N)

    a1 = []
    n_layers = 4
    for l in range(n_layers):
        n = N // (2 ** l)
        i = 0
        while i + n <= N:
            tmp = torch.zeros(N)
            tmp[i : i + n] = 1
            i += n
            a1.append(tmp)

    a1 = torch.stack(a1, dim=0)  # (2**l-1, N)

    W_new = W[:, :, None] * a1.T[None, :, :]  # (K, N, A)
    W_new = rearrange(W_new, 'k n a -> (k a) n')


    # plt.matshow(a1.data.numpy(), origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    # from IPython import embed; embed(using=False); os._exit(0)
    
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W_new.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W_new.T
    z = y @ W_dec.T
    print((x - z).abs().mean(), (x - z).abs().max())

    # Plot
    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(W_dec[:, 0].real)
    axs[1].plot(W_dec[:, 10].real)
    axs[2].plot(W_dec[:, 11].real)
    axs[3].plot(W_dec[:, 1000].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# Fractional FFT inverse 256
def add5():

    device = "cuda"
    n_fft = 256

    N = n_fft
    n = torch.arange(0, N)
    window = torch.hann_window(n_fft)

    # k1 = torch.arange(0, N // 2 + 1)
    r = 4
    k1 = torch.arange(0, N // 2 + 1e-6, 1/r)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    # W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k1, n)) / math.sqrt(N)  # (N/2+1, N)
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) / math.sqrt(r)  # (N, N)
    # W *= window
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W.T
    z = y @ W_dec.T
    print("x recon loss: ", (x - z).abs().mean(), (x - z).abs().max())
    print("W-W_dec: ", (W - W_dec.conj().T).abs().mean()) 

    # Plot
    fig, axs = plt.subplots(5, 2, sharex=True)
    axs[0, 0].plot(W[0].real)
    axs[1, 0].plot(W[1].real)
    axs[2, 0].plot(W[2].real)
    axs[3, 0].plot(W[3].real)
    axs[0, 1].plot(W_dec[:, 0].real)
    axs[1, 1].plot(W_dec[:, 1].real)
    axs[2, 1].plot(W_dec[:, 2].real)
    axs[3, 1].plot(W_dec[:, 3].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# Random bases inverse 256. Not match.
def add6():

    device = "cuda"
    n_fft = 256

    N = n_fft
    n = torch.arange(0, N)
    window = torch.hann_window(n_fft)

    # k1 = torch.arange(0, N // 2 + 1)
    r = 4
    k1 = torch.arange(0, N // 2 + 1e-6, 1/r)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    # W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k1, n)) / math.sqrt(N)  # (N/2+1, N)
    # W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) / math.sqrt(r)  # (N, N)
    W = torch.randn(N, N).to(torch.complex64) / math.sqrt(N)
    # W *= window
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W.T
    z = y @ W_dec.T
    print("x recon loss: ", (x - z).abs().mean(), (x - z).abs().max())
    print("W-W_dec: ", (W - W_dec.conj().T).abs().mean()) 

    # Plot
    fig, axs = plt.subplots(5, 2, sharex=True)
    axs[0, 0].plot(W[0].real)
    axs[1, 0].plot(W[1].real)
    axs[2, 0].plot(W[2].real)
    axs[3, 0].plot(W[3].real)
    axs[0, 1].plot(W_dec[:, 0].real)
    axs[1, 1].plot(W_dec[:, 1].real)
    axs[2, 1].plot(W_dec[:, 2].real)
    axs[3, 1].plot(W_dec[:, 3].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# Fractional FFT + Gabor, perfect reconstruct.
def add7():

    device = "cuda"
    n_fft = 256

    N = n_fft
    n = torch.arange(0, N)
    window = torch.hann_window(n_fft)

    # k1 = torch.arange(0, N // 2 + 1)
    r = 4
    k1 = torch.arange(0, N // 2 + 1e-6, 1/r)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    #
    # W_enc = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k1, n)) / math.sqrt(N)  # (N/2+1, N)
    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) / math.sqrt(r * 2)  # (N, N)
    _W2 = torch.cat([torch.ones(N*r, N//2), torch.zeros(N*r, N//2)], dim=-1) * W
    _W3 = torch.cat([torch.zeros(N*r, N//2), torch.ones(N*r, N//2)], dim=-1) * W
    W = torch.cat([W, _W2, _W3], dim=0)
    # W *= window
    
    # Inverse
    t1 = time.time()
    W_dec = torch.linalg.pinv(W.to(device)).cpu()  # (N, N)
    # W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) 
    print("time: {:.2f} s".format(time.time() - t1))

    # 
    x = torch.randn(1, n_fft).to(torch.complex64)
    y = x @ W.T
    z = y @ W_dec.T
    print("x recon loss: ", (x - z).abs().mean(), (x - z).abs().max())
    print("W-W_dec: ", (W - W_dec.conj().T).abs().mean()) 

    # Plot
    fig, axs = plt.subplots(5, 2, sharex=True)
    axs[0, 0].plot(W[0].real)
    axs[1, 0].plot(W[1].real)
    axs[2, 0].plot(W[2].real)
    axs[3, 0].plot(W[3].real)
    axs[0, 1].plot(W_dec[:, 0].real)
    axs[1, 1].plot(W_dec[:, 1].real)
    axs[2, 1].plot(W_dec[:, 2].real)
    axs[3, 1].plot(W_dec[:, 3].real)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)



class AA(nn.Module):
    def __init__(self, r):
        super().__init__()

        self.r = r
        self.n_ffts = [512, 2048, 8192]
        # self.hop_length = [512, 2048]
        # self.register_buffer(name="window", tensor=torch.hann_window(2048))
        self.register_buffer(name="window", tensor=torch.ones(2048))

    def encode(self, waveform):

        B, C, T = waveform.shape

        x = rearrange(waveform, 'b c l -> (b c) l')  # (b*c, l)

        # for i in range(self.r):
        #     N = self.n_ffts[1]
        #     from IPython import embed; embed(using=False); os._exit(0)
            # a = torch.exp(-1.j * 2 * math.pi / N * i / self.r * torch.arange(0, N + 1e-6))
        N = self.n_ffts[1]
        y = torch.stft(
            input=x, 
            n_fft=N,
            hop_length=N // 2,
            window=self.window,
            normalized=True,
            return_complex=True
        )  # (b*c, f, t)

        complex_sp = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)  # (b, c, t, f)
        return complex_sp


    def decode(self, complex_sp: Tensor) -> Tensor:
        r"""Reconstruct waveforms from STFT.

        b: batch_size
        c: channels_num
        t: frames_num
        f: freq_bins
        l: audio_samples

        Args:
            complex_sp: (b, c, t, f)

        Returns:
            out: (b, c, l)
        """

        B, C, T, F = complex_sp.shape

        x = rearrange(complex_sp, 'b c t f -> (b c) f t')  # (b*c, f, t)

        x = torch.istft(
            input=x, 
            n_fft=self.n_ffts[1],
            hop_length=self.n_ffts[1] // 2, 
            window=self.window,
            normalized=True,
        )  # (b*c, l)

        out = rearrange(x, '(b c) l -> b c l', b=B, c=C)  # (b, c, l)
        
        return out


def get_enc_dec_W(N, r):
    n = torch.arange(0, N)
    window = torch.hann_window(N)

    k1 = torch.arange(0, N // 2 + 1e-6, 1/r)
    k1_flip = N - torch.flip(k1, dims=[0])[1 : -1]
    k2 = torch.cat([k1, k1_flip], dim=0)

    W = torch.exp(-1.j * 2 * math.pi / N * torch.outer(k2, n)) / math.sqrt(N) / math.sqrt(r)  # (N, N)
    # W_dec = torch.linalg.pinv(W.to(device)).cpu()  # (N, N)
    W_dec = torch.exp(1.j * 2 * math.pi / N * torch.outer(n, k2)) / math.sqrt(N) 
    return W, W_dec


def stft(x, n_fft, hop_length):
    window = torch.ones(n_fft).to(x.device)
    x = F.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")  # (any, L)
    x = x.unfold(dimension=-1, size=n_fft, step=hop_length).contiguous()  # (b, t, n)
    x *= window
    out = torch.fft.rfft(x) / math.sqrt(n_fft)  # (any, t, f)
    return out


def istft(x, n_fft, hop_length, length=None):
    # x: (any, L)
    x = torch.fft.irfft(x)  # (b, c, t, f)

    # Overlap-add
    out = fold(
        x=rearrange(x, 'b c t n -> (b c) n t'), 
        hop_length=hop_length, 
        window=None
    )  # (b*c, l)
    out = rearrange(out, '(b c) l -> b c l', b=x.shape[0])
    
    # Remove padding
    out = out[..., n_fft // 2 :]
    
    if length is not None:
        out = out[..., 0 : length]

    return out


def stft_fractional(x, n_fft, hop_length, r):

    # window = torch.ones(n_fft).to(x.device)
    window = torch.hann_window(n_fft).to(x.device)

    x1 = F.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")  # (any, L)
    x2 = x1.unfold(dimension=-1, size=n_fft, step=hop_length).contiguous()  # (any, t, n)
    x3 = x2 * window

    x4 = torch.stack([x3 for _ in range(r)], dim=-1)  # (any, t, n, r)
    n = torch.arange(0, n_fft)
    k_prime = torch.arange(0, 1, 1 / r)
    a = torch.exp(-1.j * 2 * math.pi / n_fft * torch.outer(n, k_prime))  # (n, r)
    
    x5 = x4.to(torch.complex64) * a
    x6 = torch.fft.fft(x5, dim=-2, norm="ortho") / math.sqrt(r)  # (any, n, r) 
    x7 = rearrange(x6, '... f r -> ... (f r)')
    out = x7[..., 0 : x7.shape[-1] // 2 + 1]

    # from IPython import embed; embed(using=False); os._exit(0)
    
    return out


def istft_fractional(x, n_fft, hop_length, r, length=None):
    # x: (any, f*r)

    # window = torch.ones(n_fft).to(x.device)
    window = torch.hann_window(n_fft).to(x.device)

    out = x
    x_flip = torch.flip(out[..., 1 : -1], dims=[-1]).conj()
    x7 = torch.cat([x, x_flip], dim=-1)  # (any, f*r)
    x6 = rearrange(x7, '... (f r) -> ... f r', r=r)  # (any, f, r)

    x5 = torch.fft.ifft(x6, dim=-2, norm="ortho") / math.sqrt(r)  # (any, n, r)

    n = torch.arange(0, n_fft)
    k_prime = torch.arange(0, 1, 1 / r)
    a = torch.exp(1.j * 2 * math.pi / n_fft * torch.outer(n, k_prime))  # (n, r)

    x4 = x5 * a
    x3 = x4.sum(dim=-1)

    # Overlap-add
    x1 = fold(
        x=rearrange(x3, 'b c t n -> (b c) n t'), 
        hop_length=hop_length, 
        window=None
    )  # (b*c, l)
    x1 = rearrange(x1, '(b c) l -> b c l', b=x.shape[0])

    # x1 /= cola

    if True:
        # frame_length, num_frames = x3.shape[-2:]  # (t, n)
        frame_length = x3.shape[-1]
        num_frames = x3.shape[-2]
        L = frame_length + (num_frames - 1) * hop_length
        # from IPython import embed; embed(using=False); os._exit(0)
        win_norm = F.fold(
            window[None, :, None].repeat(1, 1, num_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        )  # (1, 1, 1, L)
        win_norm = win_norm.squeeze(dim=[0, 1, 2])  # (l,)

        x1 /= torch.clamp(win_norm, 1e-8)  # (b, l)
    
    # Remove padding
    x0 = x1[..., n_fft // 2 :].real
    
    if length is not None:
        x0 = x0[..., 0 : length]

    # from IPython import embed; embed(using=False); os._exit(0)

    return x0


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

    frame_length, n_frames = x.shape[-2:]  # (t, n)
    L = frame_length + (n_frames - 1) * hop_length

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
            window[None, :, None].repeat(1, 1, n_frames),  # (1, n, t),
            output_size=(1, L),
            kernel_size=(1, frame_length),
            stride=(1, hop_length)
        )  # (1, 1, 1, L)
        win_norm = win_norm.squeeze(dim=[0, 1, 2])  # (l,)

        out /= torch.clamp(win_norm, 1e-8)  # (b, l)

    return out


def add8():

    N = 2048
    r = 1
    W_enc, W_dec = get_enc_dec_W(N, r)
    fourier = AA(r)

    x = torch.randn(4, 2, 1024 * 64)
    # x = torch.zeros(4, 2, 1024 * 64)
    x[0, 0, 0] = 1

    y = fourier.encode(x)

    z = fourier.decode(y)

    print((x - z).abs().mean())

    # matrix multiplication
    y2 = x[:, :, 0 : N].to(torch.complex64) @ W_enc.T
    print((y[:, :, 1, :] - y2[:, :, 0 : 4097]).abs().mean())
    # z2 = y @ W_dec.T

    # rfft
    y3 = stft(x, N, N//2)
    z3 = istft(x, N, N//2)

    from IPython import embed; embed(using=False); os._exit(0)


def add9():

    N = 2048
    r = 1
    x = torch.randn(4, 2, 1024 * 64)

    # Matrix multiplication
    W_enc, W_dec = get_enc_dec_W(N, r)
    y2 = x[:, :, 0 : N].to(torch.complex64) @ W_enc.T

    # rfft
    y3 = stft(x, N, N//2)
    z3 = istft(y3, N, N//2)

    # print((y3[:, :, 1, :] - y2[:, :, 0 : 1025]).abs().mean())


def add10():

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    N = 2048
    r = 4

    x = torch.randn(4, 2, 1024 * 64)

    # Matrix multiplication
    W_enc, W_dec = get_enc_dec_W(N, r)
    y2 = x[:, :, 0 : N].to(torch.complex64) @ W_enc.T

    # rfft
    # y3 = stft_fractional(x, N, N//2, r)
    # z3 = istft(y3, N, N//2)

    #
    # print((y3[:, :, 1, :] - y2[:, :, 0 : 1025]).abs().mean())

    y4 = stft_fractional(x, N, N//4, r)
    z4 = istft_fractional(y4, N, N//4, r, x.shape[-1])
    print((x - z4).abs().mean())
    from IPython import embed; embed(using=False); os._exit(0)



# def add10():

    


if __name__ == '__main__':

    add10()