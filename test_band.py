import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
from mss.utils import fast_sdr
from scipy.signal import fftconvolve
import librosa
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchaudio


def add():

    sr = 48000
    numtaps = 10001
    window_type = 'hamming'

    x, _ = librosa.load(path="../mss2/assets/music_10s.wav", sr=sr, mono=False)
    x = Tensor(x)[None, :, 0 : -1]

    aa = Band(sr)
    subbands = aa.encode(x)

    # z = aa.add1(subbands)


    bb = BB()
    z = bb.encode(subbands, aa.f_center)

    #
    y = torch.sum(subbands, dim=0)
    sdr1 = fast_sdr(x.numpy(), y.numpy())
    sdr2 = fast_sdr(x.numpy(), z.numpy())

    print(sdr1, sdr2)
    
    # print(sdr)

    # for i in range(len(y_subbands)):
    #     soundfile.write(file=f"_tmp/{i}.wav", data=y_subbands[i], samplerate=fs)

    from IPython import embed; embed(using=False); os._exit(0)


class Band(nn.Module):
    def __init__(self, sr):
        super().__init__()

        self.sr = sr
        self.n_bands = 64
        self.bandwidth = 1200
        self.filter_len = 10001
        self.window_type = 'hamming'

        freqs = self.mel_linear(self.n_bands, self.sr, self.bandwidth)

        w = torch.empty((len(freqs) - 1, self.filter_len))

        w[0] = self.lowpass(freqs[1])
        for i in range(1, len(freqs) - 2):
            w[i] = self.bandpass(freqs[i], freqs[i+1])
        w[-1] = self.highpass(freqs[-2])

        self.register_buffer("w", w[:, None, :])  # (n_out, n_in, kernel_size)

        self.f_center = (freqs[0 : -1] + freqs[1 :]) / 2
        # from IPython import embed; embed(using=False); os._exit(0)
        # for i in range(len(self.w)):
        #     plt.plot(np.abs(np.fft.fft(w[60, :].cpu().numpy())))
        #     # plt.plot(w[20, :].cpu().numpy())
        #     plt.savefig("_zz.pdf")
        #     asdf

    def mel_linear(self, n_bands, sr, bandwidth):
        freqs = librosa.mel_frequencies(n_mels=n_bands, fmin=0, fmax=sr//2)
        idx = np.argmax(np.diff(freqs) >= bandwidth)
        mel_part = freqs[: idx + 1]
        linear_part = np.arange(mel_part[-1] + bandwidth, sr//2 + 1, bandwidth)
        freqs = np.concatenate([mel_part, linear_part])
        freqs[-1] = sr // 2
        return freqs

    def lowpass(self, f: float) -> Tensor:
        b = firwin(
            numtaps=self.filter_len, 
            cutoff=f/(self.sr/2), 
            pass_zero=True, 
            window=self.window_type
        )
        return torch.from_numpy(b)

    def bandpass(self, f1: float, f2: float) -> Tensor:
        b = firwin(
            numtaps=self.filter_len, 
            cutoff=[f1/(self.sr/2), f2/(self.sr/2)], 
            pass_zero=False, 
            window=self.window_type
        )
        return torch.from_numpy(b)

    def highpass(self, f):
        b = firwin(
            numtaps=self.filter_len, 
            cutoff=f/(self.sr/2), 
            pass_zero=False, 
            window=self.window_type
        )
        return torch.from_numpy(b)

    def encode(self, x):
        B = x.shape[0]
        x = rearrange(x, 'b c l -> (b c) 1 l')  # (b*c, 1, l)
        x = fftconvolve(x, torch.flip(self.w, dims=[2]))  # (b*c, k, l)
        subbands = rearrange(x, '(b c) k l -> k b c l', b=B)
        return subbands

    def add1(self, subbands):
        K, B, C, L = subbands.shape
        factor = 2
        x = subbands[:, :, :, 0 :: factor].contiguous()

        x_up = torch.zeros((K, B, C, L), device=x.device)
        x_up[:, :, :, ::factor] = x

        y = fftconvolve(x_up[0, :, 0:1, :], self.w[0:1, :, :]) * factor
        fast_sdr(subbands[0,0,0,0:10000].numpy(), y[0,0,0:10000].numpy())

        # fig, axes = plt.subplots(2, 1, sharex=True)
        # axes[0].plot(subbands[0, 0, 0, 0:10000].cpu().numpy())
        # axes[1].plot(y[0, 0, 0:10000].cpu().numpy())
        plt.plot(subbands[0, 0, 0, 0:10000].cpu().numpy())
        plt.plot(y[0, 0, 0:10000].cpu().numpy())
        plt.savefig("_zz.pdf")

        from IPython import embed; embed(using=False); os._exit(0)

'''
class BB(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, subbands):
        K, B, C, L = subbands.shape
        x = subbands[:, :, :, 0 :: 10].contiguous()

        y = rearrange(x, 'k b c l -> (k b c) l')
        y = upsample(y, 10)
        y = rearrange(y, '(k b c) l -> k b c l', k=K, b=B, c=C)
        y = y[..., 0 : L]

        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(subbands[0, 0, 0, 0:10000].cpu().numpy())
        axes[1].plot(y[0, 0, 0, 0:10000].cpu().numpy())
        plt.savefig("_zz.pdf")

        fast_sdr(subbands[0,0,0,0:10000].numpy(), y[0,0,0,0:10000].numpy())

        from IPython import embed; embed(using=False); os._exit(0)
        x = rearrange(x, 'k b c l -> (k b c) l')
        x = upsample(x, 10)
        x = rearrange(x, '(k b c) l -> k b c l', k=K, b=B, c=C)
        x = x[..., 0 : L]
        return x
        # from IPython import embed; embed(using=False); os._exit(0)
'''


class BB(nn.Module):
    def __init__(self):
        super().__init__()
        # model = Fourier(n_fft=2048, hop_length=480)
        # x = torch.randn(8, 4, 96000)  # (b, c, l)
        # stft = model.stft(x)  # (b, c, t, f)
        # out = model.istft(stft)  # (b, c, l)

    def encode(self, subbands, f_center):

        K, B, C, L = subbands.shape
        # stft = model.stft(x)  # (b, c, t, f)
        # out = model.istft(stft)  # (b, c, l)

        x = rearrange(subbands, 'k b c l -> (k b c) l')

        n_fft = 2048
        hop_length = 480

        if False:
            # 62 dB
            # Get analytical signal
            x = torch.stft(
                input=x, 
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft).to(subbands.device),
                normalized=True,
                return_complex=True,
            )  # (k*b*c, f, t)

            x = torch.istft(
                input=x, 
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft).to(x.device),
                normalized=True,
            )  # (b*c, l)

            x = rearrange(x, '(k b c) l -> k b c l', k=K, b=B, c=C)  # (k, b, c, l)

        if True:
            # 62 dB
            # Get analytical signal
            x = torch.stft(
                input=x, 
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft).to(subbands.device),
                normalized=True,
                return_complex=True,
                onesided=True
            )  # (k*b*c, f, t)

            x[:, 1 : -1, :] *= 2

            # x = F.pad(x, (0, 0, n_fft//2-1, 0))  # (k*b*c, f, t)
            x = F.pad(x, (0, 0, 0, n_fft//2-1))  # (k*b*c, f, t)

            x = torch.istft(
                input=x, 
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft).to(x.device),
                normalized=True,
                return_complex=True,
                onesided=False
            )  # (b*c, l)

            x = rearrange(x, '(k b c) l -> k b c l', k=K, b=B, c=C)  # (k, b, c, l)

        if True:
            # Move freq to center
            f_center = torch.from_numpy(f_center).to(subbands.device)
            a = torch.exp(-1.j * (f_center/24000*math.pi)[:, None] * torch.arange(x.shape[-1])[None, :])  # (k, l)
            x.mul_(a[:, None, None, :]) 

        if False:
            # Visualize
            X = torch.stft(
                input=x[50, 0, 0], n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft).to(subbands.device),
                normalized=True,
                return_complex=True
            )  # (k*b*c, f, t)
            plt.matshow(np.abs(X.data.cpu().numpy()), origin='lower', aspect='auto', cmap='jet')
            plt.savefig("_zz.pdf")

        if False:
            # Low pass
            # TODO

            factor = 10
            x2 = x[:, :, :, 0::factor]

            x2 = rearrange(x2, 'k b c l -> (k b c) l')
            x2 = upsample(x2.real, 10) + 1.j * upsample(x2.imag, 10)
            x3 = rearrange(x2, '(k b c) l -> k b c l', k=K, b=B, c=C)

            # fast_sdr(x.numpy(), x3.numpy())
            fast_sdr(x[0,0,0,0:1000].numpy(), x3[0,0,0,0:1000].numpy())

            fig, axes = plt.subplots(2, 1, sharex=True)
            axes[0].plot(x.real[0, 0, 0, 0:1000].cpu().numpy())
            axes[1].plot(x3.real[0, 0, 0, 0:1000].cpu().numpy())
            plt.savefig("_zz.pdf")

            tmp = np.fft.fft(x[0,0,0,0:1000].numpy())
            plt.figure()
            plt.plot(np.abs(tmp))
            plt.savefig("_zz2.pdf")

        if True:
            factor = 10
            x2 = x[:, :, :, 0::factor]

            x2 = rearrange(x2, 'k b c l -> (k b c) l')
            # x2 = torch.complex(upsample(x2.real, 10), upsample(x2.imag, 10))
            x2 = upsample(x2.real, 10) + 1.j * upsample(x2.imag, 10)
            x3 = rearrange(x2, '(k b c) l -> k b c l', k=K, b=B, c=C)

            # fig, axes = plt.subplots(2, 1, sharex=True)
            # axes[0].plot(x.real[0, 0, 0, -1000:].cpu().numpy())
            # axes[1].plot(x3.real[0, 0, 0, -1000:].cpu().numpy())
            # plt.savefig("_zz.pdf")
            # print(fast_sdr(x.real.numpy(), x3.real.numpy()))
            # print(fast_sdr(x.real[0, 0, 0, 1000:].numpy(), x3.real[0, 0, 0, 1000:].numpy()))
            # print(fast_sdr(x.real[0, 0, 0, -1000:].numpy(), x3.real[0, 0, 0, -1000:].numpy()))
            # print(fast_sdr(x.real[50, 0, 0, 10000:11000].numpy(), x3.real[50, 0, 0, 10000:11000].numpy()))

            x = x3
            # from IPython import embed; embed(using=False); os._exit(0)

        if True:
            a = torch.exp(1.j * (f_center/24000*math.pi)[:, None] * torch.arange(x.shape[-1])[None, :])  # (k, l)
            x.mul_(a[:, None, None, :]) 

        y = x.real

        print(fast_sdr(subbands.numpy(), y.numpy()))
        from IPython import embed; embed(using=False); os._exit(0)

        
        
        # X = librosa.stft(y=subbands[50, 0, 0].cpu().numpy(), n_fft=2048, hop_length=480)
        # plt.matshow(np.abs(X), origin='lower', aspect='auto', cmap='jet')
        # plt.savefig("_zz.pdf")

        from IPython import embed; embed(using=False); os._exit(0)

        x = subbands[:, :, :, 0 :: 10].contiguous()

        y = rearrange(x, 'k b c l -> (k b c) l')
        y = upsample(y, 10)
        y = rearrange(y, '(k b c) l -> k b c l', k=K, b=B, c=C)
        y = y[..., 0 : L]

        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(subbands[0, 0, 0, 0:10000].cpu().numpy())
        axes[1].plot(y[0, 0, 0, 0:10000].cpu().numpy())
        plt.savefig("_zz.pdf")

        fast_sdr(subbands[0,0,0,0:10000].numpy(), y[0,0,0,0:10000].numpy())

        from IPython import embed; embed(using=False); os._exit(0)
        x = rearrange(x, 'k b c l -> (k b c) l')
        x = upsample(x, 10)
        x = rearrange(x, '(k b c) l -> k b c l', k=K, b=B, c=C)
        x = x[..., 0 : L]
        return x
        # from IPython import embed; embed(using=False); os._exit(0)


def fftconvolve(x, h):
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


# def hz_to_mel(f):
    # return 2595 * math.log10(1 + f / 700.0)



def upsample(x, factor):
    """
    Args:
        x: (b, c, l)
    """

    filter_len = 10001
    B, L = x.shape
    x_up = torch.zeros(B, L*factor, device=x.device, dtype=x.dtype)
    x_up[:, ::factor] = x

    # sinc filter
    t = torch.arange(-filter_len//2, filter_len//2, device=x.device)
    t = t / factor
    sinc = torch.where(t == 0, torch.ones_like(t), torch.sin(torch.pi * t) / (torch.pi * t))
    window = torch.hann_window(filter_len)
    h = sinc * window
    h = h / h.sum()
    h *= factor

    out = fftconvolve(x_up[:, None, :], h[None, None, :])[:, 0, :]
    return out


def add2():

    sr = 48000
    x, _ = librosa.load(path="../mss2/assets/music_10s.wav", sr=sr, mono=True)
    x = Tensor(x)[None, :]

    x = x[:, ::10]
    upsample(x, 10)


if __name__ == '__main__':
    add()