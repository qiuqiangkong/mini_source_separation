import math
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import librosa
import time
from torch.utils.checkpoint import checkpoint

from mss.models.fourier import Fourier


'''
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
    # print(out[0, 0, 50, 1000])
    
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
    # print(x3[0, 0, 50, 1000])
    # from IPython import embed; embed(using=False); os._exit(0)

    # Overlap-add
    x1 = fold(
        x=rearrange(x3, 'b c t n -> (b c) n t'), 
        hop_length=hop_length, 
        window=None
    )  # (b*c, l)
    x1 = rearrange(x1, '(b c) l -> b c l', b=x.shape[0])

    # x1 /= cola
    # print(x1[0, 0, 10000])
    
    
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
    print(x1[0, 0, 10000:10010])
    return x0
'''

def stft_fractional_(x: Tensor, n_fft: int, hop_length: int, r: int, window: Tensor) -> Tensor:
    r"""

    b: batch_size
    L: audio_length
    t: n_frames
    n: frame_length
    r: 
    f: freq_bins

    Args:
        x: (b, L)

    Returns:
        out: (b, n, f)
    """
    # from IPython import embed; embed(using=False); os._exit(0)
    N = n_fft
    x1 = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, L)
    x2 = x1.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
    x3 = x2 * window

    x4 = torch.stack([x3 for _ in range(r)], dim=-1)  # (b, t, n, r)
    n = torch.arange(0, N)
    k_prime = torch.arange(0, 1, 1 / r)
    a = torch.exp(-1.j * 2 * math.pi / N * torch.outer(n, k_prime)).to(x.device)  # (n, r)
    
    x5 = x4.to(torch.complex64) * a
    x6 = torch.fft.fft(x5, dim=-2, norm="ortho") / math.sqrt(r)  # (b, n, r) 
    x7 = rearrange(x6, 'b t f r -> b t (f r)')
    out = x7[..., 0 : x7.shape[-1] // 2 + 1]

    # print(out[0, 50, 1000])

    return out


'''
def stft_fractional(x: Tensor, n_fft: int, hop_length: int, r: int, window: Tensor) -> Tensor:
    r"""

    b: batch_size
    L: audio_length
    t: n_frames
    n: frame_length
    r: 
    f: freq_bins

    Args:
        x: (b, L)

    Returns:
        out: (b, n, f)
    """
    
    # torch.cuda.synchronize()
    # t1 = time.time()
    # from IPython import embed; embed(using=False); os._exit(0)
    N = n_fft
    x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, L)
    x = x.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
    x.mul_(window)  # (b, t, n)
    x = x.unsqueeze(-1)  # (b, t, n, 1)
    # torch.cuda.synchronize()
    # print("b1", time.time() - t1)
    # torch.cuda.synchronize()

    # t1 = time.time()
    n = torch.arange(0, N, device=x.device)  # (n,)
    k_prime = torch.arange(0, 1, 1 / r, device=x.device)  # (r,)
    a = torch.exp(-1.j * 2 * math.pi / N * torch.outer(n, k_prime))  # (n, r)
    # torch.cuda.synchronize()
    # print("b2", time.time() - t1)
    # torch.cuda.synchronize()

    # t1 = time.time()
    # from IPython import embed; embed(using=False); os._exit(0)
    x = x * a  # (b, t, n, r)
    # torch.cuda.synchronize()
    # print("b3", time.time() - t1)
    # torch.cuda.synchronize()
    # t1 = time.time()

    if True:
        x = torch.fft.fft(x, dim=-2, norm="ortho") / math.sqrt(r)  # (b, t, f, r)
        x = rearrange(x, 'b t f r -> b t (f r)')  # (b, t, f*r)
        out = x[..., 0 : N * r // 2 + 1]  # (b, t, f*r)
    else:
        # from IPython import embed; embed(using=False); os._exit(0)
        x = torch.fft.rfft(torch.view_as_real(x), dim=-3, norm="ortho") / math.sqrt(r)
        x = x[..., 0] + 1.j * x[..., 1]
        x = rearrange(x, 'b t f r -> b t (f r)')  # (b, t, f*r)
        out = x[..., 0 : N * r // 2 + 1]  # (b, t, f*r)
        # from IPython import embed; embed(using=False); os._exit(0)

    # torch.cuda.synchronize()
    # print("b4", time.time() - t1)
    # torch.cuda.synchronize()

    return out
    # return 0
'''

def stft_fractional(x: Tensor, n_fft: int, hop_length: int, r: int, window: Tensor) -> Tensor:
    r"""

    b: batch_size
    L: audio_length
    t: n_frames
    n: frame_length
    r: 
    f: freq_bins

    Args:
        x: (b, L)

    Returns:
        out: (b, n, f)
    """
    
    N = n_fft
    x = F.pad(x, (N // 2, N // 2), mode="reflect")  # (b, L)
    x = x.unfold(dimension=-1, size=N, step=hop_length).contiguous()  # (b, t, n)
    x.mul_(window)  # (b, t, n)
    
    
    if False:
        x = x.unsqueeze(-1)  # (b, t, n, 1)
        n = torch.arange(0, N, device=x.device)  # (n,)
        k_prime = torch.arange(0, 1, 1 / r, device=x.device)  # (r,)
        a = torch.exp(-1.j * 2 * math.pi / N * torch.outer(n, k_prime))  # (n, r)
        x = x * a  # (b, t, n, r)
        x = torch.fft.fft(x, dim=-2, norm="ortho") / math.sqrt(r)  # (b, t, f, r)
        x = rearrange(x, 'b t f r -> b t (f r)')  # (b, t, f*r)
        out = x[..., 0 : N * r // 2 + 1]  # (b, t, f*r)
    else:
        x = x.to(torch.complex64)
        n = torch.arange(0, N, device=x.device)  # (n,)
        a = torch.exp(-1.j * 2 * math.pi / N * n / r)  # (n,)
        B, T, N = x.shape
        out = torch.zeros((B, T, N // 2 + 1, r), dtype=torch.complex64, device=x.device)

        for i in range(r):    
            y = torch.fft.fft(x, dim=-1, norm="ortho") / math.sqrt(r)  # (b, t, f)
            out[:, :, :, i] = y[:, :, 0 : N // 2 + 1]
            x.mul_(a)

        out = rearrange(out, 'b t f r -> b t (f r)')
        out = out[..., 0 : N * r // 2 + 1]  # (b, t, f*r)

    return out
    # return 0

'''
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
'''

def istft_fractional_(
    x: Tensor, 
    n_fft: int, 
    hop_length: int, 
    r: int, 
    window: Tensor, 
    length=None
) -> Tensor:
    r"""

    m: n_audios
    L: audio_length
    t: n_frames
    n: frame_length
    r: 
    f: freq_bins

    Args:
        x: (m, f)

    Returns:
        out: (m, L)
    """

    out = x
    x_flip = torch.flip(out[..., 1 : -1], dims=[-1]).conj()
    x7 = torch.cat([x, x_flip], dim=-1)  # (m, f*r)
    x6 = rearrange(x7, '... (f r) -> ... f r', r=r)  # (any, f, r)


    x5 = torch.fft.ifft(x6, dim=-2, norm="ortho") / math.sqrt(r)  # (any, n, r)

    n = torch.arange(0, n_fft)
    k_prime = torch.arange(0, 1, 1 / r)
    a = torch.exp(1.j * 2 * math.pi / n_fft * torch.outer(n, k_prime)).to(x.device)  # (n, r)

    x4 = x5 * a
    x3 = x4.sum(dim=-1)

    # Overlap-add
    x1 = fold(
        x=x3, 
        hop_length=hop_length, 
        window=window
    )  # (b*c, l)
    
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 1, sharex=True)
    # axes[0].plot(x1[0, 0:100].abs())
    # axes[1].plot(x1[0, 0:100].imag)
    # plt.savefig("_zz.pdf")
    # Remove padding
    x0 = x1[:, n_fft // 2 :].real
    
    if length is not None:
        x0 = x0[:, 0 : length]

    return x0


'''
def istft_fractional(
    x: Tensor, 
    n_fft: int, 
    hop_length: int, 
    r: int, 
    window: Tensor, 
    length=None
) -> Tensor:
    r"""

    b: batch_size
    L: audio_length
    t: n_frames
    n: frame_length
    r: 
    f: freq_bins

    Args:
        x: (m, f)

    Returns:
        out: (m, L)
    """
    # from IPython import embed; embed(using=False); os._exit(0)
    # torch.cuda.empty_cache()

    # from IPython import embed; embed(using=False); os._exit(0)
    # t1 = time.time()
    x_flip = torch.flip(x[..., 1 : -1], dims=[-1]).conj()
    x = torch.cat([x, x_flip], dim=-1)  # (b, f*r)
    x = rearrange(x, '... (f r) -> ... f r', r=r)  # (b, f, r)
    x = torch.fft.ifft(x, dim=-2, norm="ortho") / math.sqrt(r)  # (b, t, n, r)
    # print("b1", time.time() - t1)

    # t1 = time.time()
    n = torch.arange(0, n_fft, device=x.device)  # (n,)
    k_prime = torch.arange(0, 1, 1 / r, device=x.device)  # (r,)
    a = torch.exp(1.j * 2 * math.pi / n_fft * torch.outer(n, k_prime))  # (n, r)
    # print("b2", time.time() - t1)

    # t1 = time.time()
    x.mul_(a)  # (b, t, n, r)
    x = x.sum(dim=-1)  # (b, t, n)
    # print("b3", time.time() - t1)

    # t1 = time.time()
    # Overlap-add
    x = fold(
        x=x, 
        hop_length=hop_length, 
        window=window
    )  # (b*c, l)
    # print("b4", time.time() - t1)
    
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 1, sharex=True)
    # axes[0].plot(x1[0, 0:100].abs())
    # axes[1].plot(x1[0, 0:100].imag)
    # plt.savefig("_zz.pdf")
    # Remove padding

    # t1 = time.time()
    x = x[:, n_fft // 2 :].real
    
    if length is not None:
        x = x[:, 0 : length]

    # print("b5", time.time() - t1)
    return x
'''

def istft_fractional(
    x: Tensor, 
    n_fft: int, 
    hop_length: int, 
    r: int, 
    window: Tensor, 
    length=None
) -> Tensor:
    r"""

    b: batch_size
    L: audio_length
    t: n_frames
    n: frame_length
    r: 
    f: freq_bins

    Args:
        x: (m, f)

    Returns:
        out: (m, L)
    """
    
    if False:
        x_flip = torch.flip(x[..., 1 : -1], dims=[-1]).conj()
        x = torch.cat([x, x_flip], dim=-1)  # (b, f*r)
        x = rearrange(x, '... (f r) -> ... f r', r=r)  # (b, f, r)
        x = torch.fft.ifft(x, dim=-2, norm="ortho") / math.sqrt(r)  # (b, t, n, r)
        
        n = torch.arange(0, n_fft, device=x.device)  # (n,)
        k_prime = torch.arange(0, 1, 1 / r, device=x.device)  # (r,)
        a = torch.exp(1.j * 2 * math.pi / n_fft * torch.outer(n, k_prime))  # (n, r)
        
        x.mul_(a)  # (b, t, n, r)
        x = x.sum(dim=-1)  # (b, t, n)
        
        x = fold(
            x=x, 
            hop_length=hop_length, 
            window=window
        )  # (b*c, l)
        
        x = x[:, n_fft // 2 :].real
        
        if length is not None:
            x = x[:, 0 : length]
        out = x
    else:
        N = n_fft
        n = torch.arange(0, N, device=x.device)  # (n,)
        
        B, T = x.shape[0 : 2]
        out = torch.zeros((B, T, N), device=x.device)
        
        for i in range(r):
            y = x[:, :, i :: r]
            if i == 0:
                y_flip = torch.flip(y[..., 1 : -1], dims=[-1]).conj()
            else:
                y_flip = torch.flip(x[:, :, r - i :: r], dims=[-1]).conj()

            y = torch.cat([y, y_flip], dim=-1)  # (b, f)
            a = torch.exp(1.j * 2 * math.pi / N * n / r * i)  # (n, r)
            y = torch.fft.ifft(y, dim=-1, norm="ortho") / math.sqrt(r)  # (b, t, n, r)
            out.add_((y * a).real)
        
        out = fold(
            x=out, 
            hop_length=hop_length, 
            window=window
        )  # (b*c, l)
        
        out = out[:, n_fft // 2 :]#.real
        
        if length is not None:
            out = out[:, 0 : length]

    return out


def fold(x: Tensor, hop_length: int, window: Tensor | None):
    r"""

    b: batch_size
    t: n_frames
    n: frame_samples
    l: segment_samples

    Args:
        x: (b, n, t)

    Returns:
        x: (b, l)
    """

    n_frames, frame_length = x.shape[-2:]  # (t, n)
    L = frame_length + (n_frames - 1) * hop_length
    
    # Overlap-add
    x = F.fold(
        input=rearrange(x, 'b t n -> b n t'),  # (b, n, t)
        output_size=(1, L),
        kernel_size=(1, frame_length),
        stride=(1, hop_length)
    )  # (b, 1, 1, l)
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


def add():

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    N = 2048
    r = 4

    x = torch.randn(4, 2, 1024 * 64)

    y4 = stft_fractional(x, N, N//4, r)
    z4 = istft_fractional(y4, N, N//4, r, x.shape[-1])
    print((x - z4).abs().mean())
    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    # window = torch.ones(n_fft).to(x.device)
    

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    N = 2048
    r = 4

    x = torch.randn(10, 1024 * 64)
    window = torch.hann_window(N).to(x.device)

    # y = stft_fractional(x[:, None, :], N, N//4, r)[:, 0, :, :]
    # z = istft_fractional(y[:, None, :, :], N, N//4, r, x.shape[-1])[:, 0, :]
    # print((x - z).abs().mean())

    y2 = stft_fractional(x, N, N//4, r, window)
    z2 = istft_fractional(y2, N, N//4, r, window, x.shape[-1])
    print((x - z2).abs().mean())
    from IPython import embed; embed(using=False); os._exit(0)


class GaborTransform(nn.Module):
    def __init__(self, n_ffts: list, r: int):
        super().__init__()

        self.n_ffts = n_ffts
        self.r = r
        self.n_windows = len(self.n_ffts)

        for n_fft in self.n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))

    def encode(self, x):

        outs = []
        B = x.shape[0]

        for i in range(self.n_windows):
            n_fft = self.n_ffts[i]
            window = getattr(self, f"window_{n_fft}")
            x1 = rearrange(x, 'b c l -> (b c) l')

            if False:
                x1 = checkpoint(stft_fractional, x1, n_fft, n_fft // 4, self.r, window) / math.sqrt(self.n_windows)
            else:
                x1 = stft_fractional(x1, n_fft, n_fft // 4, self.r, window) / math.sqrt(self.n_windows)
                # from IPython import embed; embed(using=False); os._exit(0)
            x1 = rearrange(x1, '(b c) t f -> b c t f', b=B)
            outs.append(x1)

        return outs

    def decode(self, x, length):

        outs = []
        B = x[0].shape[0]

        torch.cuda.synchronize()
        t1 = time.time()
        for i in range(self.n_windows):
            N = self.n_ffts[i]
            window = getattr(self, f"window_{N}")
            x1 = rearrange(x[i], 'b c t f -> (b c) t f')
            x1 = istft_fractional(x1, N, N // 4, self.r, window, length) / math.sqrt(self.n_windows)
            x1 = rearrange(x1, '(b c) l -> b c l', b=B)
            outs.append(x1)
        # torch.cuda.synchronize()
        # print("b1", time.time() - t1)
        # torch.cuda.synchronize()
        # t1 = time.time()
        out = torch.stack(outs, dim=0).sum(0)
        # torch.cuda.synchronize()
        # print("b2", time.time() - t1)
        # torch.cuda.synchronize()

        return out
        

# Test GaborTransform, noise input
def add3():

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    N = 2048
    r = 4

    x = torch.randn(10, 2, 1024 * 64)

    fourier = GaborTransform()
    y = fourier.encode(x)
    z = fourier.decode(y, x.shape[-1])
    print((x - z).abs().mean())

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3)
    axes[0].matshow(y[0][0, 0].abs().data.numpy().T, origin='lower', aspect='auto', cmap='jet')
    axes[1].matshow(y[1][0, 0].abs().data.numpy().T, origin='lower', aspect='auto', cmap='jet')
    axes[2].matshow(y[2][0, 0].abs().data.numpy().T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


# Test GaborTransform, music input
def add4():

    sr = 48000
    device = "cuda"
    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)

    x = Tensor(audio)[None, None, 0 : sr * 5].to(device)
    x = x.repeat((10, 2, 1))

    pad_t = x.shape[-1] % 32768
    x = F.pad(x, pad=(0, pad_t))
    # x = x[:, :, 0 : 2048 * 32]
    
    
    fourier = GaborTransform(
        n_ffts=[128, 512, 2048, 8192, 32768],
        r=16,
    ).to(device)

    for _ in range(10):
        torch.cuda.synchronize()
        t1 = time.time()
        # from IPython import embed; embed(using=False); os._exit(0)
        y = fourier.encode(x)

        torch.cuda.synchronize()
        print("time1: {}".format(time.time() - t1))
        torch.cuda.synchronize()
        t1 = time.time()
        z = fourier.decode(y, x.shape[-1])
        torch.cuda.synchronize()
        print("time2: {}".format(time.time() - t1))
        torch.cuda.synchronize()

        print((x - z).abs().mean())

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 3)
    # axes[0].matshow(y[0][0, 0].data.cpu().abs().numpy().T, origin='lower', aspect='auto', cmap='jet')
    # axes[1].matshow(y[1][0, 0].data.cpu().abs().numpy().T, origin='lower', aspect='auto', cmap='jet')
    # axes[2].matshow(y[2][0, 0].data.cpu().abs().numpy().T, origin='lower', aspect='auto', cmap='jet')
    # plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add5():

    sr = 48000
    device = "cuda"
    audio_path = "./assets/music_10s.wav"
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)

    x = Tensor(audio)[None, None, 0 : sr * 10].to(device)
    x = x.repeat((10, 1, 1))

    fourier = Fourier(n_fft=2048, hop_length=512).to(device)
    # from IPython import embed; embed(using=False); os._exit(0)
    y = fourier.stft(x)
    z = fourier.istft(y)

if __name__ == '__main__':

    add4()  