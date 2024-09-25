import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from models.fourier import Fourier
# from models.rotary import RotaryEmbedding


FREQ_BINS_PER_BAND = [
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  12, 12, 12, 12, 12, 12, 12, 12,
  24, 24, 24, 24, 24, 24, 24, 24,
  48, 48, 48, 48, 48, 48, 48, 48,
  64, 64, 64, 65,
]


class BSRoformer8a(Fourier):
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        input_channels: int = 2,
        freq_bins_per_band: list = FREQ_BINS_PER_BAND,
        t1: int = 51,
        f1: int = 16,
        t2: int = 4,
        f2: int = 4,
        depth: int = 12,
        dim: int = 384,
        n_heads: int = 12
    ):
        super().__init__(n_fft, hop_length)

        self.input_channels = input_channels
        self.depth = depth
        self.dim = dim
        self.n_heads = n_heads

        self.cmplx_num = 2
        self.t1 = t1
        self.f1 = f1
        self.t2 = t2
        self.f2 = f2
        
        self.head_dim = self.dim // self.n_heads

        freq_bins_per_band = [self.cmplx_num * self.input_channels * f for f in freq_bins_per_band]

        h0 = 64

        self.band_split = BandSplit(
            freq_bins_per_band=freq_bins_per_band,
            dim=h0
        )
        
        self.band_combine = BandCombine(
            dim=h0,
            freq_bins_per_band=freq_bins_per_band,
            output_channels=input_channels,
            cmplx_num=self.cmplx_num
        )

        rotary_emb = RotaryEmbedding(dim=self.head_dim, freqs_for='pixel', max_freq=t1)
        rotary_t1 = rotary_emb.get_axial_freqs(t1)

        rotary_emb = RotaryEmbedding(dim=self.head_dim, freqs_for='pixel', max_freq=f1)
        rotary_f1 = rotary_emb.get_axial_freqs(f1)
        
        self.transformers = nn.ModuleList([])

        self.fc_in = nn.Linear(in_features=self.t2 * self.f2 * h0, out_features=self.dim)

        
        for _ in range(self.depth):
            self.transformers.append(
                TransformerBlock(dim=self.dim, n_heads=self.n_heads, rotary_t=rotary_t1, rotary_f=rotary_f1)
                )
        

        self.fc_out = nn.Linear(in_features=self.dim, out_features=self.t2 * self.f2 * h0)
        
    def forward(self, mixture):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)

        Constants:
            b: batch_size
            c: channels_num=2
            z: complex_num=2
        """
        # from IPython import embed; embed(using=False); os._exit(0)

        # Complex spectrum.
        complex_sp = self.stft(mixture)
        # shape: (b, c, t, f)

        B, C, T, Freq = complex_sp.shape

        x = self.process_image(complex_sp)
        # shape: (b, c, t, f)

        x = torch.view_as_real(x)
        # shape: (b, c, t, f, z)

        x = self.band_split(x)
        # shape: (b, d, t, f)

        assert self.t1 * self.t2 == x.shape[2]
        assert self.f1 * self.f2 == x.shape[3]

        # x = self.fc_in(x)

        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b t1 f1 (t2 f2 d)', t2=self.t2, f2=self.f2)
        x = self.fc_in(x)
        
        for transformer in self.transformers:

            x = transformer(x)
        
        x = self.fc_out(x)
        x = rearrange(x, 'b t1 f1 (t2 f2 d) -> b d (t1 t2) (f1 f2)', t2=self.t2, f2=self.f2)

        x = self.band_combine(x)
        # shape: (b, c, t, f, z)

        x = torch.view_as_complex(x)
        # shape: (b, c, t, f)

        mask = self.unprocess_image(x, T)

        sep_stft = mask * complex_sp

        output = self.istft(sep_stft)
        # (b, c, samples_num)
        # from IPython import embed; embed(using=False); os._exit(0)

        return output

    def process_image(self, x):

        B, C, T, Freq = x.shape

        pad_len = self.t1 * self.t2 - T
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        return x

    def unprocess_image(self, x, time_steps):

        output = x[:, :, 0 : time_steps, :]

        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        r"""https://github.com/meta-llama/llama/blob/main/llama/model.py"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.weight
        return output


class BandSplit(nn.Module):
    
    def __init__(
        self,
        freq_bins_per_band: list[int],
        dim: int,
    ):
        super().__init__()
        
        self.freq_bins_per_band = freq_bins_per_band
        self.band_nets = nn.ModuleList([])

        for in_dim in self.freq_bins_per_band:
            self.band_nets.append(nn.Linear(in_dim, dim))
        
    def forward(self, x):
        r"""

        Args:
            x: (b, c, t, f, z)

        Outputs:
            output: (b, d, t, f)
        """
        
        x = rearrange(x, 'b c t f z -> b t (f c z)')

        band_xs = torch.split(
            tensor=x, 
            split_size_or_sections=self.freq_bins_per_band, 
            dim=-1
        )

        xs = []
        for x, net in zip(band_xs, self.band_nets):
            x = net(x)  # (b, t, d)
            xs.append(x)

        x = torch.stack(xs, dim=-1)  # (b, t, d, f)
        x = rearrange(x, 'b t d f -> b d t f')  # (b d t f)

        return x


class BandCombine(nn.Module):
    
    def __init__(
        self,
        dim: int,
        freq_bins_per_band: list[int],
        output_channels: int,
        cmplx_num: int
    ):
        super().__init__()
        
        self.freq_bins_per_band = freq_bins_per_band
        self.output_channels = output_channels
        self.cmplx_num = cmplx_num

        self.band_nets = nn.ModuleList([])

        for out_dim in self.freq_bins_per_band:
            self.band_nets.append(nn.Linear(dim, out_dim))
        
    def forward(self, x):
        r"""

        Args:
            x: (b, d, t, f)

        Outputs:
            output: (b, c, t, f, z)
        """

        x = rearrange(x, 'b d t f -> b t d f')
        band_xs = torch.unbind(x, dim=-1)
        
        xs = []
        for x, net in zip(band_xs, self.band_nets):    
            x = net(x)  # (b, t, d)
            xs.append(x)

        x = torch.cat(xs, dim=-1)  # (b, t, f*c*z)
        x = rearrange(x, 'b t (f c z) -> b c t f z', c=self.output_channels, z=self.cmplx_num)

        return x


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim: int, n_heads: int, rotary_t: torch.Tensor, rotary_f):
        super().__init__()
        
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.rotary_t = rotary_t
        self.rotary_f = rotary_f
        # self.register_buffer('rotary', rotary)

        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # assert self.flash, "Must have flash attention."
        
        self.c_attn = nn.Linear(dim, 5 * dim, bias=False)  # 3 indicates qkv outputs.
        self.c_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        r"""
        Args:
            x: (b, t, h*d)

        Constants:
            b: batch_size
            t: time steps
            r: 3
            h: heads_num
            d: heads_dim
        """
        device = x.device
        B, T, F, D = x.size()

        q_t, q_f, k_t, k_f, v = rearrange(self.c_attn(x), 'b t f (r h d) -> r b h t f d', r=5, h=self.n_heads)

        ##
        q_t = rearrange(q_t, 'b h t f d -> (b f) h t d')
        k_t = rearrange(k_t, 'b h t f d -> (b f) h t d')
        v = rearrange(v, 'b h t f d -> (b f) h t d')

        q_t = apply_rotary_emb(self.rotary_t.to(device), q_t)
        k_t = apply_rotary_emb(self.rotary_t.to(device), k_t)

        v = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v, attn_mask=None, dropout_p=0, is_causal=False)

        ##
        q_f = rearrange(q_f, 'b h t f d -> (b t) h f d')
        k_f = rearrange(k_f, 'b h t f d -> (b t) h f d')
        v = rearrange(v, '(b f) h t d -> (b t) h f d', b=B)
    
        q_f = apply_rotary_emb(self.rotary_f.to(device), q_f)
        k_f = apply_rotary_emb(self.rotary_f.to(device), k_f)

        v = torch.nn.functional.scaled_dot_product_attention(q_f, k_f, v, attn_mask=None, dropout_p=0, is_causal=False)

        y = rearrange(v, '(b t) h f d -> b t f (h d)', b=B)

        y = self.c_proj(y)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rotary_t: torch.Tensor, rotary_f):
        
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = Attention(dim=dim, n_heads=n_heads, rotary_t=rotary_t, rotary_f=rotary_f)
        self.mlp = MLP(dim=dim)
        

    def forward(
        self,
        x: torch.Tensor,
    ):
        r"""x: (b, t, f, d)"""
        
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x
