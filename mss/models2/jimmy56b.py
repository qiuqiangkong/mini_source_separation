from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# from mss.models.attention import BlockV2 as Block, RMSNorm
from mss.models2.attention import Block as Block, RMSNorm
from mss.models2.bandsplit42a import BandSplit
from mss.models.fourier import Fourier
from mss.models.rope import RoPE


class RMSNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        r"""RMSNorm over channel dimension for 2D feature map.

        Args:
            x: (b, d, t, f)

        Outputs:
            output: (b, d, t, f)
        """

        norm_x = x.norm(2, dim=1, keepdim=True)
        rms = norm_x * (x.shape[1] ** -0.5)
        x = x / (rms + self.eps)
        return x * self.scale


class GEGLUFusion(nn.Module):
    """
    Robust GEGLU fusion module. 
    Concatenates features, projects to gate/value, and fuses.
    """

    def __init__(self, dim, dim_mult=2 ):
        super().__init__()

        # 1. Normalization (Pre-norm) - separate norms for decoder and encoder
        self.norm_dec = RMSNorm2d(dim)
        self.norm_enc = RMSNorm2d(dim)

        # 2. Projection
        # We take 2 * dim inputs (enc + dec)
        # We project to 2 * (dim * dim_mult) because GEGLU requires splitting
        hidden_dim = int(dim * dim_mult)
        self.proj_in = nn.Conv2d(dim * 2, hidden_dim * 2, kernel_size=1)
        
        # 3. Output projection
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x_dec, x_enc):
        """
        Args:
            x_dec: Decoder features (B, C, H, W) -> Acts as the residual backbone
            x_enc: Encoder features (B, C, H', W')
        """
        # 1. Spatial Alignment (Crucial for U-Nets)
        if x_enc.shape[-2:] != x_dec.shape[-2:]:
            x_enc = F.interpolate(x_enc, size=x_dec.shape[-2:], mode='bilinear', align_corners=False)

        # 2. Prepare inputs
        # We apply the residual connection to the Decoder stream, so we save x_dec
        residual = x_dec
        
        # Norm and Concat
        # (Using separate norms for decoder and encoder features)
        x_dec = self.norm_dec(x_dec)
        x_enc = self.norm_enc(x_enc)
        
        combined = torch.cat([x_dec, x_enc], dim=1)

        # 3. Project and Split (The "GEGLU" mechanism)
        # Project to [B, 2*hidden, H, W]
        gate_value = self.proj_in(combined) 
        
        # Split into two chunks along channel dim
        gate, value = gate_value.chunk(2, dim=1)

        # 4. Gating
        hidden = F.gelu(gate) * value

        # 5. Output Project + Residual
        return residual + self.proj_out(hidden)

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class TimeMixBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, dim_mult=4):
        super().__init__()
        hidden_dim = int(dim * dim_mult * 2 // 3)
        padding = kernel_size // 2
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = RMSNorm(dim)
        
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        h = self.pwconv(x.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        gate, value = self.proj(h).chunk(2, dim=-1)
        gate = self.act(gate)
        gate = self.grn(gate)
        h = gate * value
        h = self.proj_out(h)
        return x + h


class BSRoformerBlock(nn.Module):
    """Band-Split RoFormer block with time and/or frequency attention."""

    def __init__(self, dim, n_heads, axis='tf', use_time_mix=False):
        super().__init__()
        assert axis in ['t', 'f', 'tf', 'ft'], "Axis must be one of 't', 'f', 'tf', or 'ft'."
        self.axis = axis
        self.use_time_mix = use_time_mix
        if 't' in axis:
            if use_time_mix:
                self.time_block = TimeMixBlock(dim)
            else:
                self.time_block = Block(dim, n_heads)
        self.freq_block = Block(dim, n_heads) if 'f' in axis else None

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        """
        Apply time and/or frequency attention.

        Args:
            x: Input tensor of shape (b, d, t, f).
            rope: Rotary position embedding.

        Returns:
            Output tensor of shape (b, d, t, f).
        """
        B = x.shape[0]

        if self.time_block is not None:
            # Time attention
            x = rearrange(x, 'b d t f -> (b f) t d')
            if self.use_time_mix:
                x = self.time_block(x)
            else:
                x = self.time_block(x, rope=rope, pos=None)
            x = rearrange(x, '(b f) t d -> b d t f', b=B)

        if self.freq_block is not None:
            # Frequency attention
            x = rearrange(x, 'b d t f -> (b t) f d')
            x = self.freq_block(x, rope=rope, pos=None)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)

        return x


class Jimmy56b(Fourier):
    """Band-Split RoFormer for music source separation."""

    def __init__(
        self,
        audio_channels=2,
        sample_rate=48000,
        n_fft=2048,
        hop_length=480,
        n_bands=256,
        band_dim=64,
        dim_sp=96,
        dim=384,
        dim_head=32,
        patch_size=[4, 4],
        n_layers=12,
        n_pre_layers=1,
        n_post_layers=1,
        rope_len=8192,
        **kwargs
    ) -> None:
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            normalized=True
        )
        
        self.ac = audio_channels
        self.patch_size = patch_size

        # Band split
        self.bandsplit = BandSplit(
            sr=sample_rate,
            n_fft=n_fft,
            n_bands=n_bands,
            in_channels=2,  # real + imag
            out_channels=band_dim
        )

        # RoPE
        self.rope = RoPE(head_dim=dim_head, max_len=rope_len)

        # Blocks
        self.patch = nn.Conv2d(band_dim * audio_channels, dim_sp, kernel_size=1)
        self.unpatch = nn.Conv2d(dim_sp, band_dim * audio_channels, kernel_size=1)
        self.down = nn.Conv2d(dim_sp, dim, kernel_size=patch_size, stride=patch_size)
        self.up = nn.ConvTranspose2d(dim, dim_sp, kernel_size=patch_size, stride=patch_size)
        self.fusion = GEGLUFusion(dim_sp)

        self.pre_blocks = nn.ModuleList([BSRoformerBlock(dim_sp, dim_sp // dim_head, axis='tf', use_time_mix=True) for _ in range(n_pre_layers)])
        self.post_blocks = nn.ModuleList([BSRoformerBlock(dim_sp, dim_sp // dim_head, axis='tf', use_time_mix=True) for _ in range(n_post_layers)])
        self.blocks = nn.ModuleList([BSRoformerBlock(dim, dim // dim_head, axis='tf') for _ in range(n_layers)])

    def forward(self, audio: Tensor) -> Tensor:
        """
        Separate audio sources.

        Args:
            audio: Input audio of shape (b, c, l).

        Returns:
            Separated audio of shape (b, c, l).
        """
        # --- Encode ---
        # Complex spectrum
        complex_sp = self.stft(audio)
        T0 = complex_sp.shape[2]

        x = torch.view_as_real(complex_sp)

        # Pad STFT
        x = self.pad_tensor(x)

        # Convert STFT to mel scale
        x = self.bandsplit.transform(x)

        x = self.patch(rearrange(x, 'b c t f i -> b (c i) t f'))

        for block in self.pre_blocks:
            x = block(x, rope=self.rope)
        h = x
        x = self.down(x)

        for block in self.blocks:
            x = block(x, rope=self.rope) 

        x = self.up(x)
        x = self.fusion(x, h)
        for block in self.post_blocks:
            x = block(x, rope=self.rope)

        # --- Decode ---
        # Unpatchify
        x = self.unpatch(x)
        x = rearrange(x, 'b (c i) t f -> b c t f i', c=self.ac)

        # Convert mel scale STFT to original STFT
        x = self.bandsplit.inverse_transform(x)

        # Unpad
        x = x[:, :, 0:T0, :, :]

        # Get complex mask
        mask = torch.view_as_complex(x)

        # Calculate STFT of separated audio
        sep_stft = mask * complex_sp

        # ISTFT
        output = self.istft(sep_stft)

        return output

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        """
        Pad a spectrum so it can be evenly divided by downsample ratio.

        Args:
            x: Input tensor, e.g., (b, c, t=201, f).

        Returns:
            Padded tensor, e.g., (b, c, t=204, f).
        """
        # Pad last frames, e.g., 201 -> 204
        pad_t = -x.shape[2] % self.patch_size[0]
        x = F.pad(x, pad=(0, 0, 0, 0, 0, pad_t))

        return x


if __name__ == "__main__":
    model = BSRoformer()
    dummy_audio = torch.randn(1, 2, 48000 * 2)
    output = model(dummy_audio)
    print(output.shape)
