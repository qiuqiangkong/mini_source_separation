import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np

from models.fourier import Fourier


class UNet(Fourier):
    def __init__(self):
        super(UNet, self).__init__(
            n_fft=2048, 
            hop_length=441, 
            return_complex=True, 
            normalized=True
        )

        self.downsample_ratio = 16
        
        # freq_bins = self.n_fft // 2 + 1
        
        self.audio_channels = 2
        self.cmplx_num = 2
        in_channels = self.audio_channels * self.cmplx_num

        self.encoder_block1 = EncoderBlock(in_channels, 16)
        self.encoder_block2 = EncoderBlock(16, 64)
        self.encoder_block3 = EncoderBlock(64, 256)
        self.encoder_block4 = EncoderBlock(256, 1024)
        self.middle = EncoderBlock(1024, 1024)
        self.decoder_block1 = DecoderBlock(1024, 256)
        self.decoder_block2 = DecoderBlock(256, 64)
        self.decoder_block3 = DecoderBlock(64, 16)
        self.decoder_block4 = DecoderBlock(16, 16)

        self.last_conv = nn.Conv2d(
            in_channels=16, 
            out_channels=in_channels, 
            kernel_size=1, 
            padding=0,
        )

    def forward(self, mixture):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """

        # Complex spectrum.
        complex_sp = self.stft(mixture)
        # shape: (batch_size, channels_num, time_steps, freq_bins)

        x = torch.view_as_real(complex_sp)
        x = rearrange(x, 'b c t f k -> b (c k) t f')        
        # shape: (B, C, T, F)

        # process a spectrum that can be evenly divided by downsample_ratio.
        x = self.process_image(x)

        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)
        x4, latent4 = self.encoder_block4(x3)
        _, h = self.middle(x4)
        x5 = self.decoder_block1(h, latent4)
        x6 = self.decoder_block2(x5, latent3)
        x7 = self.decoder_block3(x6, latent2)
        x8 = self.decoder_block4(x7, latent1)

        x = self.last_conv(x8)

        x = rearrange(x, 'b (c k) t f -> b c t f k', k=self.cmplx_num).contiguous()
        mask = torch.view_as_complex(x)
        
        # Unprocess a spectrum to the original shape.
        mask = self.unprocess_image(mask, time_steps=complex_sp.shape[2])

        sep_stft = mask * complex_sp

        output = self.istft(sep_stft)

        return output

    '''
    def forward(self, mixture):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """

        # Complex spectrum.
        complex_sp = self.stft(mixture)
        # shape: (batch_size, channels_num, time_steps, freq_bins)

        mag = torch.abs(complex_sp)
        angle = torch.angle(complex_sp)
        # shape: (batch_size, channels_num, time_steps, freq_bins, complex=2)

        # Cut a spectrum that can be evenly divided by downsample_ratio.
        x = self.cut_image(mag)

        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)
        x4, latent4 = self.encoder_block4(x3)
        _, h = self.middle(x4)
        x5 = self.decoder_block1(h, latent4)
        x6 = self.decoder_block2(x5, latent3)
        x7 = self.decoder_block3(x6, latent2)
        x8 = self.decoder_block4(x7, latent1)

        x = torch.sigmoid(self.last_conv(x8))

        # Patch a spectrum to the original shape.
        mask = self.patch_image(x, time_steps=mag.shape[2])

        # Predict the spectrum of the target signal.
        x = (mask * mag) * torch.exp(1.j * angle)

        output = self.istft(x)

        return output
    '''

    def process_image(self, x):
        """Cut a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (B, C, 201, 1025)
        
        Outpus:
            output: E.g., (B, C, 208, 1024)
        """

        B, C, T, Freq = x.shape

        pad_len = (
            int(np.ceil(T / self.downsample_ratio)) * self.downsample_ratio
            - T
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        output = x[:, :, :, 0 : Freq - 1]

        return output

    def unprocess_image(self, x, time_steps):
        """Patch a spectrum to the original shape. E.g.,
        
        Args:
            x: E.g., (B, C, 208, 1024)
        
        Outpus:
            output: E.g., (B, C, 201, 1025)
        """
        x = F.pad(x, pad=(0, 1))

        output = x[:, :, 0 : time_steps, :]

        return output


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size):
        r"""Residual block."""
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps, freq_bins)
        """
        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        h = self.conv2(F.leaky_relu_(self.bn2(h)))

        if self.is_shortcut:
            return self.shortcut(x) + h
        else:
            return x + h


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(EncoderBlock, self).__init__()

        self.pool_size = 2

        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """

        latent = self.conv_block(x)

        output = F.avg_pool2d(latent, kernel_size=self.pool_size)
        
        return output, latent 


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(DecoderBlock, self).__init__()

        stride = 2

        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=stride,
            stride=stride,
            padding=(0, 0),
            bias=False,
        )

        self.conv_block = ConvBlock(in_channels * 2, out_channels, kernel_size)

    def forward(self, x, latent):
        """
        Args:
            x: (batch_size, in_channels, time_steps // 2, freq_bins // 2)

        Returns:
            output: (batch_size, out_channels, time_steps, freq_bins)
        """

        x = self.upsample(x)

        x = torch.cat((x, latent), dim=1)

        output = self.conv_block(x)
        
        return output