import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np


class WavUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels = self.audio_channels
        in_channels = 2
        self.downsample_ratio = 4**6

        self.encoder_block1 = EncoderBlock(in_channels, 32)
        self.encoder_block2 = EncoderBlock(32, 64)
        self.encoder_block3 = EncoderBlock(64, 128)
        self.encoder_block4 = EncoderBlock(128, 256)
        self.encoder_block5 = EncoderBlock(256, 512)
        self.encoder_block6 = EncoderBlock(512, 1024)
        self.middle = EncoderBlock(1024, 1024)
        self.decoder_block1 = DecoderBlock(1024, 512)
        self.decoder_block2 = DecoderBlock(512, 256)
        self.decoder_block3 = DecoderBlock(256, 128)
        self.decoder_block4 = DecoderBlock(128, 64)
        self.decoder_block5 = DecoderBlock(64, 32)
        self.decoder_block6 = DecoderBlock(32, 32) 

        self.last_conv = nn.Conv1d(
            in_channels=32, 
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
        time_steps = mixture.shape[-1]

        x = self.process_image(mixture)

        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)
        x4, latent4 = self.encoder_block4(x3)
        x5, latent5 = self.encoder_block5(x4)
        x6, latent6 = self.encoder_block6(x5)
        _, h = self.middle(x6)
        x7 = self.decoder_block1(h, latent6)
        x8 = self.decoder_block2(x7, latent5)
        x9 = self.decoder_block3(x8, latent4)
        x10 = self.decoder_block4(x9, latent3)
        x11 = self.decoder_block5(x10, latent2)
        x12 = self.decoder_block6(x11, latent1)

        x = self.last_conv(x12)
        
        output = self.unprocess_image(x, time_steps)

        return output

    def process_image(self, x):
        """Cut a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (B, C, 201, 1025)
        
        Outpus:
            output: E.g., (B, C, 208, 1024)
        """

        B, C, T = x.shape

        pad_len = (
            int(np.ceil(T / self.downsample_ratio)) * self.downsample_ratio
            - T
        )
        x = F.pad(x, pad=(0, pad_len))

        return x

    def unprocess_image(self, x, time_steps):
        """Patch a spectrum to the original shape. E.g.,
        
        Args:
            x: E.g., (B, C, 208, 1024)
        
        Outpus:
            output: E.g., (B, C, 201, 1025)
        """
        x = F.pad(x, pad=(0, 1))

        output = x[:, :, 0 : time_steps]

        return output


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size):
        r"""Residual block."""
        super(ConvBlock, self).__init__()

        # padding = [kernel_size[0] // 2, kernel_size[1] // 2]
        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels)
        # self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # self.conv2 = nn.Conv1d(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     bias=False,
        # )

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps, freq_bins)
        """
        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        # h = self.conv2(F.leaky_relu_(self.bn2(h)))

        return h



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.pool_size = 4
        kernel_size = 9

        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """

        latent = self.conv_block(x)

        output = F.avg_pool1d(latent, kernel_size=self.pool_size)
        
        return output, latent 


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        stride = 4
        kernel_size = 9

        self.upsample = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=stride,
            stride=stride,
            padding=0,
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