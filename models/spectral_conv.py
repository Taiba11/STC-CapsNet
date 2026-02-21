"""
Spectral Convolution (2D) Module for STC-CapsNet (Section II-B.1).

Applies 2D convolutions across both time and frequency axes to capture
localized spectral patterns in the spectrogram.

    y(t,f) = sum_{k1} sum_{k2} x(t-k1, f-k2) * w(k1, k2)    (Eq. 4)

Captures: pitch modulations, harmonics, frequency shifts, spectral distortions
"""

import torch
import torch.nn as nn


class SpectralConvLayer(nn.Module):
    """
    Single spectral (2D) convolution layer with BatchNorm and activation.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (tuple): 2D kernel (time, frequency).
        stride (int): Convolution stride.
        padding: Padding (default: same).
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3),
        stride: int = 1,
        padding: int = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, freq, time)
        Returns:
            (batch, out_channels, freq, time)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SpectralConvBlock(nn.Module):
    """
    Multi-layer spectral (2D) convolution block.

    Stacks 2D convolutions to capture localized time-frequency patterns
    such as harmonic structures, pitch anomalies, and spectral distortions.

    Args:
        in_channels (int): Input channels.
        channels (list): Output channels per layer.
        kernel_sizes (list): Kernel sizes per layer, each a [H, W] pair.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        channels: list = None,
        kernel_sizes: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels = channels or [128, 256]
        kernel_sizes = kernel_sizes or [[3, 3], [3, 3]]

        layers = []
        prev_ch = in_channels
        for ch, ks in zip(channels, kernel_sizes):
            layers.append(SpectralConvLayer(prev_ch, ch, tuple(ks), dropout=dropout))
            prev_ch = ch

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, freq, time)
        Returns:
            (batch, out_channels, freq, time) — spectral feature maps.
        """
        return self.layers(x)
