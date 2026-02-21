"""
Temporal Convolution (1D) Module for STC-CapsNet (Section II-B.1).

Applies 1D convolutions along the time axis of the spectrogram to capture
both short-term and long-term temporal dependencies in the audio signal.

    y(t) = sum_{k=0}^{K-1} x(t-k) * w_k          (Eq. 3)

Short-term dependencies: phoneme transitions, micro-timing cues
Long-term dependencies: phrase-level patterns, rhythm, pacing
"""

import torch
import torch.nn as nn


class TemporalConvLayer(nn.Module):
    """
    Single temporal convolution layer with BatchNorm and activation.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): 1D kernel size along time axis.
        stride (int): Convolution stride.
        padding (int): Padding (default: causal-style).
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, time_steps)
        Returns:
            (batch, out_channels, time_steps)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TemporalConvBlock(nn.Module):
    """
    Multi-layer temporal convolution block.

    Stacks multiple 1D convolution layers with increasing receptive field
    to capture both short-term and long-term time-domain dependencies.

    Args:
        in_channels (int): Input channels (1 for single-channel spectrogram).
        channels (list): Output channels for each layer.
        kernel_sizes (list): Kernel sizes for each layer.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list = None,
        kernel_sizes: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels = channels or [64, 128, 256]
        kernel_sizes = kernel_sizes or [7, 5, 3]

        assert len(channels) == len(kernel_sizes), \
            "channels and kernel_sizes must have same length"

        layers = []
        prev_ch = in_channels
        for ch, ks in zip(channels, kernel_sizes):
            layers.append(TemporalConvLayer(prev_ch, ch, ks, dropout=dropout))
            prev_ch = ch

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, time_steps) — 1D signal or flattened spectrogram rows.
        Returns:
            (batch, out_channels, time_steps) — temporal feature maps.
        """
        return self.layers(x)
