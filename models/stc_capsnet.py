"""
STC-CapsNet: Spatio-Temporal Convolutional Capsule Network for Audio Deepfake Detection.

Full architecture (Section II-B):
    1. Preprocessing → Mel-spectrogram or Grayscale spectrogram
    2. Temporal Convolution (1D) — time-domain dependencies
    3. Spectral Convolution (2D) — frequency-domain patterns
    4. Primary Capsules — encode time-frequency relationships as vectors
    5. Higher Capsules — aggregate via dynamic routing
    6. Classification — margin loss, ||v_fake|| = p(fake)

Paper: Wani, Qadri & Amerini, IEEE CISM 2025
"""

import torch
import torch.nn as nn

from .temporal_conv import TemporalConvBlock
from .spectral_conv import SpectralConvBlock
from .capsule_layers import PrimaryCapsuleLayer, HigherCapsuleLayer, squash


class STCCapsNet(nn.Module):
    """
    Spatio-Temporal Convolutional Capsule Network.

    Supports both mel-spectrogram and grayscale spectrogram inputs.

    Args:
        in_channels (int): Input channels (1 for grayscale/mel, 3 for RGB mel).
        num_classes (int): Number of output classes (2 = real/fake).
        temporal_channels (list): Channel sizes for temporal conv layers.
        temporal_kernels (list): Kernel sizes for temporal conv layers.
        spectral_channels (list): Channel sizes for spectral conv layers.
        spectral_kernels (list): Kernel sizes for spectral conv layers.
        primary_num_caps (int): Number of primary capsule types.
        primary_cap_dim (int): Dimension of primary capsule vectors.
        primary_kernel (int): Kernel for primary capsule conv.
        primary_stride (int): Stride for primary capsule conv.
        higher_cap_dim (int): Dimension of higher capsule vectors.
        routing_iterations (int): Dynamic routing iterations.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        temporal_channels: list = None,
        temporal_kernels: list = None,
        spectral_channels: list = None,
        spectral_kernels: list = None,
        primary_num_caps: int = 8,
        primary_cap_dim: int = 32,
        primary_kernel: int = 9,
        primary_stride: int = 2,
        higher_cap_dim: int = 16,
        routing_iterations: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        temporal_channels = temporal_channels or [64, 128, 256]
        temporal_kernels = temporal_kernels or [7, 5, 3]
        spectral_channels = spectral_channels or [128, 256]
        spectral_kernels = spectral_kernels or [[3, 3], [3, 3]]

        self.num_classes = num_classes
        self.in_channels = in_channels

        # --- Stage 1: Temporal Convolution (1D) ---
        # Operates along time axis for each frequency band
        self.temporal_conv = TemporalConvBlock(
            in_channels=in_channels,
            channels=temporal_channels,
            kernel_sizes=temporal_kernels,
            dropout=dropout,
        )

        # --- Stage 2: Spectral Convolution (2D) ---
        # Operates across both time and frequency axes
        self.spectral_conv = SpectralConvBlock(
            in_channels=temporal_channels[-1],
            channels=spectral_channels,
            kernel_sizes=spectral_kernels,
            dropout=dropout,
        )

        # --- Stage 3: Primary Capsules ---
        self.primary_capsules = PrimaryCapsuleLayer(
            in_channels=spectral_channels[-1],
            num_capsules=primary_num_caps,
            capsule_dim=primary_cap_dim,
            kernel_size=primary_kernel,
            stride=primary_stride,
        )

        # --- Stage 4: Higher Capsules (lazy init) ---
        self._num_classes = num_classes
        self._higher_cap_dim = higher_cap_dim
        self._primary_cap_dim = primary_cap_dim
        self._routing_iterations = routing_iterations
        self.higher_capsules = None
        self._initialized = False

    def _init_higher_capsules(self, num_routes, device):
        """Lazily initialize higher capsules once spatial dims are known."""
        self.higher_capsules = HigherCapsuleLayer(
            num_capsules=self._num_classes,
            num_routes=num_routes,
            in_dim=self._primary_cap_dim,
            out_dim=self._higher_cap_dim,
            routing_iterations=self._routing_iterations,
        ).to(device)
        self._initialized = True

    def forward(self, x):
        """
        Full forward pass.

        Args:
            x: (batch, channels, freq_bins, time_frames) — spectrogram input.
                For mel-spectrogram: (batch, 1, n_mels, time)
                For grayscale: (batch, 1, freq, time)

        Returns:
            v_higher: (batch, num_classes, higher_cap_dim) — capsule output vectors.
                     ||v_higher[:, k, :]|| = probability of class k.
        """
        batch, C, F, T = x.shape

        # --- Temporal Convolution (1D) ---
        # Reshape: treat each frequency bin as a separate feature channel
        # (batch, C, F, T) → (batch * F, C, T) for 1D conv along time
        x_temporal = x.permute(0, 2, 1, 3).contiguous().view(batch * F, C, T)
        x_temporal = self.temporal_conv(x_temporal)
        # (batch * F, temporal_out_ch, T')
        temporal_ch = x_temporal.size(1)
        T_new = x_temporal.size(2)

        # Reshape back to 2D: (batch, temporal_out_ch, F, T')
        x_2d = x_temporal.view(batch, F, temporal_ch, T_new)
        x_2d = x_2d.permute(0, 2, 1, 3).contiguous()

        # --- Spectral Convolution (2D) ---
        x_spectral = self.spectral_conv(x_2d)
        # (batch, spectral_out_ch, F', T')

        # --- Primary Capsules ---
        primary_caps = self.primary_capsules(x_spectral)
        # (batch, num_total_capsules, primary_cap_dim)

        # --- Higher Capsules with Dynamic Routing ---
        if not self._initialized:
            self._init_higher_capsules(primary_caps.size(1), x.device)

        v_higher = self.higher_capsules(primary_caps)
        # (batch, num_classes, higher_cap_dim)

        return v_higher

    def predict(self, x):
        """
        Predict class and confidence.

        Args:
            x: Spectrogram input.
        Returns:
            predictions: (batch,) — predicted class indices.
            confidences: (batch, num_classes) — capsule norms.
        """
        v = self.forward(x)
        confidences = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)
        predictions = confidences.argmax(dim=1)
        return predictions, confidences

    def get_num_params(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
