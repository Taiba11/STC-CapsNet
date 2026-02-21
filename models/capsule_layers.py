"""
Capsule Layers for STC-CapsNet (Section II-B.2).

Implements:
    - squash activation
    - PrimaryCapsuleLayer: converts conv features to capsule vectors
    - HigherCapsuleLayer: aggregates via dynamic routing (Eq. 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(tensor, dim=-1):
    """
    Squash activation — normalizes vector length to [0, 1].

    Args:
        tensor: Input tensor.
        dim: Dimension to compute norm over.
    Returns:
        Squashed tensor.
    """
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(squared_norm + 1e-8)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * tensor / norm


class PrimaryCapsuleLayer(nn.Module):
    """
    Primary Capsule Layer (Section II-B.2).

    Converts 2D convolution feature maps into capsule vectors.
    Each capsule type has its own conv filter producing a vector output
    whose magnitude = likelihood of feature existence and
    orientation = instantiation parameters (time-frequency position, amplitude).

    Args:
        in_channels (int): Input channels from spectral conv block.
        num_capsules (int): Number of capsule types.
        capsule_dim (int): Dimension of each capsule vector.
        kernel_size (int): Conv kernel size.
        stride (int): Conv stride.
    """

    def __init__(
        self,
        in_channels: int,
        num_capsules: int = 8,
        capsule_dim: int = 32,
        kernel_size: int = 9,
        stride: int = 2,
    ):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        self.capsules = nn.ModuleList([
            nn.Conv2d(
                in_channels, capsule_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, H, W) — spectral feature maps.
        Returns:
            (batch, num_total_capsules, capsule_dim) — primary capsule outputs.
        """
        outputs = [cap(x) for cap in self.capsules]
        # Each: (batch, capsule_dim, H', W')
        outputs = torch.stack(outputs, dim=1)
        # (batch, num_capsules, capsule_dim, H', W')

        batch_size = outputs.size(0)
        outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()
        outputs = outputs.view(batch_size, -1, self.capsule_dim)
        # (batch, num_capsules * H' * W', capsule_dim)

        return squash(outputs)


class HigherCapsuleLayer(nn.Module):
    """
    Higher Capsule Layer with Dynamic Routing (Section II-B.2).

    Each higher-level capsule aggregates inputs from multiple primary
    capsules. Dynamic routing (Eq. 5) determines coupling coefficients
    based on agreement between predictions.

        c_ij = exp(b_ij) / sum_k exp(b_ik)

    Args:
        num_capsules (int): Number of output capsules (2 = real/fake).
        num_routes (int): Number of input capsules from primary layer.
        in_dim (int): Input capsule vector dimension.
        out_dim (int): Output capsule vector dimension.
        routing_iterations (int): Number of routing iterations.
    """

    def __init__(
        self,
        num_capsules: int = 2,
        num_routes: int = -1,
        in_dim: int = 32,
        out_dim: int = 16,
        routing_iterations: int = 3,
    ):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations

        # Transformation weight matrices W_ij
        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, out_dim, in_dim) * 0.01
        )

    def forward(self, x):
        """
        Dynamic routing between primary and higher capsules.

        Args:
            x: (batch, num_routes, in_dim) — primary capsule vectors.
        Returns:
            (batch, num_capsules, out_dim) — higher capsule output vectors.
        """
        batch_size = x.size(0)

        # Prediction vectors: u_hat = W @ x
        x_expanded = x.unsqueeze(2).unsqueeze(4)
        # (batch, num_routes, 1, in_dim, 1)

        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x_expanded).squeeze(-1)
        # (batch, num_routes, num_capsules, out_dim)

        # Initialize log priors b_ij = 0
        b_ij = torch.zeros(
            batch_size, self.num_routes, self.num_capsules, 1,
            device=x.device,
        )

        # Dynamic routing iterations (Algorithm from Section II-B.2)
        for iteration in range(self.routing_iterations):
            # Coupling coefficients via softmax (Eq. 5)
            c_ij = F.softmax(b_ij, dim=2)

            # Weighted sum: s_j = sum_i c_ij * u_hat_ij
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # Squash
            v_j = squash(s_j, dim=-1)

            # Update b_ij based on agreement
            if iteration < self.routing_iterations - 1:
                agreement = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + agreement

        return v_j.squeeze(1)
