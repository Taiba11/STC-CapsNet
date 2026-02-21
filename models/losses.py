"""
Margin Loss for STC-CapsNet (Section II-C, Eq. 7).

    L_k = T_k * max(0, m+ - ||v_k||)^2 + λ * (1 - T_k) * max(0, ||v_k|| - m-)^2
"""

import torch
import torch.nn as nn


class MarginLoss(nn.Module):
    """
    Margin Loss for capsule classification.

    Encourages the correct class capsule to have ||v_k|| >= m+
    and the incorrect class capsule to have ||v_k|| <= m-.

    Args:
        m_plus (float): Positive margin. Default: 0.9
        m_minus (float): Negative margin. Default: 0.1
        lambda_val (float): Down-weighting factor. Default: 0.5
    """

    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val

    def forward(self, v_k, targets):
        """
        Args:
            v_k: (batch, num_classes, capsule_dim) — capsule output vectors.
            targets: (batch,) — integer labels.
        Returns:
            Scalar loss.
        """
        # ||v_k||: capsule norms = probabilities (Eq. 6)
        v_k_norm = torch.sqrt((v_k ** 2).sum(dim=-1) + 1e-8)

        # One-hot targets
        num_classes = v_k.size(1)
        T_k = torch.zeros(v_k.size(0), num_classes, device=v_k.device)
        T_k.scatter_(1, targets.unsqueeze(1), 1.0)

        # Margin loss (Eq. 7)
        left = T_k * torch.clamp(self.m_plus - v_k_norm, min=0.0) ** 2
        right = self.lambda_val * (1.0 - T_k) * torch.clamp(
            v_k_norm - self.m_minus, min=0.0
        ) ** 2

        return (left + right).sum(dim=-1).mean()
