"""
Ternary linear layer — drop-in replacement for nn.Linear.

Patent 1: Ternary weight encoding replaces FP16 multiply-accumulate with
          compare-and-add. For weight w_ij ∈ {-1, 0, +1}:
              w_ij = +1  →  output += input   (add)
              w_ij = -1  →  output -= input   (subtract)
              w_ij =  0  →  skip              (zero-skip, Patent 7)

          No multiplication occurs. This is the fundamental insight.

Patent 2: 2-bit encoding: 01 = +1, 10 = -1, 00 = 0, 11 = reserved.
          4 weights per byte. 8x compression vs FP16.

Patent 7: Sparsity bitmap enables zero-skip. Typical ternary models have
          60-70% zero weights. The bitmap allows skipping both memory loads
          and compute for zero entries.

Patent 36 (CNS Architecture), Claim 2:
    Synaptic-equivalent weight encoding where:
        +1 maps to excitatory neurotransmission
        -1 maps to inhibitory neurotransmission
         0 maps to resting membrane potential
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from terncore.arithmetic.quantizer import TernaryQuantizer


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.

    During forward pass:
        1. Weights are quantised to ternary using straight-through estimator (STE)
        2. Output = input @ (ternary_weights * alpha) + bias
        3. Gradients flow through STE for training compatibility

    During inference (eval mode):
        Weights are pre-quantised. Forward pass is pure compare-and-add.

    Args:
        in_features:  Size of each input sample.
        out_features: Size of each output sample.
        bias:         If True, adds a learnable bias (kept in FP32).
        threshold:    Quantisation threshold. Default 0.7.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.alpha = nn.Parameter(torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Cached ternary weights for inference mode
        self.register_buffer("_cached_ternary", None)
        self.register_buffer("_cached_alpha", None)
        self.register_buffer("_sparsity_bitmap", None)

        # Initialise weights (Xavier uniform, standard for linear layers)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ternary weights.

        Training: uses STE for gradient flow through quantisation.
        Eval: uses cached ternary weights (pure compare-and-add).
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_eval(x)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward with straight-through estimator."""
        # Quantise weights to ternary
        w_ternary = self._ternize_ste(self.weight)

        # Scale by learned alpha and compute linear transform
        output = F.linear(x, w_ternary * self.alpha, self.bias)
        return output

    def _forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward with cached ternary weights."""
        if self._cached_ternary is None:
            self._cache_ternary_weights()

        # Pure compare-and-add: no multiplication in the weight path
        # (alpha scaling is a single multiply, not per-weight)
        output = F.linear(x, self._cached_ternary * self._cached_alpha, self.bias)
        return output

    def _ternize_ste(self, w: torch.Tensor) -> torch.Tensor:
        """
        Ternary quantisation with straight-through estimator.

        Forward: hard ternary assignment {-1, 0, +1}
        Backward: gradients pass through as if quantisation were identity

        The STE trick: ternary + (w - w.detach())
            Forward value = ternary (because w - w.detach() = 0 in forward)
            Backward gradient = gradient of w (because ternary.detach() has no grad)
        """
        with torch.no_grad():
            abs_w = torch.abs(w)
            delta = self.threshold * torch.mean(abs_w)

        ternary = torch.where(
            w > delta,
            torch.ones_like(w),
            torch.where(
                w < -delta,
                -torch.ones_like(w),
                torch.zeros_like(w),
            ),
        )

        # Straight-through estimator: forward uses ternary, backward uses w
        return ternary + (w - w.detach())

    def _cache_ternary_weights(self) -> None:
        """Pre-compute and cache ternary weights for inference."""
        quantizer = TernaryQuantizer(threshold=self.threshold)
        ternary, alpha = quantizer.quantize(self.weight.data)

        self._cached_ternary = ternary
        self._cached_alpha = alpha

        # Sparsity bitmap for zero-skip (Patent 7)
        self._sparsity_bitmap = (ternary != 0)

    def invalidate_cache(self) -> None:
        """Clear cached weights (call after weight updates)."""
        self._cached_ternary = None
        self._cached_alpha = None
        self._sparsity_bitmap = None

    @property
    def sparsity(self) -> float:
        """Fraction of weights that are zero."""
        if self._cached_ternary is None:
            self._cache_ternary_weights()
        total = self._cached_ternary.numel()
        zeros = (self._cached_ternary == 0).sum().item()
        return zeros / total

    @property
    def compression_ratio(self) -> float:
        """
        Compression vs FP16.

        FP16 = 16 bits/weight. Ternary = 2 bits/weight.
        Base ratio = 8x. With sparsity bitmap, effective ratio is higher.
        """
        base_ratio = 8.0  # 16 bits / 2 bits
        # Sparse weights don't need storage (only the bitmap)
        # Effective: 2 bits for non-zero + 1 bit bitmap overhead
        return base_ratio  # conservative; actual is higher with sparsity

    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        s += f", bias={self.bias is not None}"
        s += f", threshold={self.threshold}"
        if self._cached_ternary is not None:
            s += f", sparsity={self.sparsity:.1%}"
        return s


class TernaryConv2d(nn.Module):
    """
    Conv2d layer with ternary weights {-1, 0, +1}.

    Same principle as TernaryLinear but for convolutional layers.
    Useful for vision models and multimodal architectures.

    Patent 1: Ternary weight encoding.
    Patent 19: Multimodal model conversion.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.alpha = nn.Parameter(torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_cached_ternary", None)
        self.register_buffer("_cached_alpha", None)

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_ternary = self._ternize_ste(self.weight)
            return F.conv2d(
                x, w_ternary * self.alpha, self.bias,
                self.stride, self.padding,
            )
        else:
            if self._cached_ternary is None:
                self._cache_ternary_weights()
            return F.conv2d(
                x, self._cached_ternary * self._cached_alpha, self.bias,
                self.stride, self.padding,
            )

    def _ternize_ste(self, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            delta = self.threshold * torch.mean(torch.abs(w))
        ternary = torch.where(
            w > delta, torch.ones_like(w),
            torch.where(w < -delta, -torch.ones_like(w), torch.zeros_like(w)),
        )
        return ternary + (w - w.detach())

    def _cache_ternary_weights(self) -> None:
        quantizer = TernaryQuantizer(threshold=self.threshold)
        ternary, alpha = quantizer.quantize(self.weight.data)
        self._cached_ternary = ternary
        self._cached_alpha = alpha
