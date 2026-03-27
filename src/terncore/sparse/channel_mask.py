"""
Sparse channel masking — eliminate zero channels before ANE dispatch.

Patent 7:  Sparsity-aware execution — skip computation for zero weights.
Patent 37: Zero-weight clock-gating via structural pruning.
Patent 9:  Hierarchical sparsity across model layers.

Problem: ANE processes full dense matrices. 43% of ternary weights are zero
but still consume ANE compute cycles.

Solution: Two-stage sparsity elimination for CoreML/ANE:
  1. Structural: remove entirely-zero output rows (channels) from weight
     matrices. Smaller matmul → proportional ANE speedup.
  2. Element-wise: apply coremltools threshold pruning to mark remaining
     zeros as sparse. ANE can skip individual zero elements.

Combined with 2-bit palettization, this yields:
  - Fewer output channels (structural)
  - Sparse representation of remaining zeros (element-wise)
  - 2-bit weight encoding (palettization)

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ChannelMaskStats",
    "SparseChannelLinear",
    "analyze_channel_sparsity",
    "apply_channel_mask",
]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class ChannelMaskStats:
    """Sparsity analysis for a single linear layer."""

    name: str
    out_features: int
    in_features: int
    element_sparsity: float      # fraction of zero elements
    dead_output_channels: int    # rows that are entirely zero
    dead_input_channels: int     # columns that are entirely zero
    pruned_out_features: int     # output dimension after row pruning
    pruned_in_features: int      # input dimension after column pruning
    structural_reduction: float  # fraction of compute eliminated structurally
    remaining_sparsity: float    # element sparsity in the pruned submatrix


@dataclass
class ModelMaskStats:
    """Aggregate sparsity analysis across all layers."""

    layers: list[ChannelMaskStats] = field(default_factory=list)
    total_original_ops: int = 0
    total_pruned_ops: int = 0
    total_dead_rows: int = 0
    total_dead_cols: int = 0

    @property
    def structural_speedup(self) -> float:
        if self.total_pruned_ops == 0:
            return 1.0
        return self.total_original_ops / self.total_pruned_ops

    @property
    def theoretical_speedup(self) -> float:
        """Upper bound: structural pruning + perfect element-wise skip."""
        remaining_ops = 0
        for layer in self.layers:
            pruned_ops = layer.pruned_out_features * layer.pruned_in_features
            # Remaining element sparsity means more ops can be skipped
            remaining_ops += pruned_ops * (1.0 - layer.remaining_sparsity)
        if remaining_ops == 0:
            return float("inf")
        return self.total_original_ops / remaining_ops


def analyze_channel_sparsity(
    model: nn.Module,
    threshold: float = 0.7,
) -> ModelMaskStats:
    """
    Analyze per-layer channel sparsity for ternary-quantized model.

    Identifies dead output channels (all-zero rows) and dead input channels
    (all-zero columns) in each linear layer's weight matrix.

    Args:
        model:     PyTorch model with ternary-quantized linear layers.
        threshold: Ternary quantization threshold.

    Returns:
        ModelMaskStats with per-layer and aggregate analysis.
    """
    stats = ModelMaskStats()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        w = module.weight.data  # [out, in]

        # Quantize to ternary to get exact sparsity pattern
        abs_w = w.abs()
        mean_abs = abs_w.mean(dim=1, keepdim=True)
        delta = threshold * mean_abs
        codes = torch.zeros_like(w, dtype=torch.int8)
        codes[w > delta] = 1
        codes[w < -delta] = -1

        total = codes.numel()
        n_zeros = (codes == 0).sum().item()

        # Dead output channels: rows where ALL elements are zero
        row_nonzero = (codes != 0).any(dim=1)  # [out]
        dead_rows = (~row_nonzero).sum().item()
        active_rows = row_nonzero.sum().item()

        # Dead input channels: columns where ALL elements are zero
        col_nonzero = (codes != 0).any(dim=0)  # [in]
        dead_cols = (~col_nonzero).sum().item()
        active_cols = col_nonzero.sum().item()

        # Sparsity in pruned submatrix
        if active_rows > 0 and active_cols > 0:
            pruned = codes[row_nonzero][:, col_nonzero]
            remaining_sparsity = (pruned == 0).sum().item() / pruned.numel()
        else:
            remaining_sparsity = 0.0

        original_ops = w.shape[0] * w.shape[1]
        pruned_ops = active_rows * active_cols
        structural_reduction = 1.0 - (pruned_ops / original_ops) if original_ops > 0 else 0.0

        layer_stats = ChannelMaskStats(
            name=name,
            out_features=w.shape[0],
            in_features=w.shape[1],
            element_sparsity=n_zeros / total,
            dead_output_channels=dead_rows,
            dead_input_channels=dead_cols,
            pruned_out_features=active_rows,
            pruned_in_features=active_cols,
            structural_reduction=structural_reduction,
            remaining_sparsity=remaining_sparsity,
        )
        stats.layers.append(layer_stats)
        stats.total_original_ops += original_ops
        stats.total_pruned_ops += pruned_ops
        stats.total_dead_rows += dead_rows
        stats.total_dead_cols += dead_cols

    return stats


# ---------------------------------------------------------------------------
# Sparse channel linear layer
# ---------------------------------------------------------------------------

class SparseChannelLinear(nn.Module):
    """
    Linear layer with structurally pruned zero channels.

    Physically removes dead output rows and dead input columns from the
    weight matrix. Dispatches a smaller dense matmul, then scatters
    results back to the full output dimension.

    This converts element-wise sparsity into dimension reduction —
    something ANE can directly benefit from (smaller matrix = fewer cycles).

    Patent 37: Zero-weight clock-gating → structural channel elimination.

    Args:
        compact_weight: Weight matrix with dead rows/cols removed [M', N'].
        alpha:          Per-layer ternary scaling factor.
        active_rows:    Index tensor mapping compact → full output channels.
        active_cols:    Index tensor mapping compact → full input channels.
        full_out:       Original output dimension.
        full_in:        Original input dimension.
        bias:           Optional bias tensor (full output dimension).
    """

    def __init__(
        self,
        compact_weight: torch.Tensor,
        alpha: float,
        active_rows: torch.Tensor,
        active_cols: torch.Tensor,
        full_out: int,
        full_in: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.full_out = full_out
        self.full_in = full_in

        # Store compact weight as parameter (for CoreML tracing)
        self.compact = nn.Linear(
            compact_weight.shape[1], compact_weight.shape[0], bias=False
        )
        self.compact.weight = nn.Parameter(compact_weight.float() * alpha)

        # Index buffers (int64 for torch.index_select)
        self.register_buffer("active_rows", active_rows.long())
        self.register_buffer("active_cols", active_cols.long())

        if bias is not None:
            self.register_buffer("bias", bias.float())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Select active input channels: [..., N] → [..., N']
        x_compact = torch.index_select(x, -1, self.active_cols)

        # Compact matmul: [..., N'] → [..., M']
        y_compact = self.compact(x_compact)

        # Scatter to full output dimension: [..., M'] → [..., M]
        shape = list(x.shape[:-1]) + [self.full_out]
        y = torch.zeros(shape, dtype=y_compact.dtype, device=y_compact.device)
        y[..., self.active_rows] = y_compact

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        threshold: float = 0.7,
    ) -> "SparseChannelLinear":
        """
        Convert nn.Linear with ternary weights to SparseChannelLinear.

        Quantizes weights, identifies dead channels, builds compact layer.
        """
        w = linear.weight.data
        out_f, in_f = w.shape

        # Ternary quantization
        abs_w = w.abs()
        mean_abs = abs_w.mean(dim=1, keepdim=True)
        delta = threshold * mean_abs
        codes = torch.zeros_like(w, dtype=torch.int8)
        codes[w > delta] = 1
        codes[w < -delta] = -1

        # Scaling factor (mean of non-zero absolute values per row)
        mask = codes != 0
        if mask.any():
            alpha = abs_w[mask].mean().item()
        else:
            alpha = 1.0

        # Find active channels
        row_active = (codes != 0).any(dim=1)
        col_active = (codes != 0).any(dim=0)

        active_rows = torch.where(row_active)[0]
        active_cols = torch.where(col_active)[0]

        # If no dead channels, keep originals to avoid empty-tensor issues
        if active_rows.numel() == out_f:
            active_rows = torch.arange(out_f)
        if active_cols.numel() == in_f:
            active_cols = torch.arange(in_f)

        # Build compact weight matrix
        compact_codes = codes[active_rows][:, active_cols]
        compact_weight = compact_codes.float()

        return cls(
            compact_weight=compact_weight,
            alpha=alpha,
            active_rows=active_rows,
            active_cols=active_cols,
            full_out=out_f,
            full_in=in_f,
            bias=linear.bias.data if linear.bias is not None else None,
        )


# ---------------------------------------------------------------------------
# Model-level channel masking
# ---------------------------------------------------------------------------

def apply_channel_mask(
    model: nn.Module,
    threshold: float = 0.7,
    min_dead_channels: int = 1,
) -> dict:
    """
    Replace linear layers with SparseChannelLinear where beneficial.

    Only replaces layers that have at least `min_dead_channels` dead
    output or input channels — otherwise the index_select/scatter overhead
    exceeds the matmul savings.

    Args:
        model:             PyTorch model to modify in-place.
        threshold:         Ternary quantization threshold.
        min_dead_channels: Minimum dead channels to trigger replacement.

    Returns:
        Dict with replacement statistics.
    """
    replacements = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        w = module.weight.data
        abs_w = w.abs()
        mean_abs = abs_w.mean(dim=1, keepdim=True)
        delta = threshold * mean_abs
        codes = torch.zeros_like(w, dtype=torch.int8)
        codes[w > delta] = 1
        codes[w < -delta] = -1

        dead_rows = (~(codes != 0).any(dim=1)).sum().item()
        dead_cols = (~(codes != 0).any(dim=0)).sum().item()

        if dead_rows >= min_dead_channels or dead_cols >= min_dead_channels:
            sparse_layer = SparseChannelLinear.from_linear(module, threshold)

            # Replace in model
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], sparse_layer)

            replacements.append({
                "name": name,
                "original_shape": (w.shape[0], w.shape[1]),
                "compact_shape": (
                    sparse_layer.compact.weight.shape[0],
                    sparse_layer.compact.weight.shape[1],
                ),
                "dead_rows": dead_rows,
                "dead_cols": dead_cols,
            })

    return {
        "replaced_layers": len(replacements),
        "details": replacements,
    }
