"""
Streaming weight loader for sharded safetensors models.

Reads model.safetensors.index.json and yields one transformer block's
worth of tensors at a time, keeping peak memory at ~1.7 GB for a 70B
model instead of the 140 GB required to load the full model.

Part of tern-core v0.5.0: streaming shard-by-shard conversion pipeline.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch

_BLOCK_PATTERN = re.compile(r"\.layers\.(\d+)\.")


@dataclass
class WeightBlock:
    """A group of weight tensors belonging to one transformer block."""

    block_idx: int
    weights: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def linear_names(self) -> list[str]:
        """Names of weight tensors that look like Linear layers (2-D)."""
        return [n for n, t in self.weights.items() if t.ndim == 2]


@dataclass
class NonBlockWeights:
    """Weights that sit outside the transformer block stack."""

    weights: dict[str, torch.Tensor] = field(default_factory=dict)


def _parse_block_idx(weight_name: str) -> Optional[int]:
    """Extract the transformer block index from a weight name, or None."""
    m = _BLOCK_PATTERN.search(weight_name)
    return int(m.group(1)) if m else None


def _load_tensor(shard_path: Path, tensor_name: str) -> torch.Tensor:
    """Load a single tensor from a safetensors shard."""
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


class ShardedWeightIterator:
    """Iterate over transformer blocks from a sharded safetensors model.

    Reads the weight map once, groups tensors by transformer block index,
    then yields ``WeightBlock`` objects one block at a time.  Each tensor
    is loaded lazily from its shard file and discarded by the caller after
    processing.

    For weights outside the block stack (``embed_tokens``, ``lm_head``,
    ``model.norm``), a final ``NonBlockWeights`` object is yielded.

    Usage::

        loader = ShardedWeightIterator("./llama70b")
        for item in loader:
            if isinstance(item, WeightBlock):
                print(f"Block {item.block_idx}: {list(item.weights.keys())}")
            elif isinstance(item, NonBlockWeights):
                print(f"Non-block: {list(item.weights.keys())}")
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No safetensors index found at {index_path}. "
                f"Expected a sharded model with model.safetensors.index.json."
            )
        with open(index_path) as f:
            index = json.load(f)

        self.weight_map: dict[str, str] = index["weight_map"]
        self.total_size: int = index.get("metadata", {}).get("total_size", 0)

        # Pre-compute block grouping
        self._block_weights: dict[int, list[str]] = defaultdict(list)
        self._non_block_weights: list[str] = []
        for wname in self.weight_map:
            bidx = _parse_block_idx(wname)
            if bidx is not None:
                self._block_weights[bidx].append(wname)
            else:
                self._non_block_weights.append(wname)

    @property
    def num_blocks(self) -> int:
        return len(self._block_weights)

    @property
    def num_weights(self) -> int:
        return len(self.weight_map)

    @property
    def block_indices(self) -> list[int]:
        return sorted(self._block_weights.keys())

    def eligible_linear_names(self) -> list[str]:
        """All weight names that are eligible for ternary conversion.

        Returns 2-D tensor names inside transformer blocks, excluding
        layernorm weights.  This mirrors autoscan._eligible_linear_names
        but works from the index without loading any tensors.
        """
        skip = ("layernorm", "layer_norm", "rmsnorm", "embed", "lm_head",
                "output", "classifier")
        names = []
        for bidx in self.block_indices:
            for wname in self._block_weights[bidx]:
                if any(s in wname.lower() for s in skip):
                    continue
                # Only include weight tensors (exclude bias by name convention)
                if wname.endswith(".weight"):
                    names.append(wname)
        return names

    def iter_blocks(self) -> Iterator[WeightBlock | NonBlockWeights]:
        """Yield transformer blocks in order, then non-block weights."""
        for bidx in self.block_indices:
            block = WeightBlock(block_idx=bidx)
            for wname in sorted(self._block_weights[bidx]):
                shard_file = self.model_dir / self.weight_map[wname]
                block.weights[wname] = _load_tensor(shard_file, wname)
            yield block

        if self._non_block_weights:
            nb = NonBlockWeights()
            for wname in sorted(self._non_block_weights):
                shard_file = self.model_dir / self.weight_map[wname]
                nb.weights[wname] = _load_tensor(shard_file, wname)
            yield nb

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor, Optional[int]]]:
        """Yield (name, tensor, block_idx_or_None) one tensor at a time.

        Lower-level than iter_blocks — loads and yields exactly one tensor
        at a time for minimum memory usage.
        """
        for bidx in self.block_indices:
            for wname in sorted(self._block_weights[bidx]):
                shard_file = self.model_dir / self.weight_map[wname]
                yield wname, _load_tensor(shard_file, wname), bidx

        for wname in sorted(self._non_block_weights):
            shard_file = self.model_dir / self.weight_map[wname]
            yield wname, _load_tensor(shard_file, wname), None

    def __iter__(self):
        return self.iter_blocks()
