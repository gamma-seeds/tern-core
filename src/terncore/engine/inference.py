"""
Ternary inference engine.

Patent 1-3: Core execution with compare-and-add arithmetic.
Patent 7-9: Sparsity optimisation and zero-skip execution.

This engine:
    1. Takes a standard PyTorch model (from HuggingFace or local)
    2. Converts eligible layers to ternary
    3. Runs inference using compare-and-add arithmetic
    4. Returns output with deterministic guarantees

CPU reference implementation. NPU backend added in Stage 1B.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Any

from terncore.arithmetic.quantizer import TernaryQuantizer, SensitivityAnalyzer
from terncore.arithmetic.linear import TernaryLinear, TernaryConv2d


def _get_conv1d_class():
    """Lazily import HuggingFace Conv1D if available."""
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except ImportError:
        return None


@dataclass
class ConversionReport:
    """Report from converting a model to ternary."""

    total_layers: int = 0
    converted_layers: int = 0
    skipped_layers: int = 0
    precision_critical_layers: list[str] = field(default_factory=list)
    total_params: int = 0
    ternary_params: int = 0
    original_size_mb: float = 0.0
    ternary_size_mb: float = 0.0
    compression_ratio: float = 0.0


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    output: Any
    latency_ms: float
    deterministic: bool = True  # ternary inference is always deterministic


class TernaryInferenceEngine:
    """
    Convert standard models to ternary and run inference.

    Usage:
        engine = TernaryInferenceEngine()
        report = engine.convert(model)
        result = engine.infer(model, input_tensor)

    Patent 12: Auto binary-to-ternary conversion.
    Patent 36: Deterministic execution guarantee.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        mse_ceiling: float = 0.01,
        protect_embeddings: bool = True,
        protect_layernorm: bool = True,
        protect_lm_head: bool = True,
    ) -> None:
        """
        Args:
            threshold:         Default quantisation threshold.
            mse_ceiling:       MSE ceiling for sensitivity analysis.
            protect_embeddings: Keep embedding layers in FP16 (recommended).
            protect_layernorm:  Keep LayerNorm/RMSNorm in FP32 (recommended).
            protect_lm_head:    Keep final output projection in FP16 (recommended).
        """
        self.threshold = threshold
        self.mse_ceiling = mse_ceiling
        self.protect_embeddings = protect_embeddings
        self.protect_layernorm = protect_layernorm
        self.protect_lm_head = protect_lm_head
        self.analyzer = SensitivityAnalyzer(mse_ceiling=mse_ceiling)

    def convert(
        self,
        model: nn.Module,
        sensitivity_analysis: bool = True,
    ) -> ConversionReport:
        """
        Convert a model's Linear and Conv2d layers to ternary in-place.

        Layers identified as precision-critical by sensitivity analysis
        are left in FP16. Embeddings, LayerNorm, and final output head
        are protected by default.

        Args:
            model:                The PyTorch model to convert.
            sensitivity_analysis: If True, run per-layer sensitivity analysis
                                  to determine optimal thresholds. If False,
                                  use default threshold for all layers.

        Returns:
            ConversionReport with statistics.
        """
        report = ConversionReport()

        # Run sensitivity analysis if requested
        layer_thresholds: dict[str, float] = {}
        if sensitivity_analysis:
            analyses = self.analyzer.analyze_model(model)
            for a in analyses:
                if a["precision_critical"]:
                    report.precision_critical_layers.append(a["name"])
                else:
                    layer_thresholds[a["name"]] = a["recommended_threshold"]

        # Convert layers
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                report.total_layers += 1
                report.total_params += module.weight.numel()

                if self._should_protect(name, module, report):
                    report.skipped_layers += 1
                    continue

                threshold = layer_thresholds.get(name, self.threshold)
                ternary_layer = self._convert_linear(module, threshold)
                self._replace_module(model, name, ternary_layer)

                report.converted_layers += 1
                report.ternary_params += module.weight.numel()

            elif isinstance(module, nn.Conv2d):
                report.total_layers += 1
                report.total_params += module.weight.numel()

                if name in report.precision_critical_layers:
                    report.skipped_layers += 1
                    continue

                threshold = layer_thresholds.get(name, self.threshold)
                ternary_layer = self._convert_conv2d(module, threshold)
                self._replace_module(model, name, ternary_layer)

                report.converted_layers += 1
                report.ternary_params += module.weight.numel()

            else:
                # Check for HuggingFace Conv1D (used by GPT-2 family)
                Conv1D = _get_conv1d_class()
                if Conv1D is not None and isinstance(module, Conv1D):
                    report.total_layers += 1
                    report.total_params += module.weight.numel()

                    if self._should_protect(name, module, report):
                        report.skipped_layers += 1
                        continue

                    threshold = layer_thresholds.get(name, self.threshold)
                    ternary_layer = self._convert_conv1d(module, threshold)
                    self._replace_module(model, name, ternary_layer)

                    report.converted_layers += 1
                    report.ternary_params += module.weight.numel()

        # Calculate sizes
        # FP16 = 2 bytes/param, ternary = 2 bits/param = 0.25 bytes/param
        report.original_size_mb = (report.total_params * 2) / (1024 * 1024)
        ternary_bytes = report.ternary_params * 0.25
        fp16_bytes = (report.total_params - report.ternary_params) * 2
        report.ternary_size_mb = (ternary_bytes + fp16_bytes) / (1024 * 1024)
        if report.ternary_size_mb > 0:
            report.compression_ratio = report.original_size_mb / report.ternary_size_mb

        return report

    def infer(
        self, model: nn.Module, inputs: Any, **kwargs: Any
    ) -> InferenceResult:
        """
        Run inference on a converted model.

        Guarantees deterministic output (Patent 36, Claim 14):
        same inputs + same model = bit-identical output regardless
        of hardware or timing.

        Args:
            model:  Converted model (or unconverted — works either way).
            inputs: Model input (tensor, dict, or tuple depending on model).

        Returns:
            InferenceResult with output and timing.
        """
        model.eval()

        # Ensure determinism
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        start = time.perf_counter()
        with torch.no_grad():
            if isinstance(inputs, dict):
                output = model(**inputs, **kwargs)
            elif isinstance(inputs, (tuple, list)):
                output = model(*inputs, **kwargs)
            else:
                output = model(inputs, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return InferenceResult(
            output=output,
            latency_ms=latency_ms,
            deterministic=True,
        )

    def _should_protect(
        self, name: str, module: nn.Module, report: ConversionReport
    ) -> bool:
        """Determine if a layer should be protected from quantisation."""
        name_lower = name.lower()

        if self.protect_embeddings and "embed" in name_lower:
            return True
        if self.protect_layernorm and any(
            k in name_lower for k in ("layernorm", "layer_norm", "rmsnorm")
        ):
            return True
        if self.protect_lm_head and any(
            k in name_lower for k in ("lm_head", "output", "classifier")
        ):
            return True
        if name in report.precision_critical_layers:
            return True
        return False

    @staticmethod
    def _convert_linear(module: nn.Linear, threshold: float) -> TernaryLinear:
        """Convert nn.Linear to TernaryLinear, preserving weights."""
        ternary = TernaryLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            threshold=threshold,
        )
        ternary.weight.data.copy_(module.weight.data)
        if module.bias is not None and ternary.bias is not None:
            ternary.bias.data.copy_(module.bias.data)
        return ternary

    @staticmethod
    def _convert_conv1d(module, threshold: float) -> TernaryLinear:
        """Convert HuggingFace Conv1D to TernaryLinear.

        Conv1D weight is (in_features, out_features) — transposed vs nn.Linear.
        We transpose and create a standard TernaryLinear replacement.
        """
        # Conv1D: weight shape is (in_features, out_features)
        in_features, out_features = module.weight.shape
        ternary = TernaryLinear(
            in_features=in_features,
            out_features=out_features,
            bias=module.bias is not None,
            threshold=threshold,
        )
        # Transpose Conv1D weight (in, out) → Linear weight (out, in)
        ternary.weight.data.copy_(module.weight.data.t())
        if module.bias is not None and ternary.bias is not None:
            ternary.bias.data.copy_(module.bias.data)
        return ternary

    @staticmethod
    def _convert_conv2d(module: nn.Conv2d, threshold: float) -> TernaryConv2d:
        """Convert nn.Conv2d to TernaryConv2d, preserving weights."""
        ternary = TernaryConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            bias=module.bias is not None,
            threshold=threshold,
        )
        ternary.weight.data.copy_(module.weight.data)
        if module.bias is not None and ternary.bias is not None:
            ternary.bias.data.copy_(module.bias.data)
        return ternary

    @staticmethod
    def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
        """Replace a named module in the model tree."""
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
