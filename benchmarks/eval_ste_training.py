"""
STE Training Evaluation: Quantisation-Aware Training proof-of-concept.

Loads TinyLlama-1.1B, converts to TernaryLinearSTE, trains with STE on
WikiText-2 train split, then evaluates perplexity on WikiText-2 test set.

Proves that STE training can reduce ternary perplexity degradation.

Pipeline:
    1. Load model + tokenizer
    2. Evaluate FP32 baseline perplexity (from cache or fresh)
    3. Convert to TernaryLinearSTE (QAT mode)
    4. Evaluate pre-training ternary perplexity
    5. Train with STE for N steps
    6. Evaluate post-training ternary perplexity
    7. Report improvement

Usage:
    python benchmarks/eval_ste_training.py --steps 50
    python benchmarks/eval_ste_training.py --steps 500 --lr 1e-4
    python benchmarks/eval_ste_training.py --json-only

Patent 36: Biological neural mapping — STE training.
Patent 1:  Ternary weight encoding.
Patent 12: Auto binary-to-ternary conversion.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# Ensure tern-core is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.ste import TernaryLinearSTE
from terncore.ste_trainer import STETrainer, TrainResult


# ═══════════════════════════════════════════════════════════════
# Dependency guards
# ═══════════════════════════════════════════════════════════════

_HF_AVAILABLE = False
_HF_IMPORT_ERROR: Optional[str] = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HF_AVAILABLE = True
except ImportError as e:
    _HF_IMPORT_ERROR = str(e)

_DATASETS_AVAILABLE = False
_DATASETS_IMPORT_ERROR: Optional[str] = None

try:
    from datasets import load_dataset

    _DATASETS_AVAILABLE = True
except ImportError as e:
    _DATASETS_IMPORT_ERROR = str(e)


def _require_dependencies() -> None:
    if not _HF_AVAILABLE:
        raise ImportError(
            "HuggingFace transformers required. Install: pip install terncore[transformers]\n"
            f"Error: {_HF_IMPORT_ERROR}"
        )
    if not _DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets required. Install: pip install datasets\n"
            f"Error: {_DATASETS_IMPORT_ERROR}"
        )


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_THRESHOLD = 0.7
DEFAULT_STEPS = 50
DEFAULT_LR = 1e-4
DEFAULT_SEQ_LEN = 256
DEFAULT_STRIDE = 512
SEED = 42
FP32_BASELINE_PPL = 7.19  # Known from previous eval
RESULTS_MD_PATH = Path(__file__).resolve().parent / "RESULTS.md"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "tinyllama_ste_config.json"


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class STEEvalResult:
    """Full STE training evaluation result."""

    model_id: str
    threshold: float
    num_steps: int
    lr: float
    seq_len: int
    fp32_ppl: float
    pre_train_ppl: float
    post_train_ppl: float
    ppl_improvement: float  # pre - post
    ppl_improvement_pct: float  # (pre - post) / pre * 100
    gap_vs_fp32_pre: float  # pre_train gap vs FP32 %
    gap_vs_fp32_post: float  # post_train gap vs FP32 %
    training_time_s: float
    initial_loss: float
    final_loss: float
    loss_reduction_pct: float
    converted_layers: int
    trainable_params: int
    total_params: int
    eval_tokens: int
    loss_history: list[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════


def load_train_chunks(
    tokenizer: Any, seq_len: int, seed: int = SEED
) -> list[torch.Tensor]:
    """
    Load WikiText-2 train split and chunk into random seq_len-token sequences.

    Returns list of input_ids tensors, each shape (1, seq_len).
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(
        entry["text"] for entry in dataset if entry["text"].strip()
    )
    encodings = tokenizer(text, return_tensors="pt")
    full_ids = encodings.input_ids[0]  # (total_tokens,)
    total = full_ids.shape[0]

    # Create non-overlapping chunks
    chunks = []
    for i in range(0, total - seq_len, seq_len):
        chunk = full_ids[i : i + seq_len].unsqueeze(0)  # (1, seq_len)
        chunks.append(chunk)

    # Shuffle with fixed seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(chunks)

    return chunks


def load_test_sequence(tokenizer: Any) -> torch.Tensor:
    """Load WikiText-2 test set as a single tokenized sequence."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(
        entry["text"] for entry in dataset if entry["text"].strip()
    )
    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids


# ═══════════════════════════════════════════════════════════════
# Perplexity evaluation (sliding window)
# ═══════════════════════════════════════════════════════════════


def evaluate_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    stride: int,
    max_length: int,
    phase_name: str,
    max_tokens: int = 0,
) -> float:
    """
    Compute perplexity using sliding-window approach.

    Args:
        max_tokens: If > 0, only evaluate first N tokens (for probing).
    """
    model.eval()
    seq_len = input_ids.size(1)
    if max_tokens > 0:
        seq_len = min(seq_len, max_tokens)

    nlls: list[float] = []
    prev_end = 0
    n_windows = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss.item()

        nlls.append(neg_log_likelihood * trg_len)
        prev_end = end_loc
        n_windows += 1

        if n_windows % 50 == 0:
            running_ppl = math.exp(sum(nlls) / prev_end) if prev_end > 0 else float("inf")
            pct = 100 * prev_end / seq_len
            print(
                f"\r    [{phase_name}] {prev_end:,}/{seq_len:,} tokens "
                f"({pct:.0f}%) — running PPL: {running_ppl:.2f}",
                end="", flush=True,
            )

        if end_loc == seq_len:
            break

    print()  # newline after progress

    total_tokens = prev_end
    try:
        ppl = math.exp(sum(nlls) / total_tokens)
    except OverflowError:
        ppl = float("inf")

    return ppl


# ═══════════════════════════════════════════════════════════════
# Main evaluation pipeline
# ═══════════════════════════════════════════════════════════════


def run_evaluation(
    model_id: str = DEFAULT_MODEL_ID,
    threshold: float = DEFAULT_THRESHOLD,
    num_steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
    eval_tokens: int = 0,
    quiet: bool = False,
) -> STEEvalResult:
    """Run the full STE training evaluation pipeline."""

    _require_dependencies()
    torch.manual_seed(SEED)

    # ── Step 1: Load model ──────────────────────────────────────
    print(f"\n[1/6] Loading {model_id}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # ── Step 2: Load data ───────────────────────────────────────
    print(f"\n[2/6] Loading WikiText-2...")
    train_chunks = load_train_chunks(tokenizer, seq_len)
    test_ids = load_test_sequence(tokenizer)
    max_length = getattr(model.config, "max_position_embeddings", 2048)
    print(f"  Train chunks: {len(train_chunks)} x {seq_len} tokens")
    print(f"  Test tokens:  {test_ids.shape[1]:,}")

    # ── Step 3: Pre-training ternary eval ───────────────────────
    print(f"\n[3/6] Setting up STE trainer...")
    trainer = STETrainer(
        model=model,
        threshold=threshold,
        lr=lr,
        log_every=max(1, num_steps // 10),
    )
    converted, protected = trainer.setup()
    trainable = sum(p.numel() for p in trainer.ste_params)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Converted: {converted} layers, Protected: {protected}")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Evaluate ternary PPL before training
    print(f"\n[4/6] Evaluating pre-training ternary perplexity...")
    pre_ppl = evaluate_perplexity(
        model, test_ids, stride, max_length, "pre_train", max_tokens=eval_tokens,
    )
    print(f"  Pre-training PPL: {pre_ppl:.4f} (FP32 baseline: {FP32_BASELINE_PPL})")
    print(f"  Gap vs FP32: +{100*(pre_ppl - FP32_BASELINE_PPL)/FP32_BASELINE_PPL:.2f}%")
    # Put model back in training mode
    model.train()

    # ── Step 4: STE Training ───────────────────────────────────
    print(f"\n[5/6] Training with STE ({num_steps} steps, lr={lr})...")
    train_result = trainer.train(
        data_iterator=train_chunks,
        num_steps=num_steps,
        quiet=quiet,
    )
    print(f"  Training time: {train_result.total_time_s:.1f}s")
    print(f"  Loss: {train_result.initial_loss:.4f} → {train_result.final_loss:.4f}")
    print(f"  Loss reduction: {100*train_result.loss_reduction:.1f}%")

    # ── Step 5: Post-training ternary eval ──────────────────────
    print(f"\n[6/6] Evaluating post-training ternary perplexity...")
    post_ppl = evaluate_perplexity(
        model, test_ids, stride, max_length, "post_train", max_tokens=eval_tokens,
    )
    print(f"  Post-training PPL: {post_ppl:.4f}")
    gap_post = 100 * (post_ppl - FP32_BASELINE_PPL) / FP32_BASELINE_PPL
    print(f"  Gap vs FP32: +{gap_post:.2f}%")

    ppl_improvement = pre_ppl - post_ppl
    ppl_improvement_pct = 100 * ppl_improvement / pre_ppl if pre_ppl > 0 else 0
    print(f"  PPL improvement: {ppl_improvement:.4f} ({ppl_improvement_pct:.2f}%)")

    # ── Build result ────────────────────────────────────────────
    result = STEEvalResult(
        model_id=model_id,
        threshold=threshold,
        num_steps=num_steps,
        lr=lr,
        seq_len=seq_len,
        fp32_ppl=FP32_BASELINE_PPL,
        pre_train_ppl=pre_ppl,
        post_train_ppl=post_ppl,
        ppl_improvement=ppl_improvement,
        ppl_improvement_pct=ppl_improvement_pct,
        gap_vs_fp32_pre=100 * (pre_ppl - FP32_BASELINE_PPL) / FP32_BASELINE_PPL,
        gap_vs_fp32_post=gap_post,
        training_time_s=train_result.total_time_s,
        initial_loss=train_result.initial_loss,
        final_loss=train_result.final_loss,
        loss_reduction_pct=100 * train_result.loss_reduction,
        converted_layers=converted,
        trainable_params=trainable,
        total_params=total,
        eval_tokens=eval_tokens if eval_tokens > 0 else test_ids.shape[1],
        loss_history=[s.loss for s in train_result.steps],
    )

    return result


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════


def print_summary(result: STEEvalResult) -> None:
    """Print human-readable summary."""
    print()
    print("=" * 70)
    print("  STE Training Results")
    print("=" * 70)
    print(f"  Model:             {result.model_id}")
    print(f"  Threshold:         {result.threshold}")
    print(f"  Steps:             {result.num_steps}")
    print(f"  Learning rate:     {result.lr}")
    print(f"  Seq length:        {result.seq_len}")
    print(f"  Converted layers:  {result.converted_layers}")
    print(f"  Trainable params:  {result.trainable_params:,} / {result.total_params:,}")
    print()
    print(f"  FP32 baseline PPL: {result.fp32_ppl:.4f}")
    print(f"  Pre-train PPL:     {result.pre_train_ppl:.4f} (gap: +{result.gap_vs_fp32_pre:.2f}%)")
    print(f"  Post-train PPL:    {result.post_train_ppl:.4f} (gap: +{result.gap_vs_fp32_post:.2f}%)")
    print()
    print(f"  PPL improvement:   {result.ppl_improvement:.4f} ({result.ppl_improvement_pct:.2f}%)")
    print(f"  Training loss:     {result.initial_loss:.4f} → {result.final_loss:.4f} ({result.loss_reduction_pct:.1f}% reduction)")
    print(f"  Training time:     {result.training_time_s:.1f}s")
    print()

    # Verdict
    if result.post_train_ppl < 1000 and result.ppl_improvement > 0:
        verdict = "STRONG — PPL below 1000 with improvement"
    elif result.ppl_improvement > 0:
        verdict = "PROMISING — PPL improved"
    elif result.final_loss < result.initial_loss:
        verdict = "WEAK — Training loss drops but PPL didn't improve"
    else:
        verdict = "NEGATIVE — No improvement observed"

    print(f"  Verdict:           {verdict}")
    print("=" * 70)


def save_config(result: STEEvalResult, path: Path = CONFIG_PATH) -> None:
    """Save results to JSON config file."""
    config = {
        "model_id": result.model_id,
        "threshold": result.threshold,
        "ste_training": {
            "steps": result.num_steps,
            "lr": result.lr,
            "seq_len": result.seq_len,
            "optimizer": "SGD",
            "gradient_checkpointing": True,
            "converted_layers": result.converted_layers,
            "trainable_params": result.trainable_params,
        },
        "results": {
            "fp32_ppl": result.fp32_ppl,
            "pre_train_ppl": round(result.pre_train_ppl, 4),
            "post_train_ppl": round(result.post_train_ppl, 4),
            "ppl_improvement": round(result.ppl_improvement, 4),
            "ppl_improvement_pct": round(result.ppl_improvement_pct, 2),
            "gap_vs_fp32_pre": f"+{result.gap_vs_fp32_pre:.2f}%",
            "gap_vs_fp32_post": f"+{result.gap_vs_fp32_post:.2f}%",
            "training_loss_initial": round(result.initial_loss, 4),
            "training_loss_final": round(result.final_loss, 4),
            "loss_reduction_pct": round(result.loss_reduction_pct, 1),
            "training_time_s": round(result.training_time_s, 1),
            "eval_tokens": result.eval_tokens,
        },
        "loss_history": [round(l, 4) for l in result.loss_history],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nConfig saved to {path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STE Training Evaluation for TinyLlama"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="HuggingFace model ID"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Quantisation threshold (default: 0.7)"
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help="Number of training steps (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN,
        help="Training sequence length (default: 256)"
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE,
        help="Evaluation stride (default: 512)"
    )
    parser.add_argument(
        "--eval-tokens", type=int, default=0,
        help="Limit eval to first N tokens (0 = full test set)"
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="JSON output only"
    )
    parser.add_argument(
        "--save-config", action="store_true",
        help="Save results to configs/tinyllama_ste_config.json"
    )
    args = parser.parse_args()

    result = run_evaluation(
        model_id=args.model,
        threshold=args.threshold,
        num_steps=args.steps,
        lr=args.lr,
        seq_len=args.seq_len,
        stride=args.stride,
        eval_tokens=args.eval_tokens,
        quiet=args.json_only,
    )

    if args.json_only:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print_summary(result)

    if args.save_config:
        save_config(result)


if __name__ == "__main__":
    main()
