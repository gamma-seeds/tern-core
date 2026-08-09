# BUILD 6 Track B2 — FP16-KV Baseline Report

> **Hardware:** Apple M4 Pro · 64 GB · Darwin 25.5.0 · PyTorch 2.7.0
> **Date:** 2026-08-08T06:52:34.141247+00:00
> **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Icon 1 — the measured 1× rung

| Metric | Value | Tag |
|--------|:-----:|:---:|
| KV cache at 56 tokens | 2.41 MB | Measured |
| Bytes per token | 45056 B | Measured |
| Bytes/token/layer | 2048 B | Measured |
| KV dtype | torch.float32 | Measured |
| Decode throughput | 27.8 tok/s | Measured |
| Baseline PPL | 5.54 | Measured |

## KV cache scaling

| Sequence length | KV cache | Bytes/token |
|:---:|:---:|:---:|
| 64 | 2.75 MB | 45056 B |
| 128 | 5.50 MB | 45056 B |
| 256 | 11.00 MB | 45056 B |
| 512 | 22.00 MB | 45056 B |

Linear scaling confirmed — KV grows at 45056 bytes per token.

## Compression ladder (Icon 1)

| Rung | Ratio | Source | Tag |
|------|:-----:|--------|:---:|
| FP16-KV baseline | 1× | This report | Measured |
| TurboQuant b_mse=3 (bytes) | 3.88× | Track A replication | Measured |
| TurboQuant b_mse=3 (bits) | 5.33× | Codebook math (16/3) | Derived |
| Pure ternary KV (ceiling) | ~10× | Projected | Projected |

## Notes

- KV cache stores in torch.float32 (model loaded as float32). Actual bytes per token per layer: 2048 B.
- Theoretical FP16 per token per layer: 1024 B (2 × 4 KV heads × 64 head_dim × 2 bytes).
- This establishes the '1×' denominator for all KV compression ratios.
