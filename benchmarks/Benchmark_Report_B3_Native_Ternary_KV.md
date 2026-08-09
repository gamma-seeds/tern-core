# BUILD 6 Track B3 — Native-Ternary KV Compression Report

> **Hardware:** Apple M4 Pro · 64 GB · Darwin 25.5.0
> **Date:** 2026-08-09T00:42:59.524265+00:00
> **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
> **Method:** Native ternary {-1, 0, +1} quantisation of KV cache entries

## Quality gates

- **93% floor**: no individual prompt below 93% argmax match
- **99% bell mean**: average across prompts ≥ 99% argmax match

## Threshold sweep

| Threshold | Mean match | Min match | Compression | Zero ratio | Verdict |
|:---------:|:----------:|:---------:|:-----------:|:----------:|:-------:|
| 0.3 | 4.2% | 1.0% | 14.2× | 22% | FLOOR_FAIL |
| 0.4 | 3.2% | 1.0% | 14.2× | 28% | FLOOR_FAIL |
| 0.5 | 4.4% | 1.0% | 14.2× | 35% | FLOOR_FAIL |
| 0.6 | 6.0% | 1.0% | 14.2× | 41% | FLOOR_FAIL |
| 0.7 | 5.0% | 2.0% | 14.2× | 47% | FLOOR_FAIL |
| 0.8 | 3.2% | 2.0% | 14.2× | 52% | FLOOR_FAIL |
| 0.9 | 3.4% | 1.0% | 14.2× | 57% | FLOOR_FAIL |

## No threshold passes both gates

Quality degrades below the 93% floor or 99% mean at all thresholds tested.
