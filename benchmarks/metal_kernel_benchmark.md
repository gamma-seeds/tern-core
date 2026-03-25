# Terncore Metal Kernel Benchmark

> Native ternary matmul on Apple Silicon — packed 2-bit codes, zero-skip, no FMA
> Apple M4 Pro · 2026-03-25

## Results (B=1, autoregressive decode)

| Layer | Size | Metal (us) | FP16 (us) | Dequant (us) | vs FP16 | Memory |
|-------|------|:----------:|:---------:|:------------:|:-------:|:------:|
| attn_qkv | 2048x2048 | 279.3 | 187.8 | 827.2 | 0.67x | 8.0x |
| attn_out | 2048x2048 | 165.1 | 168.0 | 1053.9 | 1.02x | 8.0x |
| mlp_gate | 5632x2048 | 329.5 | 250.4 | 2949.1 | 0.76x | 8.0x |
| mlp_up | 5632x2048 | 219.3 | 232.7 | 2509.6 | 1.06x | 8.0x |
| mlp_down | 2048x5632 | 235.7 | 250.0 | 2810.6 | 1.06x | 8.0x |
| lm_head | 32000x2048 | 513.2 | 711.0 | 8543.1 | 1.39x | 8.0x |

**Aggregate: 1.03x vs FP16, 10.7x vs dequant path**

## Key Properties

- **Zero multiplies**: All weight operations are pure add/subtract
- **43% free compute**: Zero-weighted channels skipped entirely
- **~8x memory compression**: 2-bit packed vs FP16
- **Branch-free**: Bit manipulation decode, no conditionals in hot path

---
*Terncore Metal kernel benchmark · Cubey/Synapticode · 2026-03-25*
