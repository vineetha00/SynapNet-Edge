# SynapNet-Edge: Scaled Experiment Summary

## Model
- Architecture: SynapNet-Edge hybrid (SSM + sparse-attn + episodic memory)
- Parameters: ~8.7M  (192d × 6 blocks, 6 heads, 32 memory slots)
- Vocab: 4096, classes: 64, max_len: 8192

## Table 1 — CAJQ vs. Uniform Quantization (NIAH-Single)

| Variant | Eff. Bits | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 |
|---|---|---|---|---|---|
| fp16 | 16.0 | 0.656 | 0.672 | 0.578 | 0.422 |
| int8_uniform | 8.0 | 0.656 | 0.672 | 0.562 | 0.438 |
| int4_uniform | 4.0 | 0.672 | 0.688 | 0.594 | 0.438 |
| cajq | 13.8 | 0.656 | 0.672 | 0.516 | 0.422 |

## Table 2 — BAEE Memory-Pressure Eviction Comparison

Setup: 16 needles, target repeated 3× in early portion of context, budget = 10% of total writes (90% forced eviction)

### seq_len = 1024 (budget=6/64, forced eviction = 91%)

| Policy | Target Retention | Task Acc |
|---|---|---|
| baee_salience | 0.422 | 0.016 |
| fifo | 0.000 | 0.016 |
| lru | 0.000 | 0.016 |
| random | 0.078 | 0.000 |

### seq_len = 2048 (budget=12/128, forced eviction = 91%)

| Policy | Target Retention | Task Acc |
|---|---|---|
| baee_salience | 0.594 | 0.062 |
| fifo | 0.000 | 0.031 |
| lru | 0.000 | 0.031 |
| random | 0.031 | 0.047 |

## Table 3 — Consumer Hardware Throughput (tokens/sec)

### Apple Silicon (MPS)

| Variant | Bits | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|---|---|---|
| fp16 | 16.0 | 36731 | 32585 | 56255 | 45746 | 27786 |
| int4_uniform | 4.0 | 27637 | 28747 | 52426 | 42142 | 23977 |
| cajq | 13.8 | 4436 | 4630 | 8789 | 15168 | 27196 |

### ARM CPU

| Variant | Bits | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 |
|---|---|---|---|---|---|
| fp16 | 16.0 | 12494 | 11541 | 18562 | 14600 |
| int4_uniform | 4.0 | 10670 | 10820 | 17765 | 14279 |
| cajq | 13.8 | 10919 | 10942 | 17652 | 13068 |

## Headline Numbers

- At 1024 tokens with 91% forced eviction, **BAEE retains the target needle 42% of the time** vs. 0% for FIFO/LRU (BAEE is ≥5× better than random).
- At 2048 tokens with 91% forced eviction, **BAEE retains the target needle 59% of the time** vs. 0% for FIFO/LRU (BAEE is ≥19× better than random).
- cajq: 0.656 → 0.422 NIAH-single accuracy from ctx=512 → 4096 (eff_bits=13.8)
- int4_uniform: 0.672 → 0.438 NIAH-single accuracy from ctx=512 → 4096 (eff_bits=4.0)
- fp16: 0.656 → 0.422 NIAH-single accuracy from ctx=512 → 4096 (eff_bits=16.0)
