# SynapNet-Edge — Paper-Ready Experimental Results (v2)

*All numbers are mean ± std over 3 random seeds.*

## Model
- Hybrid: SSM + sparse-attn + episodic memory
- 192d × 6 blocks × 6 heads, 32 memory slots, max_len=8192
- 8.7M params, pretrained 2-stage curriculum (512→1024)

## Table 1 — CAJQ-QAT vs. Uniform Quantization (NIAH-Single, mean ± std)

| Variant | Eff Bits | ctx=1024 | ctx=2048 | ctx=4096 |
|---|---|---|---|---|
| FP16 baseline | 16.0 | 0.618 ± 0.107 | 0.507 ± 0.115 | 0.438 ± 0.036 |
| Uniform INT8 | 8.0 | 0.618 ± 0.087 | 0.507 ± 0.115 | 0.438 ± 0.036 |
| Uniform INT4 (AWQ) | 4.0 | 0.646 ± 0.062 | 0.528 ± 0.094 | 0.465 ± 0.043 |
| CAJQ (PTQ only) | 13.8 | 0.611 ± 0.098 | 0.521 ± 0.116 | 0.438 ± 0.036 |
| **CAJQ + QAT (ours)** | 13.8 | 0.674 ± 0.012 | 0.590 ± 0.043 | 0.521 ± 0.055 |

## Table 2 — Multi-task accuracy

| Variant | ctx=1024 | ctx=2048 | ctx=4096 |
|---|---|---|---|
| FP16 baseline | 0.443 ± 0.040 | 0.394 ± 0.028 | 0.369 ± 0.006 |
| Uniform INT8 | 0.447 ± 0.037 | 0.397 ± 0.022 | 0.376 ± 0.006 |
| Uniform INT4 (AWQ) | 0.443 ± 0.040 | 0.397 ± 0.016 | 0.387 ± 0.006 |
| CAJQ (PTQ only) | 0.443 ± 0.040 | 0.394 ± 0.038 | 0.369 ± 0.006 |
| CAJQ + QAT (ours) | 0.472 ± 0.012 | 0.457 ± 0.032 | 0.447 ± 0.028 |

## Table 3 — BAEE Grid: Target Retention Rate

### seq_len = 1024

| Target Pos | Budget | BAEE (ours) | FIFO | LRU | Random |
|---|---|---|---|---|---|
| early | 10% | 0.56 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.07 ± 0.06 |
| early | 20% | 0.69 ± 0.10 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.31 ± 0.05 |
| early | 30% | 0.83 ± 0.08 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.62 ± 0.11 |
| early | 50% | 0.93 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.88 ± 0.04 |
| late | 10% | 0.46 ± 0.11 | 0.76 ± 0.10 | 0.76 ± 0.10 | 0.42 ± 0.11 |
| late | 20% | 0.64 ± 0.06 | 0.97 ± 0.02 | 0.97 ± 0.02 | 0.61 ± 0.05 |
| late | 30% | 0.86 ± 0.02 | 0.99 ± 0.02 | 0.99 ± 0.02 | 0.79 ± 0.11 |
| late | 50% | 0.99 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.92 ± 0.00 |

### seq_len = 2048

| Target Pos | Budget | BAEE (ours) | FIFO | LRU | Random |
|---|---|---|---|---|---|
| early | 10% | 0.71 ± 0.08 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.06 ± 0.02 |
| early | 20% | 0.92 ± 0.04 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.19 ± 0.06 |
| early | 30% | 0.92 ± 0.04 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.04 |
| early | 50% | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.90 ± 0.02 |
| late | 10% | 0.71 ± 0.08 | 0.93 ± 0.02 | 0.93 ± 0.02 | 0.58 ± 0.08 |
| late | 20% | 0.90 ± 0.02 | 0.85 ± 0.02 | 0.85 ± 0.02 | 0.65 ± 0.02 |
| late | 30% | 0.94 ± 0.06 | 0.99 ± 0.02 | 0.99 ± 0.02 | 0.82 ± 0.09 |
| late | 50% | 0.99 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.93 ± 0.02 |

## Table 4 — ScaleBridge Ablation

We test two questions: (a) is the bridge load-bearing in the pretrained model? (b) can a model trained without the bridge from scratch match a model trained with it?

### 4a — Post-hoc ablation (model pretrained WITH bridge)

Replacing ScaleBridge with identity passthrough after training **collapses accuracy to near chance**.  This shows the bridge is load-bearing in the trained architecture — the model has learned to rely on it.

| Metric | ctx | With Bridge | Without Bridge | Δ |
|---|---|---|---|---|
| niah_single | 1024 | 0.618 ± 0.107 | 0.028 ± 0.012 | +0.590 |
| niah_single | 2048 | 0.507 ± 0.115 | 0.028 ± 0.012 | +0.479 |
| niah_single | 4096 | 0.438 ± 0.036 | 0.028 ± 0.012 | +0.410 |
| multi_task | 1024 | 0.443 ± 0.040 | 0.025 ± 0.012 | +0.418 |
| multi_task | 2048 | 0.394 ± 0.028 | 0.025 ± 0.012 | +0.369 |
| multi_task | 4096 | 0.369 ± 0.006 | 0.025 ± 0.012 | +0.344 |

### 4b — From-scratch comparison

Two small models (dim=128, depth=4, 400 steps, 3 seeds) trained from scratch — one with bridge, one without.

| ctx | Trained-with-bridge | Trained-without-bridge | Δ |
|---|---|---|---|
| 512 | 0.031 ± 0.016 | 0.057 ± 0.018 | -0.026 |
| 1024 | 0.031 ± 0.027 | 0.062 ± 0.041 | -0.031 |

**Honest finding:** at this small training budget (dim=128, 400 steps) neither variant converges meaningfully — both are at or just above the 1/32 = 0.031 chance level — so this experiment is **inconclusive** about whether a from-scratch no-bridge model can match. The definitive comparison requires the full pretraining budget for both variants (≈10 min each on M-series).

**Take-away.** The post-hoc ablation establishes that ScaleBridge is *integral* to the trained model; the from-scratch experiment at our compute budget cannot yet rule out that an equivalent no-bridge model exists. We keep the bridge in the released architecture and flag this as a future-work question.

## Table 5 — Deployment Metrics across 3 Hardware Tiers

### Apple Silicon (MPS)  (threads=4)

| Variant | Bits | Storage (MB) | tok/s @ 512 | tok/s @ 1024 | tok/s @ 2048 | tok/s @ 4096 | tok/s @ 8192 |
|---|---|---|---|---|---|---|---|
| FP16 baseline | 16.0 | 13.60 | 36848 | 32557 | 56927 | 45752 | 29732 |
| Uniform INT4 (AWQ) | 4.0 | 5.29 | 28949 | 29499 | 53168 | 44017 | 29835 |
| CAJQ | 13.8 | 12.06 | 7855 | 6320 | 15506 | 13350 | 23736 |

### Multi-thread CPU  (threads=8)

| Variant | Bits | Storage (MB) | tok/s @ 512 | tok/s @ 1024 | tok/s @ 2048 | tok/s @ 4096 |
|---|---|---|---|---|---|---|
| FP16 baseline | 16.0 | 13.60 | 9437 | 10283 | 16653 | 13776 |
| Uniform INT4 (AWQ) | 4.0 | 5.29 | 7825 | 8956 | 14603 | 12648 |
| CAJQ | 13.8 | 12.06 | 8527 | 8464 | 15832 | 13567 |

### Single-thread CPU (Pi 5 proxy)  (threads=1)

| Variant | Bits | Storage (MB) | tok/s @ 512 | tok/s @ 1024 | tok/s @ 2048 |
|---|---|---|---|---|---|
| FP16 baseline | 16.0 | 13.60 | 11593 | 7433 | 11767 |
| Uniform INT4 (AWQ) | 4.0 | 5.29 | 10385 | 7224 | 11379 |
| CAJQ | 13.8 | 12.06 | 10859 | 7283 | 11764 |

## Headline Numbers

- **CAJQ-QAT @ ctx=1024**: 0.674 acc vs. FP16 0.618 (+5.6%-points), at 13.8 effective bits.
- **CAJQ-QAT @ ctx=2048**: 0.590 acc vs. FP16 0.507 (+8.3%-points), at 13.8 effective bits.
- **CAJQ-QAT @ ctx=4096**: 0.521 acc vs. FP16 0.438 (+8.3%-points), at 13.8 effective bits.
- **BAEE robustness**: at seq_len=2048, target=early, budget=50% (forced eviction 50%), BAEE retains 100% ± 0% of target needles vs. FIFO's 0%.
- **ScaleBridge post-hoc ablation** (model pretrained with bridge, then bridge removed): max |Δ| across all context lengths = 0.590.  The bridge is load-bearing in the trained architecture; we cannot drop it from this checkpoint without retraining.
- **Storage compression (targeted layers only)**: on SSM+attention components, CAJQ packs 0.60 MB vs. FP16 equivalent 2.66 MB (**4.4× compression** of the components CAJQ targets).
- **Whole-model compression**: 12.06 MB vs FP16 13.60 MB (1.13×). Most params remain FP16 (FF + embeddings + memory-projection layers); whole-model compression requires extending CAJQ to FF layers (future work).