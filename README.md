# SynapNet-Edge

**Efficient long-context AI inference on consumer hardware — quantised hybrid SSM + sparse attention + episodic-memory architecture, reproducible end-to-end in ~25 minutes on a MacBook.**

📄 **Paper:** arXiv preprint — link coming soon
🤗 **Checkpoints:** https://huggingface.co/vineethavc/synapnet-edge
🧪 **Companion repo (base architecture):** `SynapNet_Exp`

SynapNet-Edge is an original research framework. The base SynapNet architecture is developed in the companion repository [SynapNet_Exp].

---

## Three Core Contributions

### 1. Component-Aware Joint Quantization (CAJQ)

Different quantization strategies per architectural component:

| Component | Method | Bits | Rationale |
|-----------|--------|------|-----------|
| SSM layers | ParetoQ-style QAT | **2-bit** | Near-zero-mean depthwise weights; learned per-channel step size preserves accuracy |
| Sparse attention | SmoothQuant + AWQ | **INT4** | Per-channel smoothing migrates activation outliers to weights; group-wise INT4 |
| Episodic memory | Per-entry symmetric | **INT8** | Per-slot scale absorbs entry diversity |
| Interface layer | FP16 ScaleBridge | **FP16** | Absorbs scale mismatches between mixed-precision pathways |

**Compression** (measured at the 8.7M-param reference model): **4.4× compression on targeted SSM+attention parameters (0.60 MB vs 2.66 MB FP16-equivalent); 1.13× whole-model storage reduction** (FF, embeddings, and memory-projection layers remain FP16 in this configuration — extending CAJQ to FFs is straightforward future work).

**Accuracy** (NIAH-single, mean ± std over 3 seeds, evaluated at 1024–4096 tokens): CAJQ-QAT reaches **0.674 ± 0.012 at ctx 1024, 0.590 ± 0.043 at ctx 2048, 0.521 ± 0.055 at ctx 4096**, matching or exceeding the FP16 reference at every evaluated context length and reducing seed variance ~2.6×.

### 2. Budget-Aware Episodic Eviction (BAEE)

A lightweight retention-score classifier (**~3.3K parameters**) that progressively compresses memory entries under RAM constraints:

```
FP16 (hot) → INT8 (warm) → summary token (cold) → eviction
```

- **RetentionClassifier**: 3-layer MLP; validation ROC-AUC = **0.907 ± 0.005** (3 seeds), robust across a 10× learning-rate range and graceful under label noise.
- **Budget enforcement**: triggers compression stages when total memory exceeds a configurable threshold (default: 256 MB).
- **vs FIFO / LRU**: BAEE retains semantically important entries regardless of recency. Under 90% forced eviction with the target needle in the *early* portion of an 8K-token stream, BAEE retains the target **71% ± 8%** of the time vs **0%** for FIFO/LRU.
- **vs learned KV-cache eviction**: head-to-head 432-cell grid against H2O, Scissorhands, SnapKV, PyramidKV, and a Locret-style proxy positions BAEE competitively across budget × position regimes (see `paper/figures/v3_fig_kv_policy_grid.pdf`).

### 3. Consumer Hardware Benchmarking Suite

Three hardware tiers profiled:
- **Apple Silicon (MPS)** — measured directly
- **Multi-thread CPU** — measured directly
- **Single-thread CPU (Raspberry Pi 5 proxy)** — measured directly (real Pi 5 throughput would be ~1.5–2× lower per-core)

Metrics reported: parameter memory, on-disk storage, activation memory, episodic-store growth, runtime RSS, sustained throughput under 45-second thermal stress, and an energy/token estimate from `powermetrics` (with rated-TDP fallback).

Benchmarks: RULER-style (NIAH, variable tracking, frequency aggregation) + LongBench-style (6 task categories) + NeedleBench-style (SNIA / MKN / RoN / CN / ADN), all with self-contained synthetic data — no external downloads required.

---

## Architecture

```
Input tokens
    │
    ▼
Token + Position Embedding
    │
    ├─ [× depth] SynapBlockWithEpisodic
    │       │
    │       ├─ SimpleSSM (depthwise conv, 2-bit QAT)
    │       │
    │       ├─ SparseEventAttention (INT4 AWQ + SmoothQuant)
    │       │       └─ salience mask → BAEE scoring
    │       │
    │       ├─ WriteableMemory (INT8 per-entry)
    │       │       ├─ write: top-K salient tokens → slots
    │       │       └─ read: cross-attention to slots
    │       │
    │       └─ ScaleBridge (FP16 — normalises 3 pathway outputs)
    │
    ▼
LayerNorm → Head (classification or LM)
```

---

## Installation

```bash
git clone https://github.com/vineetha00/SynapNet-Edge
cd SynapNet-Edge
pip install -e .
# Optional extras:
pip install -e ".[dev]"      # + psutil, pyyaml, scipy, tqdm
pip install -e ".[mlx]"      # + Apple MLX for iPhone/Mac deployment
pip install -e ".[full]"     # + HuggingFace datasets for real LongBench
```

---

## Quickstart

### Pretrain the reference 8.7M model (~10 min on M-series MPS)

```bash
python scripts/pretrain_scaled.py \
  --dim 192 --depth 6 --heads 6 \
  --curriculum 512 1024 \
  --device mps
```

### Train with CAJQ (3-phase QAT, ~15 min for 3 seeds)

```bash
python scripts/exp_cajq_qat_multiseed.py \
  --context-lengths 1024 2048 4096 \
  --seeds 42 43 44 \
  --device mps
```

### Run BAEE grid vs all baselines (432 cells, ~35 min)

```bash
python scripts/exp_baee_grid_multiseed.py \
  --policies baee_salience fifo lru random h2o scissorhands snapkv pyramidkv locret_proxy \
  --budgets 0.10 0.20 0.30 0.50 \
  --positions early late \
  --seeds 42 43 44 \
  --device mps
```

### Hardware deployment profile (3 tiers, ~6 min)

```bash
python scripts/exp_deployment_metrics.py \
  --tiers apple_silicon_mps cpu_multi cpu_single \
  --variants fp16 int4_uniform cajq
```

### Generate all paper figures

```bash
python scripts/generate_paper_figures_v3.py
```

---

## Usage in Code

```python
from synapnet_edge import SynapNetEdge, SynapNetEdgeConfig, apply_cajq, BAEEMemoryManager
from synapnet_edge.quantization.cajq import CAJQConfig
from synapnet_edge.training.calibration import build_calib_loader

# Build the reference 8.7M model
cfg = SynapNetEdgeConfig(
    dim=192, depth=6, vocab_size=4096, max_len=8192,
    num_classes=64, heads=6, episodic_slots=32,
)
model = SynapNetEdge(cfg)

# Apply CAJQ quantization (PTQ; QAT entry point in scripts/exp_cajq_qat_multiseed.py)
calib_loader = build_calib_loader(n_samples=128, seq_len=1024)
model = apply_cajq(model, CAJQConfig(device="mps"),
                    calib_loader=calib_loader, mode="ptq")

# Streaming inference with BAEE
manager = BAEEMemoryManager(dim=192, n_layers=6, budget_mb=256.0)
logits, debug = model.forward_streaming(
    input_ids, chunk_size=512, baee_manager=manager
)
```

---

## Repository Structure

```
SynapNet-Edge/
├── synapnet_edge/
│   ├── models/                          # SSM / SparseAttn / Episodic / SynapBlock / full model
│   ├── quantization/                    # CAJQ + 2-bit / INT4 / INT8 / ScaleBridge
│   ├── memory/
│   │   ├── baee.py                      # BAEEMemoryManager + RetentionClassifier (~3.3K params)
│   │   └── kv_cache_policies.py         # H2O / Scissorhands / SnapKV / PyramidKV / Locret
│   ├── benchmarks/                      # RULER / LongBench / hardware / Pareto
│   ├── baselines/                       # Mamba-2 / Llama-AWQ / Falcon-H1 / EM-LLM proxies
│   ├── training/                        # QAT trainer + calibration
│   ├── data/long_context_tasks.py       # NIAH / MultiKey / VarTrack / FA / NeedleBench / MemPressure
│   └── utils/                           # profiling + visualisation
├── scripts/                             # 18 reproducible experiment scripts
├── configs/                             # YAML configs for CAJQ / BAEE / benchmarks
├── results/                             # JSON outputs for every experiment
├── paper/figures/                       # 20+ publication-quality PDFs + v3 paper summary
├── CITATION.cff                         # citation metadata
├── LICENSE                              # MIT
└── README.md
```

---

## Baselines

| Model | Architecture | Bits | Role |
|-------|-------------|------|------|
| Mamba-2 (proxy) | GRU-SSM | INT4 uniform | Approximates Mamba-2 selective scan |
| Llama-3.2 (proxy) | Full dense attention | INT4 AWQ | O(T²) cost reference |
| Falcon-H1 / Hymba (proxy) | Hybrid SSM + attention | FP16 | Upper-bound hybrid reference |
| EM-LLM | Transformer + external memory | FP16 | Memory-mechanism ablation |
| H2O / Scissorhands / SnapKV / PyramidKV / Locret-proxy | KV-cache eviction policies | — | Eviction-policy baselines (BAEE comparison) |

---

## Ablations

- **Quantization strategy**: FP16 / Uniform INT8 / Uniform INT4 / CAJQ-PTQ / **CAJQ-QAT (ours)** — see `scripts/exp_cajq_qat_multiseed.py`
- **Eviction policy**: **BAEE (ours)** / FIFO / LRU / Random / H2O / Scissorhands / SnapKV / PyramidKV / Locret — see `scripts/exp_baee_grid_multiseed.py`
- **ScaleBridge**: post-hoc removal collapses accuracy (load-bearing in trained model); from-scratch comparison is inconclusive at our compute budget — see `scripts/exp_scale_bridge_ablation.py`
- **Classifier stability**: seed × LR × label-noise grid — see `scripts/exp_classifier_stability.py`

---

## Citation

```bibtex
@article{synapnet_edge_2026,
  title={SynapNet-Edge: Component-Aware Quantization and Budget-Aware Eviction for Hybrid Long-Context Models on Consumer Hardware},
  author={Vallish Kumar, Vineetha},
  year={2026},
}
```

---

## License

Released under the [MIT License](LICENSE).

---

## Acknowledgements

Quantization methods inspired by [ParetoQ](https://arxiv.org/abs/2408.08296),
[SmoothQuant](https://arxiv.org/abs/2211.10438), and [AWQ](https://arxiv.org/abs/2306.00978).
Memory eviction inspired by [EM-LLM](https://arxiv.org/abs/2407.09450),
[MemGPT](https://arxiv.org/abs/2310.08560), and the KV-cache compression line of work
([H2O](https://arxiv.org/abs/2306.14048), [Scissorhands](https://arxiv.org/abs/2305.17118),
[SnapKV](https://arxiv.org/abs/2404.14469), [PyramidKV](https://arxiv.org/abs/2406.02069)).
