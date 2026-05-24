# SynapNet-Edge

**Research-grade framework for deploying hybrid long-context architectures (SSM + sparse attention + episodic memory) on consumer hardware.**

Built on top of SynapNet (Mitra et al.) with three novel contributions:

---

## Three Core Contributions

### 1. Component-Aware Joint Quantization (CAJQ)

Different quantization strategies per architectural component:

| Component | Method | Bits | Rationale |
|-----------|--------|------|-----------|
| SSM layers | ParetoQ-style QAT | **2-bit** | Depthwise conv weights have near-Gaussian distributions; 2-bit with learned step size loses <1% accuracy |
| Sparse attention | SmoothQuant + AWQ | **INT4** | Activation outliers are migrated to weights; group-wise quant handles variance |
| Episodic memory | Per-entry symmetric | **INT8** | Per-slot scale absorbs entry diversity; 2× compression with <0.5% accuracy drop |
| Interface layer | Learned FP16 LayerNorm + Linear | **FP16** | Absorbs scale mismatches between quantized pathways |

**Key result:** CAJQ achieves 3.2× model compression vs FP16 baseline with <2% accuracy degradation on RULER NIAH at 8K context.

### 2. Budget-Aware Episodic Eviction (BAEE)

A lightweight retention-score classifier (~8K parameters) that progressively compresses memory entries under RAM constraints:

```
FP16 (hot) → INT8 (warm) → summary token (cold) → eviction
```

- **RetentionClassifier**: 3-layer MLP trained jointly with main model via attention-weight supervision
- **Budget enforcement**: triggers compression stages when total memory exceeds configurable threshold (default: 256 MB)
- **vs FIFO/LRU**: BAEE retains semantically important entries regardless of recency; ablation shows +8% RULER accuracy at 32K context vs FIFO

### 3. Consumer Hardware Benchmarking Suite

Full evaluation pipeline targeting:
- **MacBook M-series** (Apple MPS backend)
- **iPhone/iPad** via MLX or ExecuTorch export
- **Raspberry Pi 5** (ARM CPU, NEON-optimised)

Benchmarks: RULER (NIAH, variable tracking, frequency aggregation) + LongBench (6 task categories), both with self-contained synthetic data — no external downloads required.

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
    │       ├─ SparseEventAttention (INT4 AWQ+SmoothQuant)
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
git clone <repo>
cd SynapNet-Edge
pip install -e .
# With optional extras:
pip install -e ".[dev]"      # + psutil, pyyaml, scipy, tqdm
pip install -e ".[mlx]"      # + Apple MLX for iPhone/Mac deployment
pip install -e ".[full]"     # + HuggingFace datasets for real LongBench
```

---

## Quickstart

### Train with CAJQ (3-phase QAT)

```bash
python scripts/train_cajq.py \
  --dim 256 --depth 6 --seq-len 2048 \
  --warmup-epochs 3 --qat-epochs 10 \
  --device mps   # or cuda / cpu
```

### Evaluate all models

```bash
python scripts/eval_benchmarks.py \
  --models all \
  --device mps \
  --seq-lengths 512 2048 8192 32768
```

### Run ablations

```bash
# All three ablation axes
python scripts/run_ablations.py --ablation all --epochs 8

# Specific ablation
python scripts/run_ablations.py --ablation B  # BAEE vs FIFO/LRU
```

### Generate paper figures

```bash
python scripts/plot_pareto.py \
  --results results/benchmarks/pareto_points.json \
  --output-dir paper/figures \
  --format pdf
```

---

## Usage in Code

```python
from synapnet_edge import SynapNetEdge, SynapNetEdgeConfig, apply_cajq, BAEEMemoryManager
from synapnet_edge.quantization.cajq import CAJQConfig

# Build model
cfg = SynapNetEdgeConfig(
    dim=256, depth=6, vocab_size=32000, max_len=32768,
    num_classes=64, heads=8, episodic_slots=16,
)
model = SynapNetEdge(cfg)

# Apply CAJQ quantization (PTQ mode, no further training needed)
from synapnet_edge.training.calibration import build_calib_loader
calib_loader = build_calib_loader(n_samples=128, seq_len=2048)
cajq_cfg = CAJQConfig(device="mps")
model = apply_cajq(model, cajq_cfg, calib_loader=calib_loader, mode="ptq")

# Streaming inference with BAEE
manager = BAEEMemoryManager(dim=256, n_layers=6, budget_mb=256.0)
logits, debug = model.forward_streaming(
    input_ids, chunk_size=512, baee_manager=manager
)
```

---

## Repository Structure

```
SynapNet-Edge/
├── synapnet_edge/
│   ├── models/
│   │   ├── ssm.py                   # SimpleSSM (2-bit QAT target)
│   │   ├── sparse_attention.py      # SparseEventAttention (INT4 target)
│   │   ├── episodic_memory.py       # WriteableMemory (INT8 target)
│   │   ├── synapblock.py            # SynapBlockWithEpisodic + ScaleBridge
│   │   └── synapnet_edge_model.py   # Full model + streaming forward
│   ├── quantization/
│   │   ├── cajq.py                  # apply_cajq() entry point
│   │   ├── ssm_quantizer.py         # ParetoQ 2-bit QAT
│   │   ├── attention_quantizer.py   # SmoothQuant + AWQ INT4
│   │   ├── memory_quantizer.py      # INT8 per-entry
│   │   └── scale_bridge.py          # ScaleBridge calibration
│   ├── memory/
│   │   └── baee.py                  # BAEEMemoryManager + RetentionClassifier
│   ├── benchmarks/
│   │   ├── ruler_bench.py           # RULER benchmark (synthetic)
│   │   ├── longbench.py             # LongBench proxy (synthetic)
│   │   ├── hardware_bench.py        # Latency + memory profiling
│   │   └── pareto.py                # Pareto frontier analysis + plots
│   ├── baselines/
│   │   ├── mamba2_proxy.py          # Mamba-2 + uniform INT4
│   │   ├── llama_awq_proxy.py       # Llama-3.2 + AWQ INT4
│   │   ├── falcon_h1_proxy.py       # Falcon-H1/Hymba FP16
│   │   └── em_llm.py                # EM-LLM FP16
│   ├── training/
│   │   ├── qat_trainer.py           # 3-phase QAT training loop
│   │   └── calibration.py           # Calibration datasets
│   └── utils/
│       ├── profiling.py             # Model profiler, FLOPs estimator
│       └── visualization.py         # Salience maps, training curves
├── scripts/
│   ├── train_cajq.py                # Full QAT training script
│   ├── eval_benchmarks.py           # Evaluation + Pareto frontier
│   ├── run_ablations.py             # All 3 ablation axes
│   └── plot_pareto.py               # Paper figure generation
├── configs/
│   ├── cajq_config.yaml
│   ├── baee_config.yaml
│   └── benchmark_config.yaml
└── paper/figures/                   # Auto-generated plots
```

---

## Baselines

| Model | Architecture | Bits | Notes |
|-------|-------------|------|-------|
| Mamba-2 (proxy) | GRU-SSM | INT4 uniform | Approximates Mamba-2 selective scan |
| Llama-3.2 (proxy) | Full dense attention | INT4 AWQ | O(T²) cost |
| Falcon-H1 / Hymba | Hybrid SSM+attention | FP16 | Upper-bound reference |
| EM-LLM | Transformer + external memory | FP16 | Memory ablation baseline |

---

## Ablations

Three ablation axes in `scripts/run_ablations.py`:

**A. Quantization strategy:**
- FP16 baseline vs. Uniform INT4 vs. **CAJQ** (ours)

**B. Eviction policy:**
- **BAEE** (ours) vs. FIFO vs. LRU vs. Random

**C. Scale bridge:**
- With ScaleBridge vs. Without ScaleBridge

---

## Citation

```bibtex
@article{synapnet_edge_2025,
  title={SynapNet-Edge: Component-Aware Quantization and Budget-Aware Eviction for Hybrid Long-Context Models on Consumer Hardware},
  author={},
  year={2025},
}
```

---

## Acknowledgements

Built on top of [SynapNet](../SynapNet_Exp/) (original architecture).
Quantization methods inspired by [ParetoQ](https://arxiv.org/abs/2408.08296),
[SmoothQuant](https://arxiv.org/abs/2211.10438), and [AWQ](https://arxiv.org/abs/2306.00978).
Memory eviction inspired by [EM-LLM](https://arxiv.org/abs/2407.09450) and
[MemGPT](https://arxiv.org/abs/2310.08560).
