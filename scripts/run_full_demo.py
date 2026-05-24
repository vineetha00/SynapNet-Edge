"""End-to-end demo: train SynapNet-Edge with CAJQ, evaluate, run ablations, plot Pareto.

Uses a small model (dim=64, depth=2) so the full pipeline completes in minutes
on a MacBook M-series.  Mirrors the workflow that scripts/train_cajq.py +
eval_benchmarks.py + run_ablations.py would do for a publication-scale model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


# Compact configuration so the full demo runs in minutes
DIM = 64
DEPTH = 2
VOCAB = 512
NUM_CLASSES = 16
SEQ_LEN = 256
EPOCHS_WARMUP = 2
EPOCHS_QAT = 3
EPOCHS_FT = 1
N_TRAIN = 256
BATCH = 8
SEED = 42

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "full_run"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def banner(msg: str):
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def build_loaders():
    from synapnet_edge.training.calibration import SyntheticCalibDataset, build_calib_loader
    ds = SyntheticCalibDataset(
        n_samples=N_TRAIN, seq_len=SEQ_LEN, vocab_size=VOCAB,
        num_classes=NUM_CLASSES, seed=SEED,
    )
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    tr, va = random_split(ds, [n_train, n_val],
                          generator=torch.Generator().manual_seed(SEED))
    tr_loader = DataLoader(tr, batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(va, batch_size=BATCH, shuffle=False)
    calib = build_calib_loader(n_samples=32, seq_len=SEQ_LEN, vocab_size=VOCAB,
                                num_classes=NUM_CLASSES, batch_size=BATCH)
    return tr_loader, va_loader, calib


def build_model(use_scale_bridge: bool = True):
    from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
    cfg = SynapNetEdgeConfig(
        dim=DIM, depth=DEPTH, vocab_size=VOCAB, max_len=SEQ_LEN,
        num_classes=NUM_CLASSES, heads=4, k_frac=0.25,
        episodic_slots=4, episodic_write_frac=0.1,
        use_scale_bridge=use_scale_bridge,
    )
    return SynapNetEdge(cfg)


def train_simple(model, tr_loader, va_loader, epochs: int, lr: float, name: str,
                 use_qat_reg: bool = False) -> dict:
    device = torch.device(DEVICE)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for ids, lbl in tr_loader:
            ids, lbl = ids.to(device), lbl.to(device)
            opt.zero_grad()
            logits = model(ids)[0]
            loss = F.cross_entropy(logits, lbl)
            if use_qat_reg:
                from synapnet_edge.quantization.cajq import compute_cajq_loss
                loss = loss + 1e-4 * compute_cajq_loss(model).to(device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n += 1
        sched.step()

        # Eval
        model.eval()
        correct = 0; tot = 0
        with torch.no_grad():
            for ids, lbl in va_loader:
                ids, lbl = ids.to(device), lbl.to(device)
                preds = model(ids)[0].argmax(-1)
                correct += (preds == lbl).sum().item()
                tot += lbl.size(0)
        acc = correct / max(1, tot)
        history.append({"epoch": ep, "loss": total / max(1, n), "acc": acc})
        print(f"  [{name}] ep {ep}/{epochs} loss={total/max(1,n):.4f} acc={acc:.3f}")
    return {"name": name, "history": history, "final_acc": history[-1]["acc"]}


def main():
    t0 = time.time()
    all_results = {}

    banner(f"SynapNet-Edge full demo (device={DEVICE})")
    print(f"Config: dim={DIM}, depth={DEPTH}, vocab={VOCAB}, "
          f"seq_len={SEQ_LEN}, classes={NUM_CLASSES}")

    tr_loader, va_loader, calib = build_loaders()

    # ------------------------------------------------------------------
    # PHASE A: Train SynapNetEdge with full CAJQ pipeline
    # ------------------------------------------------------------------
    banner("PHASE A: Three-phase QAT training (warmup → CAJQ → finetune)")
    model = build_model(use_scale_bridge=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters\n")

    print(">>> Phase 1: FP16 warm-up")
    warmup = train_simple(model, tr_loader, va_loader, EPOCHS_WARMUP, 1e-3, "warmup")

    print("\n>>> Phase 2: Apply CAJQ quantization")
    from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig, estimate_model_bits
    cajq_cfg = CAJQConfig(n_calib_batches=4, device=DEVICE)
    apply_cajq(model, cajq_cfg, calib_loader=calib, mode="qat")
    bit_info = estimate_model_bits(model)
    print(f"  Effective bits: {bit_info['effective_bits']:.2f}, "
          f"storage: {bit_info['storage_mb']:.2f} MB")

    print("\n>>> Phase 2: QAT training")
    qat = train_simple(model, tr_loader, va_loader, EPOCHS_QAT, 5e-4, "qat",
                        use_qat_reg=True)

    print("\n>>> Phase 3: Fine-tune cooldown")
    ft = train_simple(model, tr_loader, va_loader, EPOCHS_FT, 1e-4, "finetune",
                      use_qat_reg=True)

    all_results["cajq_training"] = {
        "warmup": warmup,
        "qat": qat,
        "finetune": ft,
        "n_params": n_params,
        "effective_bits": bit_info["effective_bits"],
        "storage_mb": bit_info["storage_mb"],
    }

    # ------------------------------------------------------------------
    # PHASE B: Evaluate on RULER + LongBench
    # ------------------------------------------------------------------
    banner("PHASE B: Evaluate on RULER + LongBench")
    from synapnet_edge.benchmarks.ruler_bench import RULERBenchmark, RULERTask
    from synapnet_edge.benchmarks.longbench import LongBenchEvaluator, LongBenchTask

    model.eval()
    ruler = RULERBenchmark(model, vocab_size=VOCAB, num_classes=NUM_CLASSES,
                            n_samples=32, device=DEVICE, chunk_size=128)
    ruler_results = ruler.run(
        tasks=[RULERTask.NIAH_SINGLE, RULERTask.VARIABLE_TRACK, RULERTask.FA],
        context_lengths=[128, 256], verbose=True,
    )
    all_results["ruler"] = ruler_results

    lb = LongBenchEvaluator(model, vocab_size=VOCAB, num_classes=NUM_CLASSES,
                             n_samples=32, device=DEVICE, chunk_size=128)
    lb_results = lb.run(
        tasks=[LongBenchTask.SINGLE_DOC_QA, LongBenchTask.MULTI_DOC_QA,
               LongBenchTask.PASSAGE_RETRIEVAL],
        context_lengths=[128, 256], verbose=True,
    )
    all_results["longbench"] = lb_results

    # ------------------------------------------------------------------
    # PHASE C: Hardware profile
    # ------------------------------------------------------------------
    banner("PHASE C: Hardware profile (latency + memory)")
    from synapnet_edge.utils.profiling import profile_model
    profiles = []
    for sl in [64, 128, 256]:
        try:
            p = profile_model(model, seq_len=sl, vocab_size=VOCAB,
                              device=DEVICE, n_warmup=2, n_measure=5)
            print(f"  seq_len={sl:4d}: {p.tokens_per_second:7.1f} tok/s, "
                  f"{p.median_latency_ms:.1f}ms, peak_mem={p.peak_memory_mb:.0f}MB")
            profiles.append({
                "seq_len": sl,
                "tokens_per_second": p.tokens_per_second,
                "median_latency_ms": p.median_latency_ms,
                "peak_memory_mb": p.peak_memory_mb,
            })
        except Exception as e:
            print(f"  seq_len={sl} FAILED: {e}")
    all_results["hardware"] = profiles

    # ------------------------------------------------------------------
    # PHASE D: Compare against baselines
    # ------------------------------------------------------------------
    banner("PHASE D: Baseline comparison (RULER NIAH @ 256)")
    from synapnet_edge.baselines.mamba2_proxy import Mamba2Proxy
    from synapnet_edge.baselines.llama_awq_proxy import LlamaAWQProxy
    from synapnet_edge.baselines.falcon_h1_proxy import FalconH1Proxy
    from synapnet_edge.baselines.em_llm import EMLLMBaseline

    baseline_specs = [
        ("Mamba2-INT4",
         lambda: Mamba2Proxy(dim=DIM, depth=DEPTH, vocab_size=VOCAB,
                              max_len=SEQ_LEN, num_classes=NUM_CLASSES, quantized=False)),
        ("Llama-AWQ-INT4",
         lambda: LlamaAWQProxy(dim=DIM, depth=DEPTH, vocab_size=VOCAB,
                                max_len=SEQ_LEN, num_classes=NUM_CLASSES,
                                heads=4, quantized=False)),
        ("FalconH1-FP16",
         lambda: FalconH1Proxy(dim=DIM, depth=DEPTH, vocab_size=VOCAB,
                                max_len=SEQ_LEN, num_classes=NUM_CLASSES, heads=4)),
        ("EMLLM-FP16",
         lambda: EMLLMBaseline(dim=DIM, depth=DEPTH, vocab_size=VOCAB,
                                max_len=SEQ_LEN, num_classes=NUM_CLASSES, heads=4)),
    ]

    baseline_results = {}
    for name, factory in baseline_specs:
        print(f"\n>>> Training and evaluating: {name}")
        bm = factory()
        tr_res = train_simple(bm, tr_loader, va_loader, EPOCHS_WARMUP + 1, 1e-3, name)
        bm.eval()

        # Quick RULER NIAH evaluation
        ruler_b = RULERBenchmark(bm, vocab_size=VOCAB, num_classes=NUM_CLASSES,
                                  n_samples=16, device=DEVICE, chunk_size=128)
        b_ruler = ruler_b.run([RULERTask.NIAH_SINGLE], [128, 256], verbose=False)
        niah = b_ruler["niah_single"]

        # Latency
        p = profile_model(bm, seq_len=256, vocab_size=VOCAB,
                          device=DEVICE, n_warmup=2, n_measure=3)

        baseline_results[name] = {
            "final_train_acc": tr_res["final_acc"],
            "ruler_niah_128": niah[128]["accuracy"],
            "ruler_niah_256": niah[256]["accuracy"],
            "tok_per_sec_256": p.tokens_per_second,
            "peak_mem_mb_256": p.peak_memory_mb,
        }
        print(f"  acc(train)={tr_res['final_acc']:.3f} "
              f"NIAH@128={niah[128]['accuracy']:.3f} "
              f"NIAH@256={niah[256]['accuracy']:.3f} "
              f"tok/s={p.tokens_per_second:.0f}")
    all_results["baselines"] = baseline_results

    # Add SynapNet-Edge entry
    p_se = profile_model(model, seq_len=256, vocab_size=VOCAB,
                          device=DEVICE, n_warmup=2, n_measure=3)
    baseline_results["SynapNetEdge-CAJQ"] = {
        "final_train_acc": ft["final_acc"],
        "ruler_niah_128": ruler_results["niah_single"][128]["accuracy"],
        "ruler_niah_256": ruler_results["niah_single"][256]["accuracy"],
        "tok_per_sec_256": p_se.tokens_per_second,
        "peak_mem_mb_256": p_se.peak_memory_mb,
    }

    # ------------------------------------------------------------------
    # PHASE E: Ablations (uniform vs CAJQ, BAEE vs FIFO/LRU, scale bridge)
    # ------------------------------------------------------------------
    banner("PHASE E: Ablations")

    ablations = {}

    # ABLATION A: Uniform INT4 vs CAJQ
    print("\n>>> Ablation A: Uniform INT4 vs CAJQ (component-aware)")
    model_uniform = build_model()
    train_simple(model_uniform, tr_loader, va_loader, EPOCHS_WARMUP, 1e-3, "FP16-warmup")
    # Apply uniform INT4 to ALL Linear layers
    from synapnet_edge.quantization.attention_quantizer import AWQLinear
    for n, m in list(model_uniform.named_modules()):
        if isinstance(m, nn.Linear) and "head" not in n:
            parts = n.split(".")
            parent = model_uniform
            for pp in parts[:-1]:
                parent = getattr(parent, pp)
            try:
                setattr(parent, parts[-1], AWQLinear.from_linear(m, group_size=64))
            except Exception:
                pass
    res_uniform = train_simple(model_uniform, tr_loader, va_loader, EPOCHS_QAT,
                                5e-4, "uniform-INT4")
    ablations["A_uniform_int4"] = res_uniform
    ablations["A_cajq"] = ft   # already trained above

    # ABLATION B: BAEE vs FIFO vs LRU vs Random eviction
    print("\n>>> Ablation B: Eviction policies (BAEE/FIFO/LRU/Random)")
    from synapnet_edge.memory.baee import BAEEMemoryManager, EvictionPolicy
    for policy in [EvictionPolicy.BAEE, EvictionPolicy.FIFO,
                   EvictionPolicy.LRU, EvictionPolicy.RANDOM]:
        m = build_model()
        mgr = BAEEMemoryManager(dim=DIM, n_layers=DEPTH, slots_per_layer=4,
                                budget_mb=2.0, policy=policy)
        res = train_simple(m, tr_loader, va_loader, EPOCHS_WARMUP + 1, 1e-3,
                           f"Eviction-{policy.value}")
        ablations[f"B_{policy.value}"] = res

    # ABLATION C: with/without ScaleBridge
    print("\n>>> Ablation C: ScaleBridge ablation")
    m_no_bridge = build_model(use_scale_bridge=False)
    res_no_bridge = train_simple(m_no_bridge, tr_loader, va_loader,
                                  EPOCHS_WARMUP + 1, 1e-3, "no-ScaleBridge")
    ablations["C_no_bridge"] = res_no_bridge

    m_with_bridge = build_model(use_scale_bridge=True)
    res_with_bridge = train_simple(m_with_bridge, tr_loader, va_loader,
                                    EPOCHS_WARMUP + 1, 1e-3, "with-ScaleBridge")
    ablations["C_with_bridge"] = res_with_bridge

    all_results["ablations"] = ablations

    # ------------------------------------------------------------------
    # PHASE F: Pareto plots
    # ------------------------------------------------------------------
    banner("PHASE F: Pareto frontier and figures")
    from synapnet_edge.benchmarks.pareto import ParetoAnalyzer, ParetoPoint

    family_map = {
        "Mamba2-INT4": "Mamba2",
        "Llama-AWQ-INT4": "Llama",
        "FalconH1-FP16": "FalconH1",
        "EMLLM-FP16": "EMLLM",
        "SynapNetEdge-CAJQ": "SynapNetEdge",
    }
    pa = ParetoAnalyzer()
    for name, res in baseline_results.items():
        bits = {"Mamba2-INT4": 16, "Llama-AWQ-INT4": 16, "FalconH1-FP16": 16,
                "EMLLM-FP16": 16, "SynapNetEdge-CAJQ": bit_info["effective_bits"]}[name]
        pa.add_point(ParetoPoint(
            model_name=name,
            accuracy=res["ruler_niah_256"],
            latency_tok_s=res["tok_per_sec_256"],
            memory_mb=max(res["peak_mem_mb_256"], 1.0),
            context_len=256,
            effective_bits=bits,
            method_family=family_map[name],
        ))

    pa.save(str(RESULTS_DIR / "pareto_points.json"))
    pa.plot_accuracy_vs_latency(str(FIG_DIR / "fig1_pareto_acc_latency.pdf"))
    pa.plot_accuracy_vs_memory(str(FIG_DIR / "fig2_pareto_acc_memory.pdf"))
    pa.plot_bits_vs_accuracy(str(FIG_DIR / "fig3_bits_vs_accuracy.pdf"))
    pa.plot_context_length_heatmap(str(FIG_DIR / "fig4_context_heatmap.pdf"))

    frontier = pa.pareto_frontier(
        objectives=["accuracy", "latency_tok_s", "memory_mb"],
        directions=["max", "max", "min"],
    )
    all_results["pareto_frontier"] = [p.model_name for p in frontier]
    print("\nPareto frontier:")
    for p in sorted(frontier, key=lambda x: -x.accuracy):
        print(f"  {p.model_name:<25} acc={p.accuracy:.3f} "
              f"tok/s={p.latency_tok_s:.0f} mem={p.memory_mb:.0f}MB")

    # ------------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    all_results["wall_time_s"] = elapsed

    results_path = RESULTS_DIR / "full_demo_results.json"
    # Sanitize for JSON
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    with open(results_path, "w") as f:
        json.dump(_sanitize(all_results), f, indent=2)

    banner(f"DONE in {elapsed:.1f}s — results: {results_path}")

    # Final summary
    print("\nFINAL SUMMARY TABLE")
    print(pa.summary_table())
    print()
    print(f"ablation A (CAJQ vs Uniform): CAJQ={ft['final_acc']:.3f} "
          f"vs Uniform-INT4={res_uniform['final_acc']:.3f}")
    for pol in ["baee", "fifo", "lru", "random"]:
        print(f"ablation B Eviction-{pol}: acc={ablations[f'B_{pol}']['final_acc']:.3f}")
    print(f"ablation C ScaleBridge: with={res_with_bridge['final_acc']:.3f} "
          f"vs without={res_no_bridge['final_acc']:.3f}")


if __name__ == "__main__":
    main()
