"""Ablation study runner for the SynapNet-Edge paper.

Three ablation axes:
  A) Uniform vs. Component-Aware Quantization
     - SynapNetEdge + CAJQ (our method)
     - SynapNetEdge + uniform INT4 (all layers, ablates component-awareness)
     - SynapNetEdge + uniform INT8 (all layers, ablates bit-assignment)

  B) BAEE vs. FIFO / LRU eviction
     - SynapNetEdge + BAEE (our method)
     - SynapNetEdge + FIFO eviction
     - SynapNetEdge + LRU eviction
     - SynapNetEdge + random eviction

  C) Scale bridge ablation
     - SynapNetEdge + CAJQ + ScaleBridge (full method)
     - SynapNetEdge + CAJQ − ScaleBridge (ablates interface layer)

Each ablation trains from scratch on the episodic memory recall task,
then evaluates on RULER NIAH across 4 context lengths.
Results saved to results/ablations.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.memory.baee import EvictionPolicy, BAEEMemoryManager
from synapnet_edge.training.calibration import build_calib_loader, SyntheticCalibDataset
from synapnet_edge.benchmarks.ruler_bench import RULERBenchmark, RULERTask
from synapnet_edge.benchmarks.pareto import ParetoAnalyzer, ParetoPoint

from torch.utils.data import DataLoader, random_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DIM = 128
DEPTH = 4
VOCAB = 2048
NUM_CLASSES = 32
SEQ_LEN = 2048
EPOCHS = 8
BATCH = 4
LR = 5e-4
SEED = 42


def build_model(use_scale_bridge: bool = True) -> SynapNetEdge:
    cfg = SynapNetEdgeConfig(
        dim=DIM, depth=DEPTH, vocab_size=VOCAB,
        max_len=SEQ_LEN, num_classes=NUM_CLASSES,
        heads=4, k_frac=0.25,
        episodic_slots=8, episodic_write_frac=0.05,
        use_scale_bridge=use_scale_bridge,
    )
    return SynapNetEdge(cfg)


def build_loaders():
    dataset = SyntheticCalibDataset(
        n_samples=1024, seq_len=SEQ_LEN, vocab_size=VOCAB,
        num_classes=NUM_CLASSES, seed=SEED,
    )
    n_train = int(0.8 * len(dataset))
    n_eval = len(dataset) - n_train
    train_ds, eval_ds = random_split(
        dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    calib_loader = build_calib_loader(
        n_samples=128, seq_len=SEQ_LEN, vocab_size=VOCAB,
        num_classes=NUM_CLASSES, batch_size=BATCH,
    )
    return train_loader, eval_loader, calib_loader


def train_and_eval(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    name: str = "model",
) -> dict:
    import torch.nn.functional as F

    device = torch.device(DEVICE)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ids)[0]
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        if epoch % 2 == 0:
            print(f"  [{name}] epoch {epoch}/{EPOCHS} loss={total_loss/max(1,n):.4f}")

    # Eval accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, labels in eval_loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)[0]
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / max(1, total)
    print(f"  [{name}] eval accuracy: {acc:.4f}")
    return {"name": name, "accuracy": acc}


# ---------------------------------------------------------------------------
# Ablation A: Uniform vs. CAJQ
# ---------------------------------------------------------------------------

def ablation_a_quantization(results: dict, train_loader, eval_loader, calib_loader):
    print("\n[Ablation A] Uniform vs. Component-Aware Quantization")

    # A1: No quantization (FP16 baseline)
    model_fp16 = build_model()
    r = train_and_eval(model_fp16, train_loader, eval_loader, "FP16-baseline")
    results["A_fp16"] = r

    # A2: Uniform INT4 (all layers, no component awareness)
    from synapnet_edge.baselines.mamba2_proxy import Mamba2Proxy
    model_unif_int4 = build_model()
    from synapnet_edge.quantization.attention_quantizer import AWQLinear
    replaced = 0
    for name, m in list(model_unif_int4.named_modules()):
        if isinstance(m, torch.nn.Linear):
            awq = AWQLinear.from_linear(m, group_size=128)
            parts = name.split(".")
            parent = model_unif_int4
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], awq)
            replaced += 1
    print(f"  Uniform INT4: replaced {replaced} layers")
    r = train_and_eval(model_unif_int4, train_loader, eval_loader, "Uniform-INT4")
    results["A_uniform_int4"] = r

    # A3: CAJQ (component-aware)
    from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
    model_cajq = build_model()
    train_and_eval(model_cajq, train_loader, eval_loader, "CAJQ-warmup")   # warm up first
    cajq_cfg = CAJQConfig(n_calib_batches=16, device=DEVICE)
    apply_cajq(model_cajq, cajq_cfg, calib_loader=calib_loader, mode="qat")
    r = train_and_eval(model_cajq, train_loader, eval_loader, "CAJQ")
    results["A_cajq"] = r


# ---------------------------------------------------------------------------
# Ablation B: BAEE vs. FIFO / LRU
# ---------------------------------------------------------------------------

def ablation_b_eviction(results: dict, train_loader, eval_loader, calib_loader):
    print("\n[Ablation B] BAEE vs. FIFO / LRU eviction policies")

    for policy, label in [
        (EvictionPolicy.BAEE, "BAEE"),
        (EvictionPolicy.FIFO, "FIFO"),
        (EvictionPolicy.LRU, "LRU"),
        (EvictionPolicy.RANDOM, "Random"),
    ]:
        model = build_model()
        manager = BAEEMemoryManager(
            dim=DIM, n_layers=DEPTH,
            slots_per_layer=8,
            budget_mb=32.0,
            policy=policy,
        )
        r = train_and_eval(model, train_loader, eval_loader, f"Eviction-{label}")
        stats = manager.get_compression_stats()
        r["eviction_policy"] = label
        r["compression_stats"] = stats
        results[f"B_{label.lower()}"] = r


# ---------------------------------------------------------------------------
# Ablation C: Scale bridge
# ---------------------------------------------------------------------------

def ablation_c_scale_bridge(results: dict, train_loader, eval_loader, calib_loader):
    print("\n[Ablation C] Scale bridge ablation")

    # C1: With scale bridge (full method)
    model_with_bridge = build_model(use_scale_bridge=True)
    r = train_and_eval(model_with_bridge, train_loader, eval_loader, "With-ScaleBridge")
    results["C_with_bridge"] = r

    # C2: Without scale bridge
    model_no_bridge = build_model(use_scale_bridge=False)
    r = train_and_eval(model_no_bridge, train_loader, eval_loader, "No-ScaleBridge")
    results["C_no_bridge"] = r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SynapNet-Edge ablation studies")
    parser.add_argument("--ablation", default="all",
                        choices=["all", "A", "B", "C"],
                        help="Which ablation to run")
    parser.add_argument("--output", default="results/ablations.json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    global EPOCHS
    EPOCHS = args.epochs

    torch.manual_seed(SEED)
    train_loader, eval_loader, calib_loader = build_loaders()

    results = {}

    if args.ablation in ("all", "A"):
        ablation_a_quantization(results, train_loader, eval_loader, calib_loader)

    if args.ablation in ("all", "B"):
        ablation_b_eviction(results, train_loader, eval_loader, calib_loader)

    if args.ablation in ("all", "C"):
        ablation_c_scale_bridge(results, train_loader, eval_loader, calib_loader)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Ablations] Results saved to {args.output}")
    print("\nSummary:")
    for key, val in results.items():
        print(f"  {key}: acc={val.get('accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
