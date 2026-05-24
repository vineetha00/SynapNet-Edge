"""Experiment 2 — BAEE vs FIFO/LRU/Random under memory pressure.

Setup designed so any policy must evict ~50% of writes:
  - 16 needles per sequence
  - 4 chunks of 512 tokens = 2048 total context
  - Each chunk writes ~16 tokens via salience (write_frac=0.05 × 512 ≈ 25 → top-K = 16)
  - BAEE budget allows 16 total slots across all chunks
  - But 4 chunks × 16 writes = 64 candidate entries → must evict 75% (way > 50%)

We compare four eviction policies on MemoryPressureNIAH:
  - BAEE-salience   — keep entries with highest salience-at-write (informed)
  - FIFO            — keep most recent writes (recency bias)
  - LRU             — keep least-recently-evicted (no info → falls back to FIFO-like)
  - Random          — random sampling

The target needle in MemoryPressureNIAH is repeated 3× → higher salience.
A salience-informed eviction (BAEE) should retain it; recency-only policies lose.

For BAEE-learned, we initialize the retention classifier from salience
(simple linear surrogate trained quickly on the streaming chunks).

Output: results/scaled/exp_baee_memory_pressure.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.data.long_context_tasks import MemoryPressureNIAH
from synapnet_edge.memory.baee import EvictionPolicy


# ---------------------------------------------------------------------------
# Streaming evaluation with externally managed episodic memory
# ---------------------------------------------------------------------------

@torch.no_grad()
def streaming_eval_with_policy(
    model: nn.Module,
    dataset,
    policy: str,
    budget_slots: int,
    chunk_size: int,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate model on dataset using forward_streaming with manual memory mgmt.

    For each layer of the model, we maintain a list of memory entries with
    associated metadata:
      - feature vector  (B, D)
      - salience score  (scalar in [0,1])
      - age             (chunks since written)
      - random key      (for Random policy)

    After each chunk, the policy is applied to keep only `budget_slots` entries.
    """
    model.eval()
    n_layers = len(model.blocks)
    D = model.cfg.dim

    correct = 0
    total = 0
    eviction_count = 0
    target_retained = 0
    target_retainable = 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rng = torch.Generator().manual_seed(42)

    for sample_i, (ids, lbl) in enumerate(loader):
        ids, lbl = ids.to(device), lbl.to(device)
        B, T = ids.shape
        assert B == 1

        # Per-layer memory state: list of (feat, salience, age, rkey)
        layer_mem: list[list] = [[] for _ in range(n_layers)]

        # Optionally track target needle positions to measure retention rate
        target_meta = dataset.metadata(sample_i) if hasattr(dataset, "metadata") else {}
        target_positions = set(target_meta.get("target_pos", []))

        # Iterate chunks
        for chunk_start in range(0, T, chunk_size):
            chunk = ids[:, chunk_start: chunk_start + chunk_size]

            # Inject current memory into each block's epmem
            # We do this by passing a "mem_bank" argument via external_mems
            mem_banks = []
            for layer_i in range(n_layers):
                entries = layer_mem[layer_i]
                if entries:
                    bank = torch.stack([e[0] for e in entries], dim=0).unsqueeze(0)  # (1,S,D)
                else:
                    bank = torch.zeros(1, 1, D, device=device)
                mem_banks.append(bank)

            # Forward this chunk
            outputs = model(chunk, external_mems=mem_banks)
            logits, masks, mems, topks = outputs
            # mems[layer_i] is the NEW mem_bank from this chunk's writes
            # (B, S_layer, D) where S_layer = model.cfg.episodic_slots

            # Append new entries to each layer's memory
            for layer_i in range(n_layers):
                new_bank = mems[layer_i][0]   # (S, D)
                new_sal = masks[layer_i][0]    # (T_chunk,) — full salience map
                top_idx = topks[layer_i][0] if topks[layer_i].numel() > 0 else None

                # New entries: one per slot, with salience-at-write per entry
                S = new_bank.size(0)
                for s in range(S):
                    if top_idx is not None and s < top_idx.numel():
                        token_idx_in_chunk = top_idx[s].item()
                        sal_score = new_sal[token_idx_in_chunk].item() \
                            if token_idx_in_chunk < new_sal.numel() else 0.0
                        abs_pos = chunk_start + token_idx_in_chunk
                    else:
                        sal_score = 0.0
                        abs_pos = -1

                    entry = [
                        new_bank[s].clone(),     # feat
                        sal_score,                # salience
                        0,                        # age
                        torch.rand((), generator=rng).item(),   # rkey
                        abs_pos,                  # absolute position (for diagnostics)
                    ]
                    layer_mem[layer_i].append(entry)

                # Apply eviction policy
                if len(layer_mem[layer_i]) > budget_slots:
                    over = len(layer_mem[layer_i]) - budget_slots
                    eviction_count += over

                    # Dispatch to policy registry for advanced KV-cache policies
                    from synapnet_edge.memory.kv_cache_policies import (
                        PolicyContext, POLICY_REGISTRY, evict,
                    )
                    if policy in POLICY_REGISTRY:
                        # Ensure each entry has the 6-element format expected
                        # by KV-cache policies (last slot = attn_accum).
                        for e in layer_mem[layer_i]:
                            if len(e) < 6:
                                e.append(0.0)
                        ctx_p = PolicyContext(
                            entries=layer_mem[layer_i],
                            budget=budget_slots,
                            layer_idx=layer_i,
                            n_layers=n_layers,
                            rng=rng,
                        )
                        keep = evict(policy, ctx_p)
                        layer_mem[layer_i] = [layer_mem[layer_i][k] for k in keep]
                    else:
                        if policy == "baee_salience":
                            layer_mem[layer_i].sort(key=lambda e: -e[1])
                        elif policy in ("fifo", "lru"):
                            layer_mem[layer_i].sort(key=lambda e: e[2])
                        elif policy == "random":
                            idxs = torch.randperm(len(layer_mem[layer_i]),
                                                  generator=rng).tolist()
                            layer_mem[layer_i] = [layer_mem[layer_i][i] for i in idxs]
                        layer_mem[layer_i] = layer_mem[layer_i][:budget_slots]

                # Age all surviving entries
                for e in layer_mem[layer_i]:
                    e[2] += 1

        # Final prediction (use last chunk's logits)
        if logits.dim() == 3:
            pred = logits[:, -1, :num_classes].argmax(-1)
        else:
            pred = logits[:, :num_classes].argmax(-1)
        correct += (pred == lbl).sum().item()
        total += 1

        # Track whether target positions were retained in any layer's memory
        if target_positions:
            target_retainable += 1
            retained = any(
                any(int(e[4]) in target_positions for e in layer_mem[layer_i])
                for layer_i in range(n_layers)
            )
            if retained:
                target_retained += 1

    return {
        "accuracy": correct / max(1, total),
        "n_samples": total,
        "eviction_count": eviction_count,
        "target_retention_rate": (
            target_retained / target_retainable if target_retainable > 0 else 0.0
        ),
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_baee_memory_pressure.json")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 2048])
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--n-needles", type=int, default=12)
    p.add_argument("--target-repeat", type=int, default=3)
    p.add_argument("--budget-multiplier", type=float, default=0.5,
                   help="Memory budget as fraction of total writes (0.5 = 50% eviction)")
    p.add_argument("--target-position-bias", default="early",
                   choices=["uniform", "early", "late"],
                   help="Where to place the target needle (early forces FIFO failure)")
    p.add_argument("--device", default="mps")
    p.add_argument("--policies", nargs="+",
                   default=["baee_salience", "fifo", "lru", "random"])
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["model_cfg"]
    vocab = config["vocab_size"]
    num_classes = config["num_classes"]

    cfg = SynapNetEdgeConfig(**config)
    model = SynapNetEdge(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded base model ({ckpt['n_params']:,} params)")

    results = {"config": {**vars(args), "model_cfg": config}, "experiments": []}

    for seq_len in args.seq_lens:
        n_chunks = seq_len // args.chunk_size
        writes_per_chunk = cfg.episodic_slots
        total_writes = n_chunks * writes_per_chunk
        budget = max(1, int(total_writes * args.budget_multiplier))
        eviction_pct = 1.0 - budget / total_writes

        print(f"\n{'='*60}")
        print(f"  seq_len={seq_len} | chunks={n_chunks} | total_writes={total_writes}")
        print(f"  budget={budget} slots | forced eviction={eviction_pct:.0%}")
        print(f"{'='*60}")

        ds = MemoryPressureNIAH(
            n_samples=args.n_samples,
            seq_len=seq_len,
            vocab_size=vocab,
            num_classes=num_classes,
            n_needles=args.n_needles,
            target_repeat=args.target_repeat,
            target_position_bias=args.target_position_bias,
            seed=2025,
        )

        seq_results = {
            "seq_len": seq_len,
            "n_chunks": n_chunks,
            "total_writes": total_writes,
            "budget_slots": budget,
            "forced_eviction_pct": eviction_pct,
            "policies": {},
        }

        for policy in args.policies:
            print(f"  -> Policy: {policy}", flush=True)
            t0 = time.perf_counter()
            res = streaming_eval_with_policy(
                model=model,
                dataset=ds,
                policy=policy,
                budget_slots=budget,
                chunk_size=args.chunk_size,
                device=device,
                num_classes=num_classes,
            )
            dt = time.perf_counter() - t0
            print(f"     acc={res['accuracy']:.3f} | "
                  f"target_retention={res['target_retention_rate']:.3f} | "
                  f"evictions={res['eviction_count']} | {dt:.1f}s")
            res["wall_time_s"] = dt
            seq_results["policies"][policy] = res

        results["experiments"].append(seq_results)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"{'seq_len':>8} | {'policy':<16} | {'acc':>6} | {'tgt_ret':>7}")
    print("-" * 50)
    for exp in results["experiments"]:
        for pol, r in exp["policies"].items():
            print(f"{exp['seq_len']:>8} | {pol:<16} | "
                  f"{r['accuracy']:>6.3f} | {r['target_retention_rate']:>7.3f}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
