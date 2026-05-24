"""Experiment 2b — BAEE grid: budgets × positions × policies × 3 seeds.

Demonstrates BAEE robustness over the full eviction-pressure spectrum:

  Memory budget  : 10%, 20%, 30%, 50% of total writes
  Target position: early (first 40%), late (last 40%)
  Policies       : BAEE-salience, FIFO, LRU, Random
  Sequence len   : 1024, 2048
  Seeds          : 42, 43, 44

For each (budget, position, policy, ctx) cell, we report mean ± std of
the target-needle retention rate and task accuracy across seeds.

Output: results/scaled/exp_baee_grid.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.data.long_context_tasks import MemoryPressureNIAH

from exp_baee_memory_pressure import streaming_eval_with_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_baee_grid.json")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 2048])
    p.add_argument("--budgets", nargs="+", type=float,
                   default=[0.10, 0.20, 0.30, 0.50])
    p.add_argument("--positions", nargs="+", default=["early", "late"])
    p.add_argument("--policies", nargs="+",
                   default=["baee_salience", "fifo", "lru", "random"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--n-needles", type=int, default=16)
    p.add_argument("--target-repeat", type=int, default=3)
    p.add_argument("--device", default="mps")
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
    print(f"[exp_baee_grid] params={ckpt['n_params']:,}")
    print(f"  budgets={args.budgets} positions={args.positions} "
          f"seeds={args.seeds} seq_lens={args.seq_lens}")

    total_cells = (len(args.budgets) * len(args.positions) *
                   len(args.policies) * len(args.seq_lens) * len(args.seeds))
    print(f"  total cells: {total_cells}")

    results = {"config": vars(args), "cells": []}
    cell_idx = 0
    t_total = time.perf_counter()

    for seq_len in args.seq_lens:
        n_chunks = seq_len // args.chunk_size
        total_writes = n_chunks * cfg.episodic_slots

        for budget_mult in args.budgets:
            budget = max(1, int(total_writes * budget_mult))

            for position in args.positions:
                for policy in args.policies:
                    for seed in args.seeds:
                        cell_idx += 1
                        ds = MemoryPressureNIAH(
                            n_samples=args.n_samples,
                            seq_len=seq_len,
                            vocab_size=vocab,
                            num_classes=num_classes,
                            n_needles=args.n_needles,
                            target_repeat=args.target_repeat,
                            target_position_bias=position,
                            seed=seed * 100 + int(budget_mult * 100),
                        )

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

                        cell = {
                            "seq_len": seq_len,
                            "budget_mult": budget_mult,
                            "budget_slots": budget,
                            "total_writes": total_writes,
                            "position": position,
                            "policy": policy,
                            "seed": seed,
                            "accuracy": res["accuracy"],
                            "target_retention": res["target_retention_rate"],
                            "wall_time_s": dt,
                        }
                        results["cells"].append(cell)

                        elapsed = time.perf_counter() - t_total
                        eta = elapsed / cell_idx * (total_cells - cell_idx)
                        print(f"  [{cell_idx:3d}/{total_cells}] "
                              f"seq={seq_len} budget={budget}/{total_writes} "
                              f"pos={position} pol={policy} seed={seed}: "
                              f"ret={res['target_retention_rate']:.2f} "
                              f"acc={res['accuracy']:.2f} "
                              f"({dt:.1f}s, eta={eta/60:.1f}min)")

    # Aggregate
    print(f"\n[exp_baee_grid] done in {(time.perf_counter()-t_total)/60:.1f} min")
    print("\n=== AGGREGATE (mean ± std) ===")
    print(f"  Format: target_retention | accuracy")

    summary = {}
    for cell in results["cells"]:
        key = (cell["seq_len"], cell["budget_mult"], cell["position"], cell["policy"])
        if key not in summary:
            summary[key] = {"ret": [], "acc": []}
        summary[key]["ret"].append(cell["target_retention"])
        summary[key]["acc"].append(cell["accuracy"])

    summary_out = {}
    for k, vals in summary.items():
        ret_m = sum(vals["ret"]) / len(vals["ret"])
        acc_m = sum(vals["acc"]) / len(vals["acc"])
        ret_s = (sum((r - ret_m) ** 2 for r in vals["ret"]) / max(1, len(vals["ret"]) - 1)) ** 0.5
        acc_s = (sum((a - acc_m) ** 2 for a in vals["acc"]) / max(1, len(vals["acc"]) - 1)) ** 0.5
        summary_out[str(k)] = {
            "ret_mean": ret_m, "ret_std": ret_s,
            "acc_mean": acc_m, "acc_std": acc_s,
            "n_seeds": len(vals["ret"]),
        }

    results["summary"] = summary_out

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"Saved to {args.output}")

    # Print a compact summary grid: rows = position×budget, cols = policies
    for seq_len in args.seq_lens:
        print(f"\n--- seq_len = {seq_len} ---")
        print(f"{'pos':<7} {'budg%':>6} | " +
              " ".join(f"{p:>22}" for p in args.policies))
        for position in args.positions:
            for budget_mult in args.budgets:
                row = f"{position:<7} {int(budget_mult*100):>6}% | "
                for policy in args.policies:
                    k = str((seq_len, budget_mult, position, policy))
                    s = summary_out[k]
                    row += f"R={s['ret_mean']:.2f}±{s['ret_std']:.2f} A={s['acc_mean']:.2f}".rjust(22) + " "
                print(row)


if __name__ == "__main__":
    main()
