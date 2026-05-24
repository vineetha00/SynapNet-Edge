"""Section 5.3 (true null-result test) — Pretrain two models from scratch:
one with ScaleBridge enabled, one without.

If the no-bridge model achieves comparable accuracy when trained without
the bridge from the start, that supports the null hypothesis that
LayerNorm + alpha/beta gates already absorb pathway scale mismatches.

If the no-bridge model is significantly worse even when trained from
scratch, then the bridge contributes a real architectural inductive bias.

We use a small/fast configuration (dim=128, depth=4, ctx=512) and 3 seeds
to keep training time bounded (~3-5 min per seed × 2 variants = ~25 min total).

Output: results/scaled/exp_scale_bridge_from_scratch.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum, NIAHSingle


def train_one(use_bridge: bool, seed: int, args) -> dict:
    torch.manual_seed(seed)
    device = torch.device(args.device)

    cfg = SynapNetEdgeConfig(
        dim=args.dim, depth=args.depth, heads=args.heads,
        vocab_size=args.vocab,
        max_len=max(args.seq_len, max(args.eval_ctxs)),
        num_classes=args.num_classes,
        k_frac=0.25,
        episodic_slots=16, episodic_write_frac=0.05,
        use_scale_bridge=use_bridge,
    )
    model = SynapNetEdge(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    ds = MultiTaskCurriculum(
        n_samples=args.n_train, seq_len=args.seq_len,
        vocab_size=args.vocab, num_classes=args.num_classes, seed=seed,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01, betas=(0.9, 0.95))
    n_steps = args.n_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    history = []
    step = 0
    t0 = time.perf_counter()
    iter_loader = iter(loader)
    model.train()
    while step < n_steps:
        try: batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
        ids, lbl, _tid = batch
        ids, lbl = ids.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = model(ids)[0]
        if logits.dim() == 3:
            pred_logits = logits[:, -1, :args.num_classes]
        else:
            pred_logits = logits[:, :args.num_classes]
        loss = F.cross_entropy(pred_logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            history.append({"step": step, "loss": float(loss.detach())})
        step += 1
    train_time = time.perf_counter() - t0

    # Evaluate
    model.eval()
    eval_results = {}
    for ctx in args.eval_ctxs:
        eval_ds = NIAHSingle(64, ctx, args.vocab, args.num_classes, seed + 1000)
        eval_loader = DataLoader(eval_ds, batch_size=2)
        correct = total = 0
        with torch.no_grad():
            for ids, lbl in eval_loader:
                ids, lbl = ids.to(device), lbl.to(device)
                logits = model(ids)[0]
                pred = (logits[:, -1, :args.num_classes] if logits.dim() == 3
                        else logits[:, :args.num_classes]).argmax(-1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)
        eval_results[ctx] = correct / max(1, total)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "n_params": n_params,
        "train_time_s": train_time,
        "final_loss": history[-1]["loss"] if history else None,
        "history": history,
        "eval_niah_single": eval_results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--vocab", type=int, default=2048)
    p.add_argument("--num-classes", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--eval-ctxs", nargs="+", type=int, default=[512, 1024])
    p.add_argument("--device", default="mps")
    p.add_argument("--output", default="results/scaled/exp_scale_bridge_from_scratch.json")
    args = p.parse_args()

    print(f"[from_scratch_bridge] dim={args.dim} depth={args.depth} "
          f"steps={args.n_steps} seeds={args.seeds}")

    results = {"config": vars(args), "runs": {}}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        results["runs"][seed] = {}
        for use_bridge in [True, False]:
            tag = "with_bridge" if use_bridge else "no_bridge"
            print(f"  Training [{tag}] seed={seed}...")
            t0 = time.perf_counter()
            res = train_one(use_bridge, seed, args)
            print(f"    train_time={res['train_time_s']:.1f}s "
                  f"loss={res['final_loss']:.4f}  "
                  f"acc@{list(res['eval_niah_single'].keys())}="
                  f"{list(res['eval_niah_single'].values())}")
            results["runs"][seed][tag] = res

    # Aggregate
    print(f"\n=== AGGREGATE (mean ± std over {len(args.seeds)} seeds) ===")
    summary = {"with_bridge": {}, "no_bridge": {}}
    for tag in ["with_bridge", "no_bridge"]:
        for ctx in args.eval_ctxs:
            vals = [results["runs"][s][tag]["eval_niah_single"][ctx]
                    for s in args.seeds]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals)-1)) ** 0.5
            summary[tag][str(ctx)] = {"mean": mean, "std": std, "vals": vals}
    results["summary"] = summary

    print(f"{'ctx':>5} | {'with_bridge':>14} | {'no_bridge':>14} | Δ")
    for ctx in args.eval_ctxs:
        w = summary["with_bridge"][str(ctx)]
        n = summary["no_bridge"][str(ctx)]
        delta = w["mean"] - n["mean"]
        print(f"{ctx:>5} | {w['mean']:.3f}±{w['std']:.3f} | "
              f"{n['mean']:.3f}±{n['std']:.3f} | {delta:+.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
