"""Experiment 3 — ScaleBridge ablation: deliberate null result.

Hypothesis (null): explicit cross-pathway scale bridging is unnecessary
because LayerNorm at the start of each block, and the per-pathway
LayerNorms inside ScaleBridge, already normalise the three pathway
outputs to unit variance.  Once normalised, the learned linear
combination (alpha/beta gates + fuse layer) can absorb any residual
scale mismatch without help from a separate calibration step.

We test this by loading the same pretrained checkpoint into two model
copies — one with ScaleBridge enabled, one with it disabled (replaced
by identity passthrough) — and comparing accuracy across context lengths
and seeds.  A small or zero gap supports the null hypothesis; a large
positive gap (with > without) would contradict it.

This is reported as Section 5.3 of the paper:
  "Why we kept ScaleBridge as identity / removed it from the final model."

Output: results/scaled/exp_scale_bridge_ablation.json
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

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.models.synapblock import ScaleBridge
from synapnet_edge.data.long_context_tasks import NIAHSingle, MultiTaskCurriculum, NIAHMultiKey
from torch.utils.data import DataLoader


class IdentityBridge(nn.Module):
    """Drop-in replacement for ScaleBridge that passes pathways through."""

    def __init__(self, dim: int, n_pathways: int = 3):
        super().__init__()
        self.dim = dim
        self.n_pathways = n_pathways

    def forward(self, *pathways: torch.Tensor):
        return tuple(pathways)


def disable_bridges(model: nn.Module) -> int:
    """Replace every ScaleBridge with IdentityBridge."""
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, ScaleBridge):
            parts = name.split(".")
            parent = model
            for pp in parts[:-1]:
                parent = getattr(parent, pp)
            setattr(parent, parts[-1], IdentityBridge(m.dim, m.n_pathways))
            n += 1
    return n


@torch.no_grad()
def eval_at(model, ctx_len, n_samples, vocab, num_classes, device, seed):
    model.eval()
    out = {}
    for name, factory in [
        ("niah_single", lambda: NIAHSingle(n_samples, ctx_len, vocab, num_classes, seed)),
        ("niah_multi_key", lambda: NIAHMultiKey(n_samples, ctx_len, vocab,
                                                  num_classes, seed + 1, n_needles=4)),
        ("multi_task", lambda: MultiTaskCurriculum(
            n_samples=n_samples * 2, seq_len=ctx_len,
            vocab_size=vocab, num_classes=num_classes, seed=seed + 2)),
    ]:
        ds = factory()
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        correct = total = 0
        for batch in loader:
            if len(batch) == 3:
                ids, lbl, _ = batch
            else:
                ids, lbl = batch
            ids, lbl = ids.to(device), lbl.to(device)
            logits = model(ids)[0]
            pred = (logits[:, -1, :num_classes] if logits.dim() == 3
                    else logits[:, :num_classes]).argmax(-1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)
        out[name] = correct / max(1, total)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_scale_bridge_ablation.json")
    p.add_argument("--context-lengths", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--n-samples", type=int, default=48)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["model_cfg"]
    vocab = config["vocab_size"]
    num_classes = config["num_classes"]

    results = {"config": {**vars(args), "model_cfg": config}, "runs": {}}

    for seed in args.seeds:
        torch.manual_seed(seed)
        print(f"\n=== seed {seed} ===")
        results["runs"][seed] = {}

        for variant in ["with_bridge", "no_bridge"]:
            cfg = SynapNetEdgeConfig(**config)
            model = SynapNetEdge(cfg)
            model.load_state_dict(ckpt["model_state"])

            if variant == "no_bridge":
                n = disable_bridges(model)
                print(f"  [no_bridge] replaced {n} ScaleBridge → IdentityBridge")
            else:
                print(f"  [with_bridge] baseline")

            model.to(device).eval()
            per_ctx = {}
            for ctx in args.context_lengths:
                t0 = time.perf_counter()
                res = eval_at(model, ctx, args.n_samples, vocab,
                              num_classes, device, seed=seed)
                res["eval_time_s"] = time.perf_counter() - t0
                per_ctx[ctx] = res
                print(f"    ctx={ctx}: NIAH={res['niah_single']:.3f} "
                      f"Multi={res['multi_task']:.3f}")
            results["runs"][seed][variant] = {"per_ctx": per_ctx}

            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    # Aggregate
    print(f"\n=== AGGREGATE (mean ± std over {len(args.seeds)} seeds) ===")
    summary = {}
    for variant in ["with_bridge", "no_bridge"]:
        summary[variant] = {}
        for ctx in args.context_lengths:
            for metric in ["niah_single", "niah_multi_key", "multi_task"]:
                vals = [results["runs"][s][variant]["per_ctx"][ctx][metric]
                        for s in args.seeds]
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals)-1)) ** 0.5
                summary[variant].setdefault(str(ctx), {})[metric] = {
                    "mean": mean, "std": std, "vals": vals,
                }

    print(f"{'metric':<14} {'ctx':>5} | {'with':<14} | {'without':<14} | {'Δ (with-without)':<10}")
    for ctx in args.context_lengths:
        for metric in ["niah_single", "niah_multi_key", "multi_task"]:
            w = summary["with_bridge"][str(ctx)][metric]
            n = summary["no_bridge"][str(ctx)][metric]
            delta = w["mean"] - n["mean"]
            print(f"{metric:<14} {ctx:>5} | "
                  f"{w['mean']:.3f}±{w['std']:.3f}  | "
                  f"{n['mean']:.3f}±{n['std']:.3f}  | {delta:+.3f}")

    results["summary"] = summary

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
