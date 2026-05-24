"""Experiment 1b — CAJQ with full QAT convergence, 3 random seeds.

Goal: demonstrate that CAJQ + post-pretraining QAT *matches or exceeds*
FP16 accuracy on long-context tasks, eliminating "calibration-noise"
concerns by reporting mean ± std across 3 seeds.

Per seed, for each variant ∈ {fp16, int8_uniform, int4_uniform, cajq, cajq+qat}:
  1. Reload the pretrained base FP16 checkpoint
  2. Apply quantization (PTQ for fp16/int8/int4/cajq, then optionally short QAT)
  3. Evaluate at multiple context lengths

The CAJQ-QAT variant runs ~200 fine-tune steps with separate optimizer
parameter groups:
  - step-size params: high LR (1e-2)         — must adapt fast to clip range
  - model params:     low LR (1e-4)          — preserve pretrained features
  - regulariser:      log-step drift penalty — prevents step collapse

Output: results/scaled/exp_cajq_qat_multiseed.json
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum, NIAHSingle, NIAHMultiKey
from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
from synapnet_edge.quantization.ssm_quantizer import (
    SSMQuantizer, QuantizedSSMWrapper,
)
from exp_cajq_long_context import (
    apply_uniform_int8, apply_uniform_int4, estimate_effective_bits,
)


# ---------------------------------------------------------------------------
# QAT loop
# ---------------------------------------------------------------------------

def run_qat(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    n_steps: int,
    num_classes: int,
    lr_model: float = 1e-4,
    lr_step: float = 1e-2,
    lambda_reg: float = 1e-3,
    grad_clip: float = 1.0,
    log_every: int = 50,
) -> list[dict]:
    """Short post-PTQ QAT fine-tune.  Separate LR for step-size params."""
    step_params_ids = set()
    step_params = list(SSMQuantizer.step_parameters(model))
    for p in step_params:
        step_params_ids.add(id(p))
    other_params = [p for p in model.parameters()
                    if id(p) not in step_params_ids and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": step_params, "lr": lr_step,  "weight_decay": 0.0},
        {"params": other_params, "lr": lr_model, "weight_decay": 0.01},
    ], betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    history = []
    model.train()
    step = 0
    train_iter = iter(train_loader)
    while step < n_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if len(batch) == 3:
            ids, lbl, _tid = batch
        else:
            ids, lbl = batch
        ids, lbl = ids.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = model(ids)[0]
        if logits.dim() == 3:
            pred_logits = logits[:, -1, :num_classes]
        else:
            pred_logits = logits[:, :num_classes]
        task_loss = F.cross_entropy(pred_logits, lbl)
        reg_loss = SSMQuantizer.collect_quantization_loss(model)
        loss = task_loss + lambda_reg * reg_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            history.append({
                "step": step,
                "task_loss": float(task_loss.detach()),
                "reg_loss": float(reg_loss.detach()),
            })
            print(f"      QAT step {step:4d}: task_loss={task_loss.item():.4f} "
                  f"reg_loss={reg_loss.item():.4f}")
        step += 1
    return history


# ---------------------------------------------------------------------------
# Multi-task eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_at_context(model: nn.Module, ctx_len: int, n_samples: int,
                    vocab: int, num_classes: int, device: torch.device,
                    seed: int = 1234) -> dict:
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
            if logits.dim() == 3:
                pred = logits[:, -1, :num_classes].argmax(-1)
            else:
                pred = logits[:, :num_classes].argmax(-1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)
        out[name] = correct / max(1, total)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_cajq_qat_multiseed.json")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[1024, 2048, 4096])
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--qat-steps", type=int, default=200)
    p.add_argument("--qat-ctx", type=int, default=1024,
                   help="Context length used for QAT fine-tune")
    p.add_argument("--qat-batch", type=int, default=4)
    p.add_argument("--device", default="mps")
    p.add_argument("--variants", nargs="+",
                   default=["fp16", "int8_uniform", "int4_uniform", "cajq_ptq", "cajq_qat"])
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["model_cfg"]
    vocab = config["vocab_size"]
    num_classes = config["num_classes"]

    print(f"[exp_cajq_qat_ms] device={device} | "
          f"model params={ckpt['n_params']:,} | seeds={args.seeds}")

    all_results = {"config": {**vars(args), "model_cfg": config}, "runs": {}}

    for seed in args.seeds:
        torch.manual_seed(seed)
        print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
        all_results["runs"][seed] = {}

        for variant in args.variants:
            print(f"\n  [{variant}] seed={seed}")
            cfg = SynapNetEdgeConfig(**config)
            model = SynapNetEdge(cfg)
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()

            t0 = time.perf_counter()

            if variant == "fp16":
                pass
            elif variant == "int8_uniform":
                n = apply_uniform_int8(model)
                print(f"    INT8 applied to {n} layers")
            elif variant == "int4_uniform":
                n = apply_uniform_int4(model, group_size=64)
                print(f"    INT4 (AWQ) applied to {n} layers")
            elif variant in ("cajq_ptq", "cajq_qat"):
                # Use seed-specific calibration data
                calib_ds = MultiTaskCurriculum(
                    n_samples=32, seq_len=512,
                    vocab_size=vocab, num_classes=num_classes,
                    seed=seed + 1000,
                )
                calib = DataLoader(calib_ds, batch_size=4, shuffle=False)
                cajq_cfg = CAJQConfig(n_calib_batches=8, device=args.device,
                                      attn_group_size=64)
                apply_cajq(model, cajq_cfg, calib_loader=calib, mode="ptq")
                model.to(device)

                if variant == "cajq_qat":
                    # Short QAT fine-tune at the chosen context length
                    train_ds = MultiTaskCurriculum(
                        n_samples=args.qat_steps * args.qat_batch,
                        seq_len=args.qat_ctx,
                        vocab_size=vocab, num_classes=num_classes,
                        seed=seed + 2000,
                    )
                    loader = DataLoader(train_ds, batch_size=args.qat_batch,
                                         shuffle=True)
                    qat_hist = run_qat(
                        model, loader, device,
                        n_steps=args.qat_steps,
                        num_classes=num_classes,
                        lr_model=1e-4, lr_step=1e-2,
                        lambda_reg=1e-3,
                    )

            eff_bits = estimate_effective_bits(model)
            prep_time = time.perf_counter() - t0

            # Eval at each context length
            per_ctx = {}
            for ctx in args.context_lengths:
                t_eval = time.perf_counter()
                res = eval_at_context(model, ctx, args.n_samples,
                                       vocab, num_classes, device, seed=seed)
                res["eval_time_s"] = time.perf_counter() - t_eval
                per_ctx[ctx] = res
                print(f"    ctx={ctx:5d}: NIAH-single={res['niah_single']:.3f} "
                      f"MultiKey={res['niah_multi_key']:.3f} "
                      f"MultiTask={res['multi_task']:.3f}")

            all_results["runs"][seed][variant] = {
                "effective_bits": eff_bits,
                "prep_time_s": prep_time,
                "per_ctx": per_ctx,
            }

            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate mean ± std across seeds
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\n  AGGREGATE (mean ± std over {len(args.seeds)} seeds)\n{'='*60}")
    summary = {}
    for variant in args.variants:
        summary[variant] = {"per_ctx": {}}
        for ctx in args.context_lengths:
            for metric in ["niah_single", "niah_multi_key", "multi_task"]:
                vals = []
                for s in args.seeds:
                    rec = all_results["runs"][s][variant]["per_ctx"][ctx]
                    vals.append(rec[metric])
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
                std = var ** 0.5
                if str(ctx) not in summary[variant]["per_ctx"]:
                    summary[variant]["per_ctx"][str(ctx)] = {}
                summary[variant]["per_ctx"][str(ctx)][metric] = {
                    "mean": mean, "std": std, "vals": vals,
                }
        # Average effective bits across seeds (should be identical)
        bits = [all_results["runs"][s][variant]["effective_bits"]
                for s in args.seeds]
        summary[variant]["effective_bits"] = sum(bits) / len(bits)

    all_results["summary"] = summary

    # Print summary table
    print(f"\n{'variant':<14} | {'bits':>5} | " +
          " | ".join(f"NIAH@{c}".rjust(13) for c in args.context_lengths))
    print("-" * (24 + 18 * len(args.context_lengths)))
    for v in args.variants:
        row = f"{v:<14} | {summary[v]['effective_bits']:>5.1f} | "
        for c in args.context_lengths:
            stat = summary[v]["per_ctx"][str(c)]["niah_single"]
            row += f"{stat['mean']:.3f}±{stat['std']:.3f}".rjust(13) + " | "
        print(row.rstrip(" |"))

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(out_path, "w") as f:
        json.dump(_san(all_results), f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
