"""Experiment 1 — CAJQ vs Uniform Quantization at Long Contexts.

Compares four quantization schemes on the pretrained base model:
  1. FP16            — full-precision baseline (upper bound)
  2. Uniform INT8    — ALL Linear layers quantized to INT8 symmetric
  3. Uniform INT4    — ALL Linear layers quantized to INT4 (AWQ-style)
  4. CAJQ (ours)     — 2-bit SSM + INT4 attention + INT8 episodic memory
                       + FP16 ScaleBridge

Each variant is evaluated at multiple context lengths (512, 1024, 2048,
4096, 8192) on the multi-task curriculum.  The expectation is that CAJQ
degrades less than uniform schemes as context length grows, because the
component-specific bit allocation preserves the precision where it matters
(attention selectivity + episodic recall).

Output: results/scaled/exp_cajq_long_context.json
"""
from __future__ import annotations

import argparse
import copy
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
from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum, NIAHSingle, NIAHMultiKey
from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
from synapnet_edge.quantization.attention_quantizer import AWQLinear


# ---------------------------------------------------------------------------
# INT8 symmetric quantization (uniform baseline)
# ---------------------------------------------------------------------------

class SymmetricINT8Linear(nn.Module):
    """Per-tensor symmetric INT8 quantized Linear (uniform baseline)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("w_int8", torch.zeros(out_features, in_features,
                                                   dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(1.0))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "SymmetricINT8Linear":
        layer = cls(linear.in_features, linear.out_features,
                    bias=linear.bias is not None)
        w = linear.weight.detach().float()
        scale = w.abs().max() / 127.0
        scale = scale.clamp(min=1e-8)
        layer.w_int8.copy_((w / scale).round().clamp(-127, 127).to(torch.int8))
        layer.scale.copy_(scale)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w_int8.to(dtype=x.dtype, device=x.device) * self.scale.to(x.device)
        bias = self.bias.to(x.device) if self.bias is not None else None
        return F.linear(x, w, bias)


def apply_uniform_int8(model: nn.Module) -> int:
    """Replace all Linear layers (except final head) with INT8."""
    replaced = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and not name.endswith(".head"):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], SymmetricINT8Linear.from_linear(m))
            replaced += 1
    return replaced


def apply_uniform_int4(model: nn.Module, group_size: int = 64) -> int:
    """Replace all Linear layers (except final head) with INT4 AWQ-style."""
    replaced = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and not name.endswith(".head"):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                awq = AWQLinear.from_linear(m, group_size=group_size)
                setattr(parent, parts[-1], awq)
                replaced += 1
            except Exception as e:
                print(f"  warn: failed to replace {name}: {e}")
    return replaced


def estimate_effective_bits(model: nn.Module) -> float:
    """Effective bits across SSM/attention/memory params, accounting for
    Quantized*Wrapper containers (which contain a SimpleSSM with FP16
    weights but are evaluated at 2-bit fake quantization)."""
    from synapnet_edge.quantization.ssm_quantizer import QuantizedSSMWrapper
    visited: set[int] = set()
    total_bits = 0
    total_params = 0

    # First pass: handle wrappers (they own their children)
    for name, m in model.named_modules():
        if isinstance(m, QuantizedSSMWrapper):
            n = m.ssm.dwconv.weight.numel() + m.ssm.gate.weight.numel()
            total_bits += 2 * n
            total_params += n
            for child in m.modules():
                visited.add(id(child))

    # Second pass: remaining leaf params
    for name, m in model.named_modules():
        if id(m) in visited:
            continue
        if isinstance(m, SymmetricINT8Linear):
            n = m.in_features * m.out_features
            total_bits += 8 * n
            total_params += n
        elif isinstance(m, AWQLinear):
            n = m.in_features * m.out_features
            total_bits += 4 * n
            total_params += n
        elif isinstance(m, nn.Linear):
            n = m.in_features * m.out_features
            total_bits += 16 * n
            total_params += n
        elif isinstance(m, nn.Conv1d):
            n = m.weight.numel()
            total_bits += 16 * n
            total_params += n
    return total_bits / max(1, total_params)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_accuracy(model: nn.Module, ctx_len: int, n_samples: int,
                  vocab: int, num_classes: int, device: torch.device,
                  seed: int = 1234) -> dict:
    """Evaluate on per-task NIAH at this context length."""
    model.eval()

    # NIAH single
    niah = NIAHSingle(n_samples=n_samples, seq_len=ctx_len,
                      vocab_size=vocab, num_classes=num_classes, seed=seed)
    loader = DataLoader(niah, batch_size=2, shuffle=False)
    correct = 0; total = 0
    t0 = time.perf_counter()
    for ids, lbl in loader:
        ids, lbl = ids.to(device), lbl.to(device)
        logits = model(ids)[0]
        pred = logits[:, :num_classes].argmax(-1) if logits.dim() == 2 \
               else logits[:, -1, :num_classes].argmax(-1)
        correct += (pred == lbl).sum().item()
        total += lbl.size(0)
    niah_acc = correct / max(1, total)
    niah_time = time.perf_counter() - t0

    # Multi-key NIAH (harder)
    mkey = NIAHMultiKey(n_samples=n_samples, seq_len=ctx_len,
                        vocab_size=vocab, num_classes=num_classes,
                        seed=seed + 1, n_needles=4)
    loader = DataLoader(mkey, batch_size=2, shuffle=False)
    correct = 0; total = 0
    for ids, lbl in loader:
        ids, lbl = ids.to(device), lbl.to(device)
        logits = model(ids)[0]
        pred = logits[:, :num_classes].argmax(-1) if logits.dim() == 2 \
               else logits[:, -1, :num_classes].argmax(-1)
        correct += (pred == lbl).sum().item()
        total += lbl.size(0)
    mkey_acc = correct / max(1, total)

    # Multi-task curriculum (average)
    multi = MultiTaskCurriculum(n_samples=n_samples * 2, seq_len=ctx_len,
                                 vocab_size=vocab, num_classes=num_classes,
                                 seed=seed + 2)
    loader = DataLoader(multi, batch_size=2, shuffle=False)
    correct = 0; total = 0
    per_task_c = {}; per_task_t = {}
    for ids, lbl, tid in loader:
        ids, lbl = ids.to(device), lbl.to(device)
        logits = model(ids)[0]
        pred = logits[:, :num_classes].argmax(-1) if logits.dim() == 2 \
               else logits[:, -1, :num_classes].argmax(-1)
        correct += (pred == lbl).sum().item()
        total += lbl.size(0)
        for t in tid.unique().tolist():
            mask = (tid == t)
            per_task_c[t] = per_task_c.get(t, 0) + (pred[mask.to(device)] == lbl[mask.to(device)]).sum().item()
            per_task_t[t] = per_task_t.get(t, 0) + int(mask.sum().item())
    multi_acc = correct / max(1, total)
    per_task = {int(t): per_task_c[t] / max(1, per_task_t[t]) for t in per_task_t}

    return {
        "niah_single_acc": niah_acc,
        "niah_multi_key_acc": mkey_acc,
        "multi_task_acc": multi_acc,
        "per_task": per_task,
        "eval_time_s": niah_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_cajq_long_context.json")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--device", default="mps")
    p.add_argument("--variants", nargs="+",
                   default=["fp16", "int8_uniform", "int4_uniform", "cajq"])
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["model_cfg"]
    vocab = config["vocab_size"]
    num_classes = config["num_classes"]

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Model cfg: dim={config['dim']} depth={config['depth']} "
          f"max_len={config['max_len']} params={ckpt['n_params']:,}")

    def fresh_model():
        cfg = SynapNetEdgeConfig(**config)
        m = SynapNetEdge(cfg)
        m.load_state_dict(ckpt["model_state"])
        m.to(device).eval()
        return m

    # Calibration data (used for CAJQ AWQ calibration)
    def calib_loader():
        ds = MultiTaskCurriculum(n_samples=32, seq_len=512,
                                  vocab_size=vocab, num_classes=num_classes,
                                  seed=999)
        return DataLoader(ds, batch_size=4, shuffle=False)

    results = {"config": config, "variants": {}}

    for variant in args.variants:
        print(f"\n{'='*60}\n  Variant: {variant}\n{'='*60}")
        model = fresh_model()

        if variant == "fp16":
            pass   # no quantization
        elif variant == "int8_uniform":
            n = apply_uniform_int8(model)
            print(f"  Applied INT8 to {n} Linear layers")
        elif variant == "int4_uniform":
            n = apply_uniform_int4(model, group_size=64)
            print(f"  Applied INT4 (AWQ-style) to {n} Linear layers")
        elif variant == "cajq":
            cajq_cfg = CAJQConfig(n_calib_batches=8, device=args.device,
                                   attn_group_size=64)
            apply_cajq(model, cajq_cfg, calib_loader=calib_loader(), mode="ptq")

        eff_bits = estimate_effective_bits(model)
        print(f"  Effective bits (linear-like layers): {eff_bits:.2f}")

        variant_results = {"effective_bits": eff_bits, "per_ctx": {}}
        for ctx in args.context_lengths:
            print(f"  Evaluating @ ctx={ctx}...", flush=True)
            try:
                res = eval_accuracy(model, ctx, args.n_samples,
                                    vocab, num_classes, device)
                print(f"    NIAH-single={res['niah_single_acc']:.3f} "
                      f"MultiKey={res['niah_multi_key_acc']:.3f} "
                      f"MultiTask={res['multi_task_acc']:.3f} "
                      f"({res['eval_time_s']:.1f}s)")
                variant_results["per_ctx"][ctx] = res
            except Exception as e:
                print(f"    FAILED: {e}")
                variant_results["per_ctx"][ctx] = {"error": str(e)}

        results["variants"][variant] = variant_results
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(out_path, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"\n[exp_cajq] Saved results to {out_path}")

    # Print summary table
    print("\n=== SUMMARY (NIAH-single accuracy by ctx) ===")
    print(f"{'variant':<16} | {'bits':>4} | " +
          " | ".join(f"{c:>5}" for c in args.context_lengths))
    print("-" * (24 + 8 * len(args.context_lengths)))
    for v, vr in results["variants"].items():
        row = f"{v:<16} | {vr['effective_bits']:>4.1f} | "
        for c in args.context_lengths:
            r = vr["per_ctx"].get(c, {})
            if "niah_single_acc" in r:
                row += f"{r['niah_single_acc']:.3f} "
            else:
                row += "  -   "
        print(row)


if __name__ == "__main__":
    main()
