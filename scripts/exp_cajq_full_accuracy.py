"""Accuracy re-check for CAJQ-Full against uniform quantization and a fine-tune control.

Question this answers: on the trained SynapNet-Edge checkpoint, how much task
accuracy does each quantization scheme cost, and is CAJQ-Full worth using over
plain uniform INT4?

Design (all variants share the same checkpoint, eval data and fine-tune data):

  Post-training quantization only (no extra training):
    fp16, int8_uniform, int4_uniform, cajq_ptq,
    cajq_full_int8_ptq, cajq_full_int4_ptq

  Same short fine-tune applied after quantization (identical recipe for all):
    fp16_ft            <- control: the full-precision model gets the same steps
    int4_uniform_ft, cajq_qat, cajq_full_int8_qat, cajq_full_int4_qat

The fp16_ft control matters. In the original exp_cajq_qat_multiseed run,
"cajq_qat" beat fp16, but the quantized weights are frozen buffers during QAT:
the gradient steps mostly update the layers CAJQ left in fp16. Without a
full-precision model fine-tuned the same way, that gain can't be attributed to
quantization.

Every variant answers the exact same questions (eval data is seeded per run
seed, not per variant), and per-sample correctness is stored, so differences
are reported as paired bootstrap 95% confidence intervals rather than as two
independent means.

Output: results/scaled/exp_cajq_full_accuracy.json (written after each variant,
so a partial run still leaves usable results).
"""
from __future__ import annotations

import argparse
import json
import random
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
from synapnet_edge.quantization.ssm_quantizer import SSMQuantizer
from exp_cajq_long_context import apply_uniform_int8, apply_uniform_int4, estimate_effective_bits
from cajq_full import quantize_remaining_linears

PTQ_VARIANTS = ["fp16", "int8_uniform", "int4_uniform", "cajq_ptq",
                "cajq_full_int8_ptq", "cajq_full_int4_ptq"]
FT_VARIANTS = ["fp16_ft", "int4_uniform_ft", "cajq_qat",
               "cajq_full_int8_qat", "cajq_full_int4_qat"]
TASKS = ["niah_single", "niah_multi_key", "multi_task"]

# Pairs reported as paired differences (A minus B).
COMPARISONS = [
    ("int8_uniform", "fp16"),
    ("int4_uniform", "fp16"),
    ("cajq_ptq", "fp16"),
    ("cajq_full_int8_ptq", "fp16"),
    ("cajq_full_int4_ptq", "fp16"),
    ("cajq_full_int8_ptq", "int8_uniform"),
    ("cajq_full_int4_ptq", "int4_uniform"),
    ("fp16_ft", "fp16"),
    ("cajq_qat", "fp16_ft"),
    ("int4_uniform_ft", "fp16_ft"),
    ("cajq_full_int8_qat", "fp16_ft"),
    ("cajq_full_int4_qat", "fp16_ft"),
    ("cajq_full_int4_qat", "int4_uniform_ft"),
]


# ---------------------------------------------------------------------------
# Quantize + (optionally) fine-tune
# ---------------------------------------------------------------------------

def build_model(ckpt: dict, config: dict, device: torch.device) -> nn.Module:
    model = SynapNetEdge(SynapNetEdgeConfig(**config))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def quantize(model: nn.Module, variant: str, seed: int, vocab: int, num_classes: int,
             device: torch.device) -> nn.Module:
    base = variant.removesuffix("_ft").removesuffix("_ptq").removesuffix("_qat")
    if base == "fp16":
        return model
    if base == "int8_uniform":
        apply_uniform_int8(model)
        return model.to(device)
    if base == "int4_uniform":
        apply_uniform_int4(model, group_size=64)
        return model.to(device)
    if base.startswith("cajq"):
        # Same seed-specific calibration recipe as exp_cajq_qat_multiseed.py
        calib = DataLoader(
            MultiTaskCurriculum(n_samples=32, seq_len=512, vocab_size=vocab,
                                num_classes=num_classes, seed=seed + 1000),
            batch_size=4, shuffle=False)
        cajq_cfg = CAJQConfig(n_calib_batches=8, device=device.type, attn_group_size=64)
        apply_cajq(model, cajq_cfg, calib_loader=calib, mode="ptq")
        model.to(device)
        if base == "cajq_full_int8":
            n = quantize_remaining_linears(model, bits=8, group_size=64)
            print(f"    [cajq-full] {n} remaining Linear layers -> INT8")
        elif base == "cajq_full_int4":
            n = quantize_remaining_linears(model, bits=4, group_size=64)
            print(f"    [cajq-full] {n} remaining Linear layers -> INT4")
        elif base != "cajq":
            raise ValueError(f"unknown variant {variant!r}")
        return model.to(device)
    raise ValueError(f"unknown variant {variant!r}")


def fine_tune(model: nn.Module, seed: int, vocab: int, num_classes: int,
              device: torch.device, n_steps: int, ctx: int, batch: int,
              lr_model: float = 1e-4, lr_step: float = 1e-2,
              lambda_reg: float = 1e-3, grad_clip: float = 1.0) -> list[dict]:
    """The exp_cajq_qat_multiseed.py QAT recipe, applied identically to every
    fine-tuned variant. Quantized weights are buffers and stay frozen; step
    sizes (SSM 2-bit only) get their own high-LR group when present."""
    step_params = list(SSMQuantizer.step_parameters(model))
    step_ids = {id(p) for p in step_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in step_ids]
    groups = [{"params": other, "lr": lr_model, "weight_decay": 0.01}]
    if step_params:
        groups.insert(0, {"params": step_params, "lr": lr_step, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    train = DataLoader(
        MultiTaskCurriculum(n_samples=n_steps * batch, seq_len=ctx, vocab_size=vocab,
                            num_classes=num_classes, seed=seed + 2000),
        batch_size=batch, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    n_trainable = sum(p.numel() for p in other) + sum(p.numel() for p in step_params)
    print(f"    fine-tune: {n_steps} steps, {n_trainable:,} trainable params")

    history, step = [], 0
    model.train()
    it = iter(train)
    while step < n_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(train)
            b = next(it)
        ids, lbl = b[0].to(device), b[1].to(device)
        opt.zero_grad()
        logits = model(ids)[0]
        pred = logits[:, -1, :num_classes] if logits.dim() == 3 else logits[:, :num_classes]
        task_loss = F.cross_entropy(pred, lbl)
        reg = SSMQuantizer.collect_quantization_loss(model)
        (task_loss + lambda_reg * reg).backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"      step {step:4d}: task_loss={task_loss.item():.4f}")
            history.append({"step": step, "task_loss": float(task_loss.detach())})
        step += 1
    model.eval()
    return history


# ---------------------------------------------------------------------------
# Eval with per-sample correctness
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_correct(model: nn.Module, ctx: int, n: int, vocab: int, num_classes: int,
                 device: torch.device, seed: int) -> dict[str, list[int]]:
    """Same task factories and seeds as exp_cajq_qat_multiseed.eval_at_context,
    but returns a 0/1 list per task so variants can be compared sample-by-sample."""
    model.eval()
    out = {}
    for name, ds in [
        ("niah_single", NIAHSingle(n, ctx, vocab, num_classes, seed)),
        ("niah_multi_key", NIAHMultiKey(n, ctx, vocab, num_classes, seed + 1, n_needles=4)),
        ("multi_task", MultiTaskCurriculum(n_samples=n * 2, seq_len=ctx, vocab_size=vocab,
                                           num_classes=num_classes, seed=seed + 2)),
    ]:
        hits: list[int] = []
        for b in DataLoader(ds, batch_size=2, shuffle=False):
            ids, lbl = b[0].to(device), b[1].to(device)
            logits = model(ids)[0]
            pred = (logits[:, -1, :num_classes] if logits.dim() == 3
                    else logits[:, :num_classes]).argmax(-1)
            hits.extend((pred == lbl).long().tolist())
        out[name] = hits
    return out


def paired_bootstrap(a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0) -> dict:
    assert len(a) == len(b), "paired comparison needs identical eval sets"
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"diff": sum(d) / n, "ci95": [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]],
            "n": n, "a_only": sum(1 for x in d if x > 0), "b_only": sum(1 for x in d if x < 0)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summarize(results: dict, seeds: list[int], ctxs: list[int], variants: list[str]) -> dict:
    """Pool per-sample correctness across seeds (same questions for every variant)."""
    pooled = {}
    for v in variants:
        if not all(str(s) in results["runs"] and v in results["runs"][str(s)] for s in seeds):
            continue
        pooled[v] = {}
        for c in ctxs:
            pooled[v][str(c)] = {
                t: sum((results["runs"][str(s)][v]["correct"][str(c)][t] for s in seeds), [])
                for t in TASKS}
    acc = {v: {c: {t: sum(x) / len(x) for t, x in tasks.items()} for c, tasks in per.items()}
           for v, per in pooled.items()}
    comps = {}
    for a, b in COMPARISONS:
        if a in pooled and b in pooled:
            comps[f"{a} - {b}"] = {c: {t: paired_bootstrap(pooled[a][c][t], pooled[b][c][t])
                                       for t in TASKS} for c in pooled[a]}
    bits = {v: results["runs"][str(seeds[0])][v]["effective_bits"] for v in pooled}
    return {"accuracy": acc, "effective_bits": bits, "paired_differences": comps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_cajq_full_accuracy.json")
    p.add_argument("--context-lengths", nargs="+", type=int, default=[1024, 2048])
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--ft-steps", type=int, default=200)
    p.add_argument("--ft-ctx", type=int, default=1024)
    p.add_argument("--ft-batch", type=int, default=4)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--variants", nargs="+", default=PTQ_VARIANTS + FT_VARIANTS)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["model_cfg"]
    vocab, num_classes = config["vocab_size"], config["num_classes"]
    max_len = config.get("max_len")
    ctxs = [c for c in args.context_lengths if max_len is None or c <= max_len]
    if len(ctxs) != len(args.context_lengths):
        print(f"[warn] dropping context lengths above max_len={max_len}")
    print(f"[cajq_full_acc] device={device} params={ckpt.get('n_params', '?')} "
          f"seeds={args.seeds} ctx={ctxs} n={args.n_samples} variants={len(args.variants)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"config": {**vars(args), "context_lengths": ctxs, "model_cfg": config},
               "runs": {}}

    for seed in args.seeds:
        results["runs"][str(seed)] = {}
        for variant in args.variants:
            print(f"\n=== seed {seed} | {variant} ===", flush=True)
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            model = build_model(ckpt, config, device)
            model = quantize(model, variant, seed, vocab, num_classes, device)
            hist = None
            if variant in FT_VARIANTS:
                hist = fine_tune(model, seed, vocab, num_classes, device,
                                 args.ft_steps, args.ft_ctx, args.ft_batch)
            prep = time.perf_counter() - t0
            correct = {}
            for c in ctxs:
                correct[str(c)] = eval_correct(model, c, args.n_samples, vocab, num_classes,
                                               device, seed)
                accs = {t: sum(v) / len(v) for t, v in correct[str(c)].items()}
                print(f"    ctx={c}: " + "  ".join(f"{t}={a:.3f}" for t, a in accs.items()),
                      flush=True)
            results["runs"][str(seed)][variant] = {
                "effective_bits": estimate_effective_bits(model),
                "prep_time_s": prep,
                "total_time_s": time.perf_counter() - t0,
                "ft_history": hist,
                "correct": correct,
            }
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
            out_path.write_text(json.dumps(results, indent=1))  # raw results survive a crash

    results["summary"] = summarize(results, args.seeds, ctxs, args.variants)
    out_path.write_text(json.dumps(results, indent=1))
    s = results["summary"]
    print("\n" + "=" * 70 + "\nACCURACY (pooled over seeds; same questions for every variant)")
    for v, per in s["accuracy"].items():
        print(f"  {v:<20} bits={s['effective_bits'][v]:>4.1f}  " +
              "  ".join(f"@{c} NIAH={per[c]['niah_single']:.3f} MT={per[c]['multi_task']:.3f}"
                        for c in per))
    print("\nPAIRED DIFFERENCES (A - B, 95% bootstrap CI) — NIAH-single and multi-task")
    for name, per in s["paired_differences"].items():
        for c, tasks in per.items():
            r1, r2 = tasks["niah_single"], tasks["multi_task"]
            print(f"  {name:<42} @{c}  NIAH {r1['diff']:+.3f} [{r1['ci95'][0]:+.3f},{r1['ci95'][1]:+.3f}]"
                  f"  MT {r2['diff']:+.3f} [{r2['ci95'][0]:+.3f},{r2['ci95'][1]:+.3f}]")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
