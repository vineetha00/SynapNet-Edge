"""Experiment 4 — Comprehensive deployment metrics across 3 hardware tiers.

Hardware tiers:
  1. Apple Silicon (MPS)            — high-perf consumer
  2. Multi-thread CPU (no MPS)      — desktop-class CPU
  3. Single-thread CPU              — Raspberry Pi 5 proxy
                                      (Pi 5 has 4 ARM Cortex-A76 cores @ 2.4 GHz;
                                       constraining to 1 thread approximates
                                       per-core throughput within ~2× factor)

For each (variant, tier) we measure:
  - Storage on disk (packed weights + scales, in MB)
  - Peak inference RAM (resident set size delta)
  - Median latency (ms) at multiple context lengths
  - Throughput (tokens/sec)
  - Accuracy (from cajq experiments, joined in plot stage)

Output: results/scaled/exp_deployment.json
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum
from torch.utils.data import DataLoader

from exp_cajq_long_context import (
    apply_uniform_int8, apply_uniform_int4, estimate_effective_bits,
)


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def estimate_storage_bytes(model: nn.Module) -> dict:
    """Estimate on-disk storage assuming bit-packed weights."""
    from synapnet_edge.quantization.attention_quantizer import AWQLinear
    from synapnet_edge.quantization.ssm_quantizer import QuantizedSSMWrapper
    from exp_cajq_long_context import SymmetricINT8Linear

    total = 0
    breakdown = {"ssm_2bit": 0, "attn_int4": 0, "int8": 0, "fp16_default": 0}
    visited = set()

    for name, m in model.named_modules():
        if isinstance(m, QuantizedSSMWrapper):
            n_conv = m.ssm.dwconv.weight.numel()
            n_gate = m.ssm.gate.weight.numel()
            # 2-bit packed: 1 byte per 4 weights + 4 bytes per channel for scale
            bytes_pack = (n_conv + n_gate) / 4
            n_channels = m.ssm.dwconv.weight.shape[0] + m.ssm.gate.weight.shape[0]
            bytes_scale = n_channels * 4
            total += int(bytes_pack + bytes_scale)
            breakdown["ssm_2bit"] += int(bytes_pack + bytes_scale)
            for c in m.modules():
                visited.add(id(c))

    for name, m in model.named_modules():
        if id(m) in visited:
            continue
        if isinstance(m, AWQLinear):
            n = m.in_features * m.out_features
            # INT4 packed: 0.5 byte per weight, per-group scale + zero (8 bytes each)
            n_groups = m.scales.numel()
            bytes_w = n // 2
            bytes_meta = n_groups * 2 * 4
            # Salient channels FP16: 2 bytes per weight
            bytes_salient = 0
            if m.salient_fp16 is not None:
                bytes_salient = m.salient_fp16.numel() * 2
            total += bytes_w + bytes_meta + bytes_salient
            breakdown["attn_int4"] += bytes_w + bytes_meta + bytes_salient
        elif isinstance(m, SymmetricINT8Linear):
            n = m.in_features * m.out_features
            total += n + 4  # int8 + per-tensor scale
            breakdown["int8"] += n + 4
        elif isinstance(m, nn.Linear):
            n = m.in_features * m.out_features
            if m.bias is not None:
                n += m.out_features
            total += n * 2  # FP16
            breakdown["fp16_default"] += n * 2
        elif isinstance(m, nn.Conv1d):
            n = m.weight.numel()
            if m.bias is not None:
                n += m.bias.numel()
            total += n * 2
            breakdown["fp16_default"] += n * 2
        elif isinstance(m, nn.Embedding):
            n = m.weight.numel()
            total += n * 2
            breakdown["fp16_default"] += n * 2

    # Token + pos embeddings + final head not necessarily in modules iteration;
    # add a fallback for any nn.Parameter not seen
    return {"total_bytes": total, "total_mb": total / 1024 / 1024,
            "breakdown_bytes": breakdown,
            "breakdown_mb": {k: v / 1024 / 1024 for k, v in breakdown.items()}}


@torch.no_grad()
def benchmark_latency(model, ctx_len, vocab, device, n_warmup=2, n_measure=5):
    model.eval()
    ids = torch.randint(0, vocab, (1, ctx_len), device=device)

    def _sync():
        if device.type == "cuda": torch.cuda.synchronize()
        elif device.type == "mps": torch.mps.synchronize()

    gc.collect()
    mem_start = rss_mb()

    for _ in range(n_warmup):
        model(ids); _sync()

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model(ids); _sync()
        times.append(time.perf_counter() - t0)

    mem_end = rss_mb()
    times.sort()
    median = times[len(times) // 2]
    return {
        "ctx_len": ctx_len,
        "median_latency_ms": median * 1000,
        "tokens_per_second": ctx_len / median,
        "rss_start_mb": mem_start,
        "rss_end_mb": mem_end,
        "delta_rss_mb": max(0, mem_end - mem_start),
    }


def build_variant(variant: str, ckpt: dict, config: dict, device: torch.device):
    cfg = SynapNetEdgeConfig(**config)
    model = SynapNetEdge(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    if variant == "fp16":
        pass
    elif variant == "int8_uniform":
        apply_uniform_int8(model)
        model.to(device)
    elif variant == "int4_uniform":
        apply_uniform_int4(model, group_size=64)
        model.to(device)
    elif variant == "cajq":
        calib_ds = MultiTaskCurriculum(
            n_samples=32, seq_len=512,
            vocab_size=config["vocab_size"],
            num_classes=config["num_classes"], seed=999,
        )
        calib = DataLoader(calib_ds, batch_size=4, shuffle=False)
        cajq_cfg = CAJQConfig(n_calib_batches=8,
                              device=str(device).replace("cuda:0", "cuda"),
                              attn_group_size=64)
        apply_cajq(model, cajq_cfg, calib_loader=calib, mode="ptq")
        model.to(device)
    return model


def configure_thread_tier(tier: str) -> None:
    if tier == "cpu_single":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    elif tier == "cpu_multi":
        # Use all cores
        try:
            n = os.cpu_count() or 4
            torch.set_num_threads(n)
        except RuntimeError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_deployment.json")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--variants", nargs="+",
                   default=["fp16", "int4_uniform", "cajq"])
    p.add_argument("--tiers", nargs="+",
                   default=["apple_silicon_mps", "cpu_multi", "cpu_single"])
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["model_cfg"]
    vocab = config["vocab_size"]

    results = {"config": vars(args), "tiers": {}}

    for tier in args.tiers:
        if tier == "apple_silicon_mps":
            if not torch.backends.mps.is_available():
                print(f"  skip {tier}: MPS unavailable")
                continue
            device = torch.device("mps")
        elif tier == "cpu_multi":
            device = torch.device("cpu")
            configure_thread_tier("cpu_multi")
        elif tier == "cpu_single":
            device = torch.device("cpu")
            configure_thread_tier("cpu_single")
        else:
            print(f"  skip unknown tier {tier}")
            continue

        n_threads = torch.get_num_threads()
        print(f"\n{'='*60}")
        print(f"  Tier: {tier} (device={device}, threads={n_threads})")
        print(f"{'='*60}")

        tier_results = {"device": str(device), "n_threads": n_threads,
                        "variants": {}}

        for variant in args.variants:
            print(f"\n  --- {variant} ---")
            model = build_variant(variant, ckpt, config, device)
            eff_bits = estimate_effective_bits(model)
            storage = estimate_storage_bytes(model)
            print(f"    eff_bits={eff_bits:.1f}  on-disk≈{storage['total_mb']:.2f} MB")
            print(f"    breakdown: " +
                  ", ".join(f"{k}={v:.2f}MB" for k, v in storage["breakdown_mb"].items()
                             if v > 0))

            per_ctx = []
            for ctx in args.context_lengths:
                # Bound runtime on single-thread CPU
                if tier == "cpu_single" and ctx > 2048:
                    print(f"    ctx={ctx}: SKIP (too slow on single-thread CPU)")
                    continue
                if tier == "cpu_multi" and ctx > 4096:
                    print(f"    ctx={ctx}: SKIP")
                    continue
                try:
                    bench = benchmark_latency(
                        model, ctx, vocab, device,
                        n_warmup=2,
                        n_measure=3 if tier == "cpu_single" else 5,
                    )
                    per_ctx.append(bench)
                    print(f"    ctx={ctx:>5}: {bench['tokens_per_second']:>8.1f} tok/s, "
                          f"{bench['median_latency_ms']:>7.1f} ms, "
                          f"RSS={bench['rss_end_mb']:>5.0f}MB")
                except Exception as e:
                    print(f"    ctx={ctx}: FAILED ({e})")

            tier_results["variants"][variant] = {
                "effective_bits": eff_bits,
                "storage": storage,
                "per_ctx": per_ctx,
            }
            del model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

        results["tiers"][tier] = tier_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary table
    print(f"\n=== DEPLOYMENT SUMMARY ===")
    for tier, tr in results["tiers"].items():
        print(f"\n{tier}:")
        for variant, vr in tr["variants"].items():
            tok_at_2048 = next((r["tokens_per_second"] for r in vr["per_ctx"]
                                if r["ctx_len"] == 2048), None)
            print(f"  {variant:<14} bits={vr['effective_bits']:.1f} "
                  f"storage={vr['storage']['total_mb']:.2f}MB "
                  f"@2048: {tok_at_2048:.0f if tok_at_2048 else 0} tok/s"
                  if tok_at_2048 else
                  f"  {variant:<14} bits={vr['effective_bits']:.1f} "
                  f"storage={vr['storage']['total_mb']:.2f}MB")


if __name__ == "__main__":
    main()
