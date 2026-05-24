"""Experiment 3 — Consumer hardware benchmarking.

Measures latency, throughput, and peak memory for SynapNet-Edge variants
on two hardware tiers:
  - Apple Silicon (MPS)        — high-perf consumer
  - ARM CPU simulation (cpu)   — low-power proxy for Raspberry Pi 5

For each variant, runs full forward passes at multiple context lengths
and reports tokens/sec and peak RSS.

Compares the four quantization variants from Experiment 1.

Output: results/scaled/exp_hardware.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum
from torch.utils.data import DataLoader

# Reuse quantization apply functions from exp 1
sys.path.insert(0, str(Path(__file__).parent))
from exp_cajq_long_context import (
    apply_uniform_int8, apply_uniform_int4,
    estimate_effective_bits,
)


def get_peak_mem_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


@torch.no_grad()
def benchmark(model: nn.Module, ctx_len: int, vocab: int, device: torch.device,
              n_warmup: int = 2, n_measure: int = 5) -> dict:
    model.eval()
    ids = torch.randint(0, vocab, (1, ctx_len), device=device)

    def _sync():
        if device.type == "cuda": torch.cuda.synchronize()
        elif device.type == "mps": torch.mps.synchronize()

    gc.collect()
    start_mem = get_peak_mem_mb()

    for _ in range(n_warmup):
        model(ids)
        _sync()

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model(ids)
        _sync()
        times.append(time.perf_counter() - t0)

    end_mem = get_peak_mem_mb()
    times.sort()
    median_s = times[len(times) // 2]
    tok_per_s = ctx_len / median_s

    return {
        "ctx_len": ctx_len,
        "median_latency_ms": median_s * 1000,
        "tokens_per_second": tok_per_s,
        "peak_mem_mb": end_mem,
        "delta_mem_mb": max(0, end_mem - start_mem),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_hardware.json")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--devices", nargs="+", default=["mps", "cpu"],
                   help="Hardware tiers: mps=Apple Silicon, cpu=ARM-like CPU proxy")
    p.add_argument("--variants", nargs="+",
                   default=["fp16", "int4_uniform", "cajq"])
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["model_cfg"]
    vocab = config["vocab_size"]
    num_classes = config["num_classes"]

    results = {"config": config, "platforms": {}}

    for device_str in args.devices:
        if device_str == "mps" and not torch.backends.mps.is_available():
            print(f"  [{device_str}] not available — skipping")
            continue
        device = torch.device(device_str)
        platform_key = "apple_silicon_mps" if device_str == "mps" else "arm_cpu_proxy"
        print(f"\n{'='*60}")
        print(f"  Platform: {platform_key} (device={device_str})")
        print(f"{'='*60}")

        platform_results = {"platform": platform_key, "variants": {}}

        for variant in args.variants:
            print(f"\n  Variant: {variant}")
            # Build fresh
            cfg = SynapNetEdgeConfig(**config)
            model = SynapNetEdge(cfg)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)

            # Apply quantization
            if variant == "int4_uniform":
                n = apply_uniform_int4(model, group_size=64)
                model.to(device)
                print(f"    Applied INT4 to {n} layers")
            elif variant == "cajq":
                ds = MultiTaskCurriculum(n_samples=16, seq_len=512,
                                          vocab_size=vocab, num_classes=num_classes,
                                          seed=999)
                calib = DataLoader(ds, batch_size=4, shuffle=False)
                cajq_cfg = CAJQConfig(n_calib_batches=4, device=device_str,
                                       attn_group_size=64)
                apply_cajq(model, cajq_cfg, calib_loader=calib, mode="ptq")
                model.to(device)

            eff_bits = estimate_effective_bits(model)
            variant_results = {"effective_bits": eff_bits, "per_ctx": []}

            for ctx in args.context_lengths:
                # On CPU, skip longest contexts to keep runtime bounded
                if device_str == "cpu" and ctx > 4096:
                    print(f"    ctx={ctx}: SKIP (too slow on CPU proxy)")
                    continue
                try:
                    res = benchmark(model, ctx, vocab, device,
                                    n_warmup=2, n_measure=3 if device_str == "cpu" else 5)
                    print(f"    ctx={ctx:5d}: {res['tokens_per_second']:>7.1f} tok/s, "
                          f"{res['median_latency_ms']:>6.1f}ms, "
                          f"{res['peak_mem_mb']:>5.0f}MB")
                    variant_results["per_ctx"].append(res)
                except Exception as e:
                    print(f"    ctx={ctx}: FAILED ({type(e).__name__}: {e})")

            platform_results["variants"][variant] = variant_results
            del model
            gc.collect()
            if device.type == "mps": torch.mps.empty_cache()

        results["platforms"][platform_key] = platform_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"\n[exp_hardware] Saved to {args.output}")

    # Summary table
    print(f"\n=== HARDWARE SUMMARY ===")
    for plat, pr in results["platforms"].items():
        print(f"\nPlatform: {plat}")
        for variant, vr in pr["variants"].items():
            print(f"  {variant:<15} (bits={vr['effective_bits']:.1f}):")
            for r in vr["per_ctx"]:
                print(f"    ctx={r['ctx_len']:5d}: "
                      f"{r['tokens_per_second']:>7.1f} tok/s, "
                      f"{r['peak_mem_mb']:>5.0f}MB")


if __name__ == "__main__":
    main()
