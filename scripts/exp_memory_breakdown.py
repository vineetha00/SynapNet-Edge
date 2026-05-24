"""Detailed memory + energy breakdown.

Measures, at multiple context lengths:
  1. Parameter memory     (weights, by quantization tier)
  2. Activation memory    (per-layer forward-pass peak, via PyTorch hooks)
  3. Episodic-store memory (per-token growth during streaming)
  4. KV-cache analog footprint (the episodic memory IS the KV-cache analog)
  5. Runtime memory       (process RSS)
  6. Energy per token     (macOS powermetrics on M-series, with fallback)
  7. Sustained throughput (60s continuous workload — thermal stress)

Output: results/scaled/exp_memory_breakdown.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


# ---------------------------------------------------------------------------
# 1. Activation memory profiling (per-layer)
# ---------------------------------------------------------------------------

class ActivationProfiler:
    """Hooks every leaf module and records peak activation memory."""
    def __init__(self, model):
        self.model = model
        self._activations: dict[str, int] = {}
        self._hooks = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:   # leaf
                def _hook(m, inp, out, n=name):
                    bytes_used = 0
                    if isinstance(out, torch.Tensor):
                        bytes_used = out.numel() * out.element_size()
                    elif isinstance(out, (list, tuple)):
                        for x in out:
                            if isinstance(x, torch.Tensor):
                                bytes_used += x.numel() * x.element_size()
                    self._activations[n] = max(
                        self._activations.get(n, 0), bytes_used
                    )
                self._hooks.append(module.register_forward_hook(_hook))
        return self

    def __exit__(self, *args):
        for h in self._hooks: h.remove()

    def report(self) -> dict:
        return dict(self._activations)


@torch.no_grad()
def profile_activations(model, ctx_len, vocab, device):
    ids = torch.randint(0, vocab, (1, ctx_len), device=device)
    with ActivationProfiler(model) as prof:
        model(ids)
    acts = prof.report()
    total_bytes = sum(acts.values())
    # Group by block-level
    by_block: dict = {}
    for name, bytes_used in acts.items():
        key = ".".join(name.split(".")[:2]) if "blocks." in name else "other"
        by_block[key] = by_block.get(key, 0) + bytes_used
    return {
        "ctx_len": ctx_len,
        "total_activation_bytes": total_bytes,
        "total_activation_mb": total_bytes / 1024 / 1024,
        "per_module": acts,
        "by_block_mb": {k: v / 1024 / 1024 for k, v in by_block.items()},
    }


# ---------------------------------------------------------------------------
# 2. Episodic-store growth during streaming
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_episodic_growth(model, total_tokens, chunk_size, budget, vocab,
                             device, policy="baee_salience"):
    from synapnet_edge.memory.kv_cache_policies import PolicyContext, evict

    model.eval()
    n_layers = len(model.blocks)
    D = model.cfg.dim
    layer_mem = [[] for _ in range(n_layers)]

    growth_samples = []
    rss_samples = []
    n_chunks = total_tokens // chunk_size
    rss_samples.append((0, rss_mb()))

    for ci in range(n_chunks):
        ids = torch.randint(0, vocab, (1, chunk_size), device=device)
        mem_banks = []
        for li in range(n_layers):
            if layer_mem[li]:
                bank = torch.stack([e[0] for e in layer_mem[li]], dim=0).unsqueeze(0)
            else:
                bank = torch.zeros(1, 1, D, device=device)
            mem_banks.append(bank)
        outputs = model(ids, external_mems=mem_banks)
        mems = outputs[2]
        for li in range(n_layers):
            new_bank = mems[li][0]
            S = new_bank.size(0)
            for s in range(S):
                layer_mem[li].append([new_bank[s].clone(), 0.5, 0, 0.0, 0, 0.0])
            if len(layer_mem[li]) > budget:
                ctx = PolicyContext(entries=layer_mem[li], budget=budget,
                                    layer_idx=li, n_layers=n_layers)
                keep = evict(policy, ctx)
                layer_mem[li] = [layer_mem[li][k] for k in keep]
            for e in layer_mem[li]:
                e[2] += 1

        total_entries = sum(len(layer_mem[li]) for li in range(n_layers))
        store_bytes = total_entries * D * 2  # FP16-equivalent
        growth_samples.append({
            "tokens": (ci + 1) * chunk_size,
            "entries": total_entries,
            "store_bytes": store_bytes,
            "store_mb": store_bytes / 1024 / 1024,
        })
        if (ci + 1) % max(1, n_chunks // 10) == 0:
            rss_samples.append(((ci + 1) * chunk_size, rss_mb()))

    return {
        "policy": policy, "total_tokens": total_tokens, "budget": budget,
        "growth": growth_samples,
        "rss": rss_samples,
        "asymptotic_store_mb": growth_samples[-1]["store_mb"] if growth_samples else 0,
    }


# ---------------------------------------------------------------------------
# 3. Energy per token via macOS powermetrics (if available)
# ---------------------------------------------------------------------------

def measure_energy_powermetrics(model, ctx_len, vocab, device, duration_s=10.0):
    """Run continuous inference and sample power via macOS powermetrics.

    Returns mean power (W), tokens processed, and J/token.  Returns None
    on platforms where powermetrics is unavailable or requires sudo.
    """
    if sys.platform != "darwin":
        return None
    try:
        # Check if we can run powermetrics (may need sudo)
        subprocess.check_call(["which", "powermetrics"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except Exception:
        return None

    # We'll spawn powermetrics in background, run inference for duration_s,
    # then read aggregate power from its output.
    import tempfile
    out_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    out_file.close()
    try:
        # `--samplers cpu_power` is the lightweight sampler.  Without sudo,
        # powermetrics will refuse, in which case we fall back to a Joule
        # estimate using rated SoC power × duration.
        proc = subprocess.Popen(
            ["powermetrics", "--samplers", "cpu_power,gpu_power",
             "-i", "1000", "-n", str(int(duration_s)),
             "-o", out_file.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (PermissionError, FileNotFoundError):
        proc = None

    # Continuous inference for duration_s
    ids = torch.randint(0, vocab, (1, ctx_len), device=device)
    tokens_processed = 0
    if device.type == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        with torch.no_grad():
            model(ids)
        if device.type == "mps": torch.mps.synchronize()
        tokens_processed += ctx_len
    elapsed = time.perf_counter() - t0

    mean_power_w = None
    if proc is not None:
        proc.wait()
        try:
            with open(out_file.name) as f:
                text = f.read()
            # Parse "Combined Power" or "Package Power" lines
            import re
            matches = re.findall(r"(?:Combined Power|Package Power).*?(\d+(?:\.\d+)?)\s*mW",
                                  text, re.IGNORECASE)
            if matches:
                vals = [float(m) for m in matches]
                mean_power_w = sum(vals) / len(vals) / 1000  # mW → W
            os.unlink(out_file.name)
        except Exception:
            pass

    # Fallback: estimate from M-series typical SoC power
    if mean_power_w is None:
        # M1/M2/M3 base ≈ 15W under MPS GPU load; rough estimate
        mean_power_w = 15.0
        used_fallback = True
    else:
        used_fallback = False

    energy_j = mean_power_w * elapsed
    j_per_token = energy_j / max(1, tokens_processed)

    return {
        "ctx_len": ctx_len, "duration_s": elapsed,
        "tokens_processed": tokens_processed,
        "tokens_per_second": tokens_processed / elapsed,
        "mean_power_w": mean_power_w,
        "energy_j": energy_j,
        "energy_uj_per_token": j_per_token * 1e6,
        "used_fallback_power_estimate": used_fallback,
    }


# ---------------------------------------------------------------------------
# 4. Sustained throughput (thermal stress)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sustained_throughput(model, ctx_len, vocab, device, duration_s=60.0):
    """Run continuous inference for `duration_s` and sample throughput windows."""
    model.eval()
    ids = torch.randint(0, vocab, (1, ctx_len), device=device)

    def _sync():
        if device.type == "mps": torch.mps.synchronize()
        elif device.type == "cuda": torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        model(ids); _sync()

    windows = []
    window_size_s = 5.0
    t_start = time.perf_counter()
    t_window = t_start
    tokens_window = 0
    total_tokens = 0

    while time.perf_counter() - t_start < duration_s:
        model(ids); _sync()
        tokens_window += ctx_len
        total_tokens += ctx_len
        if time.perf_counter() - t_window >= window_size_s:
            window_elapsed = time.perf_counter() - t_window
            windows.append({
                "t_seconds": time.perf_counter() - t_start,
                "tokens_per_second": tokens_window / window_elapsed,
                "rss_mb": rss_mb(),
            })
            tokens_window = 0
            t_window = time.perf_counter()

    if not windows:
        return {"windows": [], "first_throughput": 0, "last_throughput": 0,
                "degradation_pct": 0}

    first = windows[0]["tokens_per_second"]
    last = windows[-1]["tokens_per_second"]
    return {
        "duration_s": duration_s,
        "n_windows": len(windows),
        "windows": windows,
        "first_throughput": first,
        "last_throughput": last,
        "degradation_pct": (first - last) / first * 100 if first > 0 else 0,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_memory_breakdown.json")
    p.add_argument("--device", default="mps")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096])
    p.add_argument("--energy-duration-s", type=float, default=10.0)
    p.add_argument("--sustained-duration-s", type=float, default=60.0)
    p.add_argument("--episodic-tokens", type=int, default=8192)
    args = p.parse_args()

    from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = SynapNetEdgeConfig(**ckpt["model_cfg"])
    model = SynapNetEdge(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[memory_breakdown] params={ckpt['n_params']:,}")

    results = {"config": vars(args), "n_params": ckpt["n_params"]}

    # 1. Activation memory at each context length
    print("\n=== 1. Activation memory ===")
    results["activation"] = []
    for ctx in args.context_lengths:
        r = profile_activations(model, ctx, ckpt["model_cfg"]["vocab_size"], device)
        print(f"  ctx={ctx}: total_activation={r['total_activation_mb']:.2f} MB")
        results["activation"].append(r)

    # 2. Episodic-store growth
    print("\n=== 2. Episodic-store growth ===")
    results["episodic_growth"] = []
    for policy in ["baee_salience", "fifo"]:
        r = measure_episodic_growth(
            model, total_tokens=args.episodic_tokens, chunk_size=512,
            budget=32, vocab=ckpt["model_cfg"]["vocab_size"],
            device=device, policy=policy,
        )
        print(f"  policy={policy}: asymptotic store={r['asymptotic_store_mb']:.2f} MB "
              f"after {r['total_tokens']} tokens")
        results["episodic_growth"].append(r)

    # 3. Energy per token
    print("\n=== 3. Energy / power per token ===")
    results["energy"] = []
    for ctx in [512, 2048]:
        r = measure_energy_powermetrics(
            model, ctx, ckpt["model_cfg"]["vocab_size"], device,
            duration_s=args.energy_duration_s,
        )
        if r:
            print(f"  ctx={ctx}: {r['mean_power_w']:.1f} W mean, "
                  f"{r['energy_uj_per_token']:.1f} μJ/token "
                  f"{'(fallback estimate)' if r['used_fallback_power_estimate'] else '(measured)'}")
            results["energy"].append(r)
        else:
            print(f"  ctx={ctx}: powermetrics unavailable")

    # 4. Sustained throughput
    print(f"\n=== 4. Sustained throughput ({args.sustained_duration_s}s) ===")
    results["sustained"] = sustained_throughput(
        model, ctx_len=1024, vocab=ckpt["model_cfg"]["vocab_size"],
        device=device, duration_s=args.sustained_duration_s,
    )
    print(f"  first={results['sustained']['first_throughput']:.0f} tok/s, "
          f"last={results['sustained']['last_throughput']:.0f} tok/s, "
          f"degradation={results['sustained']['degradation_pct']:.1f}%")

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
