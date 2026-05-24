"""BAEE rigor microbenchmark suite.

Measures:
  1. Asymptotic eviction overhead vs. store size N
     - Time complexity in N: O(N log N) for BAEE (top-K sort), O(N) for FIFO
  2. Per-token amortized inference overhead at fixed budget
  3. Invocation frequency: evictions per chunk vs. chunk size
  4. Memory fragmentation: peak vs. steady-state RSS over a long stream
  5. False-positive analysis: when BAEE keeps a *non-target* entry,
     does the model's downstream attention deweight it correctly?

Output: results/scaled/exp_baee_microbench.json
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
from synapnet_edge.memory.kv_cache_policies import (
    PolicyContext, evict, POLICY_REGISTRY,
)


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


# ---------------------------------------------------------------------------
# 1. Asymptotic eviction overhead vs store size
# ---------------------------------------------------------------------------

def bench_eviction_overhead(store_sizes: list[int],
                             policies: list[str],
                             n_reps: int = 50) -> dict:
    """For each (policy, N), measure pure eviction time (no model forward)."""
    results = {}
    for policy in policies:
        results[policy] = []
        for N in store_sizes:
            # Build synthetic entries
            entries = []
            for i in range(N):
                feat = torch.randn(64)
                salience = float(torch.rand(1).item())
                age = i
                rkey = float(torch.rand(1).item())
                abs_pos = i * 10
                attn_accum = float(torch.rand(1).item())
                entries.append([feat, salience, age, rkey, abs_pos, attn_accum])

            budget = max(1, N // 4)   # 25% retained
            ctx = PolicyContext(entries=entries, budget=budget,
                                layer_idx=0, n_layers=6)

            # Warmup
            for _ in range(3):
                _ = evict(policy, ctx)

            # Measure
            times = []
            for _ in range(n_reps):
                t0 = time.perf_counter()
                _ = evict(policy, ctx)
                times.append(time.perf_counter() - t0)
            times.sort()
            median_us = times[len(times)//2] * 1e6

            results[policy].append({"N": N, "budget": budget,
                                     "median_us": median_us})
            print(f"  [{policy}] N={N:>5}: median={median_us:>7.1f} μs")
    return results


# ---------------------------------------------------------------------------
# 2. Per-token amortized overhead during streaming inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def bench_per_token_overhead(
    model: nn.Module,
    seq_len: int,
    chunk_size: int,
    budget: int,
    policy: str,
    vocab: int,
    device: torch.device,
    n_samples: int = 8,
) -> dict:
    """Compare streaming inference WITHOUT and WITH eviction.

    Returns ms/token for both, and the eviction overhead (delta).
    """
    model.eval()
    n_layers = len(model.blocks)
    D = model.cfg.dim

    def _sync():
        if device.type == "mps": torch.mps.synchronize()
        elif device.type == "cuda": torch.cuda.synchronize()

    forward_only = []
    forward_with_evict = []

    for sample_i in range(n_samples):
        ids = torch.randint(0, vocab, (1, seq_len), device=device)

        # ---- No eviction (no external_mems, just chunked forward) ----
        _sync()
        t0 = time.perf_counter()
        for cs in range(0, seq_len, chunk_size):
            chunk = ids[:, cs: cs + chunk_size]
            _ = model(chunk)
        _sync()
        forward_only.append(time.perf_counter() - t0)

        # ---- With eviction ----
        layer_mem = [[] for _ in range(n_layers)]
        _sync()
        t0 = time.perf_counter()
        for cs in range(0, seq_len, chunk_size):
            chunk = ids[:, cs: cs + chunk_size]

            mem_banks = []
            for li in range(n_layers):
                if layer_mem[li]:
                    bank = torch.stack([e[0] for e in layer_mem[li]],
                                       dim=0).unsqueeze(0)
                else:
                    bank = torch.zeros(1, 1, D, device=device)
                mem_banks.append(bank)

            outputs = model(chunk, external_mems=mem_banks)
            mems = outputs[2]; masks = outputs[1]; topks = outputs[3]

            for li in range(n_layers):
                new_bank = mems[li][0]
                new_sal = masks[li][0]
                top_idx = topks[li][0] if topks[li].numel() > 0 else None
                S = new_bank.size(0)
                for s in range(S):
                    if top_idx is not None and s < top_idx.numel():
                        token_idx = top_idx[s].item()
                        sal = new_sal[token_idx].item() if token_idx < new_sal.numel() else 0.0
                    else:
                        sal = 0.0
                    layer_mem[li].append([new_bank[s].clone(), sal, 0,
                                           0.0, 0, 0.0])

                if len(layer_mem[li]) > budget:
                    ctx = PolicyContext(entries=layer_mem[li], budget=budget,
                                        layer_idx=li, n_layers=n_layers)
                    keep = evict(policy, ctx)
                    layer_mem[li] = [layer_mem[li][k] for k in keep]
                for e in layer_mem[li]:
                    e[2] += 1
        _sync()
        forward_with_evict.append(time.perf_counter() - t0)

    fo_median = sorted(forward_only)[len(forward_only) // 2]
    fe_median = sorted(forward_with_evict)[len(forward_with_evict) // 2]
    overhead_per_tok_us = (fe_median - fo_median) / seq_len * 1e6
    rel_overhead = (fe_median - fo_median) / fo_median * 100

    return {
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "budget": budget,
        "policy": policy,
        "forward_only_ms": fo_median * 1000,
        "forward_with_evict_ms": fe_median * 1000,
        "evict_overhead_per_token_us": overhead_per_tok_us,
        "relative_overhead_pct": rel_overhead,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# 3. Memory fragmentation: RSS over a long stream
# ---------------------------------------------------------------------------

@torch.no_grad()
def bench_fragmentation(model, total_tokens: int, chunk_size: int,
                         budget: int, policy: str, vocab: int,
                         device: torch.device) -> dict:
    """Run a long stream and sample RSS over time to detect leaks/fragmentation."""
    model.eval()
    n_layers = len(model.blocks)
    D = model.cfg.dim
    layer_mem = [[] for _ in range(n_layers)]

    rss_samples = [(0, rss_mb())]
    gc.collect()
    if device.type == "mps": torch.mps.empty_cache()

    n_chunks = total_tokens // chunk_size
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
        mems = outputs[2]; masks = outputs[1]; topks = outputs[3]
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

        # Sample RSS periodically
        if (ci + 1) % max(1, n_chunks // 20) == 0:
            gc.collect()
            rss_samples.append(((ci + 1) * chunk_size, rss_mb()))

    initial = rss_samples[0][1]
    peak = max(s[1] for s in rss_samples)
    final = rss_samples[-1][1]
    return {
        "policy": policy,
        "total_tokens": total_tokens,
        "chunk_size": chunk_size,
        "budget": budget,
        "rss_initial_mb": initial,
        "rss_peak_mb": peak,
        "rss_final_mb": final,
        "leak_delta_mb": final - initial,
        "fragmentation_ratio": (peak - final) / max(1, peak),
        "samples": rss_samples,
    }


# ---------------------------------------------------------------------------
# 4. False-positive analysis (BAEE keeps a non-target)
# ---------------------------------------------------------------------------

@torch.no_grad()
def bench_false_positives(
    model: nn.Module, n_samples: int, seq_len: int, chunk_size: int,
    budget: int, n_needles: int, target_repeat: int,
    vocab: int, num_classes: int, device: torch.device,
) -> dict:
    """For each retained entry post-eviction, is it a target or a distractor?
    Measure precision/recall of BAEE for keeping target needles."""
    from synapnet_edge.data.long_context_tasks import MemoryPressureNIAH
    ds = MemoryPressureNIAH(
        n_samples=n_samples, seq_len=seq_len, vocab_size=vocab,
        num_classes=num_classes, n_needles=n_needles,
        target_repeat=target_repeat,
        target_position_bias="early", seed=12345,
    )
    model.eval()
    n_layers = len(model.blocks)
    D = model.cfg.dim

    policies_to_check = ["baee_salience", "fifo", "h2o", "scissorhands",
                          "snapkv", "pyramidkv", "locret_proxy"]
    stats = {p: {"target_kept": 0, "target_total": 0,
                  "distractor_kept": 0, "distractor_total": 0,
                  "precision_numer": 0, "precision_denom": 0}
             for p in policies_to_check}

    for sample_i in range(min(n_samples, len(ds))):
        ids, _lbl = ds[sample_i]
        ids = ids.unsqueeze(0).to(device)
        meta = ds.metadata(sample_i)
        target_positions = set(meta.get("target_pos", []))

        for policy in policies_to_check:
            layer_mem = [[] for _ in range(n_layers)]
            for cs in range(0, seq_len, chunk_size):
                chunk = ids[:, cs: cs + chunk_size]
                mem_banks = []
                for li in range(n_layers):
                    if layer_mem[li]:
                        bank = torch.stack([e[0] for e in layer_mem[li]],
                                           dim=0).unsqueeze(0)
                    else:
                        bank = torch.zeros(1, 1, D, device=device)
                    mem_banks.append(bank)
                outputs = model(chunk, external_mems=mem_banks)
                mems = outputs[2]; masks = outputs[1]; topks = outputs[3]
                for li in range(n_layers):
                    new_bank = mems[li][0]; new_sal = masks[li][0]
                    top_idx = topks[li][0] if topks[li].numel() > 0 else None
                    S = new_bank.size(0)
                    for s in range(S):
                        if top_idx is not None and s < top_idx.numel():
                            tok_idx = top_idx[s].item()
                            abs_pos = cs + tok_idx
                            sal = new_sal[tok_idx].item() if tok_idx < new_sal.numel() else 0.0
                        else:
                            abs_pos = -1; sal = 0.0
                        layer_mem[li].append([new_bank[s].clone(), sal, 0,
                                               0.0, abs_pos, 0.0])
                    if len(layer_mem[li]) > budget:
                        ctx = PolicyContext(entries=layer_mem[li], budget=budget,
                                             layer_idx=li, n_layers=n_layers)
                        keep = evict(policy, ctx)
                        layer_mem[li] = [layer_mem[li][k] for k in keep]
                    for e in layer_mem[li]:
                        e[2] += 1

            # Count target vs distractor entries in final state
            for li in range(n_layers):
                for e in layer_mem[li]:
                    is_target = e[4] in target_positions or (e[4] + 1) in target_positions
                    if is_target:
                        stats[policy]["target_kept"] += 1
                    else:
                        stats[policy]["distractor_kept"] += 1
                stats[policy]["target_total"] += len(target_positions) * 1   # per-layer
                stats[policy]["distractor_total"] += (n_needles - 1) * 1

    out = {}
    for p in policies_to_check:
        s = stats[p]
        kept = s["target_kept"] + s["distractor_kept"]
        precision = s["target_kept"] / max(1, kept)   # of what we kept, what fraction is target?
        recall = s["target_kept"] / max(1, s["target_total"])
        fp_rate = s["distractor_kept"] / max(1, s["distractor_total"])
        out[p] = {
            "target_kept": s["target_kept"],
            "distractor_kept": s["distractor_kept"],
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fp_rate,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_baee_microbench.json")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = SynapNetEdgeConfig(**ckpt["model_cfg"])
    model = SynapNetEdge(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[microbench] params={ckpt['n_params']:,}, device={device}")

    out = {"config": ckpt["model_cfg"]}

    print("\n=== 1. Asymptotic eviction overhead ===")
    out["eviction_scaling"] = bench_eviction_overhead(
        store_sizes=[16, 64, 256, 1024, 4096, 16384],
        policies=["baee_salience", "fifo", "h2o", "scissorhands",
                  "snapkv", "pyramidkv", "locret_proxy"],
        n_reps=30,
    )

    print("\n=== 2. Per-token amortized overhead ===")
    out["per_token_overhead"] = []
    for policy in ["baee_salience", "fifo", "h2o", "scissorhands",
                    "snapkv", "pyramidkv", "locret_proxy"]:
        print(f"  policy={policy}")
        res = bench_per_token_overhead(
            model, seq_len=2048, chunk_size=512, budget=32,
            policy=policy, vocab=ckpt["model_cfg"]["vocab_size"],
            device=device, n_samples=4,
        )
        print(f"    no_evict={res['forward_only_ms']:.1f}ms "
              f"with_evict={res['forward_with_evict_ms']:.1f}ms "
              f"overhead={res['evict_overhead_per_token_us']:.2f}μs/tok "
              f"({res['relative_overhead_pct']:+.1f}%)")
        out["per_token_overhead"].append(res)

    print("\n=== 3. Memory fragmentation (long stream) ===")
    out["fragmentation"] = []
    for policy in ["baee_salience", "fifo"]:
        print(f"  policy={policy}")
        res = bench_fragmentation(
            model, total_tokens=8192, chunk_size=512, budget=32,
            policy=policy, vocab=ckpt["model_cfg"]["vocab_size"], device=device,
        )
        print(f"    initial={res['rss_initial_mb']:.1f}MB "
              f"peak={res['rss_peak_mb']:.1f}MB "
              f"final={res['rss_final_mb']:.1f}MB "
              f"leak={res['leak_delta_mb']:+.1f}MB")
        out["fragmentation"].append(res)

    print("\n=== 4. False-positive analysis ===")
    out["false_positives"] = bench_false_positives(
        model, n_samples=16, seq_len=2048, chunk_size=512,
        budget=12, n_needles=16, target_repeat=3,
        vocab=ckpt["model_cfg"]["vocab_size"],
        num_classes=ckpt["model_cfg"]["num_classes"],
        device=device,
    )
    print(f"  {'policy':<16} | {'precision':>9} | {'recall':>7} | {'FP rate':>8}")
    for pol, s in out["false_positives"].items():
        print(f"  {pol:<16} | {s['precision']:>9.3f} | {s['recall']:>7.3f} | "
              f"{s['false_positive_rate']:>8.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        if isinstance(o, torch.Tensor): return o.tolist()
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(out), f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
