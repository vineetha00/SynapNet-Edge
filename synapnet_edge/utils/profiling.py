"""Model profiling utilities for SynapNet-Edge.

Provides:
  - profile_model: one-shot latency + memory profile
  - ModelProfiler: context manager for fine-grained per-module profiling
  - flop_counter: estimates FLOPs for the hybrid architecture
"""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ProfileResult:
    seq_len: int
    batch_size: int
    median_latency_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    peak_memory_mb: float
    tokens_per_second: float
    param_count: int
    effective_bits: float

    def __str__(self) -> str:
        return (
            f"seq_len={self.seq_len} | batch={self.batch_size}\n"
            f"  latency:  {self.median_latency_ms:.1f}ms (median)\n"
            f"  peak mem: {self.peak_memory_mb:.1f} MB\n"
            f"  tok/sec:  {self.tokens_per_second:.1f}\n"
            f"  params:   {self.param_count:,}\n"
            f"  eff bits: {self.effective_bits:.2f}"
        )


@torch.no_grad()
def profile_model(
    model: nn.Module,
    seq_len: int = 2048,
    batch_size: int = 1,
    vocab_size: int = 32000,
    device: str = "cpu",
    n_warmup: int = 3,
    n_measure: int = 10,
) -> ProfileResult:
    """Profile model latency and memory for one sequence length."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

    def _sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    def _reset_mem():
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

    def _peak_mem() -> float:
        if dev.type == "cuda":
            return torch.cuda.max_memory_allocated(dev) / 1024 / 1024
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return -1.0

    # Warm-up
    for _ in range(n_warmup):
        model(ids)
        _sync()

    # Measure
    _reset_mem()
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model(ids)
        _sync()
        times.append((time.perf_counter() - t0) * 1000)

    peak_mb = _peak_mem()
    times_sorted = sorted(times)
    median_ms = times_sorted[n_measure // 2]
    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    tok_s = (batch_size * seq_len) / (median_ms / 1000)

    # Param count and effective bits
    n_params = sum(p.numel() for p in model.parameters())
    try:
        from synapnet_edge.quantization.cajq import estimate_model_bits
        bit_info = estimate_model_bits(model)
        eff_bits = bit_info["effective_bits"]
    except Exception:
        eff_bits = 16.0

    return ProfileResult(
        seq_len=seq_len,
        batch_size=batch_size,
        median_latency_ms=median_ms,
        mean_latency_ms=mean_ms,
        std_latency_ms=std_ms,
        peak_memory_mb=peak_mb,
        tokens_per_second=tok_s,
        param_count=n_params,
        effective_bits=eff_bits,
    )


class ModelProfiler:
    """Per-module forward-time profiler using PyTorch hooks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._times: dict[str, list[float]] = {}
        self._hooks: list = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                self._times[name] = []

                def _pre(m, inputs, n=name):
                    m._t0 = time.perf_counter()

                def _post(m, inputs, outputs, n=name):
                    self._times[n].append((time.perf_counter() - m._t0) * 1000)

                self._hooks.append(module.register_forward_pre_hook(_pre))
                self._hooks.append(module.register_forward_hook(_post))
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def report(self, top_k: int = 20) -> dict[str, float]:
        """Return average time per module, sorted by total time."""
        avg = {k: sum(v) / max(1, len(v)) for k, v in self._times.items() if v}
        sorted_avg = dict(sorted(avg.items(), key=lambda x: -x[1])[:top_k])
        return sorted_avg

    def print_report(self, top_k: int = 20) -> None:
        print("\n[ModelProfiler] Per-module average forward time (top {top_k}):")
        for name, ms in self.report(top_k).items():
            print(f"  {name:<60} {ms:.3f}ms")


def estimate_flops(
    model: nn.Module,
    seq_len: int,
    batch_size: int = 1,
    vocab_size: int = 32000,
) -> dict[str, float]:
    """Rough FLOPs estimate per component.

    Returns dict with:
      - ssm_flops:    depthwise conv FLOPs
      - attn_flops:   O(T^2) attention FLOPs
      - mem_flops:    episodic memory read FLOPs
      - total_flops:  sum
    """
    # Walk modules to estimate
    ssm_flops = attn_flops = mem_flops = ff_flops = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv1d):
            C_in, C_out = m.in_channels, m.out_channels
            K = m.kernel_size[0]
            flops = 2 * C_in * C_out * K * seq_len * batch_size
            if m.groups > 1:
                flops /= m.groups
            ssm_flops += flops
        elif isinstance(m, nn.Linear):
            flops = 2 * m.in_features * m.out_features * seq_len * batch_size
            if "attn" in name or "to_q" in name or "to_k" in name:
                attn_flops += flops
            elif "mem" in name or "epmem" in name:
                mem_flops += flops
            else:
                ff_flops += flops

    # Attention softmax: O(T^2) per head per batch
    try:
        for m in model.modules():
            if hasattr(m, "heads"):
                attn_flops += 2 * batch_size * m.heads * seq_len * seq_len
                break
    except Exception:
        pass

    total = ssm_flops + attn_flops + mem_flops + ff_flops
    return {
        "ssm_flops": ssm_flops,
        "attn_flops": attn_flops,
        "mem_flops": mem_flops,
        "ff_flops": ff_flops,
        "total_flops": total,
        "total_gflops": total / 1e9,
    }
