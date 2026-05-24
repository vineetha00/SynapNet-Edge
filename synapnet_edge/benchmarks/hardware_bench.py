"""Consumer hardware benchmarking suite.

Measures three axes for the Pareto frontier:
  1. Latency     — tokens/second (prefill + decode)
  2. Memory      — peak RAM usage in MB
  3. Accuracy    — from RULER/LongBench scores

Supported platforms (auto-detected):
  - MacBook M-series (Apple Silicon MPS backend)
  - iPhone/iPad via MLX (if mlx is installed) or ExecuTorch
  - Raspberry Pi 5 / Linux ARM (CPU with optional NEON optimisation)
  - CUDA GPU (for server-side comparison baselines)

Usage:
    bench = HardwareBenchmark(model, seq_lengths=[512, 2048, 8192])
    profile = bench.run()
    print(profile.summary())
"""
from __future__ import annotations

import gc
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    """Return a canonical platform string."""
    if torch.backends.mps.is_available():
        cpu_brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        ).stdout.strip()
        if "Apple" in cpu_brand:
            chip = cpu_brand.split()[1] if len(cpu_brand.split()) > 1 else "M-series"
            return f"apple_silicon_{chip.lower()}"
        return "mps"
    elif torch.cuda.is_available():
        return f"cuda_{torch.cuda.get_device_name(0).replace(' ', '_').lower()}"
    else:
        machine = platform.machine()
        if machine.startswith("aarch64") or machine.startswith("arm"):
            return "arm_cpu"   # Raspberry Pi 5 or similar
        return "x86_cpu"


def get_device_for_platform(platform_str: str) -> torch.device:
    if "apple_silicon" in platform_str or "mps" in platform_str:
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    elif "cuda" in platform_str:
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------

def get_peak_memory_mb(device: torch.device) -> float:
    """Return peak memory usage in MB for the given device."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elif device.type == "mps":
        # MPS doesn't expose memory stats directly; use psutil fallback
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return -1.0
    else:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return -1.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    seq_len: int
    batch_size: int
    prefill_ms: float          # time to process the first chunk
    decode_ms_per_token: float # time per decode step (streaming)
    tokens_per_second: float
    peak_memory_mb: float


@dataclass
class HardwareProfile:
    platform: str
    model_name: str
    model_bits: float             # effective bits (from CAJQConfig)
    model_params: int
    latency_records: list[LatencyRecord] = field(default_factory=list)
    accuracy_ruler: dict = field(default_factory=dict)
    accuracy_longbench: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Platform:      {self.platform}",
            f"Model:         {self.model_name}",
            f"Eff. bits:     {self.model_bits:.2f}",
            f"Parameters:    {self.model_params:,}",
            "",
            "Latency (tokens/sec) by context length:",
        ]
        for rec in self.latency_records:
            lines.append(
                f"  seq_len={rec.seq_len:>6}: {rec.tokens_per_second:>7.1f} tok/s | "
                f"peak_mem={rec.peak_memory_mb:.1f} MB"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "model_name": self.model_name,
            "model_bits": self.model_bits,
            "model_params": self.model_params,
            "latency_records": [r.__dict__ for r in self.latency_records],
            "accuracy_ruler": self.accuracy_ruler,
            "accuracy_longbench": self.accuracy_longbench,
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class HardwareBenchmark:
    """Measures inference latency, memory, and accuracy on consumer hardware."""

    SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "SynapNetEdge",
        vocab_size: int = 32000,
        seq_lengths: list[int] | None = None,
        batch_size: int = 1,
        n_warmup: int = 3,
        n_measure: int = 10,
        chunk_size: int = 512,
    ):
        self.model = model
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.seq_lengths = seq_lengths or self.SEQ_LENGTHS
        self.batch_size = batch_size
        self.n_warmup = n_warmup
        self.n_measure = n_measure
        self.chunk_size = chunk_size

        self.platform = detect_platform()
        self.device = get_device_for_platform(self.platform)

        print(f"[HardwareBench] Platform: {self.platform}, device: {self.device}")

    def run(
        self,
        run_ruler: bool = True,
        run_longbench: bool = True,
        verbose: bool = True,
    ) -> HardwareProfile:
        self.model.eval()
        self.model.to(self.device)

        # Estimate model bits
        try:
            from synapnet_edge.quantization.cajq import estimate_model_bits
            bit_info = estimate_model_bits(self.model)
            eff_bits = bit_info["effective_bits"]
        except Exception:
            eff_bits = 16.0

        n_params = sum(p.numel() for p in self.model.parameters())

        profile = HardwareProfile(
            platform=self.platform,
            model_name=self.model_name,
            model_bits=eff_bits,
            model_params=n_params,
        )

        # --- Latency benchmarking ---
        print("[HardwareBench] Running latency profiling...")
        for seq_len in self.seq_lengths:
            rec = self._measure_latency(seq_len, verbose=verbose)
            profile.latency_records.append(rec)

        # --- Accuracy benchmarks ---
        if run_ruler:
            print("[HardwareBench] Running RULER benchmark...")
            from synapnet_edge.benchmarks.ruler_bench import RULERBenchmark, RULERTask
            ruler = RULERBenchmark(
                model=self.model,
                vocab_size=self.vocab_size,
                n_samples=100,
                device=str(self.device),
                chunk_size=self.chunk_size,
            )
            profile.accuracy_ruler = ruler.run(
                context_lengths=[1024, 4096, 8192],
                verbose=verbose,
            )

        if run_longbench:
            print("[HardwareBench] Running LongBench benchmark...")
            from synapnet_edge.benchmarks.longbench import LongBenchEvaluator
            lb = LongBenchEvaluator(
                model=self.model,
                vocab_size=self.vocab_size,
                n_samples=100,
                device=str(self.device),
                chunk_size=self.chunk_size,
            )
            profile.accuracy_longbench = lb.run(
                context_lengths=[2048, 4096, 8192],
                verbose=verbose,
            )

        return profile

    @torch.no_grad()
    def _measure_latency(self, seq_len: int, verbose: bool = True) -> LatencyRecord:
        ids = torch.randint(0, self.vocab_size, (self.batch_size, seq_len),
                            device=self.device)

        def _run_once() -> float:
            reset_peak_memory(self.device)
            t0 = time.perf_counter()
            if seq_len > self.chunk_size and hasattr(self.model, "forward_streaming"):
                self.model.forward_streaming(ids, chunk_size=self.chunk_size)
            else:
                self.model(ids)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
            return time.perf_counter() - t0

        # Warm-up
        for _ in range(self.n_warmup):
            _run_once()

        # Measure
        reset_peak_memory(self.device)
        times = [_run_once() for _ in range(self.n_measure)]
        peak_mb = get_peak_memory_mb(self.device)

        avg_time = sorted(times)[len(times) // 2]   # median
        tok_per_sec = (self.batch_size * seq_len) / avg_time

        # Approximate prefill vs decode split (prefill dominates for full-pass)
        prefill_ms = avg_time * 1000 * 0.9
        decode_ms = avg_time * 1000 * 0.1 / max(1, seq_len)

        rec = LatencyRecord(
            seq_len=seq_len,
            batch_size=self.batch_size,
            prefill_ms=prefill_ms,
            decode_ms_per_token=decode_ms,
            tokens_per_second=tok_per_sec,
            peak_memory_mb=peak_mb,
        )

        if verbose:
            print(f"  seq_len={seq_len:>6}: {tok_per_sec:>7.1f} tok/s | "
                  f"peak_mem={peak_mb:.1f} MB | "
                  f"median_latency={avg_time*1000:.1f}ms")

        return rec

    def save_profile(self, profile: HardwareProfile, path: str) -> None:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        print(f"[HardwareBench] Profile saved to {path}")


# ---------------------------------------------------------------------------
# MLX export helper (iPhone/Mac via Apple MLX)
# ---------------------------------------------------------------------------

def export_to_mlx(model: nn.Module, output_dir: str) -> bool:
    """Export SynapNetEdge to MLX format for iPhone/Mac inference.

    Requires: pip install mlx mlx-lm
    Falls back gracefully if MLX is not installed.
    """
    try:
        import mlx.core as mx
        import mlx.nn as mlx_nn
        print("[MLX] MLX available. Exporting model weights...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save state dict as numpy arrays (MLX-compatible)
        import numpy as np
        state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        np.savez(str(Path(output_dir) / "weights.npz"), **state)
        print(f"[MLX] Weights saved to {output_dir}/weights.npz")
        return True
    except ImportError:
        print("[MLX] mlx not installed. Skipping MLX export.")
        return False


def export_to_executorch(model: nn.Module, output_dir: str) -> bool:
    """Export SynapNetEdge to ExecuTorch for iOS/Android deployment.

    Requires: pip install executorch
    """
    try:
        import executorch  # noqa
        print("[ExecuTorch] ExecuTorch available. Export requires TorchScript first.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        scripted = torch.jit.script(model)
        path = str(Path(output_dir) / "synapnet_edge.pt")
        scripted.save(path)
        print(f"[ExecuTorch] TorchScript saved to {path}")
        return True
    except (ImportError, RuntimeError) as e:
        print(f"[ExecuTorch] Export failed: {e}")
        return False
