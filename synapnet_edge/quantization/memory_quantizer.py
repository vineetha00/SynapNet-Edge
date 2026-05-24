"""Per-entry INT8 quantization for episodic memory banks.

Design:
  - Symmetric per-entry quantization: scale = max(|entry|) / 127
  - Each memory slot (D-dimensional vector) has its own scale factor
  - Quantized entries stored as int8, scales stored as float16
  - Bit budget: D bytes (int8) + 2 bytes (scale) vs D*2 bytes (fp16)
    → ~50% memory reduction per slot

This is Contribution 1b from the SynapNet-Edge paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def quantize_mem_bank(
    mem_bank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize episodic memory bank to INT8.

    Args:
        mem_bank: (B, S, D) FP16/32 memory slots

    Returns:
        mem_int8:  (B, S, D) int8 tensor
        scales:    (B, S)    FP16 per-entry scale factors
    """
    B, S, D = mem_bank.shape
    abs_max = mem_bank.abs().amax(dim=-1)           # (B, S)
    scales = (abs_max / 127.0).clamp(min=1e-8)      # (B, S) FP16

    # Quantize
    mem_scaled = mem_bank / scales.unsqueeze(-1)     # (B, S, D)
    mem_int8 = mem_scaled.round().clamp(-127, 127).to(torch.int8)

    return mem_int8, scales.half()


def dequantize_mem_bank(
    mem_int8: torch.Tensor,
    scales: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize INT8 memory back to float.

    Args:
        mem_int8: (B, S, D) int8
        scales:   (B, S)    FP16 per-entry scales
        target_dtype: output dtype (default float32)

    Returns:
        mem_bank: (B, S, D) dequantized float tensor
    """
    return mem_int8.to(target_dtype) * scales.to(target_dtype).unsqueeze(-1)


def memory_size_bytes(
    mem_bank: torch.Tensor,
    quantized: bool = False,
) -> int:
    """Estimate memory footprint in bytes."""
    B, S, D = mem_bank.shape
    if not quantized:
        # FP16: 2 bytes per element
        return B * S * D * 2
    else:
        # INT8 entries + FP16 scales
        return B * S * D * 1 + B * S * 2


# ---------------------------------------------------------------------------
# Quantized memory bank wrapper
# ---------------------------------------------------------------------------

class QuantizedMemoryBank:
    """Stores episodic slots in INT8, dequantizes on demand.

    Tracks compression ratio and memory savings statistics.
    """

    def __init__(self, mem_bank: torch.Tensor):
        self.shape = mem_bank.shape              # (B, S, D)
        self.device = mem_bank.device
        self.q, self.scales = quantize_mem_bank(mem_bank)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize_mem_bank(self.q, self.scales, dtype)

    @property
    def bytes_used(self) -> int:
        B, S, D = self.shape
        return B * S * D + B * S * 2     # int8 + fp16 scales

    @property
    def bytes_fp16(self) -> int:
        B, S, D = self.shape
        return B * S * D * 2

    @property
    def compression_ratio(self) -> float:
        return self.bytes_fp16 / max(1, self.bytes_used)


# ---------------------------------------------------------------------------
# MemoryQuantizer
# ---------------------------------------------------------------------------

class MemoryQuantizer:
    """Applies INT8 per-entry quantization to episodic memory banks.

    Called by BAEEMemoryManager after the write phase to compress
    warm (medium-retention) memory entries.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
        self._stats: list[dict] = []

    def compress(
        self,
        mem_bank: torch.Tensor,
        entry_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize selected entries to INT8.

        Args:
            mem_bank:   (B, S, D) FP16 memory slots
            entry_mask: (B, S) bool tensor; True = compress to INT8.
                        If None, compresses all entries.

        Returns:
            mem_int8: (B, S, D) int8 (compressed entries only; others zeroed)
            scales:   (B, S)    FP16 scales
        """
        if entry_mask is None:
            entry_mask = torch.ones(
                mem_bank.shape[:2], dtype=torch.bool, device=mem_bank.device
            )

        mem_to_compress = mem_bank * entry_mask.unsqueeze(-1).float()
        mem_int8, scales = quantize_mem_bank(mem_to_compress)

        self._stats.append({
            "n_compressed": entry_mask.sum().item(),
            "n_total": entry_mask.numel(),
            "compression_ratio": memory_size_bytes(mem_bank) / max(
                1, memory_size_bytes(mem_bank, quantized=True)
            ),
        })

        return mem_int8, scales

    def decompress(
        self,
        mem_int8: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        return dequantize_mem_bank(mem_int8, scales, self.dtype)

    def get_stats(self) -> dict:
        if not self._stats:
            return {}
        avg_ratio = sum(s["compression_ratio"] for s in self._stats) / len(self._stats)
        total_compressed = sum(s["n_compressed"] for s in self._stats)
        return {
            "avg_compression_ratio": avg_ratio,
            "total_entries_compressed": total_compressed,
            "n_compress_events": len(self._stats),
        }
