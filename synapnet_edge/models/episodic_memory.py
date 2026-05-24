"""WriteableMemory: dynamic episodic slots with BAEE-aware quantized storage.

Memory entries are stored in one of three precision tiers:
  FP16  — hot / high-retention entries
  INT8  — warm entries (quantized per-entry by MemoryQuantizer)
  evicted — removed when below budget

The BAEE eviction policy is applied externally by BAEEMemoryManager.
This module owns the read/write mechanics and per-entry dequantization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_module_device(module: nn.Module, x: torch.Tensor) -> nn.Module:
    param = next(module.parameters(), None)
    if param is not None and param.device != x.device:
        module.to(x.device)
    return module


class WriteableMemory(nn.Module):
    """Salience-driven episodic memory bank.

    write() fills slots with the top-k salient token representations.
    read() lets every timestep attend to the stored slots.

    INT8 quantization of mem_bank entries is handled externally by
    BAEEMemoryManager after each write(); this module receives either
    FP16 or dequantized-FP16 tensors for the read path.
    """

    def __init__(self, dim: int, num_slots: int = 8, k_frac: float = 0.05):
        super().__init__()
        self.num_slots = num_slots
        self.k_frac = k_frac
        self.dim = dim

        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.val_proj = nn.Linear(dim, dim, bias=False)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Per-slot scale factors for INT8 dequantization (set by MemoryQuantizer)
        self.register_buffer("slot_scales", torch.ones(num_slots))

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write(
        self,
        x: torch.Tensor,
        salience: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k tokens by salience and pack them into slots.

        Args:
            x:        (B, T, D) token hidden states
            salience: (B, T)    soft salience scores in [0, 1]

        Returns:
            mem_bank: (B, S, D) — FP16 entries ready for optional INT8 compress
            topk_idx: (B, k)   — source token indices for audit / BAEE scoring
        """
        B, T, D = x.shape
        k = max(1, int(self.k_frac * T))

        _, topk_idx = torch.topk(salience, k, dim=1)   # (B, k)

        mem_bank_list = []
        for b in range(B):
            idxs = topk_idx[b]                           # (k,)
            token_states = x[b, idxs]                   # (k, D)

            S = self.num_slots
            if token_states.size(0) < S:
                pad = torch.zeros(
                    S - token_states.size(0), D,
                    device=x.device, dtype=x.dtype
                )
                token_states = torch.cat([token_states, pad], dim=0)
            else:
                token_states = token_states[:S]

            mem_bank_list.append(token_states.unsqueeze(0))   # (1, S, D)

        mem_bank = torch.cat(mem_bank_list, dim=0)   # (B, S, D)
        return mem_bank, topk_idx

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def read(
        self,
        x: torch.Tensor,
        mem_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend from every token to episodic slots.

        Args:
            x:        (B, T, D) current hidden states
            mem_bank: (B, S, D) memory slots (FP16, possibly dequantized)

        Returns:
            ctx: (B, T, D) episodic context per timestep
        """
        B, T, D = x.shape
        _ensure_module_device(self.query_proj, x)
        _ensure_module_device(self.key_proj, mem_bank)
        _ensure_module_device(self.val_proj, mem_bank)
        _ensure_module_device(self.out_proj, x)

        Q = self.query_proj(x)         # (B, T, D)
        K = self.key_proj(mem_bank)    # (B, S, D)
        V = self.val_proj(mem_bank)    # (B, S, D)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)   # (B, T, S)
        attn_w = F.softmax(attn_logits, dim=-1)
        ctx = torch.matmul(attn_w, V)         # (B, T, D)
        return self.out_proj(ctx)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_slots={self.num_slots}, k_frac={self.k_frac}"
