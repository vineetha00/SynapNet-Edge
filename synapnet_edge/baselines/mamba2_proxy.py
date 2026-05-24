"""Mamba-2 proxy baseline with uniform INT4 quantization.

This is a structural proxy for Mamba-2 (Dao & Gu, 2024):
  - Uses stacked gated recurrent units (GRUs) as the SSM backbone
    (true selective SSM requires CUDA kernels; GRU approximates the
    linear-time recurrence property on CPU/MPS).
  - Uniform INT4 weight quantization applied to all linear layers
    (using the same AWQ-style quantizer as SynapNetEdge attention).
  - No sparse attention, no episodic memory.

Effective bits: ~4.0 (uniform INT4).
Context handling: O(L) via recurrence, no chunking needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUSSMBlock(nn.Module):
    """GRU-based SSM block approximating Mamba-2 selective scan."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.gate = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_n = self.norm(x)
        gru_out, _ = self.gru(x_n)         # (B, T, D)
        gate = torch.sigmoid(self.gate(x_n))
        mixed = gru_out * gate
        out = residual + mixed
        return out + self.ff(out)


class Mamba2Proxy(nn.Module):
    """Mamba-2 structural proxy for benchmarking.

    Compares against SynapNetEdge under identical quantization budget.
    Used in Table 2 of the SynapNet-Edge paper.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        vocab_size: int = 32000,
        max_len: int = 32768,
        num_classes: int | None = None,
        quantized: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        self.blocks = nn.ModuleList([
            GRUSSMBlock(dim) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim)

        self.is_lm = num_classes is None
        self.head = nn.Linear(dim, num_classes if num_classes else vocab_size)

        if quantized:
            self._apply_uniform_int4()

    def _apply_uniform_int4(self):
        """Apply uniform INT4 to all Linear layers (ablation: no component awareness)."""
        from synapnet_edge.quantization.attention_quantizer import AWQLinear
        replaced = 0
        for name, module in list(self.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            awq = AWQLinear.from_linear(module, group_size=128)
            parts = name.split(".")
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], awq)
            replaced += 1
        print(f"[Mamba2Proxy] Applied uniform INT4 to {replaced} Linear layers.")

    def forward(self, idx: torch.Tensor) -> tuple:
        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        if self.is_lm:
            logits = self.head(x)
        else:
            logits = self.head(x[:, -1, :])
        # Return (logits, [], [], []) to match SynapNetEdge API
        return logits, [], [], []
