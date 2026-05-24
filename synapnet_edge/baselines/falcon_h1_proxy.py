"""Falcon-H1 / Hymba-style hybrid baseline in FP16.

Represents hybrid SSM+attention architectures (Falcon-H1, Hymba):
  - Alternating SSM and full-attention blocks
  - FP16 precision throughout (no quantization)
  - Shows the memory/accuracy trade-off of FP16 hybrid models
  - Demonstrates why CAJQ is needed for edge deployment

Architecture:
  Even-indexed blocks: SSM (GRU-based recurrence)
  Odd-indexed blocks:  Dense self-attention
  All layers: FP16, no quantization

This is the FP16 upper-bound baseline that SynapNetEdge approaches
with CAJQ quantization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapnet_edge.baselines.llama_awq_proxy import RMSNorm, SwiGLUFFN, DenseAttention
from synapnet_edge.baselines.mamba2_proxy import GRUSSMBlock


class HybridBlock(nn.Module):
    """Alternating SSM / attention block (Falcon-H1 / Hymba pattern)."""

    def __init__(self, dim: int, heads: int, block_idx: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.use_attention = (block_idx % 2 == 1)
        self.norm = RMSNorm(dim)

        if self.use_attention:
            self.mixer = DenseAttention(dim, heads)
        else:
            self.gru = nn.GRU(dim, dim, batch_first=True)
            self.gate = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_n = self.norm(x)

        if self.use_attention:
            mixed = self.mixer(x_n)
        else:
            gru_out, _ = self.gru(x_n)
            mixed = gru_out * torch.sigmoid(self.gate(x_n))

        x = residual + mixed
        x = x + self.ffn(self.norm2(x))
        return x


class FalconH1Proxy(nn.Module):
    """Falcon-H1 / Hymba-style FP16 hybrid model.

    Used as the FP16 upper-bound baseline in SynapNet-Edge comparisons.
    Shows the full-precision accuracy ceiling that CAJQ tries to approach.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        vocab_size: int = 32000,
        max_len: int = 32768,
        num_classes: int | None = None,
        heads: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        self.blocks = nn.ModuleList([
            HybridBlock(dim, heads, block_idx=i)
            for i in range(depth)
        ])
        self.norm_out = RMSNorm(dim)

        self.is_lm = num_classes is None
        self.head = nn.Linear(dim, num_classes if num_classes else vocab_size)

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
        return logits, [], [], []
