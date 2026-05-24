"""Llama-3.2 structural proxy with AWQ INT4 quantization.

Approximates the Llama-3.2 architecture:
  - Full dense self-attention (no sparsity, no recurrence)
  - Pre-norm with RMSNorm
  - SwiGLU FFN
  - Grouped-query attention (GQA) approximated with standard MHA
  - AWQ INT4 on all Q/K/V/O projections

Effective bits: ~4.0 (AWQ INT4, same as attention in SynapNetEdge).
Context: O(T^2) memory — demonstrates the quadratic cost we avoid.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenseAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Hd = self.heads, self.head_dim

        def _split(t):
            return t.view(B, T, H, Hd).transpose(1, 2)

        Q, K, V = _split(self.to_q(x)), _split(self.to_k(x)), _split(self.to_v(x))
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = attn.transpose(1, 2).contiguous().view(B, T, D)
        return self.to_out(out)


class LlamaBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.67):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = DenseAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LlamaAWQProxy(nn.Module):
    """AWQ-quantized Llama-3.2 structural proxy.

    Represents the dense-attention + AWQ INT4 baseline.
    Shows quadratic memory cost vs SynapNetEdge's linear-cost SSM.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        vocab_size: int = 32000,
        max_len: int = 8192,
        num_classes: int | None = None,
        heads: int = 8,
        quantized: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        self.blocks = nn.ModuleList([
            LlamaBlock(dim, heads) for _ in range(depth)
        ])
        self.norm_out = RMSNorm(dim)

        self.is_lm = num_classes is None
        self.head = nn.Linear(dim, num_classes if num_classes else vocab_size, bias=False)

        if quantized:
            self._apply_awq_int4()

    def _apply_awq_int4(self):
        from synapnet_edge.quantization.attention_quantizer import AWQLinear
        replaced = 0
        for name, module in list(self.named_modules()):
            if isinstance(module, nn.Linear) and "head" not in name:
                parts = name.split(".")
                parent = self
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], AWQLinear.from_linear(module, group_size=128))
                replaced += 1
        print(f"[LlamaAWQProxy] Applied AWQ INT4 to {replaced} Linear layers.")

    def forward(self, idx: torch.Tensor) -> tuple:
        B, T = idx.shape
        T = min(T, self.max_len)
        idx = idx[:, :T]
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        if self.is_lm:
            logits = self.head(x)
        else:
            logits = self.head(x[:, -1, :])
        return logits, [], [], []
