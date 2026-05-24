"""EM-LLM (External Memory LLM) baseline.

EM-LLM augments a standard Transformer with an external memory module
that stores compressed key-value pairs from the past.  This is the
closest architectural baseline to SynapNet-Edge's episodic memory.

Architecture:
  - Standard Transformer encoder blocks
  - External fixed-size memory pool (learnable slots, updated via
    top-K selection — same mechanism as SynapNet-Edge WriteableMemory
    but without dynamic compression or BAEE)
  - All in FP16 (no quantization)
  - No CAJQ, no BAEE, no scale bridge

This baseline answers the ablation question:
  "How much does CAJQ + BAEE contribute beyond just adding episodic memory?"

Reference: EM-LLM: Human-like Episodic Memory for LLMs (Fountas et al., 2024).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalMemoryPool(nn.Module):
    """Fixed learnable external memory accessed via cross-attention.

    Stores top-K token representations by attention magnitude.
    No compression, no budget management — contrast with BAEE.
    """

    def __init__(self, dim: int, num_slots: int = 32, k_frac: float = 0.1):
        super().__init__()
        self.num_slots = num_slots
        self.k_frac = k_frac

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

        # Persistent memory slots (updated each forward pass via in-place assign)
        self.register_buffer("slots", torch.zeros(1, num_slots, dim))
        self._filled = False

    def update_and_read(self, x: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """Update memory with top-K tokens, then read via cross-attention.

        Args:
            x:           (B, T, D) hidden states
            attn_weights: (B, H, T, T) self-attention weights
        """
        B, T, D = x.shape
        k = max(1, int(self.k_frac * T))

        # Importance score: sum of attention received per token
        importance = attn_weights.mean(dim=1).sum(dim=-2)   # (B, T)
        _, topk_idx = torch.topk(importance, k, dim=1)      # (B, k)

        # Write top-K into slots
        mem_list = []
        for b in range(B):
            states = x[b, topk_idx[b]]   # (k, D)
            S = self.num_slots
            if states.size(0) < S:
                pad = torch.zeros(S - states.size(0), D,
                                  device=x.device, dtype=x.dtype)
                states = torch.cat([states, pad], dim=0)
            else:
                states = states[:S]
            mem_list.append(states.unsqueeze(0))

        mem = torch.cat(mem_list, dim=0)   # (B, S, D)

        # Cross-attend from x to memory
        Q = self.to_q(x)       # (B, T, D)
        K = self.to_k(mem)     # (B, S, D)
        V = self.to_v(mem)     # (B, S, D)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)   # (B, T, S)
        attn = F.softmax(logits, dim=-1)
        ctx = torch.matmul(attn, V)   # (B, T, D)
        return self.out(ctx)


class EMLLMBlock(nn.Module):
    """Transformer + external memory block."""

    def __init__(self, dim: int, heads: int, mem_slots: int = 32, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)

        self.heads = heads
        self.head_dim = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.mem = ExternalMemoryPool(dim, num_slots=mem_slots)
        self.mem_gate = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Hd = self.heads, self.head_dim
        residual = x

        x_n = self.norm1(x)

        def _split(t):
            return t.view(B, T, H, Hd).transpose(1, 2)

        Q, K, V = _split(self.to_q(x_n)), _split(self.to_k(x_n)), _split(self.to_v(x_n))
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (Hd ** 0.5)
        attn = F.softmax(attn_logits, dim=-1)
        ctx = torch.matmul(attn, V)                         # (B, H, T, Hd)
        sa_out = ctx.transpose(1, 2).contiguous().view(B, T, D)
        sa_out = self.to_out(sa_out)

        mem_ctx = self.mem.update_and_read(x_n, attn.detach())
        gate = torch.sigmoid(self.mem_gate(x_n))
        x = residual + sa_out + gate * mem_ctx

        x = x + self.ff(self.norm2(x))
        return x


class EMLLMBaseline(nn.Module):
    """EM-LLM: Transformer + external memory without CAJQ or BAEE.

    Ablation baseline for the episodic memory contribution.
    FP16 throughout — no quantization.
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        vocab_size: int = 32000,
        max_len: int = 32768,
        num_classes: int | None = None,
        heads: int = 8,
        mem_slots: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        self.blocks = nn.ModuleList([
            EMLLMBlock(dim, heads, mem_slots=mem_slots)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim)

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
