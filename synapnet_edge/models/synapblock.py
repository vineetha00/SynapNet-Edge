"""SynapBlockWithEpisodic: one hybrid block (SSM + sparse-attn + episodic memory).

The scale-bridging interface layer (ScaleBridge) sits between the three
quantized pathways and the gating/mixing stage, converting mismatched
quantized outputs back to a common FP16 feature space.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from synapnet_edge.models.ssm import SimpleSSM
from synapnet_edge.models.sparse_attention import SparseEventAttention
from synapnet_edge.models.episodic_memory import WriteableMemory


def _ensure_module_device(module: nn.Module, x: torch.Tensor) -> nn.Module:
    param = next(module.parameters(), None)
    if param is not None and param.device != x.device:
        module.to(x.device)
    return module


class SynapBlockWithEpisodic(nn.Module):
    """One SynapNet-Edge block.

    Pathways (in order of application):
      1. SimpleSSM          → 2-bit QAT (local temporal dynamics)
      2. SparseEventAttention → INT4 AWQ+SmoothQuant (global sparse mixing)
      3. WriteableMemory    → INT8 per-entry (episodic recall)

    A learned ScaleBridge normalises the three pathway outputs before
    the α/β gating to absorb scale mismatches from mixed precision.

    Returns:
      out:      (B, T, D)  fused hidden state
      sal_mask: (B, T)     salience scores for BAEE scoring and audit
      mem_bank: (B, S, D)  written episodic slots (FP16, pre-BAEE compression)
      topk_idx: (B, k)     indices of tokens written into memory
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        heads: int = 4,
        k_frac: float = 0.25,
        episodic_slots: int = 8,
        episodic_write_frac: float = 0.05,
        use_scale_bridge: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_scale_bridge = use_scale_bridge

        self.norm_in = nn.LayerNorm(dim)

        self.ssm = SimpleSSM(dim)
        self.attn = SparseEventAttention(dim, heads=heads, k_frac=k_frac)
        self.epmem = WriteableMemory(dim, num_slots=episodic_slots,
                                     k_frac=episodic_write_frac)

        # Learned FP16 scale-bridging interface layer
        # Fuses 3 pathways of potentially different quantized scales
        if use_scale_bridge:
            self.scale_bridge = ScaleBridge(dim, n_pathways=3)

        # Learned pathway gates
        self.alpha_gate = nn.Linear(dim, dim)
        self.beta_gate = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        external_mem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x_norm = self.norm_in(x)

        # Pathway 1: SSM (2-bit QAT)
        ssm_out = self.ssm(x_norm)                          # (B, T, D)

        # Pathway 2: Sparse attention (INT4 AWQ+SmoothQuant)
        attn_out, sal_mask = self.attn(x_norm)              # (B,T,D), (B,T)

        # Pathway 3: Episodic memory (INT8 per-entry)
        # Always write fresh entries from current chunk.  If external_mem
        # is provided (streaming/BAEE mode), prepend it to the read bank so
        # later tokens can attend to history *and* current writes.
        new_mem_bank, topk_idx = self.epmem.write(x_norm, sal_mask)
        if external_mem is not None and external_mem.size(1) > 0:
            # (B, S_ext + S_new, D)
            read_bank = torch.cat([external_mem, new_mem_bank], dim=1)
        else:
            read_bank = new_mem_bank
        mem_bank = new_mem_bank   # exposed to caller — fresh writes only
        epi_ctx = self.epmem.read(x_norm, read_bank)         # (B, T, D)

        # Scale bridge: normalise pathway outputs before gating
        if self.use_scale_bridge:
            ssm_out, attn_out, epi_ctx = self.scale_bridge(
                ssm_out, attn_out, epi_ctx
            )

        _ensure_module_device(self.alpha_gate, x_norm)
        _ensure_module_device(self.beta_gate, x_norm)
        alpha = torch.sigmoid(self.alpha_gate(x_norm))
        beta = torch.sigmoid(self.beta_gate(x_norm))

        mixed = ssm_out + alpha * attn_out + beta * epi_ctx
        out = residual + mixed
        _ensure_module_device(self.ff, out)
        out = out + self.ff(out)

        return out, sal_mask, mem_bank, topk_idx


class ScaleBridge(nn.Module):
    """Learned FP16 interface layer between mismatched quantized pathways.

    Each pathway output passes through its own LayerNorm (absorbing scale
    differences), then through a shared linear projection.  Running the
    bridge in FP16 regardless of the pathway quantization avoids
    catastrophic precision loss at the mixing stage.

    This is Contribution 1c from the SynapNet-Edge paper.
    """

    def __init__(self, dim: int, n_pathways: int = 3):
        super().__init__()
        self.dim = dim
        self.n_pathways = n_pathways

        # Per-pathway normalisation to absorb quantization scale differences
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_pathways)])

        # Shared projection that recombines all pathways (concat → project)
        self.fuse = nn.Linear(dim * n_pathways, dim * n_pathways)

    def forward(self, *pathways: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            *pathways: n_pathways tensors each of shape (B, T, D)

        Returns:
            Tuple of n_pathways rescaled tensors, each (B, T, D).
        """
        assert len(pathways) == self.n_pathways
        device = pathways[0].device
        if next(self.parameters()).device != device:
            self.to(device)

        # Normalise each pathway independently (runs in FP16)
        normed = [norm(p.float()).to(p.dtype)
                  for norm, p in zip(self.norms, pathways)]

        # Concatenate along feature dim, project, split back
        cat = torch.cat(normed, dim=-1)         # (B, T, D*n)
        fused = self.fuse(cat)                  # (B, T, D*n)
        splits = fused.chunk(self.n_pathways, dim=-1)   # n x (B, T, D)
        return tuple(splits)
