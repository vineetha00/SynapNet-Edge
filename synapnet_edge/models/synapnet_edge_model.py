"""SynapNetEdge: full hybrid long-context model for consumer hardware.

Architecture:
  token_embed + pos_embed
  → N × SynapBlockWithEpisodic (SSM + sparse-attn + episodic memory)
  → LayerNorm
  → classification / LM head

The model exposes two forward modes:
  - Standard forward (training, fine-tuning)
  - Chunked streaming forward (inference on long sequences under RAM budget)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn

from synapnet_edge.models.synapblock import SynapBlockWithEpisodic


@dataclass
class SynapNetEdgeConfig:
    dim: int = 256
    depth: int = 6
    vocab_size: int = 32000
    max_len: int = 32768
    num_classes: int | None = None        # None → LM head
    heads: int = 8
    k_frac: float = 0.25
    episodic_slots: int = 16
    episodic_write_frac: float = 0.05
    mlp_ratio: float = 4.0
    use_scale_bridge: bool = True
    # CAJQ quantization flags (set by apply_cajq)
    ssm_bits: int = 2
    attn_bits: int = 4
    mem_bits: int = 8
    # BAEE budget (fraction of peak RAM)
    ram_budget_fraction: float = 0.75


class SynapNetEdge(nn.Module):
    """Hybrid SSM + Sparse-Attention + Episodic-Memory backbone.

    Designed for long-context inference under consumer hardware constraints.
    Supports CAJQ quantization (applied post-init by apply_cajq()) and
    BAEE eviction (injected at inference time by BAEEMemoryManager).
    """

    def __init__(self, cfg: SynapNetEdgeConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_len, cfg.dim) * 0.02)

        self.blocks = nn.ModuleList([
            SynapBlockWithEpisodic(
                dim=cfg.dim,
                mlp_ratio=cfg.mlp_ratio,
                heads=cfg.heads,
                k_frac=cfg.k_frac,
                episodic_slots=cfg.episodic_slots,
                episodic_write_frac=cfg.episodic_write_frac,
                use_scale_bridge=cfg.use_scale_bridge,
            )
            for _ in range(cfg.depth)
        ])

        self.norm_out = nn.LayerNorm(cfg.dim)

        if cfg.num_classes is None:
            self.head = nn.Linear(cfg.dim, cfg.vocab_size)
            self.is_lm = True
        else:
            self.head = nn.Linear(cfg.dim, cfg.num_classes)
            self.is_lm = False

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        external_mems: list[torch.Tensor] | None = None,
    ) -> tuple:
        """Full-sequence forward pass.

        Args:
            idx:           (B, T) integer token IDs
            external_mems: optional list of length `depth`, each (B, S, D),
                           injected by BAEEMemoryManager during inference.

        Returns:
            logits:          (B, V) or (B, C) or (B, T, V) for LM
            debug_masks:     list[depth] of (B, T) salience tensors
            debug_mems:      list[depth] of (B, S, D) memory banks
            debug_topk_idx:  list[depth] of (B, k) written-slot indices
        """
        if next(self.parameters()).device != idx.device:
            self.to(idx.device)

        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]

        debug_masks: list[torch.Tensor] = []
        debug_mems: list[torch.Tensor] = []
        debug_topk: list[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            ext = external_mems[i] if external_mems is not None else None
            x, sal_mask, mem_bank, topk_idx = block(x, external_mem=ext)
            debug_masks.append(sal_mask)
            debug_mems.append(mem_bank)
            debug_topk.append(topk_idx)

        x = self.norm_out(x)

        if self.is_lm:
            logits = self.head(x)                    # (B, T, V)
        else:
            final_state = x[:, -1, :]                # (B, D)
            mem_pooled = debug_mems[-1].mean(dim=1)  # (B, D)
            logits = self.head(final_state + mem_pooled)   # (B, C)

        return logits, debug_masks, debug_mems, debug_topk

    # ------------------------------------------------------------------
    # Streaming (chunked) inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_streaming(
        self,
        idx: torch.Tensor,
        chunk_size: int = 512,
        baee_manager=None,
    ) -> tuple[torch.Tensor, list]:
        """Process a long sequence in chunks, accumulating episodic memory.

        Args:
            idx:          (1, T) — single-sequence streaming inference
            chunk_size:   tokens per chunk (tune for device RAM)
            baee_manager: optional BAEEMemoryManager for budget-aware eviction

        Returns:
            logits: (1, T, V) for LM or (1, C) for classification
            all_debug: list of per-chunk debug dicts
        """
        B, T = idx.shape
        assert B == 1, "Streaming inference is single-sequence only"

        accumulated_mems: list[torch.Tensor | None] = [None] * len(self.blocks)
        all_logits = []
        all_debug = []

        for start in range(0, T, chunk_size):
            chunk = idx[:, start: start + chunk_size]

            ext = []
            for layer_mem in accumulated_mems:
                ext.append(layer_mem)

            logits, masks, mems, topks = self.forward(chunk, external_mems=ext)
            all_logits.append(logits)

            # Update accumulated memory with BAEE eviction if available
            for i, (new_mem, topk) in enumerate(zip(mems, topks)):
                if baee_manager is not None:
                    new_mem = baee_manager.update(
                        layer_idx=i,
                        new_entries=new_mem,
                        salience_scores=masks[i],
                        topk_idx=topk,
                    )
                accumulated_mems[i] = new_mem

            all_debug.append({"masks": masks, "mems": mems, "topks": topks})

        if self.is_lm:
            return torch.cat(all_logits, dim=1), all_debug
        else:
            return all_logits[-1], all_debug

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def num_parameters(self, only_trainable: bool = True) -> int:
        params = self.parameters() if not only_trainable else self.parameters()
        return sum(p.numel() for p in params if p.requires_grad or not only_trainable)

    def count_parameters_by_component(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name, module in self.named_modules():
            if any(isinstance(module, t) for t in (
                nn.Linear, nn.Conv1d, nn.Embedding
            )):
                n = sum(p.numel() for p in module.parameters())
                top = name.split(".")[0] if "." in name else name
                counts[top] = counts.get(top, 0) + n
        return counts
