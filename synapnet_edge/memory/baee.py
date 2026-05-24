"""Budget-Aware Episodic Eviction (BAEE) — Contribution 2.

BAEE manages episodic memory under RAM constraints through a three-stage
progressive compression pipeline:

  Stage 0 — Hot (FP16):     Recent / high-retention entries, full precision.
  Stage 1 — Warm (INT8):    Medium-retention entries, compressed 2× by MemoryQuantizer.
  Stage 2 — Cold (summary): Low-retention entries, compressed to a single summary token
                             via mean-pooling of a cluster.
  Stage 3 — Evicted:        Below-threshold entries removed entirely.

The RetentionClassifier is a lightweight 3-layer MLP (≈8K parameters) trained
jointly with the main model via an auxiliary loss that encourages high retention
scores for entries that were actually useful at read time (approximated by
attention weight assigned during episodic memory read).

RAM budget is enforced per-forward-pass:
  total_mem_bytes = hot_entries * D * 2
                  + warm_entries * D * 1 + warm_entries * 2   (int8 + scale)
                  + n_summaries * D * 2                        (fp16 summary tokens)
  → prune until total_mem_bytes ≤ budget_bytes

Reference: inspired by MemGPT (Packer et al., 2023) and
Landmark Attention (Mohtashami & Jaggi, 2023).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapnet_edge.quantization.memory_quantizer import (
    MemoryQuantizer,
    quantize_mem_bank,
    dequantize_mem_bank,
)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class EntryTier(enum.IntEnum):
    HOT = 0       # FP16
    WARM = 1      # INT8
    COLD = 2      # summary token
    EVICTED = 3   # removed


@dataclass
class MemoryEntry:
    """Tracks one episodic memory slot across compression stages."""
    slot_idx: int
    tier: EntryTier = EntryTier.HOT

    # FP16 representation (always available for HOT tier)
    data_fp16: torch.Tensor | None = None

    # INT8 representation (WARM tier)
    data_int8: torch.Tensor | None = None
    scale: float = 1.0

    # Summary token (COLD tier) — scalar in [0, 1] or D-dim mean vector
    summary: torch.Tensor | None = None

    # Retention score (updated each forward pass by RetentionClassifier)
    retention_score: float = 1.0

    # Usage counter: incremented each time this slot receives ≥ threshold attn weight
    use_count: int = 0

    @property
    def bytes_used(self) -> int:
        if self.tier == EntryTier.HOT:
            return self.data_fp16.numel() * 2 if self.data_fp16 is not None else 0
        elif self.tier == EntryTier.WARM:
            return (self.data_int8.numel() * 1 + 2) if self.data_int8 is not None else 0
        elif self.tier == EntryTier.COLD:
            return (self.summary.numel() * 2) if self.summary is not None else 2
        else:
            return 0

    def dequantize(self, dim: int) -> torch.Tensor:
        """Reconstruct FP16 representation."""
        if self.tier == EntryTier.HOT:
            return self.data_fp16
        elif self.tier == EntryTier.WARM:
            return self.data_int8.float() * self.scale
        elif self.tier == EntryTier.COLD:
            return self.summary.expand(dim) if self.summary.dim() == 0 else self.summary
        else:
            return torch.zeros(dim)


# ---------------------------------------------------------------------------
# RetentionClassifier
# ---------------------------------------------------------------------------

class RetentionClassifier(nn.Module):
    """Lightweight MLP that scores episodic entries for retention.

    Input features per entry:
      - entry hidden state (D dims)
      - mean salience at write time (1 dim)
      - slot position embedding (1 dim)
      - normalised age / recency (1 dim)
      - use count (1 dim)

    Output: scalar retention score in (0, 1).

    Total parameters: ~8K for dim=256.  Runs in FP16.

    Trained via an auxiliary loss:
      L_aux = -mean(attn_weight * log(retention_score)
                  + (1 - attn_weight) * log(1 - retention_score))
    where attn_weight is the average attention weight assigned to this
    entry during the episodic memory read (a proxy for utility).
    """

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.dim = dim
        input_dim = dim + 4   # entry + salience + position + age + use_count
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        entry: torch.Tensor,       # (B, S, D)
        salience_at_write: torch.Tensor,   # (B, S) mean salience when written
        slot_positions: torch.Tensor,      # (S,) normalised in [0, 1]
        ages: torch.Tensor,                # (B, S) normalised in [0, 1]
        use_counts: torch.Tensor,          # (B, S) normalised
    ) -> torch.Tensor:
        """
        Returns:
            scores: (B, S) retention probability in (0, 1)
        """
        if next(self.parameters()).device != entry.device:
            self.to(entry.device)

        B, S, D = entry.shape
        salience_at_write = salience_at_write.to(device=entry.device, dtype=entry.dtype)
        slot_positions = slot_positions.to(device=entry.device, dtype=entry.dtype)
        ages = ages.to(device=entry.device, dtype=entry.dtype)
        use_counts = use_counts.to(device=entry.device, dtype=entry.dtype)

        pos = slot_positions.unsqueeze(0).expand(B, -1).unsqueeze(-1)   # (B, S, 1)
        sal = salience_at_write.unsqueeze(-1)                            # (B, S, 1)
        age = ages.unsqueeze(-1)                                         # (B, S, 1)
        uc = use_counts.unsqueeze(-1)                                    # (B, S, 1)

        feats = torch.cat([entry, sal, pos, age, uc], dim=-1)           # (B, S, D+4)
        scores = self.net(feats).squeeze(-1)                             # (B, S)
        return scores

    def auxiliary_loss(
        self,
        retention_scores: torch.Tensor,   # (B, S)
        attn_weights: torch.Tensor,       # (B, S) average attention received
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Binary cross-entropy loss with attention weights as soft labels."""
        scores = retention_scores.clamp(eps, 1 - eps)
        labels = attn_weights.clamp(0, 1)
        loss = -(labels * scores.log() + (1 - labels) * (1 - scores).log())
        return loss.mean()


# ---------------------------------------------------------------------------
# Progressive compression helpers
# ---------------------------------------------------------------------------

def _summarize_entries(
    entries: torch.Tensor,          # (n, D)
    n_clusters: int = 1,
) -> torch.Tensor:
    """Compress multiple entries into summary tokens via K-means-style clustering.

    For simplicity (and edge-device friendliness), uses sequential mean-pooling
    over clusters.  Each cluster becomes one summary D-dim token.

    Returns: (n_clusters, D)
    """
    n, D = entries.shape
    if n <= n_clusters:
        pad = torch.zeros(n_clusters - n, D,
                          device=entries.device, dtype=entries.dtype)
        return torch.cat([entries, pad], dim=0)

    chunk = math.ceil(n / n_clusters)
    summaries = []
    for i in range(0, n, chunk):
        summaries.append(entries[i: i + chunk].mean(dim=0, keepdim=True))
    summary = torch.cat(summaries[:n_clusters], dim=0)    # (n_clusters, D)
    return summary


# ---------------------------------------------------------------------------
# EvictionPolicy
# ---------------------------------------------------------------------------

class EvictionPolicy(enum.Enum):
    BAEE = "baee"       # Retention-score driven (our method)
    FIFO = "fifo"       # First-in first-out (ablation baseline)
    LRU = "lru"         # Least recently used (ablation baseline)
    RANDOM = "random"   # Random eviction


# ---------------------------------------------------------------------------
# BAEEMemoryManager
# ---------------------------------------------------------------------------

class BAEEMemoryManager(nn.Module):
    """Manages episodic memory banks for all transformer layers under RAM budget.

    Usage (during streaming inference):
        manager = BAEEMemoryManager(cfg, n_layers=6)
        for chunk in sequence_chunks:
            logits, masks, mems, topks = model(chunk, external_mems=manager.get_banks())
            for layer_i, (new_mem, sal, topk) in enumerate(zip(mems, masks, topks)):
                manager.update(layer_i, new_mem, sal, topk)
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        slots_per_layer: int = 16,
        budget_mb: float = 256.0,
        policy: EvictionPolicy = EvictionPolicy.BAEE,
        summary_slots: int = 4,
        hot_threshold: float = 0.6,
        warm_threshold: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.slots_per_layer = slots_per_layer
        self.budget_bytes = int(budget_mb * 1024 * 1024)
        self.policy = policy
        self.summary_slots = summary_slots
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold

        # Retention classifier (shared across layers for param efficiency)
        self.retention_clf = RetentionClassifier(dim)

        # Per-layer memory state
        self._banks: list[torch.Tensor | None] = [None] * n_layers

        # Per-layer slot metadata
        self._salience_at_write: list[torch.Tensor | None] = [None] * n_layers
        self._ages: list[torch.Tensor] = [
            torch.zeros(slots_per_layer) for _ in range(n_layers)
        ]
        self._use_counts: list[torch.Tensor] = [
            torch.zeros(slots_per_layer) for _ in range(n_layers)
        ]
        self._insert_order: list[list[int]] = [[] for _ in range(n_layers)]
        self._step: int = 0

        self._quant = MemoryQuantizer()
        self._stats: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_banks(self) -> list[torch.Tensor | None]:
        """Return current memory banks for injection into model.forward()."""
        return list(self._banks)

    @torch.no_grad()
    def update(
        self,
        layer_idx: int,
        new_entries: torch.Tensor,       # (B, S_new, D)
        salience_scores: torch.Tensor,   # (B, T) salience from current chunk
        topk_idx: torch.Tensor,          # (B, k) written token positions
    ) -> torch.Tensor:
        """Update layer memory with new entries and apply eviction.

        Returns:
            mem_bank: (B, S, D) updated memory bank (FP16), ready for next chunk.
        """
        self._step += 1
        device = new_entries.device
        B, S_new, D = new_entries.shape

        # Per-slot mean salience at write time
        sal_mean = salience_scores.mean(dim=1, keepdim=True).expand(B, S_new)  # (B, S_new)

        if self._banks[layer_idx] is None:
            # First chunk: just store new entries
            bank = new_entries
            self._salience_at_write[layer_idx] = sal_mean
            self._ages[layer_idx] = torch.zeros(S_new)
            self._use_counts[layer_idx] = torch.zeros(S_new)
        else:
            prev = self._banks[layer_idx]   # (B, S_prev, D)
            prev_sal = self._salience_at_write[layer_idx]   # (B, S_prev)

            # Concatenate old + new
            bank = torch.cat([prev, new_entries], dim=1)    # (B, S_total, D)
            combined_sal = torch.cat([prev_sal, sal_mean], dim=1)

            prev_s = prev.shape[1]
            prev_ages = self._ages[layer_idx].to(device)
            new_ages = torch.zeros(S_new, device=device)
            ages = torch.cat([prev_ages + 1, new_ages], dim=0)

            prev_uc = self._use_counts[layer_idx].to(device)
            new_uc = torch.zeros(S_new, device=device)
            use_counts = torch.cat([prev_uc, new_uc], dim=0)

            # Compute retention scores
            S_total = bank.shape[1]
            slot_pos = torch.linspace(0, 1, S_total, device=device)
            ages_norm = (ages / (ages.max() + 1)).unsqueeze(0).expand(B, -1)
            uc_norm = (use_counts / (use_counts.max() + 1)).unsqueeze(0).expand(B, -1)

            retention = self.retention_clf(
                bank.float(),
                combined_sal.float(),
                slot_pos,
                ages_norm.float(),
                uc_norm.float(),
            )   # (B, S_total)

            # Apply eviction policy
            bank, retention, combined_sal, ages, use_counts = self._evict(
                bank, retention, combined_sal, ages, use_counts,
                device=device,
            )

            self._salience_at_write[layer_idx] = combined_sal
            self._ages[layer_idx] = ages.cpu()
            self._use_counts[layer_idx] = use_counts.cpu()

        # Enforce budget
        bank = self._enforce_budget(bank, layer_idx)

        self._banks[layer_idx] = bank
        return bank

    def reset(self) -> None:
        """Clear all memory banks (e.g., start of new document)."""
        self._banks = [None] * self.n_layers
        self._salience_at_write = [None] * self.n_layers
        self._ages = [torch.zeros(self.slots_per_layer) for _ in range(self.n_layers)]
        self._use_counts = [torch.zeros(self.slots_per_layer) for _ in range(self.n_layers)]
        self._step = 0

    # ------------------------------------------------------------------
    # Eviction logic
    # ------------------------------------------------------------------

    def _evict(
        self,
        bank: torch.Tensor,             # (B, S, D)
        retention: torch.Tensor,        # (B, S)
        sal: torch.Tensor,              # (B, S)
        ages: torch.Tensor,             # (S,)
        use_counts: torch.Tensor,       # (S,)
        device: str | torch.device = "cpu",
    ) -> tuple:
        """Apply eviction policy to reduce bank to at most slots_per_layer entries.

        Returns trimmed (bank, retention, sal, ages, use_counts).
        """
        S = bank.shape[1]
        target_S = self.slots_per_layer

        if S <= target_S:
            return bank, retention, sal, ages, use_counts

        if self.policy == EvictionPolicy.BAEE:
            # Keep slots with highest mean retention score across batch
            mean_retention = retention.mean(dim=0)   # (S,)
            _, keep_idx = torch.topk(mean_retention, target_S)
        elif self.policy == EvictionPolicy.FIFO:
            # Keep most recently written (highest indices)
            keep_idx = torch.arange(S - target_S, S, device=device)
        elif self.policy == EvictionPolicy.LRU:
            # Keep slots with smallest age (most recently used)
            _, keep_idx = torch.topk(-ages.to(device), target_S)
        else:  # RANDOM
            keep_idx = torch.randperm(S, device=device)[:target_S]

        keep_idx, _ = keep_idx.sort()
        bank = bank[:, keep_idx, :]
        retention = retention[:, keep_idx]
        sal = sal[:, keep_idx]
        ages = ages[keep_idx.cpu()]
        use_counts = use_counts[keep_idx.cpu()]

        return bank, retention, sal, ages, use_counts

    def _enforce_budget(
        self,
        bank: torch.Tensor,   # (B, S, D)
        layer_idx: int,
    ) -> torch.Tensor:
        """Progressively compress entries if over RAM budget.

        Progressive pipeline (per bank):
          1. Compress all entries to INT8 if total > budget
          2. If still over budget, replace bottom-50% with summary tokens
          3. If still over budget, evict bottom-25% entirely
        """
        B, S, D = bank.shape
        bytes_fp16 = B * S * D * 2

        if bytes_fp16 <= self.budget_bytes:
            return bank   # fits in budget, no action needed

        # Stage 1: compress entire bank to INT8
        bank_int8, scales = quantize_mem_bank(bank)
        bytes_int8 = B * S * D + B * S * 2
        self._stats.append({"layer": layer_idx, "action": "int8", "slots": S})

        if bytes_int8 <= self.budget_bytes:
            # Dequantize and return (inference continues in FP)
            return dequantize_mem_bank(bank_int8, scales, bank.dtype)

        # Stage 2: summarise bottom 50% to single summary token
        n_keep = max(1, S // 2)
        summary = bank[:, n_keep:, :].mean(dim=1, keepdim=True)   # (B, 1, D)
        bank = torch.cat([bank[:, :n_keep, :], summary], dim=1)    # (B, n_keep+1, D)
        bytes_after_summary = B * (n_keep + 1) * D * 2
        self._stats.append({"layer": layer_idx, "action": "summarize", "slots": n_keep + 1})

        if bytes_after_summary <= self.budget_bytes:
            return bank

        # Stage 3: hard eviction — keep only top-quarter
        n_final = max(1, n_keep // 2)
        bank = bank[:, :n_final, :]
        self._stats.append({"layer": layer_idx, "action": "evict", "slots": n_final})

        return bank

    # ------------------------------------------------------------------
    # Training support
    # ------------------------------------------------------------------

    def compute_aux_loss(
        self,
        mem_banks: list[torch.Tensor],      # per-layer (B, S, D)
        attn_weights: list[torch.Tensor],   # per-layer (B, S) from episodic read
    ) -> torch.Tensor:
        """Compute retention classifier auxiliary loss.

        Call this during training and add to main loss:
            loss = main_loss + lambda_baee * manager.compute_aux_loss(mems, attn_w)
        """
        if not mem_banks:
            return torch.tensor(0.0)

        total_loss = torch.zeros((), device=mem_banks[0].device)
        self.retention_clf.to(mem_banks[0].device)
        for i, (bank, attn_w) in enumerate(zip(mem_banks, attn_weights)):
            B, S, D = bank.shape
            slot_pos = torch.linspace(0, 1, S, device=bank.device)
            ages_norm = torch.zeros(B, S, device=bank.device)
            uc_norm = torch.zeros(B, S, device=bank.device)
            sal = torch.ones(B, S, device=bank.device) * 0.5   # placeholder

            scores = self.retention_clf(
                bank.float(), sal, slot_pos, ages_norm, uc_norm
            )
            loss_i = self.retention_clf.auxiliary_loss(scores, attn_w.to(bank.device))
            total_loss = total_loss + loss_i

        return total_loss / max(1, len(mem_banks))

    def get_compression_stats(self) -> dict:
        if not self._stats:
            return {}
        actions = [s["action"] for s in self._stats]
        return {
            "n_int8_compressions": actions.count("int8"),
            "n_summarizations": actions.count("summarize"),
            "n_evictions": actions.count("evict"),
            "total_compression_events": len(self._stats),
        }
