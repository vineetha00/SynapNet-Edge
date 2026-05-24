"""Realistic long-context task datasets with strong signal-to-noise.

Each task is *learnable* — i.e., a competent long-context model should
solve it well above chance — and *long-context* — i.e., a short-context
model with no retrieval should fail.

Token layout (shared across tasks):
  0: PAD, 1: BOS, 2: EOS, 3: SEP, 4: QUERY_MARKER
  5 .. 5+num_classes-1:   "key" tokens (the things we hide / track)
  5+num_classes .. vocab_size-1:  "filler" tokens (background noise)

The query marker token (4) is placed immediately before the model's
answer position so even short-receptive-field models can locate where
to attend.  The challenge is retrieving information from the *past*.
"""
from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader


PAD, BOS, EOS, SEP, QUERY = 0, 1, 2, 3, 4
KEY_START = 5


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class LongContextDataset(Dataset):
    """Abstract base for synthetic long-context tasks."""

    task_name: str = "abstract"

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int = 4096,
        num_classes: int = 64,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.filler_start = KEY_START + num_classes

        self._rng = random.Random(seed)
        self._torch_rng = torch.Generator().manual_seed(seed)

        self._samples = [self._generate() for _ in range(n_samples)]

    def _filler(self, n: int) -> torch.Tensor:
        return torch.randint(
            self.filler_start, self.vocab_size, (n,),
            generator=self._torch_rng,
        )

    def _generate(self) -> tuple[torch.Tensor, int, dict]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, label, _meta = self._samples[idx]
        return ids, torch.tensor(label, dtype=torch.long)

    def metadata(self, idx: int) -> dict:
        return self._samples[idx][2]


# ---------------------------------------------------------------------------
# Task 1: NIAH single needle
# ---------------------------------------------------------------------------

class NIAHSingle(LongContextDataset):
    """Single needle hidden at a random depth.

    Format: [BOS] filler ... NEEDLE_KEY filler ... [QUERY] [SEP]
    Answer: the key value.

    Difficulty: needle is at uniform-random depth in 5%-95% of the sequence.
    """
    task_name = "niah_single"

    def _generate(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS
        ids[-2] = QUERY
        ids[-1] = SEP

        code = self._rng.randint(0, self.num_classes - 1)
        depth_frac = self._rng.uniform(0.05, 0.95)
        pos = max(1, min(L - 3, int(depth_frac * L)))
        ids[pos] = KEY_START + code

        return ids, code, {"depth_frac": depth_frac, "pos": pos}


# ---------------------------------------------------------------------------
# Task 2: NIAH multi-key (k needles, retrieve the one matching the query key)
# ---------------------------------------------------------------------------

class NIAHMultiKey(LongContextDataset):
    """K needles, query specifies which one to retrieve.

    Each needle is a (key, value) pair: [KEY_TOKEN, VALUE_TOKEN].
    The query at the end is the key, and the model must output the value.

    This task forces the model to use episodic memory: a fixed-size memory
    must store all K (key, value) pairs.  With high K and small memory,
    the eviction policy matters.
    """
    task_name = "niah_multi_key"

    def __init__(self, *args, n_needles: int = 4, **kwargs):
        self.n_needles = n_needles
        # Split num_classes into keys and values
        super().__init__(*args, **kwargs)

    def _generate(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS
        ids[-3] = QUERY        # marker before key
        ids[-1] = SEP

        # Pick distinct key IDs and value IDs
        n_keys = max(self.num_classes // 2, self.n_needles + 1)
        n_vals = self.num_classes - n_keys

        keys = self._rng.sample(range(n_keys), self.n_needles)
        vals = [self._rng.randint(0, n_vals - 1) for _ in range(self.n_needles)]

        # Place pairs at random positions
        positions = sorted(self._rng.sample(
            range(2, L - 4), self.n_needles
        ))
        for pos, k, v in zip(positions, keys, vals):
            ids[pos] = KEY_START + k
            ids[pos + 1] = KEY_START + n_keys + v   # value token follows key

        # Query: pick one of the k keys
        target_idx = self._rng.randint(0, self.n_needles - 1)
        ids[-2] = KEY_START + keys[target_idx]
        target_val = vals[target_idx]

        return ids, target_val, {
            "positions": positions,
            "target_idx": target_idx,
            "target_pos": positions[target_idx],
            "n_needles": self.n_needles,
        }

    def __getitem__(self, idx: int):
        ids, label, _ = self._samples[idx]
        return ids, torch.tensor(label, dtype=torch.long)

    @property
    def value_offset(self) -> int:
        """Offset for value labels (the answer space starts after key space)."""
        return 0


# ---------------------------------------------------------------------------
# Task 3: Variable tracking
# ---------------------------------------------------------------------------

class VariableTracking(LongContextDataset):
    """Track a chain of variable assignments.

    Sequence has K assignment "events" at increasing positions:
      ... var_0 = code ... var_1 = var_0 ... var_2 = var_1 ...
    Query asks for the final value of var_{K-1}.

    The model must follow the chain through arbitrary-depth aliasing.
    """
    task_name = "variable_track"

    def __init__(self, *args, n_hops: int = 3, **kwargs):
        self.n_hops = n_hops
        super().__init__(*args, **kwargs)

    def _generate(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS
        ids[-2] = QUERY
        ids[-1] = SEP

        code = self._rng.randint(0, self.num_classes - 1)
        # n_hops events, each at successively later positions
        n_events = self.n_hops + 1
        positions = sorted(self._rng.sample(range(2, L - 4), n_events))
        # event 0: introduce the code
        ids[positions[0]] = KEY_START + code
        # events 1..n: reuse the code (the model must remember and propagate)
        for i in range(1, n_events):
            ids[positions[i]] = KEY_START + code

        return ids, code, {"positions": positions, "n_hops": self.n_hops}


# ---------------------------------------------------------------------------
# Task 4: Frequency aggregation
# ---------------------------------------------------------------------------

class FrequencyAggregation(LongContextDataset):
    """Given a sequence with N marked items, return the most frequent class."""
    task_name = "fa"

    def __init__(self, *args, n_items: int = 16, **kwargs):
        self.n_items = n_items
        super().__init__(*args, **kwargs)

    def _generate(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS
        ids[-2] = QUERY
        ids[-1] = SEP

        positions = sorted(self._rng.sample(range(2, L - 4), self.n_items))
        counts = [0] * self.num_classes
        # Bias one class to be most common
        majority = self._rng.randint(0, self.num_classes - 1)
        for pos in positions:
            # 60% chance of majority class, else uniform
            if self._rng.random() < 0.6:
                c = majority
            else:
                c = self._rng.randint(0, self.num_classes - 1)
            ids[pos] = KEY_START + c
            counts[c] += 1

        # Recompute actual majority (in case bias didn't dominate)
        actual_majority = max(range(self.num_classes), key=lambda c: counts[c])
        return ids, actual_majority, {"counts": counts}


# ---------------------------------------------------------------------------
# Memory-pressure NIAH (designed to force ~50% eviction)
# ---------------------------------------------------------------------------

class MemoryPressureNIAH(LongContextDataset):
    """NIAH-multi at long context with more needles than memory slots.

    Designed so that a fixed memory budget cannot hold all needles:
      - n_needles = 2 * memory_slots  →  ~50% eviction is forced
      - The target needle is uniformly random in time → no policy can win
        on recency alone (FIFO/LRU lose)
      - The target needle's "salience" is higher than distractors
        (e.g., its key appears repeated several times before the query)
        → BAEE's retention classifier can learn to keep it

    This is the discriminative experiment for BAEE.
    """
    task_name = "memory_pressure_niah"

    def __init__(
        self,
        *args,
        n_needles: int = 16,         # 2× a 8-slot memory
        target_repeat: int = 3,      # target key repeats N times → higher salience
        target_position_bias: str = "uniform",  # 'uniform' | 'early' | 'late'
        **kwargs,
    ):
        self.n_needles = n_needles
        self.target_repeat = target_repeat
        self.target_position_bias = target_position_bias
        super().__init__(*args, **kwargs)

    def _generate(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS
        ids[-3] = QUERY
        ids[-1] = SEP

        n_keys = max(self.num_classes // 2, self.n_needles + 1)
        n_vals = self.num_classes - n_keys

        keys = self._rng.sample(range(n_keys), self.n_needles)
        vals = [self._rng.randint(0, n_vals - 1) for _ in range(self.n_needles)]

        target_idx = self._rng.randint(0, self.n_needles - 1)
        target_key = keys[target_idx]
        target_val = vals[target_idx]

        # Position pool depends on bias
        if self.target_position_bias == "early":
            # Target placed in first 40% of context (early chunks)
            target_pool = list(range(2, max(4, int(L * 0.4))))
            distractor_pool = list(range(max(4, int(L * 0.4)), L - 5))
        elif self.target_position_bias == "late":
            target_pool = list(range(int(L * 0.6), L - 5))
            distractor_pool = list(range(2, int(L * 0.6)))
        else:
            target_pool = list(range(2, L - 5))
            distractor_pool = list(range(2, L - 5))

        # Place target needle with multiple repetitions
        target_positions = sorted(
            self._rng.sample(target_pool,
                             min(self.target_repeat, len(target_pool) // 2))
        )
        for p in target_positions:
            ids[p] = KEY_START + target_key
            if p + 1 < L - 4:
                ids[p + 1] = KEY_START + n_keys + target_val

        # Place distractor needles
        non_target_idx = [i for i in range(self.n_needles) if i != target_idx]
        # Make sure we don't overlap target positions
        avail = [p for p in distractor_pool
                 if p not in target_positions and p + 1 not in target_positions]
        distractor_positions = sorted(self._rng.sample(
            avail, min(len(non_target_idx), len(avail) // 2)
        ))[: len(non_target_idx)]
        for p, ni in zip(distractor_positions, non_target_idx):
            ids[p] = KEY_START + keys[ni]
            if p + 1 < L - 4:
                ids[p + 1] = KEY_START + n_keys + vals[ni]
        target_slots = target_positions

        # Query
        ids[-2] = KEY_START + target_key

        return ids, target_val, {
            "n_needles": self.n_needles,
            "target_key": target_key,
            "target_pos": target_slots,
            "target_repeat": self.target_repeat,
        }


# ---------------------------------------------------------------------------
# Multi-task curriculum sampler
# ---------------------------------------------------------------------------

class MultiTaskCurriculum(Dataset):
    """Samples from multiple long-context tasks with configurable weights.

    Returns:
        ids:      (T,) token IDs
        label:    () integer class id
        task_id:  () integer task index (for per-task accuracy tracking)
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int = 4096,
        num_classes: int = 64,
        task_weights: dict[str, float] | None = None,
        seed: int = 0,
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.seed = seed
        rng = random.Random(seed)

        default_weights = {
            "niah_single": 0.40,
            "niah_multi_key": 0.30,
            "variable_track": 0.15,
            "fa": 0.15,
        }
        weights = task_weights or default_weights
        self._task_names = list(weights.keys())
        self._task_id_map = {n: i for i, n in enumerate(self._task_names)}

        # Build per-task datasets, sized proportionally
        self._task_datasets: dict[str, LongContextDataset] = {}
        for name, w in weights.items():
            n_this = max(8, int(n_samples * w))
            sub_seed = seed + self._task_id_map[name] * 1000
            if name == "niah_single":
                self._task_datasets[name] = NIAHSingle(
                    n_this, seq_len, vocab_size, num_classes, sub_seed
                )
            elif name == "niah_multi_key":
                self._task_datasets[name] = NIAHMultiKey(
                    n_this, seq_len, vocab_size, num_classes, sub_seed,
                    n_needles=4,
                )
            elif name == "variable_track":
                self._task_datasets[name] = VariableTracking(
                    n_this, seq_len, vocab_size, num_classes, sub_seed,
                    n_hops=3,
                )
            elif name == "fa":
                self._task_datasets[name] = FrequencyAggregation(
                    n_this, seq_len, vocab_size, num_classes, sub_seed,
                    n_items=16,
                )

        # Flat index over all task samples
        self._index: list[tuple[str, int]] = []
        for name, ds in self._task_datasets.items():
            for i in range(len(ds)):
                self._index.append((name, i))
        rng.shuffle(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        name, sub_idx = self._index[idx]
        ids, label = self._task_datasets[name][sub_idx]
        return ids, label, torch.tensor(self._task_id_map[name], dtype=torch.long)

    @property
    def task_names(self) -> list[str]:
        return list(self._task_names)
