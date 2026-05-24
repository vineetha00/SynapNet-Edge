"""Calibration dataset and dataloader utilities for PTQ/QAT.

Provides:
  - CalibrationDataset: wraps any token-sequence dataset for calibration
  - build_calib_loader: constructs a small DataLoader for quantization calibration
  - EpisodicMemoryCalibDataset: synthetic sequences for BAEE calibration
  - LongContextCalibDataset: sequences up to max_len for attention calibration
"""
from __future__ import annotations

import random
import torch
from torch.utils.data import Dataset, DataLoader


class CalibrationDataset(Dataset):
    """Wraps a pre-tokenised tensor dataset for calibration.

    Args:
        input_ids: (N, T) int64 tensor
        labels:    (N,) or (N, T) int64 tensor; if None, uses input_ids shifted by 1
    """

    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        self.input_ids = input_ids
        if labels is None:
            self.labels = input_ids[:, 1:]   # next-token prediction
        else:
            self.labels = labels

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


class SyntheticCalibDataset(Dataset):
    """Synthetic random token sequences for calibration when no real data is available.

    Generates sequences with embedded "needle" tokens to exercise
    episodic memory recall paths.
    """

    def __init__(
        self,
        n_samples: int = 128,
        seq_len: int = 2048,
        vocab_size: int = 32000,
        num_classes: int = 32,
        seed: int = 0,
    ):
        random.seed(seed)
        torch.manual_seed(seed)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.input_ids = []
        self.labels = []

        for _ in range(n_samples):
            ids = torch.randint(4, vocab_size, (seq_len,))
            code = random.randint(0, num_classes - 1)
            ids[2] = code   # embed "fact" at position 2 (as in original SynapNet)
            # Embed query tokens at end
            ids[-3:] = torch.tensor([code + vocab_size // 2,
                                     vocab_size - 1, code])
            self.input_ids.append(ids)
            self.labels.append(torch.tensor(code))

        self.input_ids = torch.stack(self.input_ids)
        self.labels = torch.stack(self.labels)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


class LongContextCalibDataset(Dataset):
    """Random long-context sequences to calibrate attention quantization.

    Targets the full max_len range to expose activation outliers
    that SmoothQuant needs to absorb.
    """

    def __init__(
        self,
        n_samples: int = 64,
        max_len: int = 8192,
        vocab_size: int = 32000,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.data = torch.randint(4, vocab_size, (n_samples, max_len))
        self.labels = torch.zeros(n_samples, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def build_calib_loader(
    n_samples: int = 128,
    seq_len: int = 2048,
    vocab_size: int = 32000,
    num_classes: int = 32,
    batch_size: int = 4,
    seed: int = 0,
) -> DataLoader:
    """Build a calibration DataLoader using synthetic data."""
    dataset = SyntheticCalibDataset(
        n_samples=n_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_classes=num_classes,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def build_long_context_calib_loader(
    n_samples: int = 64,
    max_len: int = 8192,
    vocab_size: int = 32000,
    batch_size: int = 1,
) -> DataLoader:
    """Build a long-context calibration DataLoader for attention quantization."""
    dataset = LongContextCalibDataset(n_samples=n_samples, max_len=max_len,
                                      vocab_size=vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
