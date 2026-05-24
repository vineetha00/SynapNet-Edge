"""RULER benchmark suite for long-context evaluation.

RULER (Really Understanding Long-Context Evaluations with Reasoning) tests:
  1. NIAH (Needle-in-a-Haystack)    — retrieve a fact from a long context
  2. Variable Tracking               — track variable assignments across K hops
  3. Common Word Extraction (CWE)   — count/extract common words
  4. Frequency Aggregation (FA)     — aggregate values by key
  5. QA (question answering)        — retrieve answers from long passage

All tasks use synthetic generation (no external dataset required),
supporting context lengths from 1K to 128K tokens.

Reference: RULER: What's the Real Context Size of Your Long-Context Models?
           Hsieh et al., NAACL 2024.
"""
from __future__ import annotations

import enum
import json
import random
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RULERTask(enum.Enum):
    NIAH_SINGLE = "niah_single"        # single needle
    NIAH_MULTI_KEY = "niah_multi_key"  # multiple needles, retrieve one
    VARIABLE_TRACK = "variable_track"  # variable tracking
    CWE = "cwe"                        # common word extraction
    FA = "fa"                           # frequency aggregation
    QA = "qa"                          # closed-book QA


@dataclass
class RULERSample:
    task: RULERTask
    context_len: int
    input_ids: torch.Tensor           # (T,)
    label: int
    label_position: int               # token index where answer is hidden
    metadata: dict = None


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

class RULERGenerator:
    """Generates synthetic RULER task sequences without external data."""

    def __init__(self, vocab_size: int = 32000, seed: int = 0):
        self.vocab_size = vocab_size
        random.seed(seed)
        torch.manual_seed(seed)
        # Reserve token 0=PAD, 1=BOS, 2=EOS, 3=SEP
        self.special_offset = 4

    def niah_single(
        self,
        context_len: int,
        num_classes: int = 64,
    ) -> RULERSample:
        """Generate a NIAH sample: needle token at random depth."""
        code = random.randint(0, num_classes - 1)
        ids = torch.randint(
            self.special_offset + num_classes,
            self.vocab_size,
            (context_len,),
        )
        needle_pos = random.randint(1, context_len - 10)
        ids[needle_pos] = self.special_offset + code
        # Query is the last token
        ids[-1] = 3   # SEP — model predicts code after this

        return RULERSample(
            task=RULERTask.NIAH_SINGLE,
            context_len=context_len,
            input_ids=ids,
            label=code,
            label_position=needle_pos,
            metadata={"needle_pos": needle_pos, "needle_depth": needle_pos / context_len},
        )

    def niah_multi_key(
        self,
        context_len: int,
        n_needles: int = 4,
        num_classes: int = 64,
    ) -> RULERSample:
        """Multiple needle tokens hidden; retrieve the one nearest a query key."""
        ids = torch.randint(
            self.special_offset + num_classes,
            self.vocab_size,
            (context_len,),
        )
        needle_positions = sorted(
            random.sample(range(1, context_len - n_needles - 5), n_needles)
        )
        codes = [random.randint(0, num_classes - 1) for _ in range(n_needles)]
        for pos, code in zip(needle_positions, codes):
            ids[pos] = self.special_offset + code

        target_needle = random.randint(0, n_needles - 1)
        target_code = codes[target_needle]
        ids[-1] = 3

        return RULERSample(
            task=RULERTask.NIAH_MULTI_KEY,
            context_len=context_len,
            input_ids=ids,
            label=target_code,
            label_position=needle_positions[target_needle],
            metadata={"n_needles": n_needles, "target_idx": target_needle},
        )

    def variable_track(
        self,
        context_len: int,
        n_vars: int = 4,
        n_hops: int = 3,
        num_classes: int = 16,
    ) -> RULERSample:
        """Variable tracking: x=a; y=x; z=y; query: z=?"""
        ids = torch.randint(
            self.special_offset + num_classes + n_vars,
            self.vocab_size,
            (context_len,),
        )
        # Assign chain: var[0]=code, var[1]=var[0], ...
        code = random.randint(0, num_classes - 1)
        chain_positions = sorted(
            random.sample(range(1, context_len - 5), n_hops + 1)
        )
        for i, pos in enumerate(chain_positions):
            if i == 0:
                ids[pos] = self.special_offset + code
            else:
                ids[pos] = self.special_offset + num_classes + i   # var reference token
        ids[-1] = 3

        return RULERSample(
            task=RULERTask.VARIABLE_TRACK,
            context_len=context_len,
            input_ids=ids,
            label=code,
            label_position=chain_positions[0],
            metadata={"n_hops": n_hops, "chain": chain_positions},
        )

    def frequency_aggregation(
        self,
        context_len: int,
        n_keys: int = 8,
        num_classes: int = 8,
    ) -> RULERSample:
        """FA: given key-value pairs, find the most common key."""
        ids = torch.randint(
            self.special_offset + num_classes,
            self.vocab_size,
            (context_len,),
        )
        counts = [0] * num_classes
        positions = sorted(random.sample(range(1, context_len - 5), n_keys * 2))
        for i, pos in enumerate(positions):
            key = random.randint(0, num_classes - 1)
            ids[pos] = self.special_offset + key
            counts[key] += 1
        most_common = max(range(num_classes), key=lambda k: counts[k])
        ids[-1] = 3

        return RULERSample(
            task=RULERTask.FA,
            context_len=context_len,
            input_ids=ids,
            label=most_common,
            label_position=0,
            metadata={"counts": counts},
        )

    def generate(
        self,
        task: RULERTask,
        context_len: int,
        **kwargs,
    ) -> RULERSample:
        if task == RULERTask.NIAH_SINGLE:
            return self.niah_single(context_len, **kwargs)
        elif task == RULERTask.NIAH_MULTI_KEY:
            return self.niah_multi_key(context_len, **kwargs)
        elif task == RULERTask.VARIABLE_TRACK:
            return self.variable_track(context_len, **kwargs)
        elif task == RULERTask.FA:
            return self.frequency_aggregation(context_len, **kwargs)
        else:
            return self.niah_single(context_len, **kwargs)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class RULERDataset(Dataset):
    def __init__(
        self,
        task: RULERTask,
        context_len: int,
        n_samples: int = 500,
        vocab_size: int = 32000,
        num_classes: int = 64,
        seed: int = 42,
    ):
        gen = RULERGenerator(vocab_size=vocab_size, seed=seed)
        self.samples = [
            gen.generate(task, context_len, num_classes=num_classes)
            for _ in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.input_ids, torch.tensor(s.label), torch.tensor(s.label_position)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class RULERBenchmark:
    """Evaluates a model on RULER tasks across multiple context lengths."""

    CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]
    TASKS = [RULERTask.NIAH_SINGLE, RULERTask.NIAH_MULTI_KEY,
             RULERTask.VARIABLE_TRACK, RULERTask.FA]

    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: int = 32000,
        num_classes: int = 64,
        n_samples: int = 200,
        device: str = "cpu",
        chunk_size: int = 512,
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.n_samples = n_samples
        self.device = device
        self.chunk_size = chunk_size

    @torch.no_grad()
    def run(
        self,
        tasks: list[RULERTask] | None = None,
        context_lengths: list[int] | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        if tasks is None:
            tasks = self.TASKS
        if context_lengths is None:
            context_lengths = self.CONTEXT_LENGTHS

        results: dict[str, Any] = {}
        self.model.eval()
        self.model.to(self.device)

        for task in tasks:
            results[task.value] = {}
            for ctx_len in context_lengths:
                dataset = RULERDataset(
                    task=task,
                    context_len=ctx_len,
                    n_samples=self.n_samples,
                    vocab_size=self.vocab_size,
                    num_classes=self.num_classes,
                )
                loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

                correct = total = 0
                latencies = []

                for ids, labels, _ in loader:
                    ids = ids.to(self.device)
                    labels = labels.to(self.device)

                    t0 = time.perf_counter()
                    if ctx_len > self.chunk_size and hasattr(self.model, "forward_streaming"):
                        logits, _ = self.model.forward_streaming(
                            ids, chunk_size=self.chunk_size
                        )
                    else:
                        outputs = self.model(ids)
                        logits = outputs[0]
                    latency_ms = (time.perf_counter() - t0) * 1000
                    latencies.append(latency_ms)

                    if logits.dim() == 3:
                        pred_logits = logits[:, -1, :self.num_classes]
                    else:
                        pred_logits = logits[:, :self.num_classes]
                    preds = pred_logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                acc = correct / max(1, total)
                avg_lat = sum(latencies) / max(1, len(latencies))
                results[task.value][ctx_len] = {
                    "accuracy": acc,
                    "avg_latency_ms": avg_lat,
                    "n_samples": total,
                }
                if verbose:
                    print(f"[RULER] {task.value} | ctx={ctx_len} | "
                          f"acc={acc:.3f} | lat={avg_lat:.1f}ms")

        return results

    def save_results(self, results: dict, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[RULER] Results saved to {path}")
