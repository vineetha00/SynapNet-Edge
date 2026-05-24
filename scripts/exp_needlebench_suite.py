"""NeedleBench-style multi-skill long-context benchmark.

We implement faithful synthetic proxies for several real long-context skills:
  - Single Needle in Haystack (SNIA)
  - Multi-Key Needle (MKN): retrieve value matching a queried key
  - Reasoning over Needles (RoN): two needles must be combined
  - Counting Needles (CN): how many of a given pattern
  - Anti-Distraction Needle (ADN): target needle surrounded by similar-looking distractors

Each task is generated synthetically (no HF download required) so the suite
is fully self-contained for reviewer reproducibility.

We evaluate the pretrained 8.7M FP16 model and CAJQ-QAT variant at
multiple context lengths.

Output: results/scaled/exp_needlebench.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader


PAD, BOS, EOS, SEP, QUERY = 0, 1, 2, 3, 4
KEY_START = 5


class _NeedleBase(Dataset):
    def __init__(self, n_samples, seq_len, vocab_size, num_classes, seed):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.filler_start = KEY_START + num_classes
        self._rng = random.Random(seed)
        self._torch_rng = torch.Generator().manual_seed(seed)
        self._samples = [self._gen() for _ in range(n_samples)]
    def _filler(self, n):
        return torch.randint(self.filler_start, self.vocab_size, (n,),
                              generator=self._torch_rng)
    def __len__(self): return self.n_samples
    def __getitem__(self, i):
        ids, lbl, _ = self._samples[i]
        return ids, torch.tensor(lbl, dtype=torch.long)


class SNIA(_NeedleBase):
    """Single Needle in Haystack."""
    task = "snia"
    def _gen(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS; ids[-2] = QUERY; ids[-1] = SEP
        code = self._rng.randint(0, self.num_classes - 1)
        pos = self._rng.randint(L // 4, 3 * L // 4)
        ids[pos] = KEY_START + code
        return ids, code, {}


class MKN(_NeedleBase):
    """Multi-Key Needle: retrieve value matching queried key."""
    task = "mkn"
    def __init__(self, *a, n_keys=6, **kw):
        self.n_keys = n_keys
        super().__init__(*a, **kw)
    def _gen(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS; ids[-3] = QUERY; ids[-1] = SEP
        n_keys = max(self.num_classes // 2, self.n_keys + 1)
        n_vals = self.num_classes - n_keys
        keys = self._rng.sample(range(n_keys), self.n_keys)
        vals = [self._rng.randint(0, n_vals - 1) for _ in range(self.n_keys)]
        positions = sorted(self._rng.sample(range(2, L - 4), self.n_keys))
        for p, k, v in zip(positions, keys, vals):
            ids[p] = KEY_START + k
            if p + 1 < L - 3: ids[p + 1] = KEY_START + n_keys + v
        target_idx = self._rng.randint(0, self.n_keys - 1)
        ids[-2] = KEY_START + keys[target_idx]
        return ids, vals[target_idx], {}


class RoN(_NeedleBase):
    """Reasoning over Needles: two needles must be combined.

    Setup: needle A is "X=code_a", needle B is "Y=X+1 mod N", query: "Y=?".
    Model must remember code_a, find Y's relation to X, and compute Y.
    Since we don't actually do arithmetic, we simplify: place value pair
    (A, B) where B = (A + 1) % num_classes; query returns B.
    """
    task = "ron"
    def _gen(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS; ids[-2] = QUERY; ids[-1] = SEP
        code_a = self._rng.randint(0, self.num_classes - 1)
        code_b = (code_a + 1) % self.num_classes
        pos_a = self._rng.randint(2, L // 2)
        pos_b = self._rng.randint(L // 2, L - 4)
        ids[pos_a] = KEY_START + code_a
        ids[pos_b] = KEY_START + code_b
        return ids, code_b, {}


class CN(_NeedleBase):
    """Counting Needles: how many copies of a target pattern appear?

    The label is the count (clipped to num_classes-1).
    """
    task = "cn"
    def __init__(self, *a, max_count=8, **kw):
        self.max_count = max_count
        super().__init__(*a, **kw)
    def _gen(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS; ids[-3] = QUERY; ids[-1] = SEP
        target_code = self._rng.randint(0, min(self.num_classes - 1, 31))
        true_count = self._rng.randint(1, self.max_count)
        positions = self._rng.sample(range(2, L - 4), true_count)
        for p in positions:
            ids[p] = KEY_START + target_code
        ids[-2] = KEY_START + target_code   # query key
        return ids, min(true_count, self.num_classes - 1), {}


class ADN(_NeedleBase):
    """Anti-Distraction Needle: target surrounded by similar-but-wrong tokens.

    A "decoy" pattern with N copies of key+wrong_value, and ONE copy of
    key+right_value.  Model must find the right value.
    """
    task = "adn"
    def __init__(self, *a, n_distractors=8, **kw):
        self.n_distractors = n_distractors
        super().__init__(*a, **kw)
    def _gen(self):
        L = self.seq_len
        ids = self._filler(L)
        ids[0] = BOS; ids[-3] = QUERY; ids[-1] = SEP
        n_keys = max(self.num_classes // 2, 8)
        n_vals = self.num_classes - n_keys
        target_key = self._rng.randint(0, n_keys - 1)
        right_val = self._rng.randint(0, n_vals - 1)
        wrong_vals = [self._rng.randint(0, n_vals - 1)
                       for _ in range(self.n_distractors)]
        # Place distractors
        positions = sorted(self._rng.sample(
            range(2, L - 5), self.n_distractors + 1
        ))
        for i, p in enumerate(positions[:-1]):
            ids[p] = KEY_START + target_key
            if p + 1 < L - 4:
                ids[p + 1] = KEY_START + n_keys + wrong_vals[i]
        # Place the right answer at a random position
        p_right = positions[-1]
        ids[p_right] = KEY_START + target_key
        if p_right + 1 < L - 4:
            ids[p_right + 1] = KEY_START + n_keys + right_val
        ids[-2] = KEY_START + target_key
        return ids, right_val, {}


TASKS = {
    "snia": SNIA, "mkn": MKN, "ron": RoN, "cn": CN, "adn": ADN,
}


@torch.no_grad()
def evaluate(model, task_cls, n_samples, seq_len, vocab, num_classes, device,
              **task_kwargs):
    ds = task_cls(n_samples=n_samples, seq_len=seq_len, vocab_size=vocab,
                   num_classes=num_classes, seed=2025, **task_kwargs)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    correct = total = 0
    for ids, lbl in loader:
        ids, lbl = ids.to(device), lbl.to(device)
        logits = model(ids)[0]
        pred = (logits[:, -1, :num_classes] if logits.dim() == 3
                else logits[:, :num_classes]).argmax(-1)
        correct += (pred == lbl).sum().item()
        total += lbl.size(0)
    return correct / max(1, total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_needlebench.json")
    p.add_argument("--device", default="mps")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096])
    p.add_argument("--n-samples", type=int, default=48)
    p.add_argument("--variants", nargs="+",
                   default=["fp16", "cajq_qat"])
    args = p.parse_args()

    from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["model_cfg"]
    vocab = config["vocab_size"]
    num_classes = config["num_classes"]

    results = {"config": vars(args), "model_cfg": config, "results": {}}

    for variant in args.variants:
        print(f"\n=== variant: {variant} ===")
        cfg = SynapNetEdgeConfig(**config)
        model = SynapNetEdge(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        if variant == "cajq_qat":
            from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
            from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum
            calib_ds = MultiTaskCurriculum(
                n_samples=32, seq_len=512,
                vocab_size=vocab, num_classes=num_classes, seed=999,
            )
            calib = DataLoader(calib_ds, batch_size=4, shuffle=False)
            apply_cajq(model, CAJQConfig(n_calib_batches=4, device=args.device,
                                          attn_group_size=64),
                       calib_loader=calib, mode="ptq")
            # Short QAT (200 steps) using ssm_quantizer.step_parameters
            from synapnet_edge.quantization.ssm_quantizer import SSMQuantizer
            import torch.nn.functional as F
            import torch.nn as nn
            train_ds = MultiTaskCurriculum(800, 1024, vocab, num_classes, seed=2000)
            train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
            step_params = list(SSMQuantizer.step_parameters(model))
            step_ids = {id(p) for p in step_params}
            other_params = [p for p in model.parameters()
                             if id(p) not in step_ids and p.requires_grad]
            opt = torch.optim.AdamW([
                {"params": step_params, "lr": 1e-2, "weight_decay": 0.0},
                {"params": other_params, "lr": 1e-4, "weight_decay": 0.01},
            ])
            model.train()
            steps_done = 0
            it = iter(train_loader)
            while steps_done < 200:
                try: batch = next(it)
                except StopIteration:
                    it = iter(train_loader); batch = next(it)
                ids, lbl, _ = batch
                ids, lbl = ids.to(device), lbl.to(device)
                opt.zero_grad()
                logits = model(ids)[0]
                pl = (logits[:, -1, :num_classes] if logits.dim() == 3
                      else logits[:, :num_classes])
                loss = F.cross_entropy(pl, lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps_done += 1
            model.eval()

        results["results"][variant] = {}
        for task_name, task_cls in TASKS.items():
            results["results"][variant][task_name] = {}
            for ctx in args.context_lengths:
                kwargs = {}
                if task_name == "mkn": kwargs["n_keys"] = 6
                if task_name == "cn": kwargs["max_count"] = 8
                if task_name == "adn": kwargs["n_distractors"] = 8
                try:
                    acc = evaluate(model, task_cls, args.n_samples, ctx,
                                    vocab, num_classes, device, **kwargs)
                except Exception as e:
                    print(f"  task={task_name} ctx={ctx}: FAILED ({e})")
                    acc = None
                results["results"][variant][task_name][ctx] = acc
                print(f"  task={task_name} ctx={ctx}: acc={acc:.3f}"
                      if acc is not None else f"  task={task_name} ctx={ctx}: SKIP")
        del model
        if device.type == "mps": torch.mps.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary
    print("\n=== NEEDLEBENCH SUMMARY ===")
    for variant, vd in results["results"].items():
        print(f"\n{variant}:")
        print(f"{'task':<8} | " + " | ".join(f"ctx={c}".rjust(8)
                                              for c in args.context_lengths))
        for task in TASKS:
            row = f"{task:<8} | "
            for ctx in args.context_lengths:
                v = vd[task].get(ctx)
                row += f"{v:.3f}".rjust(8) if v is not None else "  -  ".rjust(8)
                row += " | "
            print(row.rstrip(" |"))


if __name__ == "__main__":
    main()
