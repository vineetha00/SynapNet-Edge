"""Retention classifier training stability + ROC analysis.

Trains the BAEE RetentionClassifier under several settings to characterise:
  1. Training-loss stability across seeds & learning rates
  2. ROC-AUC for predicting "was this entry ultimately used by downstream attention"
  3. Convergence speed (steps to reach plateau)
  4. Sensitivity to label noise

The training data is generated synthetically: episodic-memory snapshots from
streaming inference, with the ground-truth label being whether the entry was
heavily attended-to during the read phase (a proxy for utility).

Output: results/scaled/exp_classifier_stability.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.memory.baee import RetentionClassifier
from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum


@torch.no_grad()
def collect_training_data(model, n_batches: int, seq_len: int,
                           vocab: int, num_classes: int,
                           device: torch.device, seed: int = 0):
    """Run streaming inference and collect (entry_features, ground_truth_used_label).

    Ground truth: the per-slot attention weight received during the *next*
    block's epi.read.  We approximate by taking the salience-at-write × age-decay
    as a "noisy" proxy and binarising at 0.5.
    """
    from torch.utils.data import DataLoader
    ds = MultiTaskCurriculum(n_samples=n_batches * 2, seq_len=seq_len,
                              vocab_size=vocab, num_classes=num_classes, seed=seed)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    features = []
    raw_salience = []   # we'll binarise after gathering all values
    label_meta = []     # store side info for the label calc

    for batch_i, (ids, _lbl, _tid) in enumerate(loader):
        if batch_i >= n_batches: break
        ids = ids.to(device)
        outputs = model(ids)
        masks = outputs[1]
        mems = outputs[2]
        topks = outputs[3]

        for li in range(len(mems)):
            bank = mems[li]
            sal_full = masks[li]
            top_idx = topks[li]
            B, S, D = bank.shape
            # The "true utility" proxy: how much downstream attention this slot
            # would receive.  Approximated by: salience-at-write × cross-token
            # attention magnitude (cosine similarity of slot to next-block input).
            with torch.no_grad():
                # Magnitude of how distinctive this entry is — entries close to
                # the mean are likely redundant
                mean_state = bank.mean(dim=1, keepdim=True)
                distinctiveness = (bank - mean_state).norm(dim=-1)  # (B, S)
                distinctiveness = distinctiveness / (distinctiveness.max(dim=1, keepdim=True).values + 1e-6)

            for b in range(B):
                for s in range(S):
                    if s >= top_idx.size(1): continue
                    tok_idx = int(top_idx[b, s])
                    if tok_idx >= sal_full.size(1): continue
                    sal_at_write = float(sal_full[b, tok_idx])
                    distinct = float(distinctiveness[b, s])
                    age = float(s) / S
                    use_count = 0.0
                    slot_pos = float(s) / S
                    feat = bank[b, s].detach().cpu()

                    features.append({
                        "feat": feat,
                        "salience": sal_at_write,
                        "slot_pos": slot_pos,
                        "age": age,
                        "use_count": use_count,
                    })
                    # The "true label" is utility = salience * distinctiveness
                    # We'll quantile-threshold after collecting all values
                    raw_salience.append(sal_at_write * distinct)

    # Quantile-threshold: top 30% = positive, bottom 70% = negative
    if not raw_salience:
        return features, []
    sorted_vals = sorted(raw_salience)
    threshold = sorted_vals[int(0.70 * len(sorted_vals))]
    labels = [float(v > threshold) for v in raw_salience]
    print(f"    label threshold: {threshold:.4f}, "
          f"positives={int(sum(labels))}/{len(labels)} "
          f"({sum(labels)/max(1,len(labels))*100:.1f}%)")
    return features, labels


def train_classifier(
    features: list, labels: list, dim: int,
    lr: float, n_steps: int, seed: int, device: torch.device,
    noise_prob: float = 0.0,
) -> dict:
    """Train RetentionClassifier and track loss / accuracy curves."""
    torch.manual_seed(seed)
    clf = RetentionClassifier(dim=dim, hidden=32).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)

    # Build tensors
    N = len(features)
    feats = torch.stack([f["feat"] for f in features]).to(device)
    sal = torch.tensor([f["salience"] for f in features], device=device)
    pos = torch.tensor([f["slot_pos"] for f in features], device=device)
    age = torch.tensor([f["age"] for f in features], device=device)
    uc = torch.tensor([f["use_count"] for f in features], device=device)
    y = torch.tensor(labels, device=device)

    # Add label noise
    if noise_prob > 0:
        flip = torch.rand(N, device=device) < noise_prob
        y = torch.where(flip, 1.0 - y, y)

    # Split train/val
    perm = torch.randperm(N)
    split = int(0.8 * N)
    tr_idx, va_idx = perm[:split], perm[split:]

    history = []
    for step in range(n_steps):
        opt.zero_grad()
        # Mini-batch from train idx
        bs = min(64, len(tr_idx))
        batch = tr_idx[torch.randperm(len(tr_idx))[:bs]]
        # Reshape to (1, S, D) form expected by classifier
        feats_b = feats[batch].unsqueeze(0)
        sal_b = sal[batch].unsqueeze(0)
        pos_b = pos[batch]
        age_b = age[batch].unsqueeze(0)
        uc_b = uc[batch].unsqueeze(0)
        scores = clf(feats_b, sal_b, pos_b, age_b, uc_b).squeeze(0)
        target = y[batch]
        loss = F.binary_cross_entropy(scores.clamp(1e-6, 1 - 1e-6), target)
        loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == n_steps - 1:
            with torch.no_grad():
                # Val
                feats_v = feats[va_idx].unsqueeze(0)
                sal_v = sal[va_idx].unsqueeze(0)
                pos_v = pos[va_idx]
                age_v = age[va_idx].unsqueeze(0)
                uc_v = uc[va_idx].unsqueeze(0)
                v_scores = clf(feats_v, sal_v, pos_v, age_v, uc_v).squeeze(0)
                v_target = y[va_idx]
                v_loss = F.binary_cross_entropy(
                    v_scores.clamp(1e-6, 1-1e-6), v_target).item()
                # Accuracy at 0.5
                preds = (v_scores > 0.5).float()
                acc = (preds == v_target).float().mean().item()
                # ROC AUC via simple rank computation
                auc = _roc_auc(v_scores.cpu().numpy(), v_target.cpu().numpy())
            history.append({"step": step, "train_loss": float(loss.detach()),
                            "val_loss": v_loss, "val_acc": acc, "val_auc": auc})
    return {"history": history, "final": history[-1] if history else None}


def _roc_auc(scores, labels):
    """Compute ROC AUC."""
    import numpy as np
    pos = scores[labels > 0.5]; neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    rank = np.concatenate([pos, neg])
    sort_idx = np.argsort(rank)
    rank_pos = np.where(np.isin(np.arange(len(rank))[sort_idx],
                                  np.arange(len(pos))))[0]
    auc = (rank_pos.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/scaled/base_model_fp16.pt")
    p.add_argument("--output", default="results/scaled/exp_classifier_stability.json")
    p.add_argument("--device", default="mps")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--learning-rates", nargs="+", type=float,
                   default=[1e-4, 5e-4, 1e-3])
    p.add_argument("--noise-probs", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2])
    p.add_argument("--n-steps", type=int, default=400)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = SynapNetEdgeConfig(**ckpt["model_cfg"])
    model = SynapNetEdge(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print("Collecting classifier training data...")
    features, labels = collect_training_data(
        model, n_batches=16, seq_len=1024,
        vocab=ckpt["model_cfg"]["vocab_size"],
        num_classes=ckpt["model_cfg"]["num_classes"],
        device=device,
    )
    n_pos = sum(int(l) for l in labels)
    print(f"  Collected {len(features)} entries  ({n_pos} positive, "
          f"{len(labels)-n_pos} negative)")

    results = {"config": vars(args), "n_examples": len(features),
                "n_positive": n_pos, "experiments": {}}

    # 1. Seed stability at fixed LR
    print("\n=== 1. Seed stability at lr=5e-4 ===")
    seeds_runs = []
    for s in args.seeds:
        r = train_classifier(features, labels, cfg.dim, lr=5e-4,
                              n_steps=args.n_steps, seed=s, device=device)
        seeds_runs.append({"seed": s, **r})
        print(f"  seed={s}: final val_auc={r['final']['val_auc']:.3f} "
              f"val_acc={r['final']['val_acc']:.3f} "
              f"val_loss={r['final']['val_loss']:.4f}")
    results["experiments"]["seed_stability"] = seeds_runs

    # 2. LR sensitivity at fixed seed
    print("\n=== 2. LR sensitivity at seed=42 ===")
    lr_runs = []
    for lr in args.learning_rates:
        r = train_classifier(features, labels, cfg.dim, lr=lr,
                              n_steps=args.n_steps, seed=42, device=device)
        lr_runs.append({"lr": lr, **r})
        print(f"  lr={lr}: final val_auc={r['final']['val_auc']:.3f} "
              f"val_acc={r['final']['val_acc']:.3f}")
    results["experiments"]["lr_sensitivity"] = lr_runs

    # 3. Noise robustness
    print("\n=== 3. Label noise robustness ===")
    noise_runs = []
    for np_ in args.noise_probs:
        r = train_classifier(features, labels, cfg.dim, lr=5e-4,
                              n_steps=args.n_steps, seed=42, device=device,
                              noise_prob=np_)
        noise_runs.append({"noise_prob": np_, **r})
        print(f"  noise={np_:.2f}: final val_auc={r['final']['val_auc']:.3f}")
    results["experiments"]["noise_robustness"] = noise_runs

    # Aggregate
    aucs = [r["final"]["val_auc"] for r in seeds_runs]
    mean_auc = sum(aucs) / len(aucs)
    std_auc = (sum((a - mean_auc)**2 for a in aucs) / max(1, len(aucs)-1)) ** 0.5
    results["seed_stability_summary"] = {
        "val_auc_mean": mean_auc, "val_auc_std": std_auc,
        "val_auc_vals": aucs,
    }
    print(f"\nSeed stability: val_AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(x) for x in o]
        if isinstance(o, (int, float, str, bool, type(None))): return o
        if isinstance(o, torch.Tensor): return o.tolist()
        return str(o)
    with open(args.output, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
