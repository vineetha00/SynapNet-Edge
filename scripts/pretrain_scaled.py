"""Pretrain a scaled SynapNet-Edge model on the multi-task long-context curriculum.

Default config:
  dim=256, depth=8, heads=8, max_len=2048, vocab=4096, num_classes=64
  ≈ 8M parameters — roughly Mamba-2 130M-style aspect ratio at smaller scale,
  small enough to converge on consumer hardware in <30 minutes.

Tasks: NIAH single + multi-key, variable tracking, frequency aggregation.
The training curriculum progresses from short (256) → long (2048) sequences
to encourage the model to actually use episodic memory.

Output:
  results/scaled/base_model_fp16.pt        — FP16 base checkpoint
  results/scaled/training_curve.json       — per-step loss/acc
  results/scaled/per_task_accuracy.json    — final per-task accuracy
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.data.long_context_tasks import MultiTaskCurriculum


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--episodic-slots", type=int, default=32)
    p.add_argument("--episodic-write-frac", type=float, default=0.05)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--curriculum", nargs="+", type=int, default=[512, 1024, 2048],
                   help="Curriculum: train at each context length in turn")
    p.add_argument("--samples-per-stage", type=int, default=4096)
    p.add_argument("--epochs-per-stage", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/scaled")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def lr_schedule(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int) -> dict:
    model.eval()
    correct = {}
    total = {}
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                ids, lbl, tid = batch
            else:
                ids, lbl = batch
                tid = torch.zeros(ids.size(0), dtype=torch.long)
            ids, lbl = ids.to(device), lbl.to(device)
            logits = model(ids)[0]
            if logits.dim() == 3:
                pred = logits[:, -1, :num_classes].argmax(-1)
            else:
                pred = logits[:, :num_classes].argmax(-1)
            for t in tid.unique().tolist():
                m = (tid == t)
                correct[t] = correct.get(t, 0) + (pred[m.to(device)] == lbl[m.to(device)]).sum().item()
                total[t] = total.get(t, 0) + int(m.sum().item())
    accs = {int(t): correct[t] / max(1, total[t]) for t in total}
    overall = sum(correct.values()) / max(1, sum(total.values()))
    return {"overall": overall, "per_task": accs}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build model (scale: ~8-12M params)
    # ------------------------------------------------------------------
    cfg = SynapNetEdgeConfig(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        vocab_size=args.vocab,
        max_len=max(args.max_len, max(args.curriculum)),
        num_classes=args.num_classes,
        k_frac=0.25,
        episodic_slots=args.episodic_slots,
        episodic_write_frac=args.episodic_write_frac,
        mlp_ratio=4.0,
        use_scale_bridge=True,
    )
    model = SynapNetEdge(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain_scaled] device={device} | model: {n_params/1e6:.2f}M params "
          f"({n_params:,})")
    print(f"  dim={args.dim} depth={args.depth} heads={args.heads} "
          f"slots={args.episodic_slots} max_len={cfg.max_len}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    history = []
    eval_history = []
    global_step = 0
    total_steps_est = sum(
        args.epochs_per_stage * (args.samples_per_stage // args.batch_size)
        for _ in args.curriculum
    )
    t_start = time.time()

    for stage_idx, ctx_len in enumerate(args.curriculum):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx+1}/{len(args.curriculum)}: context length {ctx_len}")
        print(f"{'='*60}")

        train_ds = MultiTaskCurriculum(
            n_samples=args.samples_per_stage,
            seq_len=ctx_len,
            vocab_size=args.vocab,
            num_classes=args.num_classes,
            seed=args.seed + stage_idx,
        )
        eval_ds = MultiTaskCurriculum(
            n_samples=max(256, args.samples_per_stage // 8),
            seq_len=ctx_len,
            vocab_size=args.vocab,
            num_classes=args.num_classes,
            seed=args.seed + 9000 + stage_idx,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        for ep in range(args.epochs_per_stage):
            model.train()
            ep_loss = 0.0
            ep_correct = 0
            ep_total = 0
            for ids, lbl, _tid in train_loader:
                lr_now = lr_schedule(global_step, args.warmup_steps,
                                     total_steps_est, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                ids, lbl = ids.to(device), lbl.to(device)
                optimizer.zero_grad()
                logits = model(ids)[0]
                if logits.dim() == 3:
                    pred_logits = logits[:, -1, :args.num_classes]
                else:
                    pred_logits = logits[:, :args.num_classes]
                loss = F.cross_entropy(pred_logits, lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                with torch.no_grad():
                    preds = pred_logits.argmax(-1)
                    ep_correct += (preds == lbl).sum().item()
                    ep_total += lbl.size(0)
                ep_loss += loss.item()

                if global_step % args.log_every == 0:
                    train_acc = ep_correct / max(1, ep_total)
                    elapsed = time.time() - t_start
                    print(f"  step {global_step:5d} | ctx={ctx_len} | "
                          f"loss={loss.item():.4f} | train_acc={train_acc:.3f} | "
                          f"lr={lr_now:.2e} | {elapsed:.0f}s")
                    history.append({
                        "step": global_step, "ctx_len": ctx_len,
                        "loss": loss.item(), "train_acc": train_acc,
                        "lr": lr_now,
                    })
                global_step += 1

            # End of epoch eval
            eval_res = evaluate(model, eval_loader, device, args.num_classes)
            print(f"  [eval] stage={stage_idx+1} ep={ep+1} ctx={ctx_len} "
                  f"overall_acc={eval_res['overall']:.3f}")
            print(f"        per-task: {eval_res['per_task']}")
            eval_history.append({
                "step": global_step, "stage": stage_idx, "ctx_len": ctx_len,
                "epoch": ep, **eval_res,
            })

    elapsed = time.time() - t_start
    print(f"\n[pretrain_scaled] Total time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    ckpt = {
        "model_state": model.state_dict(),
        "config": vars(args),
        "model_cfg": cfg.__dict__,
        "n_params": n_params,
        "training_time_s": elapsed,
    }
    torch.save(ckpt, out_dir / "base_model_fp16.pt")
    print(f"[pretrain_scaled] Checkpoint saved to {out_dir / 'base_model_fp16.pt'}")

    with open(out_dir / "training_curve.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2)
    with open(out_dir / "per_task_accuracy.json", "w") as f:
        json.dump(eval_history[-1] if eval_history else {}, f, indent=2)

    print(f"[pretrain_scaled] Final overall acc: "
          f"{eval_history[-1]['overall'] if eval_history else 'N/A':.3f}")


if __name__ == "__main__":
    main()
