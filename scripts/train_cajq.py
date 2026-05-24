"""Full CAJQ training script for SynapNetEdge.

Runs the three-phase QAT pipeline:
  Phase 1: FP16 warm-up on episodic memory recall task
  Phase 2: Apply CAJQ quantization + QAT training
  Phase 3: QAT fine-tune / cooldown

Usage:
  python scripts/train_cajq.py --dim 256 --depth 6 --epochs 10 --device mps
  python scripts/train_cajq.py --config configs/cajq_config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.training.qat_trainer import QATTrainer, QATConfig
from synapnet_edge.training.calibration import SyntheticCalibDataset, build_calib_loader
from synapnet_edge.memory.baee import BAEEMemoryManager, EvictionPolicy
from synapnet_edge.utils.visualization import plot_training_history

from torch.utils.data import DataLoader, random_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SynapNet-Edge QAT training")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--episodic-slots", type=int, default=16)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--qat-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=2048)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-baee", type=float, default=0.1)
    p.add_argument("--lambda-sal", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    torch.manual_seed(args.seed)
    print(f"[train_cajq] device={args.device}, dim={args.dim}, depth={args.depth}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    cfg = SynapNetEdgeConfig(
        dim=args.dim,
        depth=args.depth,
        vocab_size=args.vocab,
        max_len=args.seq_len,
        num_classes=args.num_classes,
        heads=args.heads,
        k_frac=0.25,
        episodic_slots=args.episodic_slots,
        episodic_write_frac=0.05,
        use_scale_bridge=True,
    )
    model = SynapNetEdge(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_cajq] Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = SyntheticCalibDataset(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    eval_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    calib_loader = build_calib_loader(
        n_samples=256, seq_len=args.seq_len,
        vocab_size=args.vocab, num_classes=args.num_classes,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # BAEE manager
    # ------------------------------------------------------------------
    baee_manager = BAEEMemoryManager(
        dim=args.dim,
        n_layers=args.depth,
        slots_per_layer=args.episodic_slots,
        budget_mb=256.0,
        policy=EvictionPolicy.BAEE,
    )

    # ------------------------------------------------------------------
    # QAT trainer
    # ------------------------------------------------------------------
    qat_cfg = QATConfig(
        warmup_epochs=args.warmup_epochs,
        qat_epochs=args.qat_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        lr_warmup=args.lr,
        lr_qat=args.lr / 2,
        lr_finetune=args.lr / 10,
        lambda_baee=args.lambda_baee,
        lambda_sal_sup=args.lambda_sal,
        n_calib_batches=32,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )

    trainer = QATTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        calib_loader=calib_loader,
        cfg=qat_cfg,
        task="classification",
        baee_manager=baee_manager,
    )

    history = trainer.train()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, str(output_dir / "training_history.pdf"))

    # Save final model
    torch.save(model.state_dict(), output_dir / "synapnet_edge_final.pt")
    print(f"\n[train_cajq] Done. Results saved to {output_dir}/")

    # Final eval
    final_acc = history[-1].get("eval_acc")
    if final_acc is not None:
        print(f"[train_cajq] Final eval accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
