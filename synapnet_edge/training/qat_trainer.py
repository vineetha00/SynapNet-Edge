"""QAT training pipeline for SynapNet-Edge.

Training strategy:
  Phase 1 (fp16 warm-up):  Train base SynapNetEdge in FP16 for `warmup_epochs`.
  Phase 2 (QAT):           Apply CAJQ (SSM 2-bit + attention INT4 calibration),
                           then continue training with fake-quantized weights
                           and the SSM step-size regularisation loss.
  Phase 3 (fine-tune QAT): Short cooldown with lower LR to stabilise quantized weights.

Losses:
  L_total = L_task + λ_qat * L_qat_reg + λ_baee * L_baee_aux

where:
  L_task:    cross-entropy (classification) or next-token prediction (LM)
  L_qat_reg: SSM step-size regularisation from SSMQuantizer
  L_baee:    retention classifier auxiliary loss from BAEEMemoryManager
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class QATConfig:
    # Training
    warmup_epochs: int = 3
    qat_epochs: int = 10
    finetune_epochs: int = 2
    batch_size: int = 4
    lr_warmup: float = 1e-3
    lr_qat: float = 5e-4
    lr_finetune: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Loss weights
    lambda_qat_reg: float = 1e-4
    lambda_baee: float = 0.1
    lambda_sal_sup: float = 0.01    # salience supervision (optional KL to target mask)

    # Quantization
    n_calib_batches: int = 32

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    eval_every: int = 1

    # Device
    device: str = "cpu"
    seed: int = 42


class QATTrainer:
    """Manages the three-phase QAT training loop for SynapNetEdge."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        calib_loader: DataLoader,
        cfg: QATConfig,
        task: str = "classification",   # "classification" or "lm"
        baee_manager: nn.Module | None = None,
        salience_targets: Callable | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.calib_loader = calib_loader
        self.cfg = cfg
        self.task = task
        self.baee_manager = baee_manager
        self.salience_targets = salience_targets

        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Run all three training phases."""
        print("=" * 60)
        print("Phase 1: FP16 warm-up")
        print("=" * 60)
        self._phase_warmup()

        print("\n" + "=" * 60)
        print("Phase 2: CAJQ — apply quantization + QAT")
        print("=" * 60)
        self._apply_cajq()
        self._phase_qat()

        print("\n" + "=" * 60)
        print("Phase 3: QAT fine-tune (cooldown)")
        print("=" * 60)
        self._phase_finetune()

        return self._history

    # ------------------------------------------------------------------
    # Training phases
    # ------------------------------------------------------------------

    def _phase_warmup(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr_warmup,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.warmup_epochs
        )
        self._run_epochs("warmup", self.cfg.warmup_epochs, optimizer, scheduler)

    def _apply_cajq(self):
        from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
        cajq_cfg = CAJQConfig(
            n_calib_batches=self.cfg.n_calib_batches,
            device=str(self.device),
        )
        apply_cajq(self.model, cajq_cfg, calib_loader=self.calib_loader, mode="qat")

    def _phase_qat(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr_qat,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.qat_epochs
        )
        self._run_epochs("qat", self.cfg.qat_epochs, optimizer, scheduler, use_qat_loss=True)

    def _phase_finetune(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr_finetune,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.finetune_epochs
        )
        self._run_epochs("finetune", self.cfg.finetune_epochs, optimizer, scheduler,
                         use_qat_loss=True)

    # ------------------------------------------------------------------
    # Core epoch loop
    # ------------------------------------------------------------------

    def _run_epochs(
        self,
        phase: str,
        n_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
        use_qat_loss: bool = False,
    ):
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for batch in self.train_loader:
                input_ids, labels = self._unpack_batch(batch)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids)
                logits = outputs[0]
                debug_masks = outputs[1]
                debug_mems = outputs[2]

                # Task loss
                if self.task == "classification":
                    loss = F.cross_entropy(logits, labels)
                else:
                    # LM: shift by 1
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        input_ids[:, 1:].reshape(-1),
                    )

                # QAT regularisation
                if use_qat_loss:
                    from synapnet_edge.quantization.cajq import compute_cajq_loss
                    loss = loss + self.cfg.lambda_qat_reg * compute_cajq_loss(self.model)

                # BAEE auxiliary loss
                if use_qat_loss and self.baee_manager is not None and debug_mems:
                    dummy_attn = [
                        torch.ones(m.shape[0], m.shape[1], device=self.device) * 0.5
                        for m in debug_mems
                    ]
                    baee_loss = self.baee_manager.compute_aux_loss(debug_mems, dummy_attn)
                    loss = loss + self.cfg.lambda_baee * baee_loss

                # Salience supervision (optional)
                if self.salience_targets is not None and debug_masks:
                    sal_target = self.salience_targets(input_ids)
                    if sal_target is not None:
                        sal_loss = sum(
                            F.kl_div(
                                m.log().clamp(min=-20),
                                sal_target.to(self.device),
                                reduction="batchmean",
                            )
                            for m in debug_masks
                        ) / len(debug_masks)
                        loss = loss + self.cfg.lambda_sal_sup * sal_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, n_batches)
            elapsed = time.time() - t0

            eval_acc = None
            if epoch % self.cfg.eval_every == 0:
                eval_acc = self._evaluate()

            record = {
                "phase": phase,
                "epoch": epoch,
                "train_loss": avg_loss,
                "eval_acc": eval_acc,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": elapsed,
            }
            self._history.append(record)
            acc_text = f"{eval_acc:.4f}" if eval_acc is not None else "N/A"
            print(f"[{phase}] epoch {epoch}/{n_epochs} | "
                  f"loss={avg_loss:.4f} | "
                  f"acc={acc_text} | "
                  f"lr={record['lr']:.2e} | {elapsed:.1f}s")

            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(phase, epoch)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        correct = total = 0
        for batch in self.eval_loader:
            input_ids, labels = self._unpack_batch(batch)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(input_ids)
            logits = outputs[0]
            if self.task == "classification":
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / max(1, total)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, phase: str, epoch: int):
        path = Path(self.cfg.checkpoint_dir) / f"synapnet_edge_{phase}_ep{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "phase": phase,
            "model_state": self.model.state_dict(),
            "history": self._history,
        }, path)
        print(f"[QATTrainer] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self._history = ckpt.get("history", [])
        print(f"[QATTrainer] Loaded checkpoint from {path}")
        return ckpt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError(f"Unexpected batch format: {type(batch)}")
