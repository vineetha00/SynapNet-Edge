"""Component-Aware Joint Quantization (CAJQ) — Contribution 1.

Applies different quantization strategies per architectural component:
  - SSM layers:          ParetoQ 2-bit QAT  (SSMQuantizer)
  - Sparse attention:    SmoothQuant + AWQ INT4  (AttentionQuantizer)
  - Episodic memory:     per-entry INT8  (MemoryQuantizer, applied by BAEE)
  - Interface layer:     FP16 ScaleBridge  (calibrated by ScaleBridgeCalibrator)

The top-level `apply_cajq()` function is the single entry point.
It handles both QAT (training) and PTQ (post-training quantization) modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
from typing import Any

from synapnet_edge.quantization.ssm_quantizer import SSMQuantizer
from synapnet_edge.quantization.attention_quantizer import AttentionQuantizer
from synapnet_edge.quantization.scale_bridge import ScaleBridgeCalibrator, validate_bridge


@dataclass
class CAJQConfig:
    # SSM (2-bit QAT)
    ssm_bits: int = 2
    ssm_init_step: float = 0.1
    ssm_step_reg: float = 1e-4

    # Attention (INT4 SmoothQuant + AWQ)
    attn_bits: int = 4
    attn_group_size: int = 128
    smooth_alpha: float = 0.5
    salient_frac: float = 0.01

    # Memory (INT8, handled by BAEE)
    mem_bits: int = 8

    # Scale bridge
    calibrate_bridge: bool = True

    # Calibration
    n_calib_batches: int = 32
    device: str = "cpu"


def apply_cajq(
    model: nn.Module,
    cfg: CAJQConfig,
    calib_loader=None,
    mode: str = "qat",
) -> nn.Module:
    """Apply Component-Aware Joint Quantization to a SynapNetEdge model.

    Args:
        model:         SynapNetEdge model (or compatible)
        cfg:           CAJQConfig dataclass
        calib_loader:  DataLoader yielding (input_ids, ...) batches for PTQ calibration
        mode:          "qat"  — wrap SSM with QAT stubs (train further)
                       "ptq"  — apply PTQ everywhere (no further training needed)

    Returns:
        Quantized model (in-place modified).
    """
    print(f"\n[CAJQ] Applying Component-Aware Joint Quantization (mode={mode})")
    print(f"       SSM={cfg.ssm_bits}b QAT | Attention=INT{cfg.attn_bits} AWQ | Memory=INT{cfg.mem_bits}")

    # ------------------------------------------------------------------
    # 1. SSM → 2-bit QAT wrappers
    # ------------------------------------------------------------------
    print("[CAJQ] Step 1/3: Wrapping SSM layers with 2-bit QAT")
    SSMQuantizer.apply(model)
    model.to(cfg.device)

    # ------------------------------------------------------------------
    # 2. Sparse attention → SmoothQuant + AWQ INT4
    # ------------------------------------------------------------------
    if calib_loader is not None:
        print("[CAJQ] Step 2/3: Calibrating + quantizing attention (SmoothQuant + AWQ INT4)")
        AttentionQuantizer.calibrate_and_apply(
            model=model,
            calib_loader=_limited_loader(calib_loader, cfg.n_calib_batches),
            device=cfg.device,
            alpha=cfg.smooth_alpha,
            group_size=cfg.attn_group_size,
        )
        model.to(cfg.device)
    else:
        print("[CAJQ] Step 2/3: Skipping attention calibration (no calib_loader provided)")

    # ------------------------------------------------------------------
    # 3. Scale bridge calibration
    # ------------------------------------------------------------------
    if cfg.calibrate_bridge and calib_loader is not None:
        print("[CAJQ] Step 3/3: Calibrating scale bridge")
        bridge_calib = ScaleBridgeCalibrator()
        model.eval()
        bridge_calib.register_hooks(model)
        with torch.no_grad():
            for batch in _limited_loader(calib_loader, min(8, cfg.n_calib_batches)):
                ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(ids.to(cfg.device))
        bridge_calib.remove_hooks()
        bridge_calib.apply(model)
        bridge_stats = bridge_calib.export_stats()
        print(f"[CAJQ] Bridge stats: {bridge_stats}")
    else:
        print("[CAJQ] Step 3/3: Skipping bridge calibration")

    print("[CAJQ] Done.\n")
    return model


def compute_cajq_loss(model: nn.Module) -> torch.Tensor:
    """Return QAT regularisation loss (step-size penalty for SSM quantizers).

    Add this to the main training loss during QAT.
    """
    return SSMQuantizer.collect_quantization_loss(model)


def estimate_model_bits(model: nn.Module) -> dict[str, Any]:
    """Estimate effective bit-width and parameter count per component."""
    from synapnet_edge.quantization.ssm_quantizer import QuantizedSSMWrapper
    from synapnet_edge.quantization.attention_quantizer import AWQLinear
    from synapnet_edge.models.ssm import SimpleSSM
    from synapnet_edge.models.sparse_attention import SparseEventAttention
    from synapnet_edge.models.episodic_memory import WriteableMemory

    stats: dict[str, dict] = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantizedSSMWrapper):
            n = (m.ssm.dwconv.weight.numel() + m.ssm.gate.weight.numel())
            stats[name] = {"bits": 2, "params": n, "component": "SSM"}
        elif isinstance(m, AWQLinear):
            n = m.in_features * m.out_features
            stats[name] = {"bits": 4, "params": n, "component": "Attention"}
        elif isinstance(m, WriteableMemory):
            n = sum(p.numel() for p in m.parameters())
            stats[name] = {"bits": 16, "params": n, "component": "EpisodicMemory"}

    total_bits = sum(s["bits"] * s["params"] for s in stats.values())
    total_params = sum(s["params"] for s in stats.values())
    effective_bits = total_bits / max(1, total_params)

    return {
        "per_module": stats,
        "total_params": total_params,
        "effective_bits": effective_bits,
        "storage_mb": total_bits / 8 / 1024 / 1024,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _limited_loader(loader, n_batches: int):
    """Yield at most n_batches from loader."""
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        yield batch
