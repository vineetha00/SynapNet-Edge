"""CAJQ-Full — extend Component-Aware Joint Quantization to the fp16 remainder.

Standard CAJQ (synapnet_edge.quantization.cajq) quantizes only three components:
    SSM        -> 2-bit QAT
    attention  -> INT4 AWQ + SmoothQuant
    memory     -> INT8 (per-entry, at runtime)
On the scaled model that leaves the FFN blocks, the pathway gates, the episodic
projections, the token embedding and the ScaleBridge in fp16 -- roughly 95% of
the compressed model's on-disk size. The measured Pi-4 deployment showed this
directly: only ~12% storage reduction, because the component-aware scheme never
touched the bulk of the weights.

`apply_cajq_full()` keeps CAJQ's component-aware core unchanged and then
quantizes the *remaining* fp16 Linear layers to INT8 or INT4. The ScaleBridge
(Contribution 1c -- the interface layer that absorbs cross-pathway scale
mismatch) is the most precision-sensitive weight to touch, so `keep_bridge_fp16`
lets you preserve it: a principled, slightly-less-aggressive variant. The
salience MLP (attention's control path) is kept fp16 by default for the same
reason; it is negligible for storage either way.

This is a storage/efficiency change. Quantizing the FFN/embeddings/bridge is
exactly where accuracy can degrade, so the accuracy of every cajq-full variant
must be re-verified with the QAT accuracy experiment before any "little accuracy
loss" claim is made.
"""
from __future__ import annotations

import torch.nn as nn

from synapnet_edge.quantization.cajq import apply_cajq, CAJQConfig
from synapnet_edge.quantization.ssm_quantizer import QuantizedSSMWrapper
from synapnet_edge.quantization.attention_quantizer import AWQLinear

# SymmetricINT8Linear lives in the long-context experiment module alongside the
# uniform quantizers; the storage/effective-bit estimators recognise this exact
# class, so cajq-full reuses it rather than defining a parallel one.
from exp_cajq_long_context import SymmetricINT8Linear


def _ids_inside_ssm_wrappers(model: nn.Module) -> set[int]:
    """Ids of every submodule owned by a QuantizedSSMWrapper.

    After CAJQ the SSM path is wrapped and already accounted for at 2-bit; its
    internal gate is still an nn.Linear, so we must skip it or we would downgrade
    a 2-bit weight to INT8."""
    skip: set[int] = set()
    for _, m in model.named_modules():
        if isinstance(m, QuantizedSSMWrapper):
            for child in m.modules():
                skip.add(id(child))
    return skip


def quantize_remaining_linears(
    model: nn.Module,
    bits: int = 8,
    group_size: int = 64,
    keep_bridge_fp16: bool = False,
    keep_salience_fp16: bool = True,
) -> int:
    """Replace still-fp16 nn.Linear layers with INT8 or INT4 in place.

    Skips, in every case: the final `head`, anything inside a QuantizedSSMWrapper,
    and the attention projections (already AWQLinear, not nn.Linear). Optionally
    skips the ScaleBridge and the salience MLPs. Returns the number of layers
    replaced."""
    if bits not in (8, 4):
        raise ValueError(f"bits must be 8 or 4, got {bits}")

    skip_ids = _ids_inside_ssm_wrappers(model)
    replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if id(module) in skip_ids:
            continue
        if name.endswith(".head"):
            continue
        if keep_bridge_fp16 and "scale_bridge" in name:
            continue
        if keep_salience_fp16 and "salience" in name:
            continue

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        if bits == 8:
            setattr(parent, parts[-1], SymmetricINT8Linear.from_linear(module))
            replaced += 1
        else:  # bits == 4
            try:
                setattr(parent, parts[-1], AWQLinear.from_linear(module, group_size=group_size))
                replaced += 1
            except Exception as e:  # noqa: BLE001 - report and keep the fp16 layer
                print(f"  [cajq-full] could not INT4 {name}: {type(e).__name__}: {e}")

    return replaced


def apply_cajq_full(
    model: nn.Module,
    cfg: CAJQConfig,
    calib_loader=None,
    mode: str = "ptq",
    rest_bits: int = 8,
    keep_bridge_fp16: bool = False,
) -> nn.Module:
    """Apply CAJQ, then quantize the remaining fp16 Linear layers.

    Args:
        model, cfg, calib_loader, mode: forwarded to apply_cajq() unchanged.
        rest_bits:        8 or 4 -- precision for the FFN / gates / epmem / (bridge).
        keep_bridge_fp16: preserve the ScaleBridge in fp16 (Contribution 1c).
    """
    apply_cajq(model, cfg, calib_loader=calib_loader, mode=mode)
    n = quantize_remaining_linears(
        model,
        bits=rest_bits,
        group_size=cfg.attn_group_size,
        keep_bridge_fp16=keep_bridge_fp16,
    )
    print(
        f"[CAJQ-Full] quantized {n} remaining Linear layers to INT{rest_bits}"
        + (" (ScaleBridge kept fp16)" if keep_bridge_fp16 else "")
    )
    return model
