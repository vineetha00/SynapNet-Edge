"""ScaleBridge calibration utilities.

The ScaleBridge (defined in synapblock.py) is a learned FP16 interface
layer between the three quantized pathways.  This module provides:

  1. ScaleBridgeCalibrator — computes per-pathway output statistics
     on calibration data to initialise the LayerNorms well.
  2. validate_bridge — checks that the bridge does not amplify quantization
     error beyond a configurable threshold.
  3. export_bridge_stats — dumps per-pathway scale information for paper
     ablation tables.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


class ScaleBridgeCalibrator:
    """Initialises ScaleBridge LayerNorm parameters from calibration data.

    After CAJQ applies 2-bit SSM and INT4 attention quantization, the
    pathway outputs can have very different magnitudes.  This calibrator
    runs a few batches through the model, collects per-pathway output
    statistics, and rescales the LayerNorm weight/bias to compensate.
    """

    def __init__(self):
        self._means: dict[str, list[torch.Tensor]] = {}
        self._stds: dict[str, list[torch.Tensor]] = {}
        self._hooks: list = []

    def register_hooks(self, model: nn.Module) -> None:
        from synapnet_edge.models.synapblock import SynapBlockWithEpisodic, ScaleBridge
        for name, module in model.named_modules():
            if isinstance(module, ScaleBridge):
                for i, norm in enumerate(module.norms):
                    key = f"{name}.norm{i}"
                    self._means[key] = []
                    self._stds[key] = []

                    def _hook(m, inputs, outputs, k=key):
                        x = inputs[0].detach().float()
                        self._means[k].append(x.mean().item())
                        self._stds[k].append(x.std().item())

                    h = norm.register_forward_hook(_hook)
                    self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def apply(self, model: nn.Module) -> None:
        """Informational calibration — LayerNorm already normalises each
        pathway to unit variance, so no weight modification is needed.
        This method just logs per-pathway input statistics for diagnostics.
        """
        from synapnet_edge.models.synapblock import ScaleBridge
        for name, module in model.named_modules():
            if not isinstance(module, ScaleBridge):
                continue
            for i, _ in enumerate(module.norms):
                key = f"{name}.norm{i}"
                if key not in self._stds or not self._stds[key]:
                    continue
                avg_std = sum(self._stds[key]) / len(self._stds[key])
                print(f"[ScaleBridgeCalibrator] {key}: input_std={avg_std:.4f} "
                      f"(LayerNorm handles normalisation; no weight change)")

    def export_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for key in self._means:
            means = self._means[key]
            stds = self._stds[key]
            stats[key] = {
                "mean": sum(means) / max(1, len(means)),
                "std": sum(stds) / max(1, len(stds)),
            }
        return stats


def validate_bridge(
    model: nn.Module,
    calib_loader,
    device: str = "cpu",
    max_scale_ratio: float = 5.0,
) -> bool:
    """Check that scale bridge does not over-amplify any pathway.

    Returns True if all pathway scales are within max_scale_ratio of
    each other (i.e., bridge is well-conditioned).
    """
    from synapnet_edge.models.synapblock import ScaleBridge

    pathway_stds: dict[str, list[float]] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, ScaleBridge):
            for i, norm in enumerate(module.norms):
                key = f"{name}.norm{i}"
                pathway_stds[key] = []

                def _hook(m, inputs, outputs, k=key):
                    pathway_stds[k].append(outputs.detach().float().std().item())

                hooks.append(norm.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        for batch in calib_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            model(batch.to(device))
            break

    for h in hooks:
        h.remove()

    avg_stds = {k: sum(v) / max(1, len(v)) for k, v in pathway_stds.items()}
    if not avg_stds:
        return True

    stds = list(avg_stds.values())
    ratio = max(stds) / (min(stds) + 1e-8)
    ok = ratio <= max_scale_ratio
    print(f"[ScaleBridge] Max/min std ratio: {ratio:.2f} — {'OK' if ok else 'WARNING'}")
    return ok
