"""SmoothQuant + AWQ-style INT4 quantization for SparseEventAttention.

Pipeline:
  1. SmoothQuant calibration: compute per-channel activation scales α
     and migrate quantization difficulty from activations → weights.
     W_smooth = W * diag(α);  X_smooth = X / diag(α)
     Both sides now have similar dynamic ranges, making INT4 weight
     quantization accurate without per-token activation quantization.

  2. AWQ-style group-wise INT4: quantize W_smooth with group_size=128,
     asymmetric per-group (zero-point + scale), using RTN (round-to-nearest).
     Salient weight channels (top-1% by activation norm) are optionally
     protected in INT8 or FP16.

Applied to: to_q, to_k, to_v, to_out in SparseEventAttention.
The salience_mlp stays FP16 (small MLP, not the compute bottleneck).

Reference implementations:
  - SmoothQuant (Xiao et al., 2022)
  - AWQ (Lin et al., 2023)
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# INT4 per-group quantization primitives
# ---------------------------------------------------------------------------

def quantize_int4_group(
    w: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-group INT4 weight quantization.

    Args:
        w:          (out_features, in_features) weight matrix
        group_size: number of input channels per quantization group

    Returns:
        w_int4:     (out_features, in_features) int8 tensor in [0, 15]
        scales:     (out_features, n_groups) FP32 per-group scale
        zeros:      (out_features, n_groups) FP32 per-group zero-point
    """
    out_f, in_f = w.shape
    # Cap group_size to in_features to avoid zero-padding contaminating min/max
    group_size = min(group_size, in_f)
    n_groups = math.ceil(in_f / group_size)

    # Pad if needed (only when in_f is not evenly divisible by group_size)
    pad = n_groups * group_size - in_f
    if pad > 0:
        # Replicate the last column instead of zero-padding to avoid corrupting min/max
        last_col = w[:, -1:].expand(-1, pad)
        w = torch.cat([w, last_col], dim=-1)

    w_grouped = w.view(out_f, n_groups, group_size)     # (O, G, Gs)

    w_min = w_grouped.min(dim=-1).values                # (O, G)
    w_max = w_grouped.max(dim=-1).values                # (O, G)

    scales = ((w_max - w_min) / 15.0).clamp(min=1e-6)   # [0, 15] range; floor scale
    zeros = -w_min / scales

    w_scaled = (w_grouped - w_min.unsqueeze(-1)) / scales.unsqueeze(-1)
    w_int4 = w_scaled.clamp(0, 15).round().to(torch.int8)

    # Remove padding
    w_int4 = w_int4.view(out_f, -1)[:, :in_f]

    return w_int4, scales, zeros


def dequantize_int4_group(
    w_int4: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    original_in_features: int,
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct FP16 weights from INT4 group-quantized form."""
    group_size = min(group_size, original_in_features)
    out_f = w_int4.shape[0]
    n_groups = scales.shape[1]

    # Pad to group-size boundary
    pad = n_groups * group_size - original_in_features
    if pad > 0:
        w_int4 = F.pad(w_int4, (0, pad))

    w_grouped = w_int4.float().view(out_f, n_groups, group_size)

    dq = (w_grouped - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)
    dq = dq.view(out_f, -1)[:, :original_in_features]
    return dq


# ---------------------------------------------------------------------------
# SmoothQuant calibration
# ---------------------------------------------------------------------------

class SmoothQuantCalibrator:
    """Collects activation statistics and computes per-channel smooth scales.

    Usage:
        calib = SmoothQuantCalibrator(alpha=0.5)
        calib.register_hooks(model)
        run calibration data through model
        calib.remove_hooks()
        calib.apply_smooth(model)
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._hooks: list = []
        self._act_maxes: dict[str, torch.Tensor] = {}

    def register_hooks(self, model: nn.Module) -> None:
        from synapnet_edge.models.sparse_attention import SparseEventAttention
        for name, module in model.named_modules():
            if isinstance(module, SparseEventAttention):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        def hook(module, inputs, outputs):
            x = inputs[0]  # (B, T, D)
            ch_max = x.abs().amax(dim=(0, 1))   # (D,)
            if name in self._act_maxes:
                self._act_maxes[name] = torch.maximum(
                    self._act_maxes[name], ch_max
                )
            else:
                self._act_maxes[name] = ch_max.clone()
        return hook

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def apply_smooth(self, model: nn.Module) -> None:
        """Absorb smooth scales into model weights and register buffers."""
        from synapnet_edge.models.sparse_attention import SparseEventAttention
        for name, module in model.named_modules():
            if not isinstance(module, SparseEventAttention):
                continue
            if name not in self._act_maxes:
                continue

            act_max = self._act_maxes[name].to(module.to_q.weight.device)
            w_max = torch.stack([
                module.to_q.weight.abs().amax(dim=0),
                module.to_k.weight.abs().amax(dim=0),
                module.to_v.weight.abs().amax(dim=0),
            ]).amax(dim=0)    # (D,)

            # SmoothQuant scale per input channel
            alpha = self.alpha
            smooth = act_max.pow(alpha) / (w_max.pow(1 - alpha) + 1e-8)
            smooth = smooth.clamp(min=1e-5)

            # Absorb into weights: W ← W * diag(smooth)
            with torch.no_grad():
                for linear in [module.to_q, module.to_k, module.to_v]:
                    linear.weight.data.mul_(smooth.unsqueeze(0))

            # Store inverse smooth in buffer (divides input during forward)
            module.smooth_scale.data = 1.0 / smooth
            print(f"[SmoothQuant] Applied to {name}, "
                  f"smooth range [{smooth.min():.3f}, {smooth.max():.3f}]")


# ---------------------------------------------------------------------------
# AWQ-style quantized Linear
# ---------------------------------------------------------------------------

class AWQLinear(nn.Module):
    """INT4 weight-only quantized Linear (AWQ-style, group_size=128).

    Stores weights in INT4 (packed as int8), dequantizes on the fly.
    Salient channels (by activation norm) are kept in FP16.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Cap group_size to in_features (matches quantize_int4_group behaviour)
        self.group_size = min(group_size, in_features)
        n_groups = math.ceil(in_features / self.group_size)

        self.register_buffer("w_int4", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, n_groups))
        self.register_buffer("zeros", torch.zeros(out_features, n_groups))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Salient channel mask: FP16 fallback for top activations
        self.register_buffer("salient_mask", torch.zeros(in_features, dtype=torch.bool))
        self.salient_fp16: nn.Parameter | None = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        salient_frac: float = 0.01,
        activation_norms: torch.Tensor | None = None,
    ) -> "AWQLinear":
        out_f, in_f = linear.weight.shape
        layer = cls(in_f, out_f, group_size=group_size,
                    bias=linear.bias is not None)

        w = linear.weight.data.float()

        # Protect salient channels in FP16
        if activation_norms is not None:
            k_salient = max(1, int(salient_frac * in_f))
            _, salient_idx = torch.topk(activation_norms, k_salient)
            layer.salient_mask[salient_idx] = True
            salient_w = w[:, layer.salient_mask]
            # Keep salient channels in FP32 for stable training (cast to runtime dtype on use)
            layer.salient_fp16 = nn.Parameter(salient_w.float())
            # Zero out salient channels in weight before quant
            w[:, layer.salient_mask] = 0.0

        w_int4, scales, zeros = quantize_int4_group(w, group_size)
        layer.w_int4.copy_(w_int4)
        layer.scales.copy_(scales)
        layer.zeros.copy_(zeros)

        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize INT4 weights
        w_int4 = self.w_int4.to(x.device)
        scales = self.scales.to(x.device)
        zeros = self.zeros.to(x.device)
        w_dq = dequantize_int4_group(
            w_int4, scales, zeros,
            self.in_features, self.group_size
        ).to(device=x.device, dtype=x.dtype)

        # Add salient FP16 channels back
        if self.salient_fp16 is not None:
            salient_mask = self.salient_mask.to(x.device)
            w_dq[:, salient_mask] = (
                w_dq[:, salient_mask]
                + self.salient_fp16.to(device=x.device, dtype=x.dtype)
            )

        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, w_dq, bias)


# ---------------------------------------------------------------------------
# High-level AWQ calibrator and applicator
# ---------------------------------------------------------------------------

class AWQCalibrator:
    """Calibrates activation norms then replaces Linear layers with AWQLinear.

    Usage:
        calib = AWQCalibrator(group_size=128)
        calib.register_hooks(model)
        run calibration batches
        calib.remove_hooks()
        calib.apply(model)
    """

    def __init__(self, group_size: int = 128, salient_frac: float = 0.01):
        self.group_size = group_size
        self.salient_frac = salient_frac
        self._hooks: list = []
        self._norms: dict[str, torch.Tensor] = {}

    def register_hooks(self, model: nn.Module) -> None:
        from synapnet_edge.models.sparse_attention import SparseEventAttention
        for name, module in model.named_modules():
            if isinstance(module, SparseEventAttention):
                for lname in ["to_q", "to_k", "to_v", "to_out"]:
                    linear = getattr(module, lname)
                    key = f"{name}.{lname}"
                    hook = linear.register_forward_hook(self._make_hook(key))
                    self._hooks.append(hook)

    def _make_hook(self, key: str) -> Callable:
        def hook(module, inputs, outputs):
            x = inputs[0].detach().float()
            ch_norm = x.abs().mean(dim=(0, 1))   # (in_features,)
            if key in self._norms:
                self._norms[key] = (self._norms[key] + ch_norm) / 2
            else:
                self._norms[key] = ch_norm
        return hook

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def apply(self, model: nn.Module) -> nn.Module:
        """Replace SparseEventAttention linears with AWQLinear in-place."""
        from synapnet_edge.models.sparse_attention import SparseEventAttention
        for name, module in model.named_modules():
            if not isinstance(module, SparseEventAttention):
                continue
            for lname in ["to_q", "to_k", "to_v", "to_out"]:
                linear = getattr(module, lname)
                key = f"{name}.{lname}"
                act_norms = self._norms.get(key)
                awq_linear = AWQLinear.from_linear(
                    linear,
                    group_size=self.group_size,
                    salient_frac=self.salient_frac,
                    activation_norms=act_norms,
                ).to(linear.weight.device)
                setattr(module, lname, awq_linear)
        print("[AWQCalibrator] Replaced attention projections with INT4 AWQLinear.")
        return model


class AttentionQuantizer:
    """Orchestrates SmoothQuant + AWQ for all attention modules."""

    @staticmethod
    def calibrate_and_apply(
        model: nn.Module,
        calib_loader,
        device: str = "cpu",
        alpha: float = 0.5,
        group_size: int = 128,
    ) -> nn.Module:
        model.eval()
        model.to(device)

        smooth_calib = SmoothQuantCalibrator(alpha=alpha)
        awq_calib = AWQCalibrator(group_size=group_size)

        smooth_calib.register_hooks(model)
        awq_calib.register_hooks(model)

        with torch.no_grad():
            for batch in calib_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                model(batch)

        smooth_calib.remove_hooks()
        awq_calib.remove_hooks()

        smooth_calib.apply_smooth(model)
        awq_calib.apply(model)

        return model
