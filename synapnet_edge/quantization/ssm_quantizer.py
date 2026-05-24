"""ParetoQ-style 2-bit QAT for SimpleSSM layers.

Key design choices (following the ParetoQ paper, ICLR 2025):
  - Symmetric uniform 2-bit quantization: levels {-3, -1, +1, +3} × (s/2)
  - Learned per-channel step size `s` (positive, trained in log space)
  - Straight-through estimator (STE) for gradient flow through round()
  - Weight quantization only; activations remain in FP16/BF16

Applied to:
  - SimpleSSM.dwconv (Conv1d depthwise weights)
  - SimpleSSM.gate   (Linear weights)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core 2-bit quantize / dequantize with STE
# ---------------------------------------------------------------------------

def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through round: forward=round, backward=identity."""
    return x - (x - x.round()).detach()


def quantize_2bit(w: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Uniform symmetric 2-bit quantization.

    Levels: {-1.5, -0.5, +0.5, +1.5} * step  (4 levels → 2 bits)

    Args:
        w:    weight tensor, any shape
        step: per-output-channel scale, shape (C,) or scalar

    Returns:
        Quantized weight in float (same shape as w).
    """
    # Broadcast step to w's shape
    if step.dim() == 1:
        shape = [-1] + [1] * (w.dim() - 1)
        step = step.view(*shape)

    w_scaled = w / (step + 1e-8)
    # Clamp to [-1.5, 1.5] then round to nearest 0.5-multiple
    w_clamped = w_scaled.clamp(-1.5, 1.5)
    w_rounded = _ste_round(w_clamped * 2) / 2    # round to nearest 0.5
    return w_rounded * step


def pack_2bit(w_quant: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Pack quantized weights to INT8 (4 levels → 2 bits, 4 per byte).

    Level encoding:  -1.5→0, -0.5→1, +0.5→2, +1.5→3

    Returns uint8 tensor of shape (*w_quant.shape[:-1], ceil(w_quant.shape[-1]/4)).
    """
    if step.dim() == 1:
        shape = [-1] + [1] * (w_quant.dim() - 1)
        step = step.view(*shape)

    w_int = ((w_quant / (step + 1e-8) + 1.5) * 2).round().clamp(0, 3).to(torch.uint8)
    flat = w_int.flatten()
    pad = (4 - flat.numel() % 4) % 4
    if pad:
        flat = F.pad(flat, (0, pad))
    packed = (flat[0::4]) | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    return packed


def unpack_2bit(packed: torch.Tensor, numel: int, step: torch.Tensor) -> torch.Tensor:
    """Unpack INT8 to float levels.

    Returns flat float tensor of length numel.
    """
    bits = torch.stack([
        (packed) & 0x03,
        (packed >> 2) & 0x03,
        (packed >> 4) & 0x03,
        (packed >> 6) & 0x03,
    ], dim=-1).flatten().float()[:numel]

    levels = (bits / 2 - 0.75) * 2   # {0,1,2,3} → {-1.5,-0.5,0.5,1.5}
    if step.dim() == 1:
        step_scalar = step.mean().item()
    else:
        step_scalar = step.item()
    return levels * step_scalar


# ---------------------------------------------------------------------------
# Per-channel step size parameter
# ---------------------------------------------------------------------------

class LearnedStepSize(nn.Module):
    """Positive scalar per output channel, trained in log space."""

    def __init__(self, num_channels: int, init_step: float = 0.1):
        super().__init__()
        self.log_step = nn.Parameter(
            torch.full((num_channels,), math.log(init_step))
        )

    @property
    def step(self) -> torch.Tensor:
        return self.log_step.exp()


# ---------------------------------------------------------------------------
# QuantizedSSMWrapper: drop-in replacement for SimpleSSM during QAT
# ---------------------------------------------------------------------------

class QuantizedSSMWrapper(nn.Module):
    """Wraps SimpleSSM and quantizes dwconv + gate weights during forward.

    During QAT forward:
      1. Quantize weights using quantize_2bit() with STE.
      2. Run the original SimpleSSM.forward() with fake-quantized weights.
      3. On export, pack weights to 2-bit via pack_2bit().
    """

    def __init__(self, ssm: "SimpleSSM"):  # noqa: F821
        super().__init__()
        self.ssm = ssm
        C_conv = ssm.dwconv.weight.shape[0]    # depthwise: (C, 1, K)
        C_gate = ssm.gate.weight.shape[0]       # linear:    (out, in)

        # Adaptive initial step size: per-channel max|w| / 1.5 so the largest
        # weight maps to level ±1.5 (the most extreme bin in 2-bit symmetric).
        # This is crucial — a global 0.1 init biases all weights to zero
        # whenever |w_max| < 0.075 (very common with std=0.02 init).
        with torch.no_grad():
            conv_w = ssm.dwconv.weight.detach()
            gate_w = ssm.gate.weight.detach()
            # conv weight: (C, 1, K) → per-output-channel
            conv_init = conv_w.abs().amax(dim=(1, 2)) / 1.5
            gate_init = gate_w.abs().amax(dim=1) / 1.5
            conv_init = conv_init.clamp(min=1e-4)
            gate_init = gate_init.clamp(min=1e-4)

        self.step_conv = LearnedStepSize(C_conv)
        self.step_gate = LearnedStepSize(C_gate)
        with torch.no_grad():
            self.step_conv.log_step.copy_(conv_init.log())
            self.step_gate.log_step.copy_(gate_init.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fake-quantize weights (STE keeps gradients flowing through quantize_2bit)
        conv_w = quantize_2bit(self.ssm.dwconv.weight, self.step_conv.step)
        gate_w = quantize_2bit(self.ssm.gate.weight, self.step_gate.step)

        # Use functional convs/linears so gradients flow through conv_w/gate_w
        # (the .data swap trick detaches the autograd graph — avoid it)
        import torch.nn.functional as F
        dwconv = self.ssm.dwconv
        x_t = x.transpose(1, 2)
        conv_out = F.conv1d(
            x_t,
            conv_w,
            bias=dwconv.bias,
            stride=dwconv.stride,
            padding=dwconv.padding,
            dilation=dwconv.dilation,
            groups=dwconv.groups,
        ).transpose(1, 2)
        gate_vals = torch.sigmoid(F.linear(x, gate_w, self.ssm.gate.bias))
        return x + gate_vals * conv_out

    def export_packed(self) -> dict:
        """Return bit-packed weight dict for deployment."""
        conv_q = quantize_2bit(
            self.ssm.dwconv.weight.detach(),
            self.step_conv.step.detach()
        )
        gate_q = quantize_2bit(
            self.ssm.gate.weight.detach(),
            self.step_gate.step.detach()
        )
        return {
            "conv_packed": pack_2bit(conv_q, self.step_conv.step.detach()),
            "conv_step": self.step_conv.step.detach(),
            "conv_numel": self.ssm.dwconv.weight.numel(),
            "conv_shape": self.ssm.dwconv.weight.shape,
            "gate_packed": pack_2bit(gate_q, self.step_gate.step.detach()),
            "gate_step": self.step_gate.step.detach(),
            "gate_numel": self.ssm.gate.weight.numel(),
            "gate_shape": self.ssm.gate.weight.shape,
        }


# ---------------------------------------------------------------------------
# High-level quantizer
# ---------------------------------------------------------------------------

class SSMQuantizer:
    """Applies 2-bit QAT to all SimpleSSM modules in a SynapNetEdge model."""

    @staticmethod
    def apply(model: nn.Module) -> nn.Module:
        from synapnet_edge.models.ssm import SimpleSSM
        replaced = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, SimpleSSM):
                continue
            wrapped = QuantizedSSMWrapper(module).to(module.dwconv.weight.device)
            # Replace in parent
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapped)
            replaced += 1
        print(f"[SSMQuantizer] Wrapped {replaced} SimpleSSM modules with 2-bit QAT.")
        return model

    @staticmethod
    def collect_quantization_loss(model: nn.Module) -> torch.Tensor:
        """Quadratic regularisation pulling log(step) toward its adaptive init.

        The old version penalised step.mean() directly, biasing all step
        sizes toward zero — which quantises every weight to zero.  The fix
        is to penalise drift from the per-channel adaptive init (computed
        from max|w| at QAT-start) so steps can adapt but cannot collapse.
        """
        loss = None
        n_terms = 0
        for m in model.modules():
            if isinstance(m, QuantizedSSMWrapper):
                if not hasattr(m, "_init_log_conv"):
                    m._init_log_conv = m.step_conv.log_step.detach().clone()
                    m._init_log_gate = m.step_gate.log_step.detach().clone()
                conv_dev = m.step_conv.log_step - m._init_log_conv.to(
                    m.step_conv.log_step.device)
                gate_dev = m.step_gate.log_step - m._init_log_gate.to(
                    m.step_gate.log_step.device)
                term = conv_dev.pow(2).mean() + gate_dev.pow(2).mean()
                loss = term if loss is None else loss + term
                n_terms += 2
        if loss is None:
            device = next(model.parameters()).device
            loss = torch.tensor(0.0, device=device)
        return loss / max(1, n_terms)

    @staticmethod
    def step_parameters(model: nn.Module):
        """Yield only the step-size parameters (for separate LR group)."""
        for m in model.modules():
            if isinstance(m, QuantizedSSMWrapper):
                yield m.step_conv.log_step
                yield m.step_gate.log_step
