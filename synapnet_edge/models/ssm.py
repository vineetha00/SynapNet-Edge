"""SimpleSSM: depthwise-conv local temporal dynamics.

Quantization hooks are injected by CAJQ at model-build time.
The forward pass is vanilla float; quantized variants swap
dwconv and gate via the `quant_config` attribute.
"""
import torch
import torch.nn as nn


def _ensure_module_device(module: nn.Module, x: torch.Tensor) -> nn.Module:
    param = next(module.parameters(), None)
    if param is not None and param.device != x.device:
        module.to(x.device)
    return module


class SimpleSSM(nn.Module):
    """Depthwise-conv SSM with gated skip.

    Quantization target: 2-bit QAT (ParetoQ-style) applied to
    dwconv weights and gate weights by CAJQApplicator.
    """

    def __init__(self, dim: int, kernel_size: int = 9):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.dwconv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)

        # Placeholder: replaced by CAJQApplicator with QuantizedSSMWrapper
        self.quant_config: dict | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        _ensure_module_device(self.dwconv, x)
        _ensure_module_device(self.gate, x)
        x_t = x.transpose(1, 2)                        # (B, D, T)
        conv_out = self.dwconv(x_t).transpose(1, 2)    # (B, T, D)
        gate_vals = torch.sigmoid(self.gate(x))         # (B, T, D)
        return x + gate_vals * conv_out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kernel_size={self.kernel_size}"
