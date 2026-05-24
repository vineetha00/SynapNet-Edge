"""SparseEventAttention: salience-gated top-K global mixing.

INT4 quantization (SmoothQuant + AWQ) targets Q/K/V/out projections.
The salience MLP stays in FP16 (control path, not compute-bound).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_module_device(module: nn.Module, x: torch.Tensor) -> nn.Module:
    param = next(module.parameters(), None)
    if param is not None and param.device != x.device:
        module.to(x.device)
    return module


class SparseEventAttention(nn.Module):
    """Multi-head attention biased toward top-K salient tokens.

    Produces:
      - attended features (B, T, D)
      - soft salience mask in [0, 1] per token (B, T)

    Quantization target: INT4 AWQ+SmoothQuant on to_q/to_k/to_v/to_out.
    The salience_mlp remains FP16 to preserve gating accuracy.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        k_frac: float = 0.25,
        sparse_threshold: int = 1024,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.k_frac = k_frac
        self.sparse_threshold = sparse_threshold

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Salience predictor — kept FP16 by CAJQ
        self.salience_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

        # SmoothQuant scaling factors (per input channel); set during calibration
        self.register_buffer("smooth_scale", torch.ones(dim))

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        H, Hd = self.heads, self.head_dim

        # Salience in full precision (control path)
        _ensure_module_device(self.salience_mlp, x)
        if self.smooth_scale.device != x.device:
            self.smooth_scale.data = self.smooth_scale.data.to(x.device)
        sal_raw = self.salience_mlp(x).squeeze(-1)          # (B, T)
        k = max(1, int(self.k_frac * T))
        topk_vals, topk_idx = torch.topk(sal_raw, k, dim=1)
        thresh = topk_vals[:, -1].unsqueeze(-1)              # (B, 1)
        soft_mask = torch.sigmoid((sal_raw - thresh) * 10.0) # (B, T)

        # SmoothQuant: scale activations before projection
        x_scaled = x * self.smooth_scale.unsqueeze(0).unsqueeze(0)

        Q = self.to_q(x_scaled)
        K = self.to_k(x_scaled)
        V = self.to_v(x_scaled)

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, Hd).transpose(1, 2)   # (B, H, T, Hd)

        Q, K, V = _split(Q), _split(K), _split(V)

        if T > self.sparse_threshold:
            # Sparse top-K attention: O(T*k) compute and memory
            idx_exp = topk_idx.view(B, 1, k, 1).expand(B, H, k, Hd)
            K_s = K.gather(2, idx_exp)                                # (B,H,k,Hd)
            V_s = V.gather(2, idx_exp)                                # (B,H,k,Hd)
            attn_logits = torch.matmul(Q, K_s.transpose(-2, -1)) / (Hd ** 0.5)
            attn = F.softmax(attn_logits, dim=-1)                     # (B,H,T,k)
            out = torch.matmul(attn, V_s)                             # (B,H,T,Hd)
        else:
            # Soft full attention biased by salience
            attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (Hd ** 0.5)
            event_bias = torch.log(soft_mask + 1e-6).unsqueeze(1).unsqueeze(1)
            attn_logits = attn_logits + event_bias
            attn = F.softmax(attn_logits, dim=-1)                     # (B,H,T,T)
            out = torch.matmul(attn, V)                               # (B,H,T,Hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.to_out(out)
        return out, soft_mask

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.heads}, "
                f"k_frac={self.k_frac}, sparse_thr={self.sparse_threshold}")
