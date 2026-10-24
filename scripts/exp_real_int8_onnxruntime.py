"""Real INT8 kernels via ONNX Runtime — measured latency drop, not simulated.

Motivation
----------
The CAJQ pipeline in this repo uses *fake quantization* (quantize→dequantize in
PyTorch), which measures accuracy effects faithfully but executes every matmul
in floating point — so quantized latency is equal to or worse than FP16.  This
experiment closes that gap: it exports the actual SynapNet-Edge reference model
(dim=192, depth=6, heads=6, vocab=4096, num_classes=64, episodic_slots=32 —
the ~8.7M-param configuration used for all paper numbers) to ONNX and runs it
through onnxruntime's *real* dynamic-INT8 kernels (QOperator format →
DynamicQuantizeLinear + MatMulInteger on arm64), comparing FP32 vs INT8
wall-clock latency on the same CPU.

What exactly is exported and benchmarked
----------------------------------------
Preferred target is the FULL SynapNetEdge forward pass (token+pos embedding →
6 × SynapBlockWithEpisodic [SimpleSSM + SparseEventAttention + WriteableMemory
+ ScaleBridge + gated FF] → LayerNorm → classification head), traced at a fixed
(batch=1, seq_len) shape, returning logits only.  If that export fails on some
torch/onnx version, the script falls back to (a) the largest exportable
subgraph — a stack of the model's real SparseEventAttention + feed-forward +
ScaleBridge pathways — and (b) a dimension-matched Linear-stack proxy, and it
RECORDS WHICH TARGET WAS ACTUALLY BENCHMARKED in the output JSON and stdout.
No silent substitution.

What is INT8 and what is not
----------------------------
Dynamic quantization converts weight-bearing MatMul/Gemm ops (attention
q/k/v/out, episodic-memory projections, ScaleBridge fuse, gates, FF, salience
MLP, head) to MatMulInteger with on-the-fly activation quantization.
Activation-x-activation matmuls (attention score/value products), softmax,
LayerNorm, the depthwise SSM conv, and gathers stay FP32 in BOTH models, so
the reported speedup is end-to-end and conservative, not a matmul microbench.

Methodology matches scripts/exp_hardware_bench.py: single-sequence forward,
CPU, and the single-thread setting (intra_op_num_threads=1) is the repo's
Raspberry Pi proxy.  Default-thread numbers are reported alongside.

Portable by construction: pure CPUExecutionProvider, no mac-only code paths;
platform (machine/system/processor) is detected at runtime and recorded in the
results JSON.  To reproduce on a Raspberry Pi (aarch64 Linux):

    python3 -m venv .venv && . .venv/bin/activate
    pip install torch onnx onnxruntime numpy
    python scripts/exp_real_int8_onnxruntime.py --runs 50 --seq-len 1024

Output: results/real_int8_onnxruntime.json + a Markdown table on stdout.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.models.sparse_attention import SparseEventAttention

# ----------------------------------------------------------------------------
# Reference model configuration — the ~8.7M-param model behind the paper
# numbers (see README Quickstart: pretrain_scaled.py --dim 192 --depth 6).
# ----------------------------------------------------------------------------
REFERENCE_CFG = dict(
    dim=192, depth=6, vocab_size=4096, max_len=8192,
    num_classes=64, heads=6, episodic_slots=32,
)


class FullModelWrapper(nn.Module):
    """SynapNetEdge forward returning logits only (ONNX needs tensor outputs)."""

    def __init__(self, model: SynapNetEdge):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _, _, _ = self.model(idx)
        return logits


class AttentionFFNStack(nn.Module):
    """Fallback subgraph: the model's REAL sparse-attention + FF pathways.

    Built from the trained-architecture modules (SparseEventAttention and each
    block's `ff`/gate layers), i.e. the INT4-targeted compute path of the paper,
    but WITHOUT the episodic-memory write loop and SSM conv.  Only used if the
    full-model export fails; the results JSON says so explicitly.
    """

    def __init__(self, model: SynapNetEdge):
        super().__init__()
        self.norm_in = nn.ModuleList([b.norm_in for b in model.blocks])
        self.attns = nn.ModuleList([b.attn for b in model.blocks])
        self.ffs = nn.ModuleList([b.ff for b in model.blocks])
        self.alpha = nn.ModuleList([b.alpha_gate for b in model.blocks])
        self.embed = model.token_embed
        self.pos = model.pos_embed
        self.norm_out = model.norm_out
        self.head = model.head

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.embed(idx) + self.pos[:, :T, :]
        for norm, attn, ff, gate in zip(self.norm_in, self.attns, self.ffs, self.alpha):
            xn = norm(x)
            a, _ = attn(xn)
            x = x + torch.sigmoid(gate(xn)) * a
            x = x + ff(x)
        x = self.norm_out(x)
        return self.head(x[:, -1, :])


class LinearProxy(nn.Module):
    """Last-resort proxy: Linear stack dimension-matched to one SynapBlock's
    weight-matmul budget (q/k/v/out + epmem projections + gates + bridge + FF),
    repeated `depth` times.  This is NOT the SynapNet-Edge model — it exists
    only to measure the INT8 kernel speedup at the model's exact matmul shapes
    when the real graph cannot be exported."""

    def __init__(self, dim: int = 192, depth: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        layers: list[nn.Module] = []
        for _ in range(depth):
            # 10 dim->dim projections per block (attn q/k/v/out, epmem x4, gates x2)
            for _ in range(10):
                layers.append(nn.Linear(dim, dim))
            layers.append(nn.Linear(dim, dim * 3))      # bridge in
            layers.append(nn.Linear(dim * 3, dim))      # bridge out
            layers.append(nn.Linear(dim, hidden))        # ff up
            layers.append(nn.Linear(hidden, dim))        # ff down
        self.stack = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dims chain cleanly: dim->dim (x10) -> 3*dim -> dim -> hidden -> dim
        for lin in self.stack:
            x = torch.relu(lin(x))
        return x


# ----------------------------------------------------------------------------
# Export + quantization
# ----------------------------------------------------------------------------

def export_onnx(module: nn.Module, example_input: torch.Tensor, path: Path) -> None:
    """Export with the TorchScript exporter (stable for this graph); fall back
    to the dynamo exporter on torch versions where the legacy path is gone."""
    module.eval()
    kwargs = dict(
        input_names=["input"], output_names=["logits"], opset_version=17,
    )
    try:
        torch.onnx.export(module, (example_input,), str(path),
                          dynamo=False, **kwargs)
    except TypeError:  # older torch: no `dynamo` kwarg
        torch.onnx.export(module, (example_input,), str(path), **kwargs)
    except Exception:
        # Legacy exporter failed/unavailable: try dynamo exporter.
        torch.onnx.export(module, (example_input,), str(path),
                          dynamo=True, **kwargs)


def quantize_int8(fp32_path: Path, int8_path: Path, weight_type_name: str) -> None:
    """Dynamic quantization, QOperator format.

    quantize_dynamic always emits QOperator-format ops (MatMulInteger +
    DynamicQuantizeLinear), which is what actually runs integer kernels on
    arm64 — NOT the QDQ fake-quant format.  Only weight-bearing MatMul/Gemm
    are quantized (MatMulConstBOnly), so attention score matmuls, softmax,
    LayerNorm and the depthwise conv stay FP32.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    weight_type = QuantType.QInt8 if weight_type_name == "qint8" else QuantType.QUInt8
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=weight_type,
        per_channel=False,
        extra_options={"MatMulConstBOnly": True},
    )


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------

def make_session(path: Path, single_thread: bool):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if single_thread:
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def bench_session(session, feed: dict, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        session.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "mean_ms": statistics.fmean(times),
        "p50_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
        "min_ms": times[0],
        "runs": runs,
        "warmup": warmup,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=50, help="timed iterations per config")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=1024,
                   help="context length (fixed-shape export)")
    p.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    p.add_argument("--onnx-dir", default="results/onnx")
    p.add_argument("--output", default="results/real_int8_onnxruntime.json")
    args = p.parse_args()

    import onnxruntime as ort

    torch.manual_seed(0)
    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    cfg = SynapNetEdgeConfig(**REFERENCE_CFG)
    model = SynapNetEdge(cfg)
    model.eval()
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[build] SynapNetEdge reference model: {n_params/1e6:.2f}M params "
          f"(dim={cfg.dim}, depth={cfg.depth}, heads={cfg.heads}, "
          f"vocab={cfg.vocab_size}, num_classes={cfg.num_classes})")

    ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len))
    feats = torch.randn(1, args.seq_len, cfg.dim)

    # Export targets, most-honest first.  The first one that exports AND
    # quantizes cleanly is benchmarked; the JSON records which it was.
    targets = [
        ("full_model",
         "Full SynapNetEdge forward (embedding -> 6x[SSM + sparse-attn + "
         "episodic memory + ScaleBridge + FF] -> head), logits only",
         FullModelWrapper(model), ids),
        ("attention_ffn_stack",
         "SUBGRAPH ONLY: the model's real SparseEventAttention + gated FF "
         "pathways x6 (no episodic write / SSM conv) — full model did not export",
         AttentionFFNStack(model), ids),
        ("linear_proxy",
         "PROXY ONLY: dimension-matched Linear stack, NOT the real model — "
         "no SynapNet subgraph could be exported",
         LinearProxy(cfg.dim, cfg.depth, cfg.mlp_ratio), feats),
    ]

    chosen = None
    export_errors = {}
    for name, desc, module, example in targets:
        fp32_path = onnx_dir / f"synapnet_edge_{name}_fp32.onnx"
        int8_path = onnx_dir / f"synapnet_edge_{name}_int8.onnx"
        try:
            print(f"[export] trying target '{name}' ...")
            export_onnx(module, example, fp32_path)
            quantize_int8(fp32_path, int8_path, args.weight_type)
            chosen = (name, desc, fp32_path, int8_path, example)
            print(f"[export] OK: {name}")
            break
        except Exception as e:  # noqa: BLE001 — record and fall through
            export_errors[name] = f"{type(e).__name__}: {e}"
            print(f"[export] FAILED for '{name}': {export_errors[name]}")

    if chosen is None:
        print("[fatal] no target could be exported", file=sys.stderr)
        sys.exit(1)

    name, desc, fp32_path, int8_path, example = chosen
    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    print(f"[size] FP32 {fp32_mb:.2f} MB -> INT8 {int8_mb:.2f} MB "
          f"({fp32_mb / int8_mb:.2f}x smaller)")

    # Confirm real integer kernels are in the quantized graph.
    import onnx as onnx_lib
    int8_ops = sorted({n.op_type for n in onnx_lib.load(str(int8_path)).graph.node})
    has_int_kernels = any(op in int8_ops for op in
                          ("MatMulInteger", "DynamicQuantizeLinear", "QLinearMatMul"))
    print(f"[check] INT8 graph ops include integer kernels: {has_int_kernels}")
    if not has_int_kernels:
        print("[warn] quantized graph contains no integer matmul ops — "
              "the INT8 numbers below would NOT reflect real INT8 kernels!",
              file=sys.stderr)

    if example.dtype in (torch.int64, torch.int32):
        feed = {"input": example.numpy().astype(np.int64)}
    else:
        feed = {"input": example.numpy().astype(np.float32)}

    # Numerical sanity: INT8 output should track FP32 output.
    sess_check_fp32 = make_session(fp32_path, single_thread=True)
    sess_check_int8 = make_session(int8_path, single_thread=True)
    out_fp32 = sess_check_fp32.run(None, feed)[0].astype(np.float64).ravel()
    out_int8 = sess_check_int8.run(None, feed)[0].astype(np.float64).ravel()
    corr = float(np.corrcoef(out_fp32, out_int8)[0, 1])
    max_abs = float(np.max(np.abs(out_fp32 - out_int8)))
    print(f"[check] FP32 vs INT8 logits: pearson r={corr:.4f}, "
          f"max|diff|={max_abs:.4f}")
    del sess_check_fp32, sess_check_int8

    thread_modes = [("single_thread", True), ("default_threads", False)]
    bench: dict[str, dict] = {}
    for mode, single in thread_modes:
        bench[mode] = {}
        for prec, path in (("fp32", fp32_path), ("int8", int8_path)):
            sess = make_session(path, single_thread=single)
            stats = bench_session(sess, feed, runs=args.runs, warmup=args.warmup)
            bench[mode][prec] = stats
            print(f"[bench] {mode:>15} {prec}: mean={stats['mean_ms']:.2f}ms "
                  f"p50={stats['p50_ms']:.2f}ms p90={stats['p90_ms']:.2f}ms")
            del sess
        bench[mode]["speedup_mean"] = (
            bench[mode]["fp32"]["mean_ms"] / bench[mode]["int8"]["mean_ms"])
        bench[mode]["speedup_p50"] = (
            bench[mode]["fp32"]["p50_ms"] / bench[mode]["int8"]["p50_ms"])

    results = {
        "experiment": "real_int8_onnxruntime",
        "benchmarked_target": name,
        "benchmarked_target_description": desc,
        "export_errors_before_fallback": export_errors,
        "model_params": n_params,
        "reference_config": REFERENCE_CFG,
        "seq_len": args.seq_len,
        "batch_size": 1,
        "weight_type": args.weight_type,
        "quant_format": "QOperator (dynamic: DynamicQuantizeLinear + MatMulInteger)",
        "quantized_op_types": ["MatMul", "Gemm"],
        "int8_graph_op_types": int8_ops,
        "int8_graph_has_integer_kernels": has_int_kernels,
        "fp32_vs_int8_logits": {"pearson_r": corr, "max_abs_diff": max_abs},
        "model_size_mb": {"fp32": fp32_mb, "int8": int8_mb,
                          "ratio": fp32_mb / int8_mb},
        "latency": bench,
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] {out_path}")

    # Markdown table
    print(f"\n### Real INT8 kernels (ONNX Runtime, {platform.machine()} "
          f"{platform.system()}) — target: `{name}`, seq_len={args.seq_len}, "
          f"batch=1\n")
    print("| Threads | FP32 mean (ms) | INT8 mean (ms) | FP32 p50 | INT8 p50 "
          "| FP32 p90 | INT8 p90 | Speedup (mean) |")
    print("|---|---|---|---|---|---|---|---|")
    for mode, _ in thread_modes:
        b = bench[mode]
        label = "1 (Pi proxy)" if mode == "single_thread" else "default"
        print(f"| {label} | {b['fp32']['mean_ms']:.2f} | {b['int8']['mean_ms']:.2f} "
              f"| {b['fp32']['p50_ms']:.2f} | {b['int8']['p50_ms']:.2f} "
              f"| {b['fp32']['p90_ms']:.2f} | {b['int8']['p90_ms']:.2f} "
              f"| **{b['speedup_mean']:.2f}x** |")
    print(f"\nModel file: {fp32_mb:.2f} MB (FP32) -> {int8_mb:.2f} MB (INT8), "
          f"{fp32_mb / int8_mb:.2f}x smaller.")


if __name__ == "__main__":
    main()
