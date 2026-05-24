"""Benchmark evaluation script: RULER + LongBench + hardware profiling.

Evaluates all model variants and generates Pareto frontier plots.

Usage:
  # Evaluate all models
  python scripts/eval_benchmarks.py --models all --device mps

  # Evaluate specific model from checkpoint
  python scripts/eval_benchmarks.py --checkpoint results/synapnet_edge_final.pt

  # Hardware profiling only
  python scripts/eval_benchmarks.py --hardware-only --seq-lengths 512 2048 8192
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.baselines.mamba2_proxy import Mamba2Proxy
from synapnet_edge.baselines.llama_awq_proxy import LlamaAWQProxy
from synapnet_edge.baselines.falcon_h1_proxy import FalconH1Proxy
from synapnet_edge.baselines.em_llm import EMLLMBaseline
from synapnet_edge.benchmarks.ruler_bench import RULERBenchmark, RULERTask
from synapnet_edge.benchmarks.longbench import LongBenchEvaluator
from synapnet_edge.benchmarks.hardware_bench import HardwareBenchmark, detect_platform
from synapnet_edge.benchmarks.pareto import ParetoAnalyzer, ParetoPoint
from synapnet_edge.utils.profiling import profile_model


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DIM = 128
DEPTH = 4
VOCAB = 32000
NUM_CLASSES = 64
HEADS = 4


def build_all_models(device: str) -> dict[str, torch.nn.Module]:
    """Build all model variants for comparison."""
    models = {}

    # SynapNetEdge (our method, FP16)
    cfg = SynapNetEdgeConfig(
        dim=DIM, depth=DEPTH, vocab_size=VOCAB,
        max_len=8192, num_classes=NUM_CLASSES,
        heads=HEADS, k_frac=0.25,
        episodic_slots=8, episodic_write_frac=0.05,
        use_scale_bridge=True,
    )
    models["SynapNetEdge-FP16"] = SynapNetEdge(cfg)

    # Mamba-2 proxy (INT4)
    models["Mamba2-INT4"] = Mamba2Proxy(
        dim=DIM, depth=DEPTH, vocab_size=VOCAB,
        max_len=32768, num_classes=NUM_CLASSES, quantized=True,
    )

    # Llama AWQ INT4
    models["Llama-AWQ-INT4"] = LlamaAWQProxy(
        dim=DIM, depth=DEPTH, vocab_size=VOCAB,
        max_len=8192, num_classes=NUM_CLASSES, heads=HEADS, quantized=True,
    )

    # Falcon-H1 FP16
    models["FalconH1-FP16"] = FalconH1Proxy(
        dim=DIM, depth=DEPTH, vocab_size=VOCAB,
        max_len=32768, num_classes=NUM_CLASSES, heads=HEADS,
    )

    # EM-LLM FP16
    models["EMLLM-FP16"] = EMLLMBaseline(
        dim=DIM, depth=DEPTH, vocab_size=VOCAB,
        max_len=32768, num_classes=NUM_CLASSES, heads=HEADS,
    )

    return {k: v.to(device) for k, v in models.items()}


METHOD_FAMILIES = {
    "SynapNetEdge-FP16": "SynapNetEdge",
    "SynapNetEdge-CAJQ": "SynapNetEdge",
    "Mamba2-INT4": "Mamba2",
    "Llama-AWQ-INT4": "Llama",
    "FalconH1-FP16": "FalconH1",
    "EMLLM-FP16": "EMLLM",
}


def main():
    parser = argparse.ArgumentParser(description="SynapNet-Edge benchmark evaluation")
    parser.add_argument("--models", default="all",
                        help="Comma-separated model names, or 'all'")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SynapNetEdge checkpoint to evaluate")
    parser.add_argument("--hardware-only", action="store_true")
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="results/benchmarks")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    platform = detect_platform()
    print(f"[eval_benchmarks] Platform: {platform}, device: {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_all_models(args.device)

    # Load checkpoint if provided
    if args.checkpoint:
        cfg = SynapNetEdgeConfig(
            dim=DIM, depth=DEPTH, vocab_size=VOCAB,
            max_len=8192, num_classes=NUM_CLASSES,
            heads=HEADS, use_scale_bridge=True,
        )
        ckpt_model = SynapNetEdge(cfg)
        ckpt_model.load_state_dict(
            torch.load(args.checkpoint, map_location=args.device)
        )
        ckpt_model.to(args.device)
        models["SynapNetEdge-CAJQ"] = ckpt_model
        print(f"[eval_benchmarks] Loaded checkpoint: {args.checkpoint}")

    pareto = ParetoAnalyzer()
    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model.eval()
        model_results = {"model_name": model_name}

        # ------------------------------------------------------------------
        # Hardware profiling
        # ------------------------------------------------------------------
        print("\n[Profiling] Latency and memory...")
        latency_results = {}
        for seq_len in args.seq_lengths:
            try:
                result = profile_model(
                    model=model,
                    seq_len=seq_len,
                    vocab_size=VOCAB,
                    device=args.device,
                    n_warmup=2,
                    n_measure=5,
                )
                latency_results[seq_len] = {
                    "tokens_per_second": result.tokens_per_second,
                    "peak_memory_mb": result.peak_memory_mb,
                    "median_latency_ms": result.median_latency_ms,
                }
                print(f"  seq={seq_len}: {result.tokens_per_second:.1f} tok/s, "
                      f"{result.peak_memory_mb:.0f} MB")
            except Exception as e:
                print(f"  seq={seq_len}: FAILED ({e})")
                latency_results[seq_len] = {}

        model_results["latency"] = latency_results

        if not args.hardware_only:
            # ------------------------------------------------------------------
            # RULER
            # ------------------------------------------------------------------
            print("\n[RULER] Running benchmark...")
            try:
                ruler = RULERBenchmark(
                    model=model,
                    vocab_size=VOCAB,
                    num_classes=NUM_CLASSES,
                    n_samples=args.n_samples,
                    device=args.device,
                    chunk_size=args.chunk_size,
                )
                ruler_results = ruler.run(
                    context_lengths=[l for l in args.seq_lengths if l <= 8192],
                    verbose=True,
                )
                model_results["ruler"] = ruler_results
            except Exception as e:
                print(f"  RULER failed: {e}")
                ruler_results = {}
                model_results["ruler"] = {}

            # ------------------------------------------------------------------
            # LongBench
            # ------------------------------------------------------------------
            print("\n[LongBench] Running benchmark...")
            try:
                lb = LongBenchEvaluator(
                    model=model,
                    vocab_size=VOCAB,
                    num_classes=NUM_CLASSES,
                    n_samples=args.n_samples,
                    device=args.device,
                    chunk_size=args.chunk_size,
                )
                lb_results = lb.run(
                    context_lengths=[l for l in args.seq_lengths if l <= 8192],
                    verbose=True,
                )
                model_results["longbench"] = lb_results
            except Exception as e:
                print(f"  LongBench failed: {e}")
                lb_results = {"aggregate_score": 0.0}
                model_results["longbench"] = {}

            # Add to Pareto analysis
            for seq_len in args.seq_lengths:
                if seq_len not in latency_results or not latency_results[seq_len]:
                    continue
                lat = latency_results[seq_len]

                # Pull RULER NIAH accuracy for this context length
                acc = 0.5   # default
                if ruler_results and "niah_single" in ruler_results:
                    niah = ruler_results["niah_single"]
                    if seq_len in niah:
                        acc = niah[seq_len].get("accuracy", 0.5)

                try:
                    from synapnet_edge.quantization.cajq import estimate_model_bits
                    bit_info = estimate_model_bits(model)
                    eff_bits = bit_info["effective_bits"]
                except Exception:
                    eff_bits = 16.0

                family = METHOD_FAMILIES.get(model_name, "Other")
                pareto.add_point(ParetoPoint(
                    model_name=model_name,
                    accuracy=acc,
                    latency_tok_s=lat.get("tokens_per_second", 0.0),
                    memory_mb=lat.get("peak_memory_mb", 0.0),
                    context_len=seq_len,
                    effective_bits=eff_bits,
                    method_family=family,
                ))

        all_results[model_name] = model_results

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval_benchmarks] Results saved to {results_path}")

    # Generate Pareto plots
    if not args.hardware_only and pareto.points:
        pareto.save(str(output_dir / "pareto_points.json"))
        pareto.plot_accuracy_vs_latency(str(output_dir / "pareto_acc_latency.pdf"))
        pareto.plot_accuracy_vs_memory(str(output_dir / "pareto_acc_memory.pdf"))
        pareto.plot_bits_vs_accuracy(str(output_dir / "bits_vs_accuracy.pdf"))
        pareto.plot_context_length_heatmap(str(output_dir / "context_accuracy_heatmap.pdf"))

        print("\n" + pareto.summary_table())

    print(f"\n[eval_benchmarks] All done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
