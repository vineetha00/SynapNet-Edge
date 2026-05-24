"""Generate all Pareto frontier figures for the paper.

Loads benchmark results and produces publication-quality plots.

Usage:
  python scripts/plot_pareto.py --results results/benchmarks/pareto_points.json
  python scripts/plot_pareto.py --results results/benchmarks/ --format pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synapnet_edge.benchmarks.pareto import ParetoAnalyzer, ParetoPoint


def main():
    p = argparse.ArgumentParser(description="Generate Pareto plots for SynapNet-Edge")
    p.add_argument("--results", default="results/benchmarks/pareto_points.json",
                   help="Path to pareto_points.json from eval_benchmarks.py")
    p.add_argument("--output-dir", default="paper/figures")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--context-len", type=int, default=None,
                   help="Filter Pareto plot to one context length")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[plot_pareto] {results_path} not found. "
              "Run eval_benchmarks.py first.")
        sys.exit(1)

    analyzer = ParetoAnalyzer.load(str(results_path))
    print(f"[plot_pareto] Loaded {len(analyzer.points)} points from {results_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = f".{args.format}"

    # Figure 1: Accuracy vs. throughput
    analyzer.plot_accuracy_vs_latency(
        output_path=str(output_dir / f"fig1_pareto_acc_latency{ext}"),
        context_len_filter=args.context_len,
    )

    # Figure 2: Accuracy vs. memory
    analyzer.plot_accuracy_vs_memory(
        output_path=str(output_dir / f"fig2_pareto_acc_memory{ext}"),
    )

    # Figure 3: Bits vs. accuracy
    analyzer.plot_bits_vs_accuracy(
        output_path=str(output_dir / f"fig3_bits_vs_accuracy{ext}"),
    )

    # Figure 4: Context length heatmap
    analyzer.plot_context_length_heatmap(
        output_path=str(output_dir / f"fig4_context_accuracy_heatmap{ext}"),
    )

    # Print Pareto frontier
    frontier = analyzer.pareto_frontier(
        objectives=["accuracy", "latency_tok_s", "memory_mb"],
        directions=["max", "max", "min"],
    )
    print(f"\n[plot_pareto] Pareto frontier ({len(frontier)} points):")
    for p in sorted(frontier, key=lambda x: -x.accuracy):
        print(f"  {p.model_name:<30} acc={p.accuracy:.3f} "
              f"tok/s={p.latency_tok_s:.0f} mem={p.memory_mb:.0f}MB "
              f"ctx={p.context_len} bits={p.effective_bits:.1f}")

    # Summary table
    print("\n" + analyzer.summary_table())


if __name__ == "__main__":
    main()
