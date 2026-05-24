"""Pareto frontier analysis and plotting for the SynapNet-Edge paper.

Generates:
  Fig 1: Accuracy vs. latency (tokens/sec) scatter plot with Pareto frontier
  Fig 2: Accuracy vs. peak memory (MB) per context length
  Fig 3: Effective bits vs. accuracy (RULER + LongBench)
  Fig 4: Context length vs. accuracy heatmap across models
  Fig 5: BAEE compression stats over sequence length

Each model appears as a point coloured by method family.
The Pareto-optimal subset is highlighted with a connecting line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ParetoPoint:
    """One configuration in the accuracy-efficiency trade-off space."""
    model_name: str
    accuracy: float           # primary accuracy metric (e.g. RULER NIAH)
    latency_tok_s: float      # tokens per second (higher = better)
    memory_mb: float          # peak RAM (lower = better)
    context_len: int
    effective_bits: float     # quantization bit-width
    method_family: str        # "SynapNetEdge", "Mamba2", "Llama", "FalconH1", "EMLLM"
    metadata: dict = field(default_factory=dict)

    @property
    def efficiency_score(self) -> float:
        """Combined efficiency: accuracy / (memory_mb * (1/latency))."""
        return self.accuracy * self.latency_tok_s / max(1.0, self.memory_mb)


class ParetoAnalyzer:
    """Computes Pareto frontier and generates publication-quality plots."""

    # Colour palette per method family
    COLOURS = {
        "SynapNetEdge": "#2196F3",    # blue (ours)
        "Mamba2": "#4CAF50",           # green
        "Llama": "#FF9800",            # orange
        "FalconH1": "#9C27B0",         # purple
        "EMLLM": "#F44336",            # red
    }
    MARKERS = {
        "SynapNetEdge": "o",
        "Mamba2": "s",
        "Llama": "^",
        "FalconH1": "D",
        "EMLLM": "v",
    }

    def __init__(self, points: list[ParetoPoint] | None = None):
        self.points: list[ParetoPoint] = points or []

    def add_point(self, point: ParetoPoint) -> None:
        self.points.append(point)

    def add_points(self, points: list[ParetoPoint]) -> None:
        self.points.extend(points)

    # ------------------------------------------------------------------
    # Pareto dominance
    # ------------------------------------------------------------------

    def pareto_frontier(
        self,
        objectives: list[str] = ("accuracy", "latency_tok_s", "memory_mb"),
        directions: list[str] = ("max", "max", "min"),
    ) -> list[ParetoPoint]:
        """Return Pareto-optimal subset (no point dominated on all objectives)."""
        if not self.points:
            return []

        vals = np.array([
            [getattr(p, obj) for obj in objectives]
            for p in self.points
        ], dtype=float)

        # Flip minimisation objectives
        for j, d in enumerate(directions):
            if d == "min":
                vals[:, j] = -vals[:, j]

        dominated = np.zeros(len(self.points), dtype=bool)
        for i in range(len(self.points)):
            for j in range(len(self.points)):
                if i == j:
                    continue
                if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                    dominated[i] = True
                    break

        return [p for p, d in zip(self.points, dominated) if not d]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_accuracy_vs_latency(
        self,
        output_path: str = "paper/figures/pareto_acc_latency.pdf",
        context_len_filter: int | None = None,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Pareto] matplotlib not installed. Skipping plot.")
            return

        pts = self.points
        if context_len_filter is not None:
            pts = [p for p in pts if p.context_len == context_len_filter]
        if not pts:
            print("[Pareto] No points to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # All points
        for family in set(p.method_family for p in pts):
            family_pts = [p for p in pts if p.method_family == family]
            ax.scatter(
                [p.latency_tok_s for p in family_pts],
                [p.accuracy for p in family_pts],
                c=self.COLOURS.get(family, "#999"),
                marker=self.MARKERS.get(family, "o"),
                s=80, alpha=0.8, label=family, zorder=3,
            )

        # Pareto frontier
        frontier = self.pareto_frontier(
            objectives=["accuracy", "latency_tok_s"],
            directions=["max", "max"],
        )
        if frontier:
            frontier_sorted = sorted(frontier, key=lambda p: p.latency_tok_s)
            ax.plot(
                [p.latency_tok_s for p in frontier_sorted],
                [p.accuracy for p in frontier_sorted],
                "k--", lw=1.5, alpha=0.7, label="Pareto frontier", zorder=4,
            )

        ax.set_xlabel("Throughput (tokens/sec)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        title = "Accuracy vs. Throughput"
        if context_len_filter:
            title += f" (ctx={context_len_filter})"
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Pareto] Saved: {output_path}")

    def plot_accuracy_vs_memory(
        self,
        output_path: str = "paper/figures/pareto_acc_memory.pdf",
        context_lengths: list[int] | None = None,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError:
            print("[Pareto] matplotlib not installed. Skipping.")
            return

        if context_lengths is None:
            context_lengths = sorted(set(p.context_len for p in self.points))

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = cm.get_cmap("viridis", len(context_lengths))

        for ci, ctx in enumerate(context_lengths):
            pts = [p for p in self.points if p.context_len == ctx]
            if not pts:
                continue
            for family in set(p.method_family for p in pts):
                fp = [p for p in pts if p.method_family == family]
                ax.scatter(
                    [p.memory_mb for p in fp],
                    [p.accuracy for p in fp],
                    c=[cmap(ci)] * len(fp),
                    marker=self.MARKERS.get(family, "o"),
                    s=60, alpha=0.75,
                    label=f"{family} ctx={ctx}" if ci == 0 else "",
                )

        ax.set_xlabel("Peak Memory (MB)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy vs. Peak Memory by Context Length", fontsize=13)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()   # lower memory = better (to the right)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Pareto] Saved: {output_path}")

    def plot_bits_vs_accuracy(
        self,
        output_path: str = "paper/figures/bits_vs_accuracy.pdf",
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(7, 5))

        for family in set(p.method_family for p in self.points):
            pts = [p for p in self.points if p.method_family == family]
            pts.sort(key=lambda p: p.effective_bits)
            ax.plot(
                [p.effective_bits for p in pts],
                [p.accuracy for p in pts],
                marker=self.MARKERS.get(family, "o"),
                color=self.COLOURS.get(family, "#999"),
                lw=1.5, markersize=7, label=family,
            )

        ax.set_xlabel("Effective Bit-Width", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Quantization Bit-Width vs. Accuracy", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Pareto] Saved: {output_path}")

    def plot_context_length_heatmap(
        self,
        output_path: str = "paper/figures/context_accuracy_heatmap.pdf",
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        models = sorted(set(p.model_name for p in self.points))
        ctx_lens = sorted(set(p.context_len for p in self.points))

        matrix = np.zeros((len(models), len(ctx_lens)))
        for i, m in enumerate(models):
            for j, c in enumerate(ctx_lens):
                matching = [p.accuracy for p in self.points
                            if p.model_name == m and p.context_len == c]
                matrix[i, j] = np.mean(matching) if matching else 0.0

        fig, ax = plt.subplots(figsize=(max(6, len(ctx_lens) * 1.2), max(4, len(models) * 0.7)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)

        ax.set_xticks(range(len(ctx_lens)))
        ax.set_xticklabels([str(c) for c in ctx_lens], rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("Context Length", fontsize=11)
        ax.set_title("Accuracy by Model and Context Length", fontsize=13)

        for i in range(len(models)):
            for j in range(len(ctx_lens)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if matrix[i, j] < 0.7 else "white")

        plt.colorbar(im, ax=ax, label="Accuracy")
        fig.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Pareto] Saved: {output_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(self) -> str:
        """Return a markdown table of all Pareto points."""
        lines = [
            "| Model | Method | Bits | Acc | Tok/s | Mem(MB) | ctx_len |",
            "|-------|--------|------|-----|-------|---------|---------|",
        ]
        for p in sorted(self.points, key=lambda x: -x.accuracy):
            lines.append(
                f"| {p.model_name} | {p.method_family} | {p.effective_bits:.1f} | "
                f"{p.accuracy:.3f} | {p.latency_tok_s:.1f} | "
                f"{p.memory_mb:.0f} | {p.context_len} |"
            )
        return "\n".join(lines)

    def save(self, path: str) -> None:
        data = [
            {
                "model_name": p.model_name,
                "accuracy": p.accuracy,
                "latency_tok_s": p.latency_tok_s,
                "memory_mb": p.memory_mb,
                "context_len": p.context_len,
                "effective_bits": p.effective_bits,
                "method_family": p.method_family,
                "metadata": p.metadata,
            }
            for p in self.points
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Pareto] Points saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ParetoAnalyzer":
        with open(path) as f:
            data = json.load(f)
        points = [ParetoPoint(**d) for d in data]
        return cls(points)
