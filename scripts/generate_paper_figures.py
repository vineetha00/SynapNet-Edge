"""Combine all experiment outputs into publication-quality figures + LaTeX tables.

Inputs:
  results/scaled/exp_cajq_long_context.json
  results/scaled/exp_baee_memory_pressure.json
  results/scaled/exp_hardware.json

Outputs:
  paper/figures/scaled_fig1_cajq_long_context.pdf
  paper/figures/scaled_fig2_baee_retention.pdf
  paper/figures/scaled_fig3_pareto_acc_throughput.pdf
  paper/figures/scaled_fig4_hardware_scaling.pdf
  paper/figures/scaled_fig5_pareto_acc_bits.pdf
  paper/figures/scaled_paper_summary.md   — LaTeX-ready tables + key numbers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent.parent / "results" / "scaled"
FIG_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: CAJQ long-context accuracy retention
# ---------------------------------------------------------------------------

def fig_cajq_long_context(data: dict):
    if not data:
        return
    colours = {"fp16": "#000000", "int8_uniform": "#FF9800",
               "int4_uniform": "#F44336", "cajq": "#2196F3"}
    markers = {"fp16": "o", "int8_uniform": "s", "int4_uniform": "^", "cajq": "D"}
    labels = {
        "fp16": "FP16 (baseline)",
        "int8_uniform": "Uniform INT8",
        "int4_uniform": "Uniform INT4 (AWQ)",
        "cajq": "CAJQ (ours)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [("niah_single_acc", "NIAH-Single Accuracy"),
               ("multi_task_acc", "Multi-Task Avg Accuracy")]

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx]
        for variant, vdata in data["variants"].items():
            ctx_vals = sorted(int(c) for c in vdata["per_ctx"].keys())
            accs = [vdata["per_ctx"][str(c)].get(metric, 0.0) for c in ctx_vals]
            ax.plot(ctx_vals, accs,
                    marker=markers.get(variant, "o"),
                    color=colours.get(variant, "#999"),
                    label=labels.get(variant, variant),
                    lw=2, markersize=8, alpha=0.85)
        ax.set_xlabel("Context Length (tokens)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ctx_vals)
        ax.set_xticklabels([str(c) for c in ctx_vals])
        ax.set_ylim(0, max(0.8, max(
            v["per_ctx"][str(c)].get(metric, 0) for v in data["variants"].values()
            for c in ctx_vals
        )) + 0.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        ax.set_title(title, fontsize=12)

    fig.suptitle("Quantization Strategy vs. Long-Context Accuracy",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "scaled_fig1_cajq_long_context.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: BAEE target retention rate vs eviction policies
# ---------------------------------------------------------------------------

def fig_baee_retention(data: dict):
    if not data:
        return
    experiments = data["experiments"]
    seq_lens = [e["seq_len"] for e in experiments]
    policies = list(experiments[0]["policies"].keys())
    pol_labels = {
        "baee_salience": "BAEE (ours)",
        "fifo": "FIFO",
        "lru": "LRU",
        "random": "Random",
    }
    pol_colours = {
        "baee_salience": "#2196F3",
        "fifo": "#FF9800",
        "lru": "#9C27B0",
        "random": "#666666",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: target retention rate
    ax = axes[0]
    width = 0.18
    x = np.arange(len(seq_lens))
    for i, pol in enumerate(policies):
        rates = [e["policies"][pol]["target_retention_rate"] for e in experiments]
        ax.bar(x + (i - 1.5) * width, rates, width,
               label=pol_labels.get(pol, pol),
               color=pol_colours.get(pol, "#999"),
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sl} tok\n(budget={e['budget_slots']}/{e['total_writes']})"
                       for sl, e in zip(seq_lens, experiments)])
    ax.set_ylabel("Target Needle Retention Rate", fontsize=11)
    ax.set_title("Memory-Pressure NIAH: Target Retention vs. Eviction Policy",
                 fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper left")

    # Annotate forced-eviction percentage
    ax.text(0.5, 0.95, f"Forced eviction: "
            f"{experiments[0]['forced_eviction_pct']:.0%}",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    # Right: task accuracy
    ax = axes[1]
    for i, pol in enumerate(policies):
        accs = [e["policies"][pol]["accuracy"] for e in experiments]
        ax.bar(x + (i - 1.5) * width, accs, width,
               label=pol_labels.get(pol, pol),
               color=pol_colours.get(pol, "#999"),
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sl} tok" for sl in seq_lens])
    ax.set_ylabel("Task Accuracy", fontsize=11)
    ax.set_title("Memory-Pressure NIAH: Task Accuracy", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "scaled_fig2_baee_retention.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Pareto frontier — accuracy vs throughput
# ---------------------------------------------------------------------------

def fig_pareto_acc_throughput(cajq_data: dict, hw_data: dict):
    if not cajq_data or not hw_data:
        return

    # Combine: for each (variant, platform), get accuracy at ctx=2048
    # and tokens/sec at ctx=2048
    fig, ax = plt.subplots(figsize=(8, 5.5))

    families = {
        "fp16": ("FP16", "#000000", "o"),
        "int4_uniform": ("Uniform-INT4", "#F44336", "^"),
        "cajq": ("CAJQ (ours)", "#2196F3", "D"),
    }
    platforms = {
        "apple_silicon_mps": ("Apple Silicon", "filled"),
        "arm_cpu_proxy": ("ARM CPU", "hollow"),
    }

    target_ctx = 2048

    points = []
    for variant, (label, colour, marker) in families.items():
        if variant not in cajq_data["variants"]:
            continue
        v = cajq_data["variants"][variant]["per_ctx"].get(str(target_ctx))
        if not v or "niah_single_acc" not in v:
            continue
        acc = v["niah_single_acc"]
        for plat_key, (plat_label, style) in platforms.items():
            if plat_key not in hw_data["platforms"]:
                continue
            if variant not in hw_data["platforms"][plat_key]["variants"]:
                continue
            pv = hw_data["platforms"][plat_key]["variants"][variant]
            # Find matching ctx
            ctx_rec = next((r for r in pv["per_ctx"] if r["ctx_len"] == target_ctx), None)
            if not ctx_rec:
                continue
            tok_s = ctx_rec["tokens_per_second"]
            mfc = colour if style == "filled" else "white"
            ax.scatter(tok_s, acc, marker=marker, s=140,
                       facecolor=mfc, edgecolor=colour, linewidth=2,
                       label=f"{label} @ {plat_label}",
                       zorder=3, alpha=0.9)
            points.append((label, plat_label, tok_s, acc))

    ax.set_xlabel(f"Throughput at ctx={target_ctx} (tokens/sec)", fontsize=11)
    ax.set_ylabel(f"NIAH-Single Accuracy at ctx={target_ctx}", fontsize=11)
    ax.set_title(f"Pareto: Accuracy vs. Throughput (context={target_ctx} tokens)",
                 fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)

    # Pareto frontier: max acc per throughput level
    if points:
        # Sort by throughput, then mark dominators
        sorted_pts = sorted(points, key=lambda p: p[2])
        frontier = []
        best_acc = -1
        for p in reversed(sorted_pts):
            if p[3] > best_acc:
                frontier.append(p)
                best_acc = p[3]
        frontier = sorted(frontier, key=lambda p: p[2])
        if len(frontier) >= 2:
            ax.plot([p[2] for p in frontier], [p[3] for p in frontier],
                    "k--", lw=1, alpha=0.5, zorder=2)

    fig.tight_layout()
    out = FIG_DIR / "scaled_fig3_pareto_acc_throughput.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: Hardware scaling — tokens/sec vs ctx_len
# ---------------------------------------------------------------------------

def fig_hardware_scaling(hw_data: dict):
    if not hw_data:
        return
    families = {
        "fp16": ("FP16", "#000000", "-"),
        "int4_uniform": ("Uniform-INT4", "#F44336", "--"),
        "cajq": ("CAJQ", "#2196F3", "-."),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax_idx, (plat_key, ax) in enumerate(zip(hw_data["platforms"], axes)):
        plat_data = hw_data["platforms"][plat_key]
        plat_label = "Apple Silicon (MPS)" if "apple" in plat_key else "ARM CPU"
        for variant, (label, colour, ls) in families.items():
            if variant not in plat_data["variants"]:
                continue
            recs = plat_data["variants"][variant]["per_ctx"]
            if not recs:
                continue
            ctxs = [r["ctx_len"] for r in recs]
            tok_s = [r["tokens_per_second"] for r in recs]
            ax.plot(ctxs, tok_s, linestyle=ls, color=colour,
                    marker="o", lw=2, markersize=7, label=label, alpha=0.85)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Context Length (tokens)", fontsize=11)
        ax.set_ylabel("Throughput (tokens/sec)", fontsize=11)
        ax.set_title(plat_label, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("Consumer Hardware Throughput Scaling", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "scaled_fig4_hardware_scaling.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 5: Pareto — Accuracy vs Bit-width
# ---------------------------------------------------------------------------

def fig_pareto_acc_bits(cajq_data: dict):
    if not cajq_data:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))

    families = {
        "fp16": ("FP16", "#000000", "o"),
        "int8_uniform": ("Uniform-INT8", "#FF9800", "s"),
        "int4_uniform": ("Uniform-INT4 (AWQ)", "#F44336", "^"),
        "cajq": ("CAJQ (ours)", "#2196F3", "D"),
    }

    ctxs_to_plot = [512, 2048, 4096]
    n = len(ctxs_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ctx in zip(axes, ctxs_to_plot):
        for variant, (label, colour, marker) in families.items():
            if variant not in cajq_data["variants"]:
                continue
            vdata = cajq_data["variants"][variant]
            v = vdata["per_ctx"].get(str(ctx))
            if not v or "niah_single_acc" not in v:
                continue
            ax.scatter(vdata["effective_bits"], v["niah_single_acc"],
                       marker=marker, s=160, color=colour,
                       label=label, zorder=3, alpha=0.85,
                       edgecolor="white", linewidth=1)
            ax.annotate(f"{v['niah_single_acc']:.2f}",
                        (vdata["effective_bits"], v["niah_single_acc"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=8)
        ax.set_xlabel("Effective Bits", fontsize=11)
        ax.set_title(f"ctx = {ctx}", fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("NIAH-Single Accuracy", fontsize=11)
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Pareto: Accuracy vs. Bit-Width Across Context Lengths",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "scaled_fig5_pareto_acc_bits.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Paper-ready summary
# ---------------------------------------------------------------------------

def write_summary(cajq, baee, hw):
    lines = ["# SynapNet-Edge: Scaled Experiment Summary", ""]
    lines.append("## Model")
    if cajq:
        c = cajq["config"]
        lines.append(f"- Architecture: SynapNet-Edge hybrid (SSM + sparse-attn + episodic memory)")
        lines.append(f"- Parameters: ~8.7M  ({c['dim']}d × {c['depth']} blocks, "
                     f"{c['heads']} heads, {c['episodic_slots']} memory slots)")
        lines.append(f"- Vocab: {c['vocab_size']}, classes: {c['num_classes']}, "
                     f"max_len: {c['max_len']}")
    lines.append("")

    # Table 1: CAJQ accuracy
    if cajq:
        lines.append("## Table 1 — CAJQ vs. Uniform Quantization (NIAH-Single)")
        lines.append("")
        ctxs = sorted(set(int(c) for v in cajq["variants"].values()
                          for c in v["per_ctx"].keys()))
        header = "| Variant | Eff. Bits | " + " | ".join(f"ctx={c}" for c in ctxs) + " |"
        sep = "|---" * (2 + len(ctxs)) + "|"
        lines.append(header)
        lines.append(sep)
        for variant, vdata in cajq["variants"].items():
            row = [variant, f"{vdata['effective_bits']:.1f}"]
            for c in ctxs:
                rec = vdata["per_ctx"].get(str(c), {})
                row.append(f"{rec.get('niah_single_acc', 0):.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Table 2: BAEE retention
    if baee:
        lines.append("## Table 2 — BAEE Memory-Pressure Eviction Comparison")
        lines.append("")
        lines.append("Setup: " +
                     f"{baee['config']['n_needles']} needles, " +
                     f"target repeated {baee['config']['target_repeat']}× in "
                     f"{baee['config']['target_position_bias']} portion of context, " +
                     f"budget = {int(baee['config']['budget_multiplier']*100)}% of "
                     "total writes ({:.0%} forced eviction)".format(
                         1 - baee['config']['budget_multiplier']))
        lines.append("")
        for exp in baee["experiments"]:
            lines.append(f"### seq_len = {exp['seq_len']} "
                         f"(budget={exp['budget_slots']}/{exp['total_writes']}, "
                         f"forced eviction = {exp['forced_eviction_pct']:.0%})")
            lines.append("")
            lines.append("| Policy | Target Retention | Task Acc |")
            lines.append("|---|---|---|")
            for pol, r in exp["policies"].items():
                lines.append(f"| {pol} | {r['target_retention_rate']:.3f} | "
                             f"{r['accuracy']:.3f} |")
            lines.append("")

    # Table 3: Hardware
    if hw:
        lines.append("## Table 3 — Consumer Hardware Throughput (tokens/sec)")
        lines.append("")
        for plat_key, plat in hw["platforms"].items():
            plat_label = "Apple Silicon (MPS)" if "apple" in plat_key else "ARM CPU"
            lines.append(f"### {plat_label}")
            lines.append("")
            variants = list(plat["variants"].keys())
            ctxs = sorted(set(r["ctx_len"]
                              for v in plat["variants"].values()
                              for r in v["per_ctx"]))
            header = "| Variant | Bits | " + " | ".join(f"ctx={c}" for c in ctxs) + " |"
            sep = "|---" * (2 + len(ctxs)) + "|"
            lines.append(header)
            lines.append(sep)
            for v in variants:
                vd = plat["variants"][v]
                row = [v, f"{vd['effective_bits']:.1f}"]
                for c in ctxs:
                    rec = next((r for r in vd["per_ctx"] if r["ctx_len"] == c), None)
                    if rec:
                        row.append(f"{rec['tokens_per_second']:.0f}")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # Key headlines for paper abstract
    lines.append("## Headline Numbers")
    lines.append("")
    if baee and baee["experiments"]:
        for exp in baee["experiments"]:
            pols = exp["policies"]
            baee_ret = pols["baee_salience"]["target_retention_rate"]
            fifo_ret = pols["fifo"]["target_retention_rate"]
            if fifo_ret > 0:
                gain = baee_ret / fifo_ret
                lines.append(f"- At {exp['seq_len']} tokens with "
                             f"{exp['forced_eviction_pct']:.0%} forced eviction, "
                             f"BAEE retains the target needle "
                             f"{baee_ret:.0%} of the time vs. {fifo_ret:.0%} for FIFO.")
            else:
                lines.append(f"- At {exp['seq_len']} tokens with "
                             f"{exp['forced_eviction_pct']:.0%} forced eviction, "
                             f"**BAEE retains the target needle {baee_ret:.0%} of the time** "
                             f"vs. {fifo_ret:.0%} for FIFO/LRU "
                             f"(BAEE is ≥{int(baee_ret / max(0.01, pols['random']['target_retention_rate']))}× better than random).")
    if cajq:
        for variant in ("cajq", "int4_uniform", "fp16"):
            vd = cajq["variants"].get(variant)
            if not vd:
                continue
            ctxs = sorted(int(c) for c in vd["per_ctx"].keys())
            if not ctxs:
                continue
            acc_long = vd["per_ctx"][str(max(ctxs))].get("niah_single_acc", 0)
            acc_short = vd["per_ctx"][str(min(ctxs))].get("niah_single_acc", 0)
            lines.append(f"- {variant}: {acc_short:.3f} → {acc_long:.3f} "
                         f"NIAH-single accuracy from ctx={min(ctxs)} → {max(ctxs)} "
                         f"(eff_bits={vd['effective_bits']:.1f})")
    lines.append("")

    out = FIG_DIR / "scaled_paper_summary.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out}")


def main():
    print("Loading experiment results...")
    cajq_data = load_json(RESULTS_DIR / "exp_cajq_long_context.json")
    baee_data = load_json(RESULTS_DIR / "exp_baee_memory_pressure.json")
    hw_data = load_json(RESULTS_DIR / "exp_hardware.json")

    print("\nGenerating figures...")
    fig_cajq_long_context(cajq_data)
    fig_baee_retention(baee_data)
    fig_pareto_acc_throughput(cajq_data, hw_data)
    fig_hardware_scaling(hw_data)
    fig_pareto_acc_bits(cajq_data)
    write_summary(cajq_data, baee_data, hw_data)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
