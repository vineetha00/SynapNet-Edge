"""Visualization utilities for SynapNet-Edge.

Publication-quality plots for:
  - Salience heatmaps (per-layer, per-token salience scores)
  - Episodic memory write histograms (which positions were stored)
  - Training history (loss / accuracy curves)
  - BAEE compression statistics (budget usage over time)
  - CAJQ quantization error analysis
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. "
                          "Install with: pip install matplotlib")


def plot_salience_heatmap(
    debug_masks: list[torch.Tensor],   # list[depth] of (B, T) tensors
    seq_len: int | None = None,
    sample_idx: int = 0,
    output_path: str = "paper/figures/salience_heatmap.pdf",
    title: str = "SynapNet-Edge Salience Masks (per layer)",
) -> None:
    """Plot per-layer salience masks as a heatmap.

    Args:
        debug_masks: from model.forward() — one (B, T) tensor per block
        sample_idx:  which batch element to visualise
        output_path: save path
    """
    plt = _ensure_mpl()

    masks_np = [m[sample_idx].float().cpu().numpy() for m in debug_masks]
    depth = len(masks_np)
    T = masks_np[0].shape[0]
    if seq_len:
        T = min(T, seq_len)
        masks_np = [m[:T] for m in masks_np]

    data = np.stack(masks_np, axis=0)   # (depth, T)

    fig, ax = plt.subplots(figsize=(min(20, T // 50 + 6), depth * 0.6 + 1))
    im = ax.imshow(data, aspect="auto", cmap="plasma", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("Token position", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_yticks(range(depth))
    ax.set_yticklabels([f"Layer {i}" for i in range(depth)], fontsize=8)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, label="Salience score")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Salience heatmap saved: {output_path}")


def plot_memory_write_histogram(
    debug_topk: list[torch.Tensor],   # list[depth] of (B, k) written-slot indices
    seq_len: int,
    sample_idx: int = 0,
    output_path: str = "paper/figures/memory_write_hist.pdf",
    title: str = "Episodic Memory Write Frequency",
) -> None:
    """Plot histogram of which token positions were written to episodic memory.

    A peaked histogram (concentrated at specific positions) indicates the
    model is learning to focus on salient tokens.  A flat histogram suggests
    random writes (uninformative episodic memory).
    """
    plt = _ensure_mpl()

    fig, axes = plt.subplots(
        len(debug_topk), 1,
        figsize=(10, 2 * len(debug_topk)),
        sharex=True,
    )
    if len(debug_topk) == 1:
        axes = [axes]

    for i, (topk, ax) in enumerate(zip(debug_topk, axes)):
        indices = topk[sample_idx].cpu().numpy().flatten()
        ax.hist(indices, bins=min(50, seq_len // 10), range=(0, seq_len),
                color="#2196F3", alpha=0.8, edgecolor="white", lw=0.3)
        ax.set_ylabel(f"L{i} count", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Token position", fontsize=11)
    axes[0].set_title(title, fontsize=12)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Memory write histogram saved: {output_path}")


def plot_training_history(
    history: list[dict],
    output_path: str = "paper/figures/training_history.pdf",
) -> None:
    """Plot training loss and accuracy over QAT phases."""
    plt = _ensure_mpl()

    phases = list(set(r["phase"] for r in history))
    phase_colours = {"warmup": "#FF9800", "qat": "#2196F3", "finetune": "#4CAF50"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    global_step = 0
    for record in history:
        c = phase_colours.get(record["phase"], "#999")
        ax1.scatter(global_step, record["train_loss"], color=c, s=15, alpha=0.8)
        if record.get("eval_acc") is not None:
            ax2.scatter(global_step, record["eval_acc"], color=c, s=15, alpha=0.8)
        global_step += 1

    ax1.set_ylabel("Training Loss", fontsize=11)
    ax1.set_title("SynapNet-Edge QAT Training", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Eval Accuracy", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=p) for p, c in phase_colours.items()]
    ax1.legend(handles=handles, fontsize=9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Training history saved: {output_path}")


def plot_compression_stats(
    baee_stats_over_time: list[dict],
    output_path: str = "paper/figures/baee_compression.pdf",
) -> None:
    """Plot BAEE compression events over sequence length."""
    plt = _ensure_mpl()

    if not baee_stats_over_time:
        print("[Viz] No BAEE stats to plot.")
        return

    steps = list(range(len(baee_stats_over_time)))
    int8_counts = [s.get("n_int8_compressions", 0) for s in baee_stats_over_time]
    summ_counts = [s.get("n_summarizations", 0) for s in baee_stats_over_time]
    evict_counts = [s.get("n_evictions", 0) for s in baee_stats_over_time]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(steps, int8_counts, label="INT8 compress", color="#2196F3", alpha=0.8)
    ax.bar(steps, summ_counts, bottom=int8_counts, label="Summarize", color="#FF9800", alpha=0.8)
    ax.bar(steps, evict_counts,
           bottom=[a + b for a, b in zip(int8_counts, summ_counts)],
           label="Evict", color="#F44336", alpha=0.8)

    ax.set_xlabel("Chunk index (streaming inference)", fontsize=11)
    ax.set_ylabel("Compression events", fontsize=11)
    ax.set_title("BAEE Compression Events Over Sequence", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] BAEE compression stats saved: {output_path}")


def plot_quantization_error(
    model_fp16: torch.nn.Module,
    model_quantized: torch.nn.Module,
    seq_len: int = 512,
    vocab_size: int = 32000,
    output_path: str = "paper/figures/quant_error.pdf",
) -> dict[str, float]:
    """Compare FP16 vs quantized model outputs; plot error distribution."""
    plt = _ensure_mpl()

    device = next(model_fp16.parameters()).device
    ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        out_fp16 = model_fp16(ids)[0].float()
        out_quant = model_quantized(ids)[0].float()

    error = (out_fp16 - out_quant).abs()
    rel_error = error / (out_fp16.abs() + 1e-8)

    flat_error = error.flatten().cpu().numpy()
    flat_rel = rel_error.flatten().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(flat_error, bins=100, color="#2196F3", alpha=0.8, edgecolor="white", lw=0.2)
    ax1.set_xlabel("|FP16 - Quantized|", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Absolute Quantization Error", fontsize=12)
    ax1.set_yscale("log")

    ax2.hist(np.clip(flat_rel, 0, 2), bins=100, color="#FF9800", alpha=0.8,
             edgecolor="white", lw=0.2)
    ax2.set_xlabel("Relative error", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Relative Quantization Error", fontsize=12)
    ax2.set_yscale("log")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Quantization error plot saved: {output_path}")

    return {
        "mean_abs_error": float(flat_error.mean()),
        "p99_abs_error": float(np.percentile(flat_error, 99)),
        "mean_rel_error": float(flat_rel.mean()),
    }
