"""
CPU Cluster Lab — Script 05: Plot Results (Paper Figures)
==========================================================
PURPOSE:
    Reads all CSV files from results/ and generates publication-ready
    figures for your research paper. Run after collecting training data.

FIGURES GENERATED (saved to results/figures/):
    01_loss_curves.png       — Training loss over steps/epochs per config
    02_throughput_bar.png    — Throughput comparison: 1-node vs N-node
    03_scaling_efficiency.png— Scaling efficiency (%) vs number of nodes
    04_network_bandwidth.png — Network benchmark: bandwidth vs tensor size
    05_time_per_step.png     — Step time breakdown across configurations

RUN:
    python3 src/05_plot_results.py               # plots all available CSVs
    python3 src/05_plot_results.py --show        # also opens plots on screen
"""

import os
import sys
import glob
import argparse
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
COLORS   = sns.color_palette("muted")
FIG_DIR  = os.path.join(config.RESULTS_DIR, "figures")
RUN_TS   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR  = os.path.join(FIG_DIR, f"run_{RUN_TS}")
DPI      = 150


def save_fig(fig, name, show=False):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    # 1. Save timestamped archive copy
    archive_path = os.path.join(RUN_DIR, name)
    fig.savefig(archive_path, dpi=DPI, bbox_inches="tight")

    # 2. Save latest copy in main directory
    latest_path = os.path.join(FIG_DIR, name)
    fig.savefig(latest_path, dpi=DPI, bbox_inches="tight")

    print(f"  Saved → {latest_path}")
    print(f"  Archived → {archive_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 1: Loss Curves ─────────────────────────────────────────────────────

def plot_loss_curves(show=False):
    cifar_files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "cifar_*.csv")))
    gpt2_files  = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "gpt2_*.csv")))

    if not cifar_files and not gpt2_files:
        print("  [skip] No training CSVs found for loss curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Loss Curves — CPU Cluster", fontweight="bold")

    # CIFAR
    ax = axes[0]
    for i, f in enumerate(cifar_files):
        df    = pd.read_csv(f)
        label = f"{df['world_size'].iloc[0]}-node cluster"
        ax.plot(df["epoch"], df["train_loss"], marker="o", label=label,
                color=COLORS[i % len(COLORS)])
    ax.set_title("ResNet-18 / CIFAR-10")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.legend()

    # GPT-2
    ax = axes[1]
    for i, f in enumerate(gpt2_files):
        df    = pd.read_csv(f)
        label = f"{df['world_size'].iloc[0]}-node cluster"
        ax.plot(df["step"], df["loss"], label=label,
                color=COLORS[i % len(COLORS)], alpha=0.8)
    ax.set_title("GPT-2 Small / WikiText-2")
    ax.set_xlabel("Step")
    ax.set_ylabel("Language Model Loss")
    ax.legend()

    save_fig(fig, "01_loss_curves.png", show)


# ── Figure 2: Throughput Bar Chart ────────────────────────────────────────────

def plot_throughput(show=False):
    # Find latest benchmark file for each node count to avoid overlapping bars
    latest_bench_by_nodes = {}
    for f in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "benchmark_*nodes_*.csv"))):
        fname = os.path.basename(f)
        parts = fname.split("_")
        if len(parts) >= 2 and "nodes" in parts[1]:
            n_str = parts[1].replace("nodes", "")
            try:
                n = int(n_str)
                latest_bench_by_nodes[n] = f
            except ValueError:
                pass

    records = []

    # From latest benchmark CSVs
    for n, f in sorted(latest_bench_by_nodes.items()):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            records.append({
                "task"      : row["task"],
                "nodes"     : int(row["world_size"]),
                "throughput": float(row["throughput"]),
                "unit"      : row["unit"],
            })

    if not records:
        print("  [skip] No benchmark data found for throughput chart")
        return

    df_all = pd.DataFrame(records)

    tasks = df_all["task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]
    fig.suptitle("Training Throughput — CPU Cluster vs Node Count", fontweight="bold")

    for ax, task in zip(axes, tasks):
        sub = df_all[df_all["task"] == task].sort_values("nodes")
        unit = sub["unit"].iloc[0]
        bars = ax.bar(
            [f"{n}-node" for n in sub["nodes"]],
            sub["throughput"],
            color=COLORS[:len(sub)],
            edgecolor="white", linewidth=1.2
        )
        ax.bar_label(bars, fmt="%.1f", padding=4)
        ax.set_title(task)
        ax.set_ylabel(unit)
        ax.set_xlabel("Cluster Configuration")
        # Give some headroom so label doesn't touch the top
        max_y = sub["throughput"].max()
        ax.set_ylim(0, max_y * 1.18)

    save_fig(fig, "02_throughput_bar.png", show)


# ── Figure 3: Scaling Efficiency ──────────────────────────────────────────────

def plot_scaling_efficiency(show=False):
    latest_bench_by_nodes = {}
    for f in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "benchmark_*nodes_*.csv"))):
        fname = os.path.basename(f)
        parts = fname.split("_")
        if len(parts) >= 2 and "nodes" in parts[1]:
            n_str = parts[1].replace("nodes", "")
            try:
                n = int(n_str)
                latest_bench_by_nodes[n] = f
            except ValueError:
                pass

    # Build throughput per task per node count using latest data
    data = {}
    for n, f in sorted(latest_bench_by_nodes.items()):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            task  = row["task"]
            nodes = int(row["world_size"])
            tput  = float(row["throughput"])
            data.setdefault(task, {})[nodes] = tput

    if not data:
        print("  [skip] Need data from multiple node counts for efficiency plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Scaling Efficiency — CPU Cluster over Gigabit Ethernet",
                 fontweight="bold")

    for i, (task, node_tput) in enumerate(data.items()):
        nodes = sorted(node_tput.keys())
        if 1 not in node_tput:
            continue
        base  = node_tput[1]
        effs  = [node_tput[n] / (n * base) * 100 for n in nodes]
        ax.plot(nodes, effs, marker="o", label=task, color=COLORS[i], linewidth=2.5, markersize=8)
        for n, e in zip(nodes, effs):
            # Stagger labels to avoid overlapping at 100%
            offset_y = 9 if (i == 0 or n > 1) else -16
            ax.annotate(f"{e:.0f}%", (n, e), textcoords="offset points",
                        xytext=(0, offset_y), ha="center", fontsize=10, fontweight="bold",
                        color=COLORS[i])

    # Perfect scaling reference line
    max_nodes = max(max(d.keys()) for d in data.values())
    ax.axhline(100, linestyle="--", color="gray", alpha=0.5, label="Perfect (100%)")

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_ylim(0, 115)
    ax.set_xticks(range(1, max_nodes + 1))
    ax.legend(loc="lower left")

    save_fig(fig, "03_scaling_efficiency.png", show)


# ── Figure 4: Network Bandwidth ───────────────────────────────────────────────

def plot_network_bandwidth(show=False):
    net_files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "net_test_*.csv")))

    if not net_files:
        print("  [skip] No net_test CSVs found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Network Benchmark — Gigabit Ethernet AllReduce Performance",
                 fontweight="bold")

    for i, f in enumerate(net_files):
        df    = pd.read_csv(f)
        nodes = int(df["world_size"].iloc[0]) if "world_size" in df.columns else "?"
        label = f"{nodes}-node"
        axes[0].plot(df["size_mb"], df["bandwidth_mbs"], marker="o",
                     label=label, color=COLORS[i % len(COLORS)])
        axes[1].plot(df["size_mb"], df["latency_ms"], marker="s",
                     label=label, color=COLORS[i % len(COLORS)])

    # Theoretical GbE line
    axes[0].axhline(940, linestyle="--", color="gray", alpha=0.5, label="GbE max (940 MB/s)")

    axes[0].set_xlabel("Tensor Size (MB)")
    axes[0].set_ylabel("AllReduce Bandwidth (MB/s)")
    axes[0].set_title("Bandwidth vs Tensor Size")
    axes[0].legend()

    axes[1].set_xlabel("Tensor Size (MB)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency vs Tensor Size")
    axes[1].legend()

    save_fig(fig, "04_network_bandwidth.png", show)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Open plots on screen after saving")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  PLOTTING RESULTS — CPU Cluster Lab")
    print(f"  Reading from : {config.RESULTS_DIR}/")
    print(f"  Saving to    : {FIG_DIR}/")
    print("=" * 60)
    print()

    plot_loss_curves(args.show)
    plot_throughput(args.show)
    plot_scaling_efficiency(args.show)
    plot_network_bandwidth(args.show)

    print()
    print("=" * 60)
    print(f"  Done — figures in {FIG_DIR}/")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
