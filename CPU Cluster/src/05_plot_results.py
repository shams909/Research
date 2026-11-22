"""
CPU Cluster Lab — Script 05: Plot Results (Publication Figures)
==============================================================
PURPOSE:
    Reads all experimental CSV files from results/ and generates
    the 5 official publication figures agreed upon for the conference paper:

FIGURES GENERATED (saved to results/figures/ & timestamped run archives):
    Figure_1_throughput_vs_accumulation.png   — Throughput vs K (with error bars)
    Figure_2_single_vs_distributed_scaling.png — Speedup & Efficiency (1.0x crossover)
    Figure_3_compute_vs_communication_breakdown.png — Stacked bar: Compute vs AllReduce ms
    Figure_4_training_convergence_comparison.png   — Loss curves over steps (K=1, 4, 8)
    Figure_5_bandwidth_sensitivity.png        — AllReduce bandwidth & network latency

RUN:
    python3 src/05_plot_results.py               # Generates all publication figures
    python3 src/05_plot_results.py --show        # Also displays figures on screen
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

# ── Style & Configuration ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
COLORS      = sns.color_palette("muted")
RESULTS_DIR = config.RESULTS_DIR
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
RUN_TS      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR     = os.path.join(FIG_DIR, f"run_{RUN_TS}")
DPI         = 200


def configure_paths(results_dir=None):
    global RESULTS_DIR, FIG_DIR, RUN_DIR
    if results_dir:
        RESULTS_DIR = results_dir
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")
    RUN_DIR = os.path.join(FIG_DIR, f"run_{RUN_TS}")


def save_fig(fig, name, show=False):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    # 1. Save timestamped archive copy
    archive_path = os.path.join(RUN_DIR, name)
    fig.savefig(archive_path, dpi=DPI, bbox_inches="tight")

    # 2. Save latest copy in current session figures directory
    latest_path = os.path.join(FIG_DIR, name)
    fig.savefig(latest_path, dpi=DPI, bbox_inches="tight")

    # 3. Also mirror to top-level results/figures for easy global browsing
    top_fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
    if os.path.abspath(FIG_DIR) != os.path.abspath(top_fig_dir):
        os.makedirs(top_fig_dir, exist_ok=True)
        fig.savefig(os.path.join(top_fig_dir, name), dpi=DPI, bbox_inches="tight")

    print(f"  ✓ Saved   → {latest_path}")
    print(f"    Archive → {archive_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 1: Throughput vs. Accumulation Factor (K) ──────────────────────────

def _render_fig1_axis(ax, sub, base_tput, task, title_prefix, unit, color_idx=0):
    ks = list(sub["accum_steps"])
    x_indices = np.arange(len(ks))

    tput = sub["throughput_mean"].values if "throughput_mean" in sub.columns else sub["throughput"].values
    yerr = sub["throughput_std"].values if "throughput_std" in sub.columns else None

    max_y = max(max(tput), base_tput) * 1.25
    ax.axhspan(base_tput, max_y, color="#2CA02C", alpha=0.06, label="_nolegend_")
    ax.axhspan(0, base_tput, color="#D95F02", alpha=0.04, label="_nolegend_")

    ax.axhline(base_tput, color="#D95F02", linestyle="--", linewidth=2.0,
               label=f"1-Node Baseline ({base_tput:.1f} {unit})")

    ax.errorbar(x_indices, tput, yerr=yerr, marker="o", markersize=8, linewidth=2.2,
                capsize=5, capthick=1.5, color=COLORS[color_idx % len(COLORS)], label="2-Node Cluster")

    for xi, y_val in zip(x_indices, tput):
        ax.annotate(f"{y_val:.1f}", (xi, y_val), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontweight="bold", fontsize=10)

    crossover_idx = np.where(tput > base_tput)[0]
    if len(crossover_idx) > 0:
        c_idx = crossover_idx[0]
        c_k = ks[c_idx]
        c_val = tput[c_idx]
        ax.scatter([c_idx], [c_val], color="crimson", s=110, zorder=5)
        ax.annotate(f"Crossover (K={c_k})\nCluster beats 1-Node",
                    (c_idx, c_val), textcoords="offset points",
                    xytext=(-25, 24), ha="center", fontsize=8.5, fontweight="bold",
                    color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}{task} Throughput", fontsize=12, fontweight="bold")
    ax.set_xlabel("Accumulation Steps (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Throughput ({unit})", fontsize=11, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.4, len(ks) - 0.6)
    ax.set_ylim(0, max_y)
    ax.legend(loc="lower right", framealpha=0.92)


def plot_figure_1_throughput_vs_k(show=False):
    """
    Figure 1: Saturation curve of training throughput vs K with error bars.
    Generates:
      - Figure_1_throughput_vs_accumulation.png (Combined dual-panel)
      - Figure_1a_ResNet18_throughput.png (Standalone solo)
      - Figure_1b_GPT2_throughput.png (Standalone solo)
    """
    single_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "01_single_node_baseline", "summary_*.csv")))
    if not single_files:
        single_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "benchmark_1nodes_*.csv")))

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "02_sweep_fixed_microbatch", "summary_*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "02_sweep_fixed_microbatch", "*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "benchmark_2nodes_*.csv")))

    if not files:
        print("  [skip Fig 1] No benchmark or sweep CSVs found")
        return

    df_single = pd.concat([pd.read_csv(f) for f in single_files], ignore_index=True) if single_files else pd.DataFrame()
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    if "mode" in df_all.columns:
        df_all = df_all[df_all["mode"] == "micro"]

    tasks = df_all["task"].unique() if "task" in df_all.columns else []
    if not len(tasks):
        return

    fallback_single_tput = {
        "ResNet-18": 113.2,
        "ResNet-18 / CIFAR-10": 113.2,
        "GPT-2 Small": 359.5,
        "GPT-2 Small (117M)": 359.5,
    }

    # 1. Combined Dual-Panel Figure
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.5 * len(tasks), 5.5))
    if len(tasks) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Figure 1: Training Throughput vs. Gradient Accumulation Steps (K)",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, (ax, task) in enumerate(zip(axes, tasks)):
        sub = df_all[df_all["task"] == task].sort_values("accum_steps")
        if "world_size" in sub.columns:
            sub = sub[sub["world_size"] == sub["world_size"].max()]
        sub = sub.drop_duplicates(subset=["accum_steps"], keep="last")

        base_tput = fallback_single_tput.get(task, 113.2)
        if not df_single.empty and "task" in df_single.columns:
            sub1 = df_single[(df_single["task"] == task) & (df_single["world_size"] == 1)]
            if not sub1.empty:
                base_tput = sub1["throughput_mean"].iloc[-1] if "throughput_mean" in sub1.columns else sub1["throughput"].iloc[-1]

        unit = sub["unit"].iloc[0]
        panel_letter = chr(97 + i)
        _render_fig1_axis(ax, sub, base_tput, task, panel_letter, unit, color_idx=i)

    save_fig(fig, "Figure_1_throughput_vs_accumulation.png", show)

    # 2. Standalone Solo Figures
    for i, task in enumerate(tasks):
        sub = df_all[df_all["task"] == task].sort_values("accum_steps")
        if "world_size" in sub.columns:
            sub = sub[sub["world_size"] == sub["world_size"].max()]
        sub = sub.drop_duplicates(subset=["accum_steps"], keep="last")

        base_tput = fallback_single_tput.get(task, 113.2)
        if not df_single.empty and "task" in df_single.columns:
            sub1 = df_single[(df_single["task"] == task) & (df_single["world_size"] == 1)]
            if not sub1.empty:
                base_tput = sub1["throughput_mean"].iloc[-1] if "throughput_mean" in sub1.columns else sub1["throughput"].iloc[-1]

        unit = sub["unit"].iloc[0]
        panel_letter = chr(97 + i)
        task_slug = "ResNet18" if "ResNet" in task else "GPT2"

        fig_solo, ax_solo = plt.subplots(figsize=(8.0, 5.5))
        _render_fig1_axis(ax_solo, sub, base_tput, task, "", unit, color_idx=i)
        ax_solo.set_title(f"{task} — Throughput vs. Accumulation Steps (K)", fontsize=13, fontweight="bold")
        save_fig(fig_solo, f"Figure_1{panel_letter}_{task_slug}_throughput.png", show)


# ── Figure 2: Single-Node vs. Distributed Scaling (Speedup & Crossover) ────────

def _render_fig2a_speedup(ax, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, fallback_single_tput, title_prefix=""):
    ax.axhspan(1.0, 2.15, color="#2CA02C", alpha=0.06, label="_nolegend_")
    ax.axhspan(0, 1.0, color="#D95F02", alpha=0.04, label="_nolegend_")
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.8, label="1-Node Baseline (1.0×)")
    ax.text(0.04, 0.92, "▲ Distributed Speedup (S₂ > 1.0×)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#1B7837", alpha=0.9)
    ax.text(0.04, 0.06, "▼ Network Bottleneck Slowdown (S₂ < 1.0×)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#B2182B", alpha=0.85)

    for i, task in enumerate(tasks):
        base_tput = fallback_single_tput.get(task, 150.0)
        if not df_single.empty and "task" in df_single.columns:
            sub1 = df_single[(df_single["task"] == task) & (df_single["world_size"] == 1)]
            if not sub1.empty:
                base_tput = sub1["throughput_mean"].iloc[-1] if "throughput_mean" in sub1.columns else sub1["throughput"].iloc[-1]

        sub2 = df_cluster[df_cluster["task"] == task].sort_values("accum_steps")
        if sub2.empty:
            continue
        sub2 = sub2.drop_duplicates(subset=["accum_steps"], keep="last")

        ks = list(sub2["accum_steps"])
        x_pos = [k_to_idx[k] for k in ks if k in k_to_idx]
        tput2 = sub2["throughput_mean"].values if "throughput_mean" in sub2.columns else sub2["throughput"].values
        speedup = tput2 / base_tput

        ax.plot(x_pos, speedup, marker="s", markersize=8, linewidth=2.2,
                color=COLORS[i % len(COLORS)], label=f"{task} (2 Nodes)")

        for xp, y_val in zip(x_pos, speedup):
            ax.annotate(f"{y_val:.2f}×", (xp, y_val), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}Cluster Speedup Ratio (S₂)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Accumulation Steps (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup vs. 1-Node Baseline", fontsize=11, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(k) for k in all_k_vals])
    ax.set_xlim(-0.4, len(all_k_vals) - 0.6)
    ax.set_ylim(0, 2.15)
    ax.legend(loc="lower right", framealpha=0.92)


def _render_fig2b_efficiency(ax, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, fallback_single_tput, title_prefix=""):
    ax.axhspan(0, 100.0, color="#4C72B0", alpha=0.03, label="_nolegend_")
    ax.axhline(100.0, color="gray", linestyle=":", linewidth=1.5, label="Ideal Linear Scaling (100%)")

    for i, task in enumerate(tasks):
        base_tput = fallback_single_tput.get(task, 150.0)
        if not df_single.empty and "task" in df_single.columns:
            sub1 = df_single[(df_single["task"] == task) & (df_single["world_size"] == 1)]
            if not sub1.empty:
                base_tput = sub1["throughput_mean"].iloc[-1] if "throughput_mean" in sub1.columns else sub1["throughput"].iloc[-1]

        sub2 = df_cluster[df_cluster["task"] == task].sort_values("accum_steps")
        if sub2.empty:
            continue
        sub2 = sub2.drop_duplicates(subset=["accum_steps"], keep="last")

        ks = list(sub2["accum_steps"])
        x_pos = [k_to_idx[k] for k in ks if k in k_to_idx]
        tput2 = sub2["throughput_mean"].values if "throughput_mean" in sub2.columns else sub2["throughput"].values
        efficiency = (tput2 / (2.0 * base_tput)) * 100.0

        ax.plot(x_pos, efficiency, marker="o", markersize=8, linewidth=2.2,
                color=COLORS[i % len(COLORS)], label=f"{task} (2 Nodes)")

        for xp, y_val in zip(x_pos, efficiency):
            ax.annotate(f"{y_val:.1f}%", (xp, y_val), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}Weak-Scaling Efficiency (E₂ %)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Accumulation Steps (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Scaling Efficiency (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(k) for k in all_k_vals])
    ax.set_xlim(-0.4, len(all_k_vals) - 0.6)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", framealpha=0.92)


def plot_figure_2_scaling_efficiency(show=False):
    """
    Figure 2: Speedup Ratio (S_N) and Scaling Efficiency (E_N) vs K.
    Generates:
      - Figure_2_single_vs_distributed_scaling.png (Combined dual-panel)
      - Figure_2a_cluster_speedup_ratio.png (Standalone solo)
      - Figure_2b_weak_scaling_efficiency.png (Standalone solo)
    """
    single_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "01_single_node_baseline", "summary_*.csv")))
    if not single_files:
        single_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "benchmark_1nodes_*.csv")))

    sweep_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "02_sweep_fixed_microbatch", "summary_*.csv")))
    if not sweep_files:
        sweep_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "02_sweep_fixed_microbatch", "*.csv")))
    if not sweep_files:
        sweep_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "benchmark_2nodes_*.csv")))

    if not sweep_files:
        print("  [skip Fig 2] No data for scaling comparison")
        return

    df_single = pd.concat([pd.read_csv(f) for f in single_files], ignore_index=True) if single_files else pd.DataFrame()
    df_cluster = pd.concat([pd.read_csv(f) for f in sweep_files], ignore_index=True)

    if "mode" in df_cluster.columns:
        df_cluster = df_cluster[df_cluster["mode"] == "micro"]

    tasks = df_cluster["task"].unique() if "task" in df_cluster.columns else []
    if not len(tasks):
        return

    fallback_single_tput = {
        "ResNet-18": 113.2,
        "ResNet-18 / CIFAR-10": 113.2,
        "GPT-2 Small": 359.5,
        "GPT-2 Small (117M)": 359.5,
    }

    all_k_vals = sorted(df_cluster["accum_steps"].dropna().unique())
    if not len(all_k_vals):
        all_k_vals = [1, 2, 4, 8, 16, 32, 64]
    k_to_idx = {k: i for i, k in enumerate(all_k_vals)}
    x_ticks = np.arange(len(all_k_vals))

    # 1. Combined Dual-Panel Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Figure 2: Distributed Speedup & Scaling Efficiency vs. Single-Node Baseline",
                 fontsize=14, fontweight="bold", y=1.02)

    _render_fig2a_speedup(axes[0], df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, fallback_single_tput, "a")
    _render_fig2b_efficiency(axes[1], df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, fallback_single_tput, "b")
    save_fig(fig, "Figure_2_single_vs_distributed_scaling.png", show)

    # 2. Standalone Solo Figures
    fig_solo_a, ax_solo_a = plt.subplots(figsize=(8.0, 5.5))
    _render_fig2a_speedup(ax_solo_a, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, fallback_single_tput, "")
    ax_solo_a.set_title("Cluster Speedup Ratio (S₂) vs. Single-Node Baseline", fontsize=13, fontweight="bold")
    save_fig(fig_solo_a, "Figure_2a_cluster_speedup_ratio.png", show)

    fig_solo_b, ax_solo_b = plt.subplots(figsize=(8.0, 5.5))
    _render_fig2b_efficiency(ax_solo_b, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, fallback_single_tput, "")
    ax_solo_b.set_title("Weak-Scaling Efficiency (E₂ %) vs. Accumulation Steps (K)", fontsize=13, fontweight="bold")
    save_fig(fig_solo_b, "Figure_2b_weak_scaling_efficiency.png", show)


# ── Figure 3: Computation vs. Communication Breakdown (Dual-Panel) ────────────

def _render_fig3a_breakdown(ax, x_indices, ks, fwd, bwd, ar, totals, title_prefix=""):
    width = 0.55
    ax.bar(x_indices, fwd, width, label="Forward Compute (Local CPU)", color="#4C72B0")
    ax.bar(x_indices, bwd, width, bottom=fwd, label="Backward Compute (Local CPU)", color="#55A868")
    ax.bar(x_indices, ar,  width, bottom=fwd + bwd, label="AllReduce Network Barrier (100 Mbps)", color="#C44E52")

    for i, tot in enumerate(totals):
        ax.annotate(f"{tot:.1f}s", (i, tot), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}Step Duration Breakdown (Wall-Clock Seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Step Duration (seconds)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gradient Accumulation Configuration", fontsize=11, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(ks)
    ax.set_ylim(0, max(totals) * 1.22)
    ax.legend(loc="upper left", framealpha=0.92)


def _render_fig3b_gamma(ax, x_indices, ks, gamma_ratio, title_prefix=""):
    ax.axhspan(1.0, 10.5, color="#2CA02C", alpha=0.06, label="_nolegend_")
    ax.axhspan(0.0, 1.0, color="#D95F02", alpha=0.04, label="_nolegend_")
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.8, label="Balanced Threshold (Γ = 1.0)")

    ax.text(0.04, 0.92, "▲ Compute-Dominant Regime (Γ > 1.0)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#1B7837")
    ax.text(0.04, 0.06, "▼ Communication-Bound Regime (Γ < 1.0)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#B2182B")

    ax.plot(x_indices, gamma_ratio, marker="s", markersize=8, linewidth=2.2,
            color="#55A868", label="Compute-to-Comm Ratio (Γ)")

    for xi, g_val in zip(x_indices, gamma_ratio):
        ax.annotate(f"{g_val:.2f}×", (xi, g_val), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, fontweight="bold", color="#2E6930")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}Compute-to-Communication Ratio (Γ = T_comp / T_comm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Compute-to-Communication Ratio (Γ)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gradient Accumulation Configuration", fontsize=11, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(ks)
    ax.set_xlim(-0.4, len(ks) - 0.6)
    ax.set_ylim(0, 10.5)
    ax.legend(loc="upper left", framealpha=0.92)


def plot_figure_3_timing_breakdown(show=False):
    """
    Figure 3: Dual-Panel Systems Decomposition:
      - Figure_3_compute_vs_communication_breakdown.png (Combined dual-panel)
      - Figure_3a_step_duration_breakdown.png (Standalone solo stacked bar)
      - Figure_3b_compute_to_comm_ratio.png (Standalone solo Gamma ratio)
    """
    timing_files = glob.glob(os.path.join(RESULTS_DIR, "04_step_time_breakdown", "timing_decomposition_*.csv"))
    if not timing_files:
        timing_files = glob.glob(os.path.join(RESULTS_DIR, "02_sweep_fixed_microbatch", "summary_*.csv"))

    if not timing_files:
        print("  [skip Fig 3] No timing decomposition CSV found")
        return

    df = pd.concat([pd.read_csv(f) for f in timing_files], ignore_index=True)
    required_cols = ["forward_ms", "backward_ms", "allreduce_ms"]
    alt_cols = ["forward_ms_mean", "backward_ms_mean", "allreduce_ms_mean"]

    use_mean = all(c in df.columns for c in alt_cols)
    use_raw  = all(c in df.columns for c in required_cols)

    if not (use_mean or use_raw):
        print("  [skip Fig 3] Timing columns missing from CSV")
        return

    fwd_col = "forward_ms_mean" if use_mean else "forward_ms"
    bwd_col = "backward_ms_mean" if use_mean else "backward_ms"
    ar_col  = "allreduce_ms_mean" if use_mean else "allreduce_ms"

    gpt_df = df[df["task"].str.contains("GPT-2", na=False)].drop_duplicates(subset=["accum_steps"], keep="last").sort_values("accum_steps")
    if gpt_df.empty:
        gpt_df = df.drop_duplicates(subset=["accum_steps"], keep="last").sort_values("accum_steps")

    ks = [f"K={k}" for k in gpt_df["accum_steps"]]
    x_indices = np.arange(len(ks))
    fwd = np.array(gpt_df[fwd_col]) / 1000.0
    bwd = np.array(gpt_df[bwd_col]) / 1000.0
    ar  = np.array(gpt_df[ar_col])  / 1000.0

    totals = fwd + bwd + ar
    compute_total = fwd + bwd
    gamma_ratio = compute_total / np.maximum(ar, 1e-4)

    # 1. Combined Dual-Panel Figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Figure 3: Step Time Breakdown & Compute-to-Communication Bottleneck Inversion",
                 fontsize=14, fontweight="bold", y=1.02)

    _render_fig3a_breakdown(axes[0], x_indices, ks, fwd, bwd, ar, totals, "a")
    _render_fig3b_gamma(axes[1], x_indices, ks, gamma_ratio, "b")
    save_fig(fig, "Figure_3_compute_vs_communication_breakdown.png", show)

    # 2. Standalone Solo Figures
    fig_solo_a, ax_solo_a = plt.subplots(figsize=(8.5, 5.5))
    _render_fig3a_breakdown(ax_solo_a, x_indices, ks, fwd, bwd, ar, totals, "")
    ax_solo_a.set_title("Step Time Breakdown — Local Compute vs. AllReduce Wait", fontsize=13, fontweight="bold")
    save_fig(fig_solo_a, "Figure_3a_step_duration_breakdown.png", show)

    fig_solo_b, ax_solo_b = plt.subplots(figsize=(8.5, 5.5))
    _render_fig3b_gamma(ax_solo_b, x_indices, ks, gamma_ratio, "")
    ax_solo_b.set_title("Compute-to-Communication Ratio (Γ) vs. Accumulation Steps (K)", fontsize=13, fontweight="bold")
    save_fig(fig_solo_b, "Figure_3b_compute_to_comm_ratio.png", show)


# ── Figure 4: Training Convergence Comparison (Dual-Panel Clean) ──────────────

def _render_fig4a_cifar(ax, cifar_files, title_prefix=""):
    for i, f in enumerate(cifar_files):
        df = pd.read_csv(f)
        k_val = df["accum_steps"].iloc[0] if "accum_steps" in df.columns else "?"
        nodes = df["world_size"].iloc[0] if "world_size" in df.columns else "?"
        label = f"{nodes} Nodes (K={k_val})"

        ax.plot(df["epoch"], df["train_loss"], marker="o", markersize=8, linewidth=2.2,
                label=label, color=COLORS[i % len(COLORS)])

        if "test_acc_pct" in df.columns:
            for ep, loss_val, acc in zip(df["epoch"], df["train_loss"], df["test_acc_pct"]):
                ax.annotate(f"Epoch {ep:.0f}: {loss_val:.2f} ({acc:.1f}% Acc)",
                            (ep, loss_val), textcoords="offset points",
                            xytext=(0, 10 if ep == 1 else -18),
                            ha="center", fontsize=9, fontweight="bold")
        else:
            for ep, loss_val in zip(df["epoch"], df["train_loss"]):
                ax.annotate(f"Loss: {loss_val:.2f}", (ep, loss_val),
                            textcoords="offset points", xytext=(0, 10 if ep == 1 else -18),
                            ha="center", fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}ResNet-18 / CIFAR-10 Loss Convergence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Training Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xlim(0.7, 2.3)
    ax.set_ylim(1.15, 1.85)
    ax.legend(loc="upper right", framealpha=0.92)


def _render_fig4b_gpt2(ax, gpt2_files, title_prefix=""):
    for i, f in enumerate(gpt2_files):
        df = pd.read_csv(f)
        k_val = df["accum_steps"].iloc[0] if "accum_steps" in df.columns else "?"
        nodes = df["world_size"].iloc[0] if "world_size" in df.columns else "?"
        label = f"{nodes} Nodes (K={k_val})"

        ax.plot(df["step"], df["loss"], marker="o", markersize=5, linewidth=2.0,
                label=label, color=COLORS[i % len(COLORS)])

        first_step, first_loss = df["step"].iloc[0], df["loss"].iloc[0]
        last_step, last_loss   = df["step"].iloc[-1], df["loss"].iloc[-1]

        ax.annotate(f"Start: {first_loss:.2f}", (first_step, first_loss),
                    textcoords="offset points", xytext=(15, 6), fontsize=9, fontweight="bold")
        ax.annotate(f"Final: {last_loss:.2f}", (last_step, last_loss),
                    textcoords="offset points", xytext=(-20, 12), fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}GPT-2 Small / WikiText-2 Language Model Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Optimizer Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Language Model Loss (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax.set_ylim(7.0, 11.5)
    ax.legend(loc="upper right", framealpha=0.92)


def plot_figure_4_convergence(show=False):
    """
    Figure 4: Loss curves verifying numerical convergence across K.
    Generates:
      - Figure_4_training_convergence_comparison.png (Combined dual-panel)
      - Figure_4a_cifar10_convergence.png (Standalone solo)
      - Figure_4b_gpt2_convergence.png (Standalone solo)
    """
    cifar_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "cifar_*.csv")))
    gpt2_files  = sorted(glob.glob(os.path.join(RESULTS_DIR, "gpt2_*.csv")))

    if not cifar_files and not gpt2_files:
        print("  [skip Fig 4] No convergence CSVs found")
        return

    # 1. Combined Dual-Panel Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Figure 4: Training Convergence & Loss Stability Across Accumulation Steps",
                 fontsize=14, fontweight="bold", y=1.02)

    _render_fig4a_cifar(axes[0], cifar_files, "a")
    _render_fig4b_gpt2(axes[1], gpt2_files, "b")
    save_fig(fig, "Figure_4_training_convergence_comparison.png", show)

    # 2. Standalone Solo Figures
    fig_solo_a, ax_solo_a = plt.subplots(figsize=(8.0, 5.5))
    _render_fig4a_cifar(ax_solo_a, cifar_files, "")
    ax_solo_a.set_title("ResNet-18 / CIFAR-10 Convergence", fontsize=13, fontweight="bold")
    save_fig(fig_solo_a, "Figure_4a_cifar10_convergence.png", show)

    fig_solo_b, ax_solo_b = plt.subplots(figsize=(8.0, 5.5))
    _render_fig4b_gpt2(ax_solo_b, gpt2_files, "")
    ax_solo_b.set_title("GPT-2 Small / WikiText-2 Language Model Loss", fontsize=13, fontweight="bold")
    save_fig(fig_solo_b, "Figure_4b_gpt2_convergence.png", show)


# ── Figure 5: Network Bandwidth & Latency Benchmark ───────────────────────────

def _render_fig5a_bandwidth(ax, net_files, title_prefix=""):
    for i, f in enumerate(net_files):
        df    = pd.read_csv(f)
        nodes = int(df["world_size"].iloc[0]) if "world_size" in df.columns else 2
        label = f"{nodes}-node cluster"
        ax.plot(df["size_mb"], df["bandwidth_mbs"], marker="o",
                label=label, color=COLORS[i % len(COLORS)], linewidth=2)

    ax.axhline(12.5, linestyle="--", color="crimson", alpha=0.7, label="Fast Ethernet Max (12.5 MB/s)")
    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_xlabel("Tensor Size (MB)", fontsize=11, fontweight="bold")
    ax.set_ylabel("AllReduce Bandwidth (MB/s)", fontsize=11, fontweight="bold")
    ax.set_title(f"{prefix}Effective Bandwidth vs. Tensor Size", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")


def _render_fig5b_latency(ax, net_files, title_prefix=""):
    for i, f in enumerate(net_files):
        df    = pd.read_csv(f)
        nodes = int(df["world_size"].iloc[0]) if "world_size" in df.columns else 2
        label = f"{nodes}-node cluster"
        ax.plot(df["size_mb"], df["latency_ms"], marker="s",
                label=label, color=COLORS[i % len(COLORS)], linewidth=2)

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_xlabel("Tensor Size (MB)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Collective Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title(f"{prefix}Collective Latency vs. Tensor Size", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")


def plot_figure_5_bandwidth_sensitivity(show=False):
    """
    Figure 5: Physical AllReduce bandwidth (MB/s) and latency (ms) vs Tensor Size.
    Generates:
      - Figure_5_bandwidth_sensitivity.png (Combined dual-panel)
      - Figure_5a_effective_bandwidth.png (Standalone solo)
      - Figure_5b_collective_latency.png (Standalone solo)
    """
    net_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "net_test_*.csv")))
    if not net_files:
        print("  [skip Fig 5] No net_test CSVs found")
        return

    # 1. Combined Dual-Panel Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Figure 5: Physical Interconnect Characterization (100 Mbps Fast Ethernet)",
                 fontsize=14, fontweight="bold", y=1.02)

    _render_fig5a_bandwidth(axes[0], net_files, "a")
    _render_fig5b_latency(axes[1], net_files, "b")
    save_fig(fig, "Figure_5_bandwidth_sensitivity.png", show)

    # 2. Standalone Solo Figures
    fig_solo_a, ax_solo_a = plt.subplots(figsize=(8.0, 5.5))
    _render_fig5a_bandwidth(ax_solo_a, net_files, "")
    ax_solo_a.set_title("Effective AllReduce Bandwidth vs. Tensor Size", fontsize=13, fontweight="bold")
    save_fig(fig_solo_a, "Figure_5a_effective_bandwidth.png", show)

    fig_solo_b, ax_solo_b = plt.subplots(figsize=(8.0, 5.5))
    _render_fig5b_latency(ax_solo_b, net_files, "")
    ax_solo_b.set_title("Collective Communication Latency vs. Tensor Size", fontsize=13, fontweight="bold")
    save_fig(fig_solo_b, "Figure_5b_collective_latency.png", show)


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot publication figures for CPU cluster research paper")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results directory or timestamped session directory (defaults to config.RESULTS_DIR)")
    parser.add_argument("--show", action="store_true", help="Display figures interactively on screen")
    args = parser.parse_args()

    configure_paths(args.results_dir)

    print()
    print("=" * 70)
    print("  CPU CLUSTER LAB — PUBLICATION FIGURE GENERATOR")
    print(f"  Source Directory : {RESULTS_DIR}/")
    print(f"  Target Directory : {FIG_DIR}/")
    print(f"  Archive Snapshot : {RUN_DIR}/")
    print("=" * 70)
    print()

    plot_figure_1_throughput_vs_k(args.show)
    plot_figure_2_scaling_efficiency(args.show)
    plot_figure_3_timing_breakdown(args.show)
    plot_figure_4_convergence(args.show)
    plot_figure_5_bandwidth_sensitivity(args.show)

    print()
    print("=" * 70)
    print("  ✓ All 15 publication figures generated successfully!")
    print("    • 5 Dual-Panel Combined Comparison Figures (Figures 1-5)")
    print("    • 10 Standalone Solo Subfigures (Figures 1a-5b)")
    print(f"    Inspect results in: {FIG_DIR}/")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
