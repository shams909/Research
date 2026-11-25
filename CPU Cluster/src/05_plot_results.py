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
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
    COLORS = sns.color_palette("muted")
except ImportError:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

# Ensure subscripts render cleanly on both Windows and Linux without missing-glyph warnings
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Style & Configuration ─────────────────────────────────────────────────────
RESULTS_DIR        = config.RESULTS_DIR
EXTRA_RESULTS_DIRS = []
FIG_DIR            = os.path.join(RESULTS_DIR, "figures")
RUN_TS             = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR            = os.path.join(FIG_DIR, f"run_{RUN_TS}")
NO_MIRROR          = False
GENERATED_FIGS     = []
DPI                = 200

FALLBACK_SINGLE_TPUT = {
    "ResNet-18": 113.15,
    "ResNet-18 / CIFAR-10": 113.15,
    "GPT-2 Small": 358.20,
    "GPT-2 Small (117M)": 358.20,
    "Whisper-Tiny": 12.0,
    "Whisper-Tiny (Speech)": 12.0,
}


def configure_paths(results_dir=None, extra_results_dirs=None, out_dir=None, no_mirror=False):
    global RESULTS_DIR, EXTRA_RESULTS_DIRS, FIG_DIR, RUN_DIR, NO_MIRROR, GENERATED_FIGS
    if results_dir:
        RESULTS_DIR = results_dir
    EXTRA_RESULTS_DIRS = list(extra_results_dirs or [])
    FIG_DIR = out_dir if out_dir else os.path.join(RESULTS_DIR, "figures")
    RUN_DIR = os.path.join(FIG_DIR, f"run_{RUN_TS}")
    NO_MIRROR = bool(no_mirror)
    GENERATED_FIGS = []


def _all_results_dirs():
    dirs = [RESULTS_DIR]
    seen = {os.path.abspath(RESULTS_DIR)}
    for d in EXTRA_RESULTS_DIRS:
        if d:
            ad = os.path.abspath(d)
            if ad not in seen:
                seen.add(ad)
                dirs.append(d)
    return dirs


def _glob_results(*rel_parts, include_extra=True):
    """Collect matching CSV paths from primary RESULTS_DIR (and optional extra session dirs)."""
    dirs = _all_results_dirs() if include_extra else [RESULTS_DIR]
    matched = []
    for d in dirs:
        matched.extend(sorted(glob.glob(os.path.join(d, *rel_parts))))
    return matched


def _task_slug(task):
    if "ResNet" in task:
        return "ResNet18"
    if "Whisper" in task:
        return "WhisperTiny"
    return "GPT2"


def _resolve_base_tput(task, df_single):
    base_tput = FALLBACK_SINGLE_TPUT.get(task, 113.15)
    if not df_single.empty and "task" in df_single.columns:
        sub1 = df_single[(df_single["task"] == task) & (df_single["world_size"] == 1)]
        if not sub1.empty:
            sub1 = _dedup_task_rows(sub1)
            base_tput = sub1["throughput_mean"].iloc[-1] if "throughput_mean" in sub1.columns else sub1["throughput"].iloc[-1]
    return base_tput


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
    if not NO_MIRROR:
        top_fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
        if os.path.abspath(FIG_DIR) != os.path.abspath(top_fig_dir):
            os.makedirs(top_fig_dir, exist_ok=True)
            fig.savefig(os.path.join(top_fig_dir, name), dpi=DPI, bbox_inches="tight")

    GENERATED_FIGS.append(name)
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


def _load_single_node_df():
    """Load single-node baseline rows across configured results directories."""
    files = []
    for d in _all_results_dirs():
        s_files = sorted(glob.glob(os.path.join(d, "01_single_node_baseline", "summary_*.csv")))
        if not s_files:
            s_files = sorted(glob.glob(os.path.join(d, "benchmark_1nodes_*.csv")))
        files.extend(s_files)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _load_micro_sweep_df():
    """Load the primary fixed-microbatch 2-node sweep without sanity-check overrides."""
    files = []
    for d in _all_results_dirs():
        summary_main = os.path.join(d, "02_sweep_fixed_microbatch", "summary_micro.csv")
        if os.path.exists(summary_main):
            d_files = [summary_main]
        else:
            d_files = sorted(glob.glob(os.path.join(d, "02_sweep_fixed_microbatch", "summary_*.csv")))
            if not d_files:
                d_files = sorted(glob.glob(os.path.join(d, "02_sweep_fixed_microbatch", "*.csv")))
            if not d_files:
                d_files = sorted(glob.glob(os.path.join(d, "benchmark_2nodes_*.csv")))
        files.extend(d_files)
    if not files:
        return pd.DataFrame()

    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "mode" in df_all.columns:
        df_all = df_all[df_all["mode"] == "micro"]
    if "world_size" in df_all.columns and not df_all.empty:
        df_all = df_all[df_all["world_size"] == df_all["world_size"].max()]
    return df_all


def _dedup_task_rows(sub):
    """Deduplicate by accum_steps while preferring highest trials_count over quick sanity runs."""
    sort_cols = ["accum_steps"]
    if "trials_count" in sub.columns:
        sort_cols = ["accum_steps", "trials_count"]
    return (
        sub.sort_values(sort_cols, kind="mergesort")
        .drop_duplicates(subset=["accum_steps"], keep="last")
        .sort_values("accum_steps", kind="mergesort")
    )


def plot_figure_1_throughput_vs_k(show=False):
    """
    Figure 1: Saturation curve of training throughput vs K with error bars.
    Generates:
      - Figure_1_throughput_vs_accumulation.png (Combined multi-panel)
      - Figure_1<a/b/c>_<Task>_throughput.png (Standalone solo)
    """
    df_all = _load_micro_sweep_df()
    if df_all.empty:
        print("  [skip Fig 1] No benchmark or sweep CSVs found")
        return

    df_single = _load_single_node_df()

    tasks = df_all["task"].unique() if "task" in df_all.columns else []
    if not len(tasks):
        return

    # 1. Combined Multi-Panel Figure
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.5 * len(tasks), 5.5))
    if len(tasks) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Figure 1: Training Throughput vs. Gradient Accumulation Steps (K)",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, (ax, task) in enumerate(zip(axes, tasks)):
        sub = _dedup_task_rows(df_all[df_all["task"] == task])
        base_tput = _resolve_base_tput(task, df_single)
        unit = sub["unit"].iloc[0]
        panel_letter = chr(97 + i)
        _render_fig1_axis(ax, sub, base_tput, task, panel_letter, unit, color_idx=i)

    save_fig(fig, "Figure_1_throughput_vs_accumulation.png", show)

    # 2. Standalone Solo Figures
    for i, task in enumerate(tasks):
        sub = _dedup_task_rows(df_all[df_all["task"] == task])
        base_tput = _resolve_base_tput(task, df_single)
        unit = sub["unit"].iloc[0]
        panel_letter = chr(97 + i)
        task_slug = _task_slug(task)

        fig_solo, ax_solo = plt.subplots(figsize=(8.0, 5.5))
        _render_fig1_axis(ax_solo, sub, base_tput, task, "", unit, color_idx=i)
        ax_solo.set_title(f"{task} — Throughput vs. Accumulation Steps (K)", fontsize=13, fontweight="bold")
        save_fig(fig_solo, f"Figure_1{panel_letter}_{task_slug}_throughput.png", show)


# ── Figure 2: Single-Node vs. Distributed Scaling (Speedup & Crossover) ────────

def _render_fig2a_speedup(ax, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, title_prefix=""):
    max_speedup = 1.0
    plotted = []
    for i, task in enumerate(tasks):
        base_tput = _resolve_base_tput(task, df_single)
        sub2 = _dedup_task_rows(df_cluster[df_cluster["task"] == task])
        if sub2.empty:
            continue

        ks = list(sub2["accum_steps"])
        x_pos = [k_to_idx[k] for k in ks if k in k_to_idx]
        tput2 = sub2["throughput_mean"].values if "throughput_mean" in sub2.columns else sub2["throughput"].values
        speedup = tput2 / base_tput
        if len(speedup):
            max_speedup = max(max_speedup, float(np.max(speedup)))
        plotted.append((i, task, x_pos, speedup))

    y_max = max(2.15, max_speedup * 1.15)
    ax.axhspan(1.0, y_max, color="#2CA02C", alpha=0.06, label="_nolegend_")
    ax.axhspan(0, 1.0, color="#D95F02", alpha=0.04, label="_nolegend_")
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.8, label="1-Node Baseline (1.0×)")
    ax.text(0.03, 0.92, "▲ Distributed Speedup ($S_2$ > 1.0×)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#1B7837", alpha=0.9)
    ax.text(0.02, 0.415, "▼ Bottleneck ($S_2$ < 1.0×)", transform=ax.transAxes,
            fontsize=9.2, fontweight="bold", color="#B2182B", alpha=0.85)

    for i, task, x_pos, speedup in plotted:
        ax.plot(x_pos, speedup, marker="s", markersize=8, linewidth=2.2,
                color=COLORS[i % len(COLORS)], label=f"{task} (2 Nodes)")

        for xp, y_val in zip(x_pos, speedup):
            ax.annotate(f"{y_val:.2f}×", (xp, y_val), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}Cluster Speedup Ratio ($S_2$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Accumulation Steps (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup vs. 1-Node Baseline", fontsize=11, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(k) for k in all_k_vals])
    ax.set_xlim(-0.4, len(all_k_vals) - 0.6)
    ax.set_ylim(0, y_max)
    ax.legend(loc="lower right", framealpha=0.92)


def _render_fig2b_efficiency(ax, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, title_prefix=""):
    ax.axhspan(0, 100.0, color="#4C72B0", alpha=0.03, label="_nolegend_")
    ax.axhline(100.0, color="gray", linestyle=":", linewidth=1.5, label="Ideal Linear Scaling (100%)")

    max_eff = 100.0
    for i, task in enumerate(tasks):
        base_tput = _resolve_base_tput(task, df_single)
        sub2 = _dedup_task_rows(df_cluster[df_cluster["task"] == task])
        if sub2.empty:
            continue

        ks = list(sub2["accum_steps"])
        x_pos = [k_to_idx[k] for k in ks if k in k_to_idx]
        tput2 = sub2["throughput_mean"].values if "throughput_mean" in sub2.columns else sub2["throughput"].values
        efficiency = (tput2 / (2.0 * base_tput)) * 100.0
        if len(efficiency):
            max_eff = max(max_eff, float(np.max(efficiency)))

        ax.plot(x_pos, efficiency, marker="o", markersize=8, linewidth=2.2,
                color=COLORS[i % len(COLORS)], label=f"{task} (2 Nodes)")

        for xp, y_val in zip(x_pos, efficiency):
            ax.annotate(f"{y_val:.1f}%", (xp, y_val), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}Weak-Scaling Efficiency ($E_2$ %)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Accumulation Steps (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Scaling Efficiency (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(k) for k in all_k_vals])
    ax.set_xlim(-0.4, len(all_k_vals) - 0.6)
    ax.set_ylim(0, max(115.0, max_eff * 1.15))
    ax.legend(loc="lower right", framealpha=0.92)


def plot_figure_2_scaling_efficiency(show=False):
    """
    Figure 2: Speedup Ratio (S_N) and Scaling Efficiency (E_N) vs K.
    Generates:
      - Figure_2_single_vs_distributed_scaling.png (Combined dual-panel)
      - Figure_2a_cluster_speedup_ratio.png (Standalone solo)
      - Figure_2b_weak_scaling_efficiency.png (Standalone solo)
    """
    df_cluster = _load_micro_sweep_df()
    if df_cluster.empty:
        print("  [skip Fig 2] No data for scaling comparison")
        return

    df_single = _load_single_node_df()

    tasks = df_cluster["task"].unique() if "task" in df_cluster.columns else []
    if not len(tasks):
        return

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

    _render_fig2a_speedup(axes[0], df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, "a")
    _render_fig2b_efficiency(axes[1], df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, "b")
    save_fig(fig, "Figure_2_single_vs_distributed_scaling.png", show)

    # 2. Standalone Solo Figures
    fig_solo_a, ax_solo_a = plt.subplots(figsize=(8.0, 5.5))
    _render_fig2a_speedup(ax_solo_a, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, "")
    ax_solo_a.set_title("Cluster Speedup Ratio ($S_2$) vs. Single-Node Baseline", fontsize=13, fontweight="bold")
    save_fig(fig_solo_a, "Figure_2a_cluster_speedup_ratio.png", show)

    fig_solo_b, ax_solo_b = plt.subplots(figsize=(8.0, 5.5))
    _render_fig2b_efficiency(ax_solo_b, df_cluster, df_single, tasks, all_k_vals, k_to_idx, x_ticks, "")
    ax_solo_b.set_title("Weak-Scaling Efficiency ($E_2$ %) vs. Accumulation Steps (K)", fontsize=13, fontweight="bold")
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
    y_max = max(10.5, float(np.max(gamma_ratio)) * 1.15) if len(gamma_ratio) else 10.5
    ax.axhspan(1.0, y_max, color="#2CA02C", alpha=0.06, label="_nolegend_")
    ax.axhspan(0.0, 1.0, color="#D95F02", alpha=0.04, label="_nolegend_")
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.8, label="Balanced Threshold (Γ = 1.0)")

    ax.text(0.04, 0.72, "▲ Compute-Dominant Regime (Γ > 1.0)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#1B7837")
    ax.text(0.97, 0.03, "▼ Communication-Bound Regime (Γ < 1.0)", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#B2182B", ha="right")

    ax.plot(x_indices, gamma_ratio, marker="s", markersize=8, linewidth=2.2,
            color="#55A868", label="Compute-to-Comm Ratio (Γ)")

    for xi, g_val in zip(x_indices, gamma_ratio):
        y_off = -13 if (0.45 <= g_val <= 0.90) else 9
        ax.annotate(f"{g_val:.2f}×", (xi, g_val), textcoords="offset points",
                    xytext=(0, y_off), ha="center", fontsize=9, fontweight="bold", color="#2E6930")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}Compute-to-Communication Ratio (Γ = T_comp / T_comm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Compute-to-Communication Ratio (Γ)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gradient Accumulation Configuration", fontsize=11, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(ks)
    ax.set_xlim(-0.4, len(ks) - 0.6)
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper left", framealpha=0.92)


def plot_figure_3_timing_breakdown(show=False):
    """
    Figure 3: Dual-Panel Systems Decomposition:
      - Figure_3_compute_vs_communication_breakdown.png (Combined dual-panel)
      - Figure_3a_step_duration_breakdown.png (Standalone solo stacked bar)
      - Figure_3b_compute_to_comm_ratio.png (Standalone solo Gamma ratio)
    """
    timing_files = []
    for d in _all_results_dirs():
        main_timing = os.path.join(d, "04_step_time_breakdown", "timing_decomposition_micro.csv")
        if os.path.exists(main_timing):
            timing_files.append(main_timing)
        else:
            d_files = sorted(glob.glob(os.path.join(d, "04_step_time_breakdown", "timing_decomposition_*.csv")))
            if not d_files:
                d_files = sorted(glob.glob(os.path.join(d, "02_sweep_fixed_microbatch", "summary_*.csv")))
            timing_files.extend(d_files)

    if not timing_files:
        print("  [skip Fig 3] No timing decomposition CSV found")
        return

    df = pd.concat([pd.read_csv(f) for f in timing_files], ignore_index=True)
    if "mode" in df.columns:
        df = df[df["mode"] == "micro"]
    if "world_size" in df.columns and not df.empty:
        df = df[df["world_size"] == df["world_size"].max()]

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

    focus_df = _dedup_task_rows(df[df["task"].str.contains("GPT-2", na=False)])
    if focus_df.empty:
        focus_df = _dedup_task_rows(df)
    if focus_df.empty:
        print("  [skip Fig 3] Empty filtered timing DataFrame")
        return

    ks = [f"K={k}" for k in focus_df["accum_steps"]]
    x_indices = np.arange(len(ks))
    fwd = np.array(focus_df[fwd_col]) / 1000.0
    bwd = np.array(focus_df[bwd_col]) / 1000.0
    ar  = np.array(focus_df[ar_col])  / 1000.0

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


# ── Figure 4: Training Convergence Comparison (Multi-Panel Clean) ─────────────

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


def _render_step_loss_curve(ax, files, title, ylabel, title_prefix="", fixed_ylim=None):
    all_losses = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        if df.empty or "step" not in df.columns or "loss" not in df.columns:
            continue
        k_val = df["accum_steps"].iloc[0] if "accum_steps" in df.columns else "?"
        nodes = df["world_size"].iloc[0] if "world_size" in df.columns else "?"
        label = f"{nodes} Nodes (K={k_val})"

        ax.plot(df["step"], df["loss"], marker="o", markersize=5, linewidth=2.0,
                label=label, color=COLORS[i % len(COLORS)])

        first_step, first_loss = df["step"].iloc[0], df["loss"].iloc[0]
        last_step, last_loss   = df["step"].iloc[-1], df["loss"].iloc[-1]
        all_losses.extend(df["loss"].tolist())

        ax.annotate(f"Start: {first_loss:.2f}", (first_step, first_loss),
                    textcoords="offset points", xytext=(15, 6), fontsize=9, fontweight="bold")
        ax.annotate(f"Final: {last_loss:.2f}", (last_step, last_loss),
                    textcoords="offset points", xytext=(-20, 12), fontsize=9, fontweight="bold")

    prefix = f"({title_prefix}) " if title_prefix else ""
    ax.set_title(f"{prefix}{title}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Optimizer Step", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    elif all_losses:
        lo, hi = min(all_losses), max(all_losses)
        pad = max(0.5, (hi - lo) * 0.25)
        ax.set_ylim(max(0.0, lo - pad * 0.5), hi + pad)
    ax.legend(loc="upper right", framealpha=0.92)


def _render_fig4b_gpt2(ax, gpt2_files, title_prefix=""):
    _render_step_loss_curve(
        ax, gpt2_files,
        "GPT-2 Small / WikiText-2 Language Model Loss",
        "Language Model Loss (Cross-Entropy)",
        title_prefix=title_prefix,
        fixed_ylim=(7.0, 11.5),
    )


def _render_fig4c_whisper(ax, whisper_files, title_prefix=""):
    _render_step_loss_curve(
        ax, whisper_files,
        "Whisper-Tiny / SpeechCommands Sequence Loss",
        "Sequence Loss (Cross-Entropy)",
        title_prefix=title_prefix,
    )


def plot_figure_4_convergence(show=False):
    """
    Figure 4: Loss curves verifying numerical convergence across available modalities.
    Generates combined multi-panel Figure_4 plus solo panels (4a/4b/4c) for found CSVs.
    """
    cifar_files   = _glob_results("cifar_*.csv")
    gpt2_files    = _glob_results("gpt2_*.csv")
    whisper_files = _glob_results("whisper_*.csv")

    active = []
    if cifar_files:
        active.append((cifar_files, _render_fig4a_cifar, "Figure_4a_cifar10_convergence.png", "ResNet-18 / CIFAR-10 Convergence"))
    if gpt2_files:
        active.append((gpt2_files, _render_fig4b_gpt2, "Figure_4b_gpt2_convergence.png", "GPT-2 Small / WikiText-2 Language Model Loss"))
    if whisper_files:
        active.append((whisper_files, _render_fig4c_whisper, "Figure_4c_whisper_convergence.png", "Whisper-Tiny / SpeechCommands Sequence Loss"))

    if not active:
        print("  [skip Fig 4] No convergence CSVs found")
        return

    # 1. Combined Multi-Panel Figure
    fig, axes = plt.subplots(1, len(active), figsize=(7.0 * len(active), 5.5))
    if len(active) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle("Figure 4: Training Convergence & Loss Stability Across Accumulation Steps",
                 fontsize=14, fontweight="bold", y=1.02)

    for idx, (files, renderer, _, _) in enumerate(active):
        renderer(axes[idx], files, chr(97 + idx))
    save_fig(fig, "Figure_4_training_convergence_comparison.png", show)

    # 2. Standalone Solo Figures
    for files, renderer, filename, solo_title in active:
        fig_solo, ax_solo = plt.subplots(figsize=(8.0, 5.5))
        renderer(ax_solo, files, "")
        ax_solo.set_title(solo_title, fontsize=13, fontweight="bold")
        save_fig(fig_solo, filename, show)


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
    net_files = _glob_results("net_test_*.csv", include_extra=False)
    if not net_files:
        net_files = _glob_results("net_test_*.csv", include_extra=True)
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


# ── Figure 6: Continuous System Telemetry Timeline ────────────────────────────

def plot_figure_6_telemetry(show=False):
    """
    Figure 6: 3-panel time-series profile of CPU%, DDR4 RAM, and Network I/O
    from continuous 1 Hz sysmon logs.
    """
    sysmon_files = _glob_results("sysmon_*.csv", include_extra=False)
    if not sysmon_files:
        sysmon_files = _glob_results("sysmon_*.csv", include_extra=True)
    if not sysmon_files:
        print("  [skip Fig 6] No sysmon_*.csv found")
        return

    best_sysmon = max(sysmon_files, key=lambda p: os.path.getsize(p))
    df = pd.read_csv(best_sysmon)
    if df.empty or "elapsed_s" not in df.columns:
        print("  [skip Fig 6] Invalid sysmon CSV format")
        return

    df["elapsed_h"] = df["elapsed_s"] / 3600.0
    df["cpu_roll"] = df["cpu_pct"].rolling(window=30, min_periods=1).mean()
    df["net_sent_roll"] = df["net_sent_mbs"].rolling(window=10, min_periods=1).mean()
    df["net_recv_roll"] = df["net_recv_mbs"].rolling(window=10, min_periods=1).mean()

    duration_h = df["elapsed_h"].max()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. CPU Usage
    axes[0].plot(df["elapsed_h"], df["cpu_roll"], color="#1f77b4", lw=1.2,
                 label="CPU Utilization (30s Rolling Avg)")
    axes[0].axhline(y=100, color="crimson", linestyle="--", alpha=0.45,
                    label="100% Logical Core Ceiling")
    axes[0].set_ylabel("CPU Usage (%)", fontweight="bold", fontsize=11)
    axes[0].set_ylim(0, 108)
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)
    axes[0].set_title(
        f"Figure 6: Real-Time Hardware Telemetry Profile ({duration_h:.1f}-Hour Continuous 1 Hz Trace)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # 2. RAM Usage
    axes[1].plot(df["elapsed_h"], df["ram_used_gb"], color="#2ca02c", lw=1.4,
                 label="Host RAM Used (GB)")
    axes[1].axhline(y=14.88, color="crimson", linestyle="--", lw=1.5,
                    label="Physical RAM Ceiling (14.88 GB DDR4)")
    axes[1].fill_between(df["elapsed_h"], 0, df["ram_used_gb"], color="#2ca02c", alpha=0.18)
    axes[1].set_ylabel("Host Memory (GB)", fontweight="bold", fontsize=11)
    axes[1].set_ylim(0, max(16.5, float(df["ram_used_gb"].max()) * 1.15))
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)

    # 3. Network Throughput
    axes[2].plot(df["elapsed_h"], df["net_sent_roll"], color="#ff7f0e", lw=1.0, alpha=0.85,
                 label="Net Egress (Sent MB/s)")
    axes[2].plot(df["elapsed_h"], df["net_recv_roll"], color="#9467bd", lw=1.0, alpha=0.75,
                 label="Net Ingress (Recv MB/s)")
    axes[2].axhline(y=12.5, color="black", linestyle=":", lw=1.5,
                    label="100 Mbps Fast Ethernet Cap (12.5 MB/s)")
    axes[2].set_ylabel("Network (MB/s)", fontweight="bold", fontsize=11)
    axes[2].set_xlabel("Elapsed Experiment Time (Hours)", fontweight="bold", fontsize=12)
    axes[2].set_ylim(0, max(16.0, float(df[["net_sent_roll", "net_recv_roll"]].max().max()) * 1.15))
    axes[2].legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)

    # Annotate canonical ResNet+GPT-2 session stages only when plotting that specific 18.4h trace
    if "20261121_200541" in os.path.abspath(best_sysmon):
        stage_markers = [
            (1.43, "Stage 4: Microbatch K-Sweep"),
            (8.86, "Stage 5: Global-Batch K-Sweep"),
            (12.56, "Stage 5b: Convergence Verification"),
            (16.06, "Stage 6: Idle Baseline"),
        ]
        for x_pos, label in stage_markers:
            if x_pos <= duration_h:
                for ax in axes:
                    ax.axvline(x=x_pos, color="#666666", linestyle="-.", alpha=0.6, lw=1)
                axes[0].text(
                    x_pos + 0.12, 70, label, fontsize=8.8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffdd", edgecolor="#999999", alpha=0.92),
                )

    plt.tight_layout()
    save_fig(fig, "Figure_6_system_telemetry_timeline.png", show)


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot publication figures for CPU cluster research paper")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results directory or timestamped session directory (defaults to config.RESULTS_DIR)")
    parser.add_argument("--extra_results_dir", action="append", default=[],
                        help="Additional session directory to merge for multi-modal comparison (repeatable)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Optional output directory for generated figures (defaults to <results_dir>/figures)")
    parser.add_argument("--no_mirror", action="store_true",
                        help="Do not mirror figures into top-level results/figures/")
    parser.add_argument("--show", action="store_true", help="Display figures interactively on screen")
    args = parser.parse_args()

    configure_paths(args.results_dir, args.extra_results_dir, args.out_dir, args.no_mirror)

    print()
    print("=" * 70)
    print("  CPU CLUSTER LAB — PUBLICATION FIGURE GENERATOR")
    print(f"  Source Directory : {RESULTS_DIR}/")
    for extra in EXTRA_RESULTS_DIRS:
        print(f"  Merged Extra Dir : {extra}/")
    print(f"  Target Directory : {FIG_DIR}/")
    print(f"  Archive Snapshot : {RUN_DIR}/")
    print("=" * 70)
    print()

    plot_figure_1_throughput_vs_k(args.show)
    plot_figure_2_scaling_efficiency(args.show)
    plot_figure_3_timing_breakdown(args.show)
    plot_figure_4_convergence(args.show)
    plot_figure_5_bandwidth_sensitivity(args.show)
    plot_figure_6_telemetry(args.show)

    print()
    print("=" * 70)
    print(f"  ✓ Generated {len(GENERATED_FIGS)} publication figures successfully!")
    print(f"    Inspect results in: {FIG_DIR}/")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
