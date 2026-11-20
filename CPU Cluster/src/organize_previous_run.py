"""
CPU Cluster Lab — Script: Organize Previous Run into Timestamped Session
========================================================================
PURPOSE:
    Collects and organizes all existing experimental CSVs, logs,
    and profiles generated from today's multi-hour run into a dedicated,
    clean session folder (results/session_YYYYMMDD_HHMMSS/), and immediately
    generates all publication figures into that session's figures/ directory.
"""

import os
import sys
import glob
import shutil
import datetime
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def organize():
    # 1. Determine session timestamp from existing log or sweep CSV
    logs = sorted(glob.glob(os.path.join(RESULTS_ROOT, "pipeline_*.log")))
    sweeps = sorted(glob.glob(os.path.join(RESULTS_ROOT, "02_sweep_fixed_microbatch", "summary_*.csv")))

    ts = None
    if logs:
        # e.g., results/pipeline_20261120_113835.log
        base = os.path.basename(logs[-1])
        parts = base.replace("pipeline_", "").replace(".log", "")
        ts = parts
    elif sweeps:
        base = os.path.basename(sweeps[-1])
        parts = base.replace(".csv", "").split("_")[-2:]
        ts = "_".join(parts)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    session_name = f"session_{ts}"
    session_dir = os.path.join(RESULTS_ROOT, session_name)
    print(f"\n=======================================================")
    print(f"  CPU CLUSTER LAB — SESSION PACKAGER")
    print(f"  Target Session Folder : {session_dir}")
    print(f"=======================================================\n")

    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "figures"), exist_ok=True)

    # 2. Copy/organize pipeline log
    if logs:
        shutil.copy2(logs[-1], os.path.join(session_dir, "pipeline.log"))
        print(f"  ✓ Copied execution log → {session_dir}/pipeline.log")

    # 3. Copy hardware profile
    hw_dir = os.path.join(RESULTS_ROOT, "00_hardware_profile")
    if os.path.exists(hw_dir):
        dest_hw = os.path.join(session_dir, "00_hardware_profile")
        os.makedirs(dest_hw, exist_ok=True)
        for item in os.listdir(hw_dir):
            s = os.path.join(hw_dir, item)
            d = os.path.join(dest_hw, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        print(f"  ✓ Copied hardware profiles → {dest_hw}/")

    # 4. Copy network test CSVs
    net_files = glob.glob(os.path.join(RESULTS_ROOT, "net_test_*.csv"))
    for nf in net_files:
        shutil.copy2(nf, session_dir)
        print(f"  ✓ Copied network benchmark → {session_dir}/{os.path.basename(nf)}")

    # 5. Copy microbatch sweep
    micro_dir = os.path.join(RESULTS_ROOT, "02_sweep_fixed_microbatch")
    if os.path.exists(micro_dir):
        dest_micro = os.path.join(session_dir, "02_sweep_fixed_microbatch")
        os.makedirs(dest_micro, exist_ok=True)
        for root, dirs, files in os.walk(micro_dir):
            rel = os.path.relpath(root, micro_dir)
            target = os.path.join(dest_micro, rel)
            os.makedirs(target, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(root, f), os.path.join(target, f))
        print(f"  ✓ Copied microbatch sweep data → {dest_micro}/")

    # 6. Copy global batch sweep
    global_dir = os.path.join(RESULTS_ROOT, "03_sweep_fixed_global_batch")
    if os.path.exists(global_dir):
        dest_global = os.path.join(session_dir, "03_sweep_fixed_global_batch")
        os.makedirs(dest_global, exist_ok=True)
        for root, dirs, files in os.walk(global_dir):
            rel = os.path.relpath(root, global_dir)
            target = os.path.join(dest_global, rel)
            os.makedirs(target, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(root, f), os.path.join(target, f))
        print(f"  ✓ Copied global batch sweep data → {dest_global}/")

    # 7. Copy timing breakdown
    timing_dir = os.path.join(RESULTS_ROOT, "04_step_time_breakdown")
    if os.path.exists(timing_dir):
        dest_timing = os.path.join(session_dir, "04_step_time_breakdown")
        os.makedirs(dest_timing, exist_ok=True)
        for item in os.listdir(timing_dir):
            s = os.path.join(timing_dir, item)
            d = os.path.join(dest_timing, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        print(f"  ✓ Copied step timing breakdown → {dest_timing}/")

    # 8. Copy telemetry files
    telemetry_files = glob.glob(os.path.join(RESULTS_ROOT, "sysmon_*.csv"))
    for tf in telemetry_files:
        shutil.copy2(tf, session_dir)
        print(f"  ✓ Copied telemetry CSV → {session_dir}/{os.path.basename(tf)}")

    # 9. Update symlink to latest
    latest_link = os.path.join(RESULTS_ROOT, "latest")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(session_name, latest_link)
        print(f"  ✓ Updated symlink results/latest → {session_name}")
    except Exception:
        pass

    # 10. Run Plotter directly for this session
    print(f"\n>>> Generating publication figures for {session_name}...")
    plot_script = os.path.join(PROJECT_ROOT, "src", "05_plot_results.py")

    # Resolve Python interpreter: prefer project virtualenv (.venv) where pandas & matplotlib live
    venv_py3 = os.path.join(PROJECT_ROOT, ".venv", "bin", "python3")
    venv_py  = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_py3):
        py_exec = venv_py3
    elif os.path.exists(venv_py):
        py_exec = venv_py
    else:
        py_exec = sys.executable

    subprocess.run([py_exec, plot_script, "--results_dir", session_dir], check=False)

    print(f"\n=======================================================")
    print(f"  ✓ Session successfully packaged and figures rendered!")
    print(f"  Folder: {session_dir}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    organize()
