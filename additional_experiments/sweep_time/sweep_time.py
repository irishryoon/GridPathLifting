"""
Sweep time length on simulated data by slicing cached simulations.

For each t_length value the script:
1. Loads pre-generated simulation from DATA_ROOT/revision_cache/
2. Takes the first t_length time steps (all neurons retained)
3. Runs the full Gardner pipeline on the slice
4. Computes global and segmental mismatch scores

Results are saved incrementally to CSV (resumable).
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf

sys.path.append(".")
sys.path.append("../.")
from pipelines import recover_traj_pipeline, compare_lifted_pipeline, segmental_compare_pipeline
from ROOT_PATH import DATA_ROOT
from grid_cell_simulation import preprocess
from trajectory import World  # noqa: F401 — required for unpickling trajectory files
from revision.csv_helpers import init_csv as _init_csv, append_csv as _append_csv, load_completed as _load_completed


SEG_LEN = 10000
TRAJ_DIR = "/data/hyoon/GridDecode/trajectory"
GRID_DIR = "/data/hyoon/GridDecode/grid_fields"
SAVE_DIR = "revision/sweep_time/sweep_time_results"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(
    t_length: int, repeat: int,
    csv_path: str, lock_path: str,
    seg_csv_path: str, seg_lock_path: str,
    traj_dir: str = TRAJ_DIR,
    grid_dir: str = GRID_DIR,
) -> tuple:
    traj_path = os.path.join(traj_dir, f"random_walk_1holes_25000_n{repeat}.pkl")
    with open(traj_path, "rb") as f:
        sim_result = pickle.load(f)
    processed_trace = preprocess(sim_result[0], goal=600000)

    activity_path = os.path.join(grid_dir, f"random_walk_1holes_25000_n{repeat}_grid_activity.npz")
    grid_activity = np.load(activity_path)["grid_activity"]  # (n_neurons, n_total_time)

    min_len = min(processed_trace.shape[0], grid_activity.shape[1])
    processed_trace = processed_trace[:min_len]
    grid_activity = grid_activity[:, :min_len]

    t_length_clamped = min(t_length, grid_activity.shape[1])
    grid_activity_slice = grid_activity[:, :t_length_clamped]
    processed_trace_slice = processed_trace[:t_length_clamped]

    try:
        _, lifted = recover_traj_pipeline(grid_activity_slice)
    except Exception as e:
        print(f"  t={t_length} repeat={repeat}: Pipeline failed ({e})")
        row = (t_length, repeat, np.nan, np.nan, np.nan)
        _append_csv(csv_path, lock_path, row)
        return row

    lifted_transformed, ms = compare_lifted_pipeline(lifted, processed_trace_slice)
    try:
        seg_scores = segmental_compare_pipeline(lifted_transformed, processed_trace_slice, SEG_LEN)
        seg_mean = float(np.mean(seg_scores))
        seg_std = float(np.std(seg_scores))
    except Exception as e:
        print(f"  t={t_length} repeat={repeat}: Segmental pipeline failed ({e})")
        seg_mean = seg_std = np.nan
        seg_scores = []

    print(f"  t={t_length} repeat={repeat}: mismatch={ms:.6f} seg_mean={seg_mean}")
    row = (t_length, repeat, ms, seg_mean, seg_std)
    _append_csv(csv_path, lock_path, row)

    for seg_idx, seg_score in enumerate(seg_scores):
        _append_csv(seg_csv_path, seg_lock_path, (t_length, repeat, seg_idx, float(seg_score)))

    return row


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_time(
    t_length_list: list[int] | None = None,
    n_repeats: int = 5,
    save_dir: str = SAVE_DIR,
    max_workers: int = 6,
):
    if t_length_list is None:
        t_length_list = [5000, 10000, 20000, 50000, 100000, 200000, 500000]

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "sweep_time.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["t_length", "repeat", "mismatch_score", "seg_mean", "seg_std"])

    seg_csv_path = os.path.join(save_dir, "sweep_time_segment_scores.csv")
    seg_lock_path = seg_csv_path + ".lock"
    _init_csv(seg_csv_path, ["t_length", "repeat", "segment_index", "segment_score"])

    completed = _load_completed(csv_path, ["t_length", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed.")

    jobs = [
        (t, r) for t in t_length_list for r in range(n_repeats)
        if (t, r) not in completed
    ]
    print(f"Launching {len(jobs)} jobs (max_workers={max_workers})...")

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, t, r, csv_path, lock_path, seg_csv_path, seg_lock_path): (t, r)
            for t, r in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nResults saved to {csv_path}")
    print(f"Segment-level scores saved to {seg_csv_path}")



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
MAX_T_LENGTH = 30000

def plot_sweep_time(csv_path: str, save_dir: str | None = None):
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    t_length_list = sorted(df["t_length"].unique())
    t_length_list_capped = [t for t in t_length_list if t <= MAX_T_LENGTH]
    repeats = sorted(df["repeat"].unique())
    x = np.array(t_length_list_capped)

    def _build_matrix(col, t_values=None):
        if t_values is None:
            t_values = t_length_list_capped
        mat = np.full((len(t_values), len(repeats)), np.nan)
        for i, t in enumerate(t_values):
            for j, r in enumerate(repeats):
                mask = (df["t_length"] == t) & (df["repeat"] == r)
                if mask.any():
                    mat[i, j] = df.loc[mask, col].values[-1]  # type: ignore
        return mat

    def _plot(mat, title, ylabel, fig_path):
        fig, ax = plt.subplots(figsize=(8, 5))
        means = np.nanmean(mat, axis=1)
        stds = np.nanstd(mat, axis=1)
        valid = ~np.isnan(means)
        ax.errorbar(x[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
        for j in range(mat.shape[1]):
            ax.scatter(x[valid], mat[valid, j], alpha=0.3, s=15, color="gray")
        ax.set_xlabel("Time Length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.show()

    _plot(_build_matrix("mismatch_score"),
          "Global Mismatch vs Time Length (Simulated)",
          "Mismatch Score",
          os.path.join(save_dir, "sweep_time_global.png"))
    _plot(_build_matrix("seg_mean"),
          "Segmental Mismatch vs Time Length (Simulated)",
          "Segmental Mismatch Score (mean)",
          os.path.join(save_dir, "sweep_time_segmental.png"))


if __name__ == "__main__":
    sweep_time(t_length_list=[1000, 2000, 3000], n_repeats=5)
    #sweep_time(t_length_list=[5000, 10000, 20000, 40000, 60000, 80000, 100000], n_repeats=5)
    plot_sweep_time(f"{SAVE_DIR}/sweep_time.csv")
