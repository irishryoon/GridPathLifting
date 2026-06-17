"""
Sweep neuron count on simulated data by slicing cached simulations.

For each n_neurons value the script:
1. Loads pre-generated simulation from DATA_ROOT/revision_cache/
2. Randomly selects n_neurons neuron rows (without replacement, seeded)
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
SAVE_DIR = "revision/sweep_neuron/sweep_neuron_results"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(
    n_neurons: int, repeat: int,
    csv_path: str, lock_path: str,
    seg_scores_csv_path: str | None = None,
    seg_scores_lock_path: str | None = None,
    write_main_csv: bool = True,
    traj_dir: str = TRAJ_DIR,
    grid_dir: str = GRID_DIR,
) -> tuple:
    traj_path = os.path.join(traj_dir, f"random_walk_1holes_25000_n{repeat}.pkl")
    with open(traj_path, "rb") as f:
        sim_result = pickle.load(f)
    processed_trace = preprocess(sim_result[0], goal=600000)

    activity_path = os.path.join(grid_dir, f"random_walk_1holes_25000_n{repeat}_grid_activity.npz")
    grid_activity = np.load(activity_path)["grid_activity"]  # (n_total_neurons, n_time)

    min_len = min(processed_trace.shape[0], grid_activity.shape[1])
    processed_trace = processed_trace[:min_len]
    grid_activity = grid_activity[:, :min_len]

    n_total_neurons = grid_activity.shape[0]
    n_neurons_clamped = min(n_neurons, n_total_neurons)

    rng = np.random.default_rng(seed=(n_neurons * 1_000_003 + repeat))
    neuron_idx = rng.choice(n_total_neurons, size=n_neurons_clamped, replace=False)
    neuron_idx.sort()

    grid_activity_slice = grid_activity[neuron_idx, :]

    try:
        _, lifted = recover_traj_pipeline(grid_activity_slice)
    except Exception as e:
        print(f"  n={n_neurons} repeat={repeat}: Pipeline failed ({e})")
        row = (n_neurons, repeat, np.nan, np.nan, np.nan)
        if write_main_csv:
            _append_csv(csv_path, lock_path, row)
        return row

    lifted_transformed, ms = compare_lifted_pipeline(lifted, processed_trace)
    seg_scores = None
    try:
        seg_scores = segmental_compare_pipeline(lifted_transformed, processed_trace, SEG_LEN)
        seg_mean = float(np.mean(seg_scores))
        seg_std = float(np.std(seg_scores))
    except Exception as e:
        print(f"  n={n_neurons} repeat={repeat}: Segmental pipeline failed ({e})")
        seg_mean = seg_std = np.nan

    print(f"  n={n_neurons} repeat={repeat}: mismatch={ms:.6f} seg_mean={seg_mean}")
    row = (n_neurons, repeat, ms, seg_mean, seg_std)
    if write_main_csv:
        _append_csv(csv_path, lock_path, row)

    if seg_scores_csv_path is not None and seg_scores is not None:
        for seg_idx, score in enumerate(seg_scores):
            seg_row = (n_neurons, repeat, seg_idx, float(score))
            _append_csv(seg_scores_csv_path, seg_scores_lock_path, seg_row)

    return row


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_neuron(
    n_neurons_list: list[int] | None = None,
    n_repeats: int = 5,
    save_dir: str = SAVE_DIR,
    max_workers: int = 6,
    save_all_segment_scores: bool = True,
):
    if n_neurons_list is None:
        n_neurons_list = [50, 100, 250, 500, 1000, 1500, 2000, 2464]

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "sweep_neuron.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["n_neurons", "repeat", "mismatch_score", "seg_mean", "seg_std"])

    seg_scores_csv_path = None
    seg_scores_lock_path = None
    if save_all_segment_scores:
        seg_scores_csv_path = os.path.join(save_dir, "sweep_neuron_all_seg_scores.csv")
        seg_scores_lock_path = seg_scores_csv_path + ".lock"
        _init_csv(seg_scores_csv_path, ["n_neurons", "repeat", "segment_index", "seg_score"])

    completed_main = _load_completed(csv_path, ["n_neurons", "repeat"])
    if completed_main:
        print(f"Resuming main CSV: {len(completed_main)} tasks already completed.")

    completed_seg = set()
    if save_all_segment_scores and seg_scores_csv_path is not None and os.path.exists(seg_scores_csv_path):
        seg_df = pd.read_csv(seg_scores_csv_path)
        if not seg_df.empty and {"n_neurons", "repeat"}.issubset(seg_df.columns):
            completed_seg = set(zip(seg_df["n_neurons"], seg_df["repeat"]))
            print(f"Resuming segment CSV: {len(completed_seg)} tasks already completed.")

    jobs = []
    for n in n_neurons_list:
        for r in range(n_repeats):
            key = (n, r)
            need_main = key not in completed_main
            need_seg = save_all_segment_scores and (key not in completed_seg)
            if need_main or need_seg:
                jobs.append((n, r, need_main))

    print(f"Launching {len(jobs)} jobs (max_workers={max_workers})...")

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, n, r, csv_path, lock_path,
                            seg_scores_csv_path, seg_scores_lock_path, write_main_csv=need_main): (n, r)
            for n, r, need_main in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nResults saved to {csv_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep_neuron(csv_path: str, save_dir: str | None = None):
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    n_neurons_list = sorted(df["n_neurons"].unique())
    repeats = sorted(df["repeat"].unique())
    x = np.array(n_neurons_list)

    def _build_matrix(col):
        mat = np.full((len(n_neurons_list), len(repeats)), np.nan)
        for i, n in enumerate(n_neurons_list):
            for j, r in enumerate(repeats):
                mask = (df["n_neurons"] == n) & (df["repeat"] == r)
                if mask.any():
                    mat[i, j] = df.loc[mask, col].values[-1]
        return mat

    def _plot(mat, title, ylabel, fig_path):
        fig, ax = plt.subplots(figsize=(8, 5))
        means = np.nanmean(mat, axis=1)
        stds = np.nanstd(mat, axis=1)
        valid = ~np.isnan(means)
        ax.errorbar(x[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
        for j in range(mat.shape[1]):
            ax.scatter(x[valid], mat[valid, j], alpha=0.3, s=15, color="gray")
        ax.set_xlabel("Number of Neurons")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.show()

    _plot(_build_matrix("mismatch_score"),
          "Global Mismatch vs Neuron Count (Simulated)",
          "Mismatch Score",
          os.path.join(save_dir, "sweep_neuron_global.png"))
    _plot(_build_matrix("seg_mean"),
          "Segmental Mismatch vs Neuron Count (Simulated)",
          "Segmental Mismatch Score (mean)",
          os.path.join(save_dir, "sweep_neuron_segmental.png"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sweep neuron count on simulated data")
    parser.add_argument("--n_neurons_list", type=int, nargs="+", default=None, help="List of n_neurons values to sweep")
    parser.add_argument("--n_repeats", type=int, default=5, help="Number of repeats per n_neurons value")
    args = parser.parse_args()
    
    n_neurons_list = args.n_neurons_list if args.n_neurons_list is not None else [50, 100, 250,  500, 1000, 1500, 2000, 2464]
    sweep_neuron(n_neurons_list=n_neurons_list, n_repeats=args.n_repeats)
    plot_sweep_neuron(f"{SAVE_DIR}/sweep_neuron.csv")
