"""
Sweep the mean downsampling interval using Poisson-distributed random subsampling.

Instead of uniform downsampling (every k-th point), this script draws
inter-sample intervals from a Poisson distribution with a given mean (lambda).
The cumulative sum of these intervals gives the random subsample indices.

For each mean interval (lambda), the script:
1. Loads pre-generated simulation from DATA_ROOT/revision_cache/
2. Randomly subsamples using Poisson-distributed intervals
3. Computes toroidal coordinates via Gardner method
4. Lifts and aligns via affine transform
5. Computes the global and segmental mismatch scores

Results are saved incrementally: every completed task is appended to the CSV
so nothing is lost if the process is interrupted.
"""

import pickle
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf

sys.path.append(".")
sys.path.append("../.")
from trajectory import World
from grid_cell_simulation import preprocess
from pipelines import recover_traj_pipeline, compare_lifted_pipeline, segmental_compare_pipeline
from ROOT_PATH import DATA_ROOT
from revision.csv_helpers import init_csv as _init_csv, append_csv as _append_csv, load_completed as _load_completed


SEG_LEN = 10000
CACHE_DIR = os.path.join(DATA_ROOT, "revision_cache")
TRAJ_DIR = "/data/hyoon/GridDecode/trajectory"
GRID_DIR = "/data/hyoon/GridDecode/grid_fields"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _load_simulation(repeat: int, cache_dir: str) -> tuple[np.ndarray, np.ndarray]:
    traj_path = os.path.join(TRAJ_DIR, f"random_walk_1holes_25000_n{repeat}.pkl")
    with open(traj_path, "rb") as f:
        sim_result = pickle.load(f)
    processed_trace = preprocess(sim_result[0], goal=600000)

    activity_path = os.path.join(GRID_DIR, f"random_walk_1holes_25000_n{repeat}_grid_activity.npz")
    grid_activity = np.load(activity_path)["grid_activity"]

    min_len = min(processed_trace.shape[0], grid_activity.shape[1])
    return processed_trace[:min_len], grid_activity[:, :min_len]

def _poisson_subsample_indices(n_total: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Generate subsample indices using Poisson-distributed inter-sample intervals."""
    intervals = rng.poisson(max(lam - 1, 0), size=n_total) + 1
    indices = np.cumsum(intervals)
    return indices[indices < n_total]


def _lift_and_score(
    mean_interval: float, repeat: int, cache_dir: str,
    csv_path: str, lock_path: str,
) -> None:
    processed_trace, grid_activity = _load_simulation(repeat, cache_dir)

    n_total = grid_activity.shape[1]
    rng = np.random.default_rng()
    indices = _poisson_subsample_indices(n_total, mean_interval, rng)

    if len(indices) < 10:
        print(f"  lam={mean_interval} repeat={repeat}: Too few samples ({len(indices)})")
        _append_csv(csv_path, lock_path, (mean_interval, repeat, np.nan, -1, np.nan))
        return

    grid_activity_ds = grid_activity[:, indices]
    processed_trace_ds = processed_trace[indices]

    try:
        _, lifted = recover_traj_pipeline(grid_activity_ds)
    except Exception as e:
        print(f"  lam={mean_interval} repeat={repeat}: Pipeline failed ({e})")
        _append_csv(csv_path, lock_path, (mean_interval, repeat, np.nan, -1, np.nan))
        return

    lifted_transformed, ms = compare_lifted_pipeline(lifted, processed_trace_ds)

    seg_len_scaled = max(1, SEG_LEN // int(mean_interval))

    if len(processed_trace_ds) < seg_len_scaled:
        print(
            f"  lam={mean_interval} repeat={repeat}: mismatch={ms:.6f} "
            f"(trace too short for segmental — reporting global)  (n_samples={len(indices)})"
        )
        _append_csv(csv_path, lock_path, (mean_interval, repeat, ms, 0, ms))
        return

    try:
        seg_scores = segmental_compare_pipeline(lifted_transformed, processed_trace_ds, seg_len_scaled)
    except Exception as e:
        print(f"  lam={mean_interval} repeat={repeat}: Segmental pipeline failed ({e})")
        _append_csv(csv_path, lock_path, (mean_interval, repeat, ms, -1, np.nan))
        return

    print(
        f"  lam={mean_interval} repeat={repeat}: mismatch={ms:.6f} "
        f"seg_len={seg_len_scaled} n_segments={len(seg_scores)}  (n_samples={len(indices)})"
    )
    for seg_idx, seg_score in enumerate(seg_scores):
        _append_csv(csv_path, lock_path, (mean_interval, repeat, ms, seg_idx, float(seg_score)))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_downsample_poisson(
    mean_intervals: list[int] | list[float] | np.ndarray | None = None,
    n_repeats: int = 5,
    save_dir: str | None = None,
    cache_dir: str = CACHE_DIR,
    max_workers: int = 6,
):
    """
    Sweep Poisson mean interval and record the mismatch score for each value.

    Args:
        mean_intervals: List of Poisson lambda values (mean inter-sample intervals).
        n_repeats: Number of repeats per lambda (uses cached simulations 0..n_repeats-1).
        save_dir: Directory to save results.
        cache_dir: Directory containing pre-generated simulation cache files.
        max_workers: Number of parallel workers.
    """
    if mean_intervals is None:
        mean_intervals = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    if save_dir is None:
        save_dir = "revision/sweep_downsampling/downsample_poisson_sweep_results"
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "downsample_poisson_sweep_1holes.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["mean_interval", "repeat", "mismatch_score", "segment_index", "seg_score"])

    completed = _load_completed(csv_path, ["mean_interval", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed, skipping them.")

    jobs = [
        (s, r) for s in mean_intervals for r in range(n_repeats)
        if (float(s), r) not in completed
    ]
    print(f"Launching {len(jobs)} jobs (max_workers={max_workers})...")

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_lift_and_score, s, r, cache_dir, csv_path, lock_path): (s, r)
            for s, r in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nResults saved to {csv_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_downsample_poisson_sweep(csv_path: str, save_dir: str | None = None):
    """Plot Poisson downsampling sweep results from the incremental CSV."""
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    mean_intervals = sorted(df["mean_interval"].unique())
    intervals_arr = np.array(mean_intervals)

    def _plot_panel(means, stds, scatter_vals, title, ylabel, fig_path):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = ~np.isnan(means)
        ax.errorbar(intervals_arr[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
        for vals in scatter_vals:
            ax.scatter(intervals_arr[valid], np.array(vals)[valid], alpha=0.3, s=15, color="gray")
        ax.set_xlabel(r"Mean Poisson Interval ($\lambda$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.show()

    # Global mismatch: one value per (mean_interval, repeat) — deduplicate
    global_df = df.drop_duplicates(subset=["mean_interval", "repeat"])
    repeats = sorted(global_df["repeat"].unique())
    global_mat = np.full((len(mean_intervals), len(repeats)), np.nan)
    for i, iv in enumerate(mean_intervals):
        for j, rp in enumerate(repeats):
            mask = (global_df["mean_interval"] == iv) & (global_df["repeat"] == rp)
            if mask.any():
                global_mat[i, j] = global_df.loc[mask, "mismatch_score"].values[0]  # type: ignore
    _plot_panel(
        np.nanmean(global_mat, axis=1), np.nanstd(global_mat, axis=1),
        [global_mat[:, j] for j in range(global_mat.shape[1])],
        "Global Mismatch — world_1holes", "Mismatch Score",
        os.path.join(save_dir, "downsample_poisson_sweep_1holes_global.png"),
    )

    # Segmental mismatch: aggregate individual segment scores per mean_interval
    seg_df = df[(df["segment_index"] >= 0) & df["seg_score"].notna()]
    seg_grouped = seg_df.groupby("mean_interval")["seg_score"]
    seg_means = np.array([seg_grouped.mean().get(iv, np.nan) for iv in mean_intervals])
    seg_stds = np.array([seg_grouped.std().get(iv, np.nan) for iv in mean_intervals])
    _plot_panel(
        seg_means, seg_stds, [],
        "Segmental Mismatch — world_1holes", "Segmental Mismatch Score (mean)",
        os.path.join(save_dir, "downsample_poisson_sweep_1holes_segmental.png"),
    )


if __name__ == "__main__":
    mean_intervals = [1,  50, 100, 200, 300, 400, 500]
    sweep_downsample_poisson(mean_intervals=mean_intervals, n_repeats=5)
    plot_downsample_poisson_sweep(
        "revision/sweep_downsampling/downsample_poisson_sweep_results/downsample_poisson_sweep_1holes.csv"
    )
