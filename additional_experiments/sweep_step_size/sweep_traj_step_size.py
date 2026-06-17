"""
Sweep the step_size parameter of the random walk trajectory (not preprocess) and evaluate
its effect on mismatch score. This tests robustness to trajectory resolution independent
of the simulation pipeline, since preprocess() no longer uses step_size.

For each step_size, the script:
1. Generates a random walk trajectory with the given step_size
2. Preprocesses and simulates grid cell activity
3. Computes toroidal coordinates via Gardner method
4. Lifts and aligns via affine transform
5. Computes the global and segmental mismatch scores
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf

sys.path.append(".")
from trajectory import random_walk, World
from grid_cell_simulation import preprocess, simulation
from pipelines import recover_traj_pipeline, compare_lifted_pipeline, segmental_compare_pipeline
from revision.csv_helpers import init_csv as _init_csv, append_csv as _append_csv, load_completed as _load_completed


WORLD_SIZE = (100, 100)
HOLES = [(30, 30, 70, 70)]  # 1-hole world
TRAJ_LENGTH = 25000
SEG_LEN = 10000
SAVE_DIR = "revision/sweep_step_size/traj_step_size_sweep_results"


def _run_single(step_size: int, repeat: int, csv_path: str, lock_path: str) -> None:
    """Run a single (step_size, repeat) job and write result to CSV incrementally."""
    world = World(WORLD_SIZE, HOLES, step_size=step_size)
    trace = random_walk(TRAJ_LENGTH, world, save=False, no_warnings=True)
    processed_trace = preprocess(trace, goal=600000)
    grid_activity = simulation(processed_trace)

    try:
        _, lifted = recover_traj_pipeline(grid_activity)
    except Exception as e:
        print(f"  step_size={step_size} repeat={repeat}: Pipeline failed ({e})")
        _append_csv(csv_path, lock_path, (step_size, repeat, np.nan, -1, np.nan))
        return

    lifted_transformed, ms = compare_lifted_pipeline(lifted, processed_trace)

    try:
        seg_scores = segmental_compare_pipeline(lifted_transformed, processed_trace, SEG_LEN)
    except Exception as e:
        print(f"  step_size={step_size} repeat={repeat}: Segmental pipeline failed ({e})")
        _append_csv(csv_path, lock_path, (step_size, repeat, ms, -1, np.nan))
        return

    print(
        f"  step_size={step_size} repeat={repeat}: "
        f"mismatch={ms:.6f}  n_segments={len(seg_scores)}"
    )
    for seg_idx, seg_score in enumerate(seg_scores):
        _append_csv(csv_path, lock_path, (step_size, repeat, ms, seg_idx, float(seg_score)))


def sweep_traj_step_size(
    step_sizes: list[int] | np.ndarray | None = None,
    n_repeats: int = 5,
    save_dir: str = SAVE_DIR,
    max_workers: int = 6,
):
    """
    Sweep step_size of the random walk trajectory and record the mismatch score for each value.

    Args:
        step_sizes: List of step_size values to sweep.
        n_repeats: Number of random walk repeats per step_size.
        save_dir: Directory to save results.
        max_workers: Number of parallel workers.
    """
    if step_sizes is None:
        step_sizes = [1, 2, 3, 4, 5, 8, 10]

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "traj_step_size_sweep_1holes.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["step_size", "repeat", "mismatch_score", "segment_index", "seg_score"])

    completed = _load_completed(csv_path, ["step_size", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed.")

    jobs = [
        (s, r) for s in step_sizes for r in range(n_repeats)
        if (s, r) not in completed
    ]
    print(f"Launching {len(jobs)} jobs (max_workers={max_workers})...")

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single, s, r, csv_path, lock_path): (s, r)
            for s, r in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nResults saved to {csv_path}")
    return csv_path


def plot_traj_step_size_sweep(csv_path: str, save_dir: str | None = None, step_sizes: list | None = None):
    """Plot trajectory step_size sweep results from saved CSV file."""
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    if step_sizes is not None:
        df = df[df["step_size"].isin(step_sizes)]
    step_sizes = sorted(df["step_size"].unique())
    x = np.array(step_sizes)

    def _plot(means, stds, scatter_vals, title, ylabel, fig_path):
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = ~np.isnan(means)
        ax.errorbar(x[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
        for vals in scatter_vals:
            ax.scatter(x[valid], np.array(vals)[valid], alpha=0.3, s=15, color="gray")
        ax.set_xlabel("Trajectory Step Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x[valid])
        ax.set_xticklabels([str(s) for s in x[valid]])
        fig.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.show()

    # Global mismatch: one value per (step_size, repeat) — deduplicate
    global_df = df.drop_duplicates(subset=["step_size", "repeat"])
    repeats = sorted(global_df["repeat"].unique())
    global_mat = np.full((len(step_sizes), len(repeats)), np.nan)
    for i, s in enumerate(step_sizes):
        for j, r in enumerate(repeats):
            mask = (global_df["step_size"] == s) & (global_df["repeat"] == r)
            if mask.any():
                global_mat[i, j] = global_df.loc[mask, "mismatch_score"].values[0]  # type: ignore
    _plot(
        np.nanmean(global_mat, axis=1), np.nanstd(global_mat, axis=1),
        [global_mat[:, j] for j in range(global_mat.shape[1])],
        "Global Mismatch — world_1holes", "Mismatch Score",
        os.path.join(save_dir, "traj_step_size_sweep_1holes_global.png"),
    )

    # Segmental mismatch: aggregate individual segment scores per step_size
    seg_df = df[(df["segment_index"] >= 0) & df["seg_score"].notna()]
    seg_grouped = seg_df.groupby("step_size")["seg_score"]
    seg_means = np.array([seg_grouped.mean().get(s, np.nan) for s in step_sizes])
    seg_stds = np.array([seg_grouped.std().get(s, np.nan) for s in step_sizes])
    _plot(
        seg_means, seg_stds, [],
        "Segmental Mismatch — world_1holes", "Segmental Mismatch Score (mean)",
        os.path.join(save_dir, "traj_step_size_sweep_1holes_segmental.png"),
    )


if __name__ == "__main__":
    step_sizes = [1, 2, 3, 4, 5, 8, 10]
    csv_path = sweep_traj_step_size(step_sizes=step_sizes, n_repeats=5)
    plot_traj_step_size_sweep(csv_path, step_sizes=step_sizes)
