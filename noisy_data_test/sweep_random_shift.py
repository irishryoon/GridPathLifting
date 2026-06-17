"""
Sweep the random shifting distance and evaluate its effect on mismatch score.

Pipeline design:
  Simulation and decoding tasks share a single process pool. For each repeat,
  a simulation task generates the trajectory, simulates grid cell activity,
  creates shifted copies, and saves them to disk. As soon as a simulation
  finishes, its decoding tasks (one per shift distance) are immediately
  submitted to the same pool so decoding starts while other simulations are
  still running.

Results are saved incrementally: every time a decoding task finishes, its
result is appended to the CSV so nothing is lost if the process is interrupted.

Results (CSV) and figure are saved to revision/random_shift_sweep_results/.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf

sys.path.append(".")
sys.path.append("../.")
from trajectory import random_walk, World
from grid_cell_simulation import preprocess, simulation
from pipelines import recover_traj_pipeline, compare_lifted_pipeline, segmental_compare_pipeline
from revision.csv_helpers import init_csv as _init_csv, append_csv as _append_csv, load_completed as _load_completed
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from affine_transform.score_mismatch import score_mismatch


SEG_LEN = 10000

WORLD_SIZE = (100, 100)
HOLES = [(30, 30, 70, 70)]  # 1-hole world
STEP_SIZE = 2  # fixed step size
TRAJ_LENGTH = 25000

TRAJ_DIR = "/data/hyoon/GridDecode/trajectory"
GRID_DIR = "/data/hyoon/GridDecode/grid_fields"


# ---------------------------------------------------------------------------
# Workers – run in child processes
# ---------------------------------------------------------------------------
def _simulate_one_repeat(r: int, tmp_dir: str) -> int:
    """Generate trajectory + activity for one repeat, save to disk."""
    world = World(WORLD_SIZE, HOLES, step_size=STEP_SIZE)
    trace = random_walk(TRAJ_LENGTH, world, save=False, no_warnings=True)
    processed_trace = preprocess(trace, goal=600000)
    grid_activity = simulation(processed_trace)

    np.save(os.path.join(tmp_dir, f"trace_r{r}.npy"), processed_trace)
    np.save(os.path.join(tmp_dir, f"activity_r{r}.npy"), grid_activity)

    return r


def _decode_one(
    activity_path: str,
    trace_path: str,
    shift_dist: int,
    repeat: int,
    csv_path: str,
    lock_path: str,
) -> None:
    """Apply random shift, then run DREiMac -> lifting -> affine -> mismatch."""
    with open(trace_path, "rb") as f:
        sim_result = pickle.load(f)
    processed_trace = preprocess(sim_result[0], goal=600000)

    activity = np.load(activity_path)["grid_activity"]
    min_len = min(processed_trace.shape[0], activity.shape[1])
    processed_trace = processed_trace[:min_len]
    activity = activity[:, :min_len]

    if shift_dist > 0:
        rng = np.random.default_rng()
        n_cells = activity.shape[0]
        shifts = rng.integers(-shift_dist, shift_dist + 1, size=n_cells)
        for i in range(n_cells):
            activity[i] = np.roll(activity[i], shifts[i])

    try:
        _, lifted = recover_traj_pipeline(activity)
    except Exception as e:
        print(f"  shift={shift_dist} repeat={repeat}: Pipeline failed ({e})")
        _append_csv(csv_path, lock_path, (shift_dist, repeat, np.nan, -1, np.nan))
        return

    lifted_transformed, ms = compare_lifted_pipeline(lifted, processed_trace)

    try:
        seg_scores = segmental_compare_pipeline(lifted_transformed, processed_trace, SEG_LEN)
    except Exception as e:
        print(f"  shift={shift_dist} repeat={repeat}: Segmental pipeline failed ({e})")
        _append_csv(csv_path, lock_path, (shift_dist, repeat, ms, -1, np.nan))
        return

    print(
        f"  shift={shift_dist} repeat={repeat}: mismatch={ms:.6f}, n_segments={len(seg_scores)}"
    )
    for seg_idx, seg_score in enumerate(seg_scores):
        _append_csv(csv_path, lock_path, (shift_dist, repeat, ms, seg_idx, float(seg_score)))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_random_shift(
    shift_distances: list[int] | np.ndarray | None = None,
    n_repeats: int = 5,
    save_dir: str | None = None,
    n_workers: int = 8,
    cache_dir: str | None = None,
):
    if shift_distances is None:
        shift_distances = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    shift_distances = list(shift_distances)

    if save_dir is None:
        save_dir = "revision/sweep_trajectory/random_shift_sweep_results"
    os.makedirs(save_dir, exist_ok=True)

    if cache_dir is None:
        from ROOT_PATH import DATA_ROOT
        cache_dir = os.path.join(DATA_ROOT, "revision_cache")

    csv_path = os.path.join(save_dir, "random_shift_sweep_1holes.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["shift_distance", "repeat", "mismatch_score", "segment_index", "seg_score"])

    completed = _load_completed(csv_path, ["shift_distance", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed, skipping them.")

    total_decode = sum(
        1 for r in range(n_repeats) for d in shift_distances
        if (d, r) not in completed
    )
    print(f"Decoding tasks: {total_decode}, workers: {n_workers}")

    done_count = 0
    with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures: dict[cf.Future, tuple[int, int]] = {}
        for r in range(n_repeats):
            act_path = os.path.join(GRID_DIR, f"random_walk_1holes_25000_n{r}_grid_activity.npz")
            trace_path = os.path.join(TRAJ_DIR, f"random_walk_1holes_25000_n{r}.pkl")
            for d in shift_distances:
                if (d, r) in completed:
                    continue
                fut = executor.submit(_decode_one, act_path, trace_path, d, r, csv_path, lock_path)
                futures[fut] = (d, r)

        for fut in cf.as_completed(futures):
            fut.result()
            done_count += 1
            if done_count % 10 == 0 or done_count == total_decode:
                print(f"  [{done_count}/{total_decode}] decode tasks completed")

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["shift_distance", "repeat", "segment_index"], keep="last")
    df = df.sort_values(["shift_distance", "repeat", "segment_index"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path} ({len(df)} rows, sorted and deduplicated)")


# ---------------------------------------------------------------------------
# Plotting (reads the incremental CSV)
# ---------------------------------------------------------------------------
def _agg_col(df: pd.DataFrame, col: str):
    shift_distances = sorted(df["shift_distance"].unique())
    repeats = sorted(df["repeat"].unique())
    pivot = {d: {} for d in shift_distances}
    for _, row in df.iterrows():
        pivot[int(row["shift_distance"])][int(row["repeat"])] = row[col]
    means, stds, all_vals = [], [], []
    for d in shift_distances:
        vals = [pivot[d].get(r, np.nan) for r in repeats]
        all_vals.append(vals)
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))
    return np.array(shift_distances), np.array(means), np.array(stds), all_vals, repeats


def plot_random_shift_sweep(csv_path: str, save_dir: str | None = None):
    """Plot global mismatch score vs shift distance."""
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    shift_arr, means, stds, all_vals, repeats = _agg_col(df, "mismatch_score")
    valid = ~np.isnan(means)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(shift_arr[valid], means[valid], yerr=stds[valid],
                fmt="o-", markersize=5, capsize=4, label="Mean ± Std")
    for r_idx in range(len(repeats)):
        vals = [all_vals[i][r_idx] for i in range(len(shift_arr))]
        ax.scatter(shift_arr[valid], np.array(vals)[valid], alpha=0.3, s=15, color="gray")
    ax.set_xlabel("Max Shift Distance (time steps)")
    ax.set_ylabel("Mismatch Score")
    ax.set_title("Global Mismatch Score — world_1holes")
    ax.set_xscale("symlog", linthresh=100)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "random_shift_sweep_1holes.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()


def plot_random_shift_sweep_segmental(csv_path: str, save_dir: str | None = None):
    """Plot segmental mean mismatch score vs shift distance."""
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    shift_distances = sorted(df["shift_distance"].unique())
    shift_arr = np.array(shift_distances)

    seg_df = df[(df["segment_index"] >= 0) & df["seg_score"].notna()]
    seg_grouped = seg_df.groupby("shift_distance")["seg_score"]
    means = np.array([seg_grouped.mean().get(d, np.nan) for d in shift_distances])
    stds = np.array([seg_grouped.std().get(d, np.nan) for d in shift_distances])
    valid = ~np.isnan(means)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(shift_arr[valid], means[valid], yerr=stds[valid],
                fmt="o-", markersize=5, capsize=4, label="Mean ± Std")
    ax.set_xlabel("Max Shift Distance (time steps)")
    ax.set_ylabel("Segmental Mismatch Score (mean)")
    ax.set_title("Segmental Mean Mismatch — world_1holes")
    ax.set_xscale("symlog", linthresh=100)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "random_shift_sweep_1holes_segmental.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()


def plot_shift_examples(
    shift_distances: list[int] | None = None,
    save_dir: str | None = None,
    cache_dir: str | None = None,
):
    """Plot neuron 0 activity for each shift distance, one row per shift, using cached files."""
    if shift_distances is None:
        shift_distances = [0, 10, 50, 100, 200, 500, 1000]

    if save_dir is None:
        save_dir = "revision/sweep_trajectory/random_shift_sweep_results"
    os.makedirs(save_dir, exist_ok=True)

    if cache_dir is None:
        from ROOT_PATH import DATA_ROOT
        cache_dir = os.path.join(DATA_ROOT, "revision_cache")

    fig, axes = plt.subplots(len(shift_distances), 1,
                             figsize=(12, 2 * len(shift_distances)), sharex=True)
    if len(shift_distances) == 1:
        axes = [axes]

    activity_path = os.path.join(GRID_DIR, "random_walk_1holes_25000_n0_grid_activity.npz")
    base_activity = np.load(activity_path)["grid_activity"]
    rng = np.random.default_rng()

    for ax, m in zip(axes, shift_distances):
        activity = base_activity.copy()
        shifts = rng.integers(-m, m + 1, size=activity.shape[0])
        for i in range(activity.shape[0]):
            activity[i] = np.roll(activity[i], int(shifts[i]))
        ax.plot(activity[0, :10000], lw=0.5)
        ax.set_ylabel(f"m={m}", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("time (steps)", fontsize=9)
    fig.suptitle("Neuron 0 activity under uniform shift (first 10000 time points)", fontsize=10)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "random_shift_examples.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()


def plot_shift_examples_five_neurons_random(
    shift_distances: list[int] | None = None,
    neuron_ids: list[int] | None = None,
    save_dir: str | None = None,
    cache_dir: str | None = None,
    repeat_idx: int = 0,
    max_time_points: int = 10000,
    seed: int | None = None,
):
    """Plot randomly shifted activity for five neurons, one PNG per max shift distance.

    For each max shift distance m, each requested neuron i is shifted by an
    independently sampled integer between -m and m, then plotted in its own
    subplot.

    If neuron_ids is not provided, this function uses a fixed set of neuron
    indices: [282, 85, 1012, 767, 1482].
    """
    if shift_distances is None:
        shift_distances = [10, 50, 100, 200, 500, 1000]
    if save_dir is None:
        save_dir = "revision/sweep_trajectory/random_shift_sweep_results"
    os.makedirs(save_dir, exist_ok=True)

    if cache_dir is None:
        from ROOT_PATH import DATA_ROOT
        cache_dir = os.path.join(DATA_ROOT, "revision_cache")

    activity_path = os.path.join(GRID_DIR, f"random_walk_1holes_25000_n{repeat_idx % 10}_grid_activity.npz")
    base_activity = np.load(activity_path)["grid_activity"]
    n_cells, n_time = base_activity.shape

    rng = np.random.default_rng(seed)

    if neuron_ids is None:
        neuron_ids = [282, 85, 1012, 767, 1482]

    for neuron_id in neuron_ids:
        if neuron_id < 0 or neuron_id >= n_cells:
            raise ValueError(f"neuron id {neuron_id} is out of range [0, {n_cells - 1}].")

    if len(neuron_ids) != 5:
        raise ValueError("neuron_ids must contain exactly 5 neuron indices.")

    print(f"Selected neurons: {neuron_ids}")

    n_plot = min(max_time_points, n_time)

    # Save original activity (no shift) for the selected neurons.
    fig, axes = plt.subplots(len(neuron_ids), 1,
                             figsize=(12, 2 * len(neuron_ids)), sharex=True)
    if len(neuron_ids) == 1:
        axes = [axes]
    for ax, neuron_id in zip(axes, neuron_ids):
        ax.plot(base_activity[neuron_id, :n_plot], lw=0.6)
        ax.set_ylabel(f"n={neuron_id}\nshift=0", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (steps)", fontsize=9)
    fig.suptitle(f"Original activity for neurons {neuron_ids} (no shift)", fontsize=10)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "random_shift_examples_5neurons_original.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.close(fig)

    for m in shift_distances:
        sampled_shifts = rng.integers(-m, m + 1, size=len(neuron_ids))

        fig, axes = plt.subplots(len(neuron_ids), 1,
                                 figsize=(12, 2 * len(neuron_ids)), sharex=True)
        if len(neuron_ids) == 1:
            axes = [axes]

        for ax, neuron_id, shift in zip(axes, neuron_ids, sampled_shifts):
            shifted = np.roll(base_activity[neuron_id], int(shift))
            ax.plot(shifted[:n_plot], lw=0.6)
            ax.set_ylabel(f"n={neuron_id}\nshift={int(shift)}", fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1].set_xlabel("time (steps)", fontsize=9)
        fig.suptitle(
            f"Randomly shifted activity for neurons {neuron_ids} (max shift={m})",
            fontsize=10,
        )
        fig.tight_layout()

        fig_path = os.path.join(save_dir, f"random_shift_examples_5neurons_maxshift_{m}.png")
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.close(fig)


def plot_shift_examples_five_neurons_pm(
    shift_distances: list[int] | None = None,
    neuron_ids: list[int] | None = None,
    save_dir: str | None = None,
    cache_dir: str | None = None,
    repeat_idx: int = 0,
    max_time_points: int = 10000,
    seed: int | None = None,
):
    """Plot shifted activity for neurons 1012 and 767 with fixed +/-m shifts.

    For each max shift distance m, neuron shifts are set by position:
    neuron 1012 uses +m and neuron 767 uses -m, then each neuron is plotted in
    its own subplot.

    This function always uses the fixed neuron indices: [1012, 767].
    """
    if shift_distances is None:
        shift_distances = [10, 50, 100, 200, 500, 1000]
    if save_dir is None:
        save_dir = "revision/sweep_trajectory/random_shift_sweep_results"
    os.makedirs(save_dir, exist_ok=True)

    if cache_dir is None:
        from ROOT_PATH import DATA_ROOT
        cache_dir = os.path.join(DATA_ROOT, "revision_cache")

    activity_path = os.path.join(GRID_DIR, f"random_walk_1holes_25000_n{repeat_idx % 10}_grid_activity.npz")
    base_activity = np.load(activity_path)["grid_activity"]
    n_cells, n_time = base_activity.shape

    neuron_ids = [1012, 767]

    for neuron_id in neuron_ids:
        if neuron_id < 0 or neuron_id >= n_cells:
            raise ValueError(f"neuron id {neuron_id} is out of range [0, {n_cells - 1}].")

    if len(neuron_ids) != 2:
        raise ValueError("neuron_ids must contain exactly 2 neuron indices.")

    print(f"Selected neurons: {neuron_ids}")

    n_plot = min(max_time_points, n_time)
    font_family = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    label_fontsize = 16
    tick_fontsize = 16
    xlabel_fontsize = 16
    title_fontsize = 20

    # Save original activity (no shift) for the selected neurons.
    fig, axes = plt.subplots(len(neuron_ids), 1,
                             figsize=(12, 2 * len(neuron_ids)), sharex=True)
    if len(neuron_ids) == 1:
        axes = [axes]
    for ax, neuron_id in zip(axes, neuron_ids):
        ax.plot(base_activity[neuron_id, :n_plot], lw=0.6)
        ax.set_ylabel(f"n={neuron_id}\nshift=0", fontsize=label_fontsize, fontname=font_family)
        ax.tick_params(labelsize=tick_fontsize)
    axes[-1].set_xlabel("time (steps)", fontsize=xlabel_fontsize, fontname=font_family)
    fig.suptitle(
        f"Original activity for neurons {neuron_ids} (no shift)",
        fontsize=title_fontsize,
        fontname=font_family,
    )
    fig.tight_layout()
    fig_path_pdf = os.path.join(save_dir, "shift_examples_5neurons_pm_original.pdf")
    plt.savefig(fig_path_pdf)
    print(f"Figure saved to {fig_path_pdf}")
    plt.close(fig)

    for m in shift_distances:
        sampled_shifts = np.array([m, -m], dtype=int)

        fig, axes = plt.subplots(len(neuron_ids), 1,
                                 figsize=(12, 2 * len(neuron_ids)), sharex=True)
        if len(neuron_ids) == 1:
            axes = [axes]

        for ax, neuron_id, shift in zip(axes, neuron_ids, sampled_shifts):
            shifted = np.roll(base_activity[neuron_id], int(shift))
            ax.plot(shifted[:n_plot], lw=0.6)
            ax.set_ylabel(
                f"n={neuron_id}\nshift={int(shift)}",
                fontsize=label_fontsize,
                fontname=font_family,
            )
            ax.tick_params(labelsize=tick_fontsize)

        axes[-1].set_xlabel("time (steps)", fontsize=xlabel_fontsize, fontname=font_family)
        fig.suptitle(
            f"Shifted activity for neurons {neuron_ids} (shift in {{-{m}, {m}}})",
            fontsize=title_fontsize,
            fontname=font_family,
        )
        fig.tight_layout()

        fig_path_pdf = os.path.join(save_dir, f"shift_examples_5neurons_pm_maxshift_{m}.pdf")
        plt.savefig(fig_path_pdf)
        print(f"Figure saved to {fig_path_pdf}")
        plt.close(fig)


if __name__ == "__main__":
    shift_distances = [0, 10, 20, 50, 100, 200, 500, 1000]
    sweep_random_shift(shift_distances=shift_distances, n_repeats=5)
    csv_path = "revision/sweep_trajectory/random_shift_sweep_results/random_shift_sweep_1holes.csv"
    plot_random_shift_sweep(csv_path)
    plot_random_shift_sweep_segmental(csv_path)
    # plot_shift_examples()
    # plot_shift_examples_five_neurons_random(shift_distances=[10, 50, 100, 200, 500, 1000])
