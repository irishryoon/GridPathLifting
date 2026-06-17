"""
Sweep non-uniform downsampling on Gardner (real) data using Poisson intervals.

For each mean interval, the script:
1. Loads Gardner spikes and trajectory once
2. Randomly subsamples time points using Poisson-distributed inter-sample intervals
3. Extracts toroidal coordinates via Gardner_coord
4. Lifts and aligns via affine transform
5. Computes global and segmental mismatch scores

Results are saved incrementally to CSV (resumable).
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf
from scipy.ndimage import gaussian_filter1d

sys.path.append(".")
from constants import GARDNER_DATA_PATH
from helper_scripts.utils import get_spikes
from toroidal_coordinates.Gardner_toroidal import Gardner_coord
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance_upgrade
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from affine_transform.score_mismatch import score_mismatch, segmental_mismatch
from revision.csv_helpers import init_csv as _init_csv, append_csv as _append_csv, load_completed as _load_completed


RAT_NAME = "R"
MOD_NAME = "1"
DAY_NAME = "day2"
SESS_NAME = "OF"
SEG_LEN = 2000
SMOOTH_SIGMA = 10.0
SAVE_DIR = "revision/sweep_downsampling/downsample_poisson_sweep_gardner_results"


def _poisson_subsample_indices(n_total: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    intervals = rng.poisson(max(lam - 1, 0), size=n_total) + 1
    indices = np.cumsum(intervals)
    return indices[indices < n_total]


def _worker(
    mean_interval: float,
    repeat: int,
    sspikes: np.ndarray,
    original_coord: np.ndarray,
    smooth_sigma: float,
    seg_len: int,
    csv_path: str,
    lock_path: str,
) -> None:
    n_total = sspikes.shape[0]
    rng = np.random.default_rng(seed=(int(round(mean_interval * 1000)) * 1_000_003 + repeat))
    indices = _poisson_subsample_indices(n_total, mean_interval, rng)

    if len(indices) < 10:
        print(f"  lam={mean_interval} repeat={repeat}: Too few samples ({len(indices)})")
        _append_csv(csv_path, lock_path, (mean_interval, repeat, np.nan, -1, np.nan))
        return

    spikes_slice = sspikes[indices, :]
    coord_slice = original_coord[indices]

    try:
        cir_coords, _, _, _ = Gardner_coord(spikes_slice)
    except Exception as e:
        print(f"  lam={mean_interval} repeat={repeat}: Gardner_coord failed ({e})")
        _append_csv(csv_path, lock_path, (mean_interval, repeat, np.nan, -1, np.nan))
        return

    lifted = toroidal_lifting_distance_upgrade(cir_coords.copy())
    if smooth_sigma > 0:
        lifted[:, 0] = gaussian_filter1d(lifted[:, 0], smooth_sigma)
        lifted[:, 1] = gaussian_filter1d(lifted[:, 1], smooth_sigma)

    transform_mat = get_transform_mat(lifted, coord_slice)
    lifted_transformed = apply_transform_mat(lifted, transform_mat)
    ms = float(score_mismatch(lifted_transformed, coord_slice))

    seg_len_scaled = max(1, seg_len // int(mean_interval))

    if len(coord_slice) < seg_len_scaled:
        print(
            f"  lam={mean_interval} repeat={repeat}: mismatch={ms:.6f} "
            f"(trace too short for segmental — reporting global)  (n_samples={len(indices)})"
        )
        _append_csv(csv_path, lock_path, (mean_interval, repeat, ms, 0, ms))
        return

    try:
        seg_scores = segmental_mismatch(lifted_transformed, coord_slice, seg_len_scaled)
    except Exception as e:
        print(f"  lam={mean_interval} repeat={repeat}: Segmental mismatch failed ({e})")
        _append_csv(csv_path, lock_path, (mean_interval, repeat, ms, -1, np.nan))
        return

    print(
        f"  lam={mean_interval} repeat={repeat}: mismatch={ms:.6f} "
        f"seg_len={seg_len_scaled} n_segments={len(seg_scores)}  (n_samples={len(indices)})"
    )
    for seg_idx, seg_score in enumerate(seg_scores):
        _append_csv(csv_path, lock_path, (mean_interval, repeat, ms, seg_idx, float(seg_score)))


def sweep_downsample_poisson_gardner(
    mean_intervals: list[int] | list[float] | np.ndarray | None = None,
    n_repeats: int = 5,
    rat_name: str = RAT_NAME,
    mod_name: str = MOD_NAME,
    day_name: str = DAY_NAME,
    sess_name: str = SESS_NAME,
    smooth_sigma: float = SMOOTH_SIGMA,
    seg_len: int = SEG_LEN,
    save_dir: str = SAVE_DIR,
    max_workers: int = 6,
):
    if mean_intervals is None:
        mean_intervals = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    print(f"Loading Gardner data (rat={rat_name}, mod={mod_name}, day={day_name}, sess={sess_name})...")
    sspikes, xx, yy, _, _ = get_spikes(
        rat_name, mod_name, day_name, sess_name,
        bType="pure", bSmooth=True, bSpeed=True,
        folder=GARDNER_DATA_PATH,
    )
    original_coord = np.column_stack((xx, yy))
    print(f"Spike matrix: {sspikes.shape}  trajectory: {original_coord.shape}")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "Gardner_downsample_poisson_sweep.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["mean_interval", "repeat", "mismatch_score", "segment_index", "seg_score"])

    completed = _load_completed(csv_path, ["mean_interval", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed.")

    jobs = [
        (mean_interval, repeat) for mean_interval in mean_intervals for repeat in range(n_repeats)
        if (float(mean_interval), repeat) not in completed
    ]
    print(f"Launching {len(jobs)} jobs (max_workers={max_workers})...")

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _worker, mean_interval, repeat, sspikes, original_coord,
                smooth_sigma, seg_len, csv_path, lock_path
            ): (mean_interval, repeat)
            for mean_interval, repeat in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nResults saved to {csv_path}")
    return csv_path


def plot_downsample_poisson_gardner(csv_path: str, save_dir: str | None = None):
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    mean_intervals = sorted(df["mean_interval"].unique())
    x = np.array(mean_intervals)

    def _plot(means, stds, scatter_vals, title: str, ylabel: str, fig_path: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = ~np.isnan(means)
        ax.errorbar(x[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
        for vals in scatter_vals:
            ax.scatter(x[valid], np.array(vals)[valid], alpha=0.3, s=15, color="gray")
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
    for i, mean_interval in enumerate(mean_intervals):
        for j, repeat in enumerate(repeats):
            mask = (global_df["mean_interval"] == mean_interval) & (global_df["repeat"] == repeat)
            if mask.any():
                global_mat[i, j] = global_df.loc[mask, "mismatch_score"].values[0]  # type: ignore
    _plot(
        np.nanmean(global_mat, axis=1), np.nanstd(global_mat, axis=1),
        [global_mat[:, j] for j in range(global_mat.shape[1])],
        "Global Mismatch vs Non-uniform Downsampling — Gardner",
        "Mismatch Score",
        os.path.join(save_dir, "Gardner_downsample_poisson_sweep_global.png"),
    )

    # Segmental mismatch: aggregate individual segment scores per mean_interval
    seg_df = df[(df["segment_index"] >= 0) & df["seg_score"].notna()]
    seg_grouped = seg_df.groupby("mean_interval")["seg_score"]
    seg_means = np.array([seg_grouped.mean().get(iv, np.nan) for iv in mean_intervals])
    seg_stds = np.array([seg_grouped.std().get(iv, np.nan) for iv in mean_intervals])
    _plot(
        seg_means, seg_stds, [],
        "Segmental Mismatch vs Non-uniform Downsampling — Gardner",
        "Segmental Mismatch Score (mean)",
        os.path.join(save_dir, "Gardner_downsample_poisson_sweep_segmental.png"),
    )


if __name__ == "__main__":
    csv_path = sweep_downsample_poisson_gardner(
        mean_intervals=[1, 2, 5, 10, 20, 50, 100, 200, 500],
        n_repeats=5,
    )
    plot_downsample_poisson_gardner(csv_path)
