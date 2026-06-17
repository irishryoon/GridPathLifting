"""
Sweep neuron count on Gardner (real) data.

For each n_neurons value the script:
1. Loads Gardner spikes once (shape: n_timepoints × n_neurons)
2. Randomly selects n_neurons neuron columns (seeded per repeat)
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
SAVE_DIR = "revision/sweep_neuron/sweep_neuron_gardner_results"


# ---------------------------------------------------------------------------
# Worker (called in subprocess — receives spike slice, not full matrix)
# ---------------------------------------------------------------------------
def _worker(
    n_neurons: int, repeat: int,
    sspikes: np.ndarray,      # (n_timepoints, n_total_neurons)
    original_coord: np.ndarray,
    smooth_sigma: float,
    seg_len: int,
    csv_path: str, lock_path: str,
    segment_csv_path: str, segment_lock_path: str,
) -> tuple:
    n_total_neurons = sspikes.shape[1]
    n_neurons_clamped = min(n_neurons, n_total_neurons)

    rng = np.random.default_rng(seed=(n_neurons * 1_000_003 + repeat))
    neuron_idx = rng.choice(n_total_neurons, size=n_neurons_clamped, replace=False)
    neuron_idx.sort()

    spikes_slice = sspikes[:, neuron_idx]  # (n_timepoints, n_neurons_clamped)

    try:
        cir_coords, _, _, _ = Gardner_coord(spikes_slice)
    except Exception as e:
        print(f"  n={n_neurons} repeat={repeat}: Gardner_coord failed ({e})")
        row = (n_neurons, repeat, np.nan, np.nan, np.nan)
        _append_csv(csv_path, lock_path, row)
        return row

    lifted = toroidal_lifting_distance_upgrade(cir_coords.copy())
    if smooth_sigma > 0:
        lifted[:, 0] = gaussian_filter1d(lifted[:, 0], smooth_sigma)
        lifted[:, 1] = gaussian_filter1d(lifted[:, 1], smooth_sigma)

    transform_mat = get_transform_mat(lifted, original_coord)
    lifted_transformed = apply_transform_mat(lifted, transform_mat)
    ms = float(score_mismatch(lifted_transformed, original_coord))

    try:
        seg_scores = segmental_mismatch(lifted_transformed, original_coord, seg_len)
        seg_mean = float(np.mean(seg_scores))
        seg_std = float(np.std(seg_scores))

        # Save every segmental mismatch score to a dedicated CSV.
        for seg_idx, seg_score in enumerate(seg_scores):
            seg_row = (n_neurons, repeat, seg_idx, float(seg_score))
            _append_csv(segment_csv_path, segment_lock_path, seg_row)
    except Exception as e:
        print(f"  n={n_neurons} repeat={repeat}: Segmental mismatch failed ({e})")
        seg_mean = seg_std = np.nan

    print(f"  n={n_neurons} repeat={repeat}: mismatch={ms:.6f} seg_mean={seg_mean}")
    row = (n_neurons, repeat, ms, seg_mean, seg_std)
    _append_csv(csv_path, lock_path, row)
    return row


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_neuron_gardner(
    n_neurons_list: list[int] | None = None,
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
    if n_neurons_list is None:
        n_neurons_list = [10, 20, 40, 60, 80, 100, 111]

    print(f"Loading Gardner data (rat={rat_name}, mod={mod_name}, day={day_name}, sess={sess_name})...")
    sspikes, xx, yy, _, _ = get_spikes(
        rat_name, mod_name, day_name, sess_name,
        bType="pure", bSmooth=True, bSpeed=True,
        folder=GARDNER_DATA_PATH,
    )
    original_coord = np.column_stack((xx, yy))
    print(f"Spike matrix: {sspikes.shape}  trajectory: {original_coord.shape}")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "Gardner_sweep_neuron.csv")
    lock_path = csv_path + ".lock"
    segment_csv_path = os.path.join(save_dir, "Gardner_sweep_neuron_segment_scores.csv")
    segment_lock_path = segment_csv_path + ".lock"
    _init_csv(csv_path, ["n_neurons", "repeat", "mismatch_score", "seg_mean", "seg_std"])
    _init_csv(segment_csv_path, ["n_neurons", "repeat", "segment_idx", "segment_mismatch_score"])

    completed = _load_completed(csv_path, ["n_neurons", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed.")

    jobs = [
        (n, r) for n in n_neurons_list for r in range(n_repeats)
        if (n, r) not in completed
    ]
    print(f"Launching {len(jobs)} jobs (max_workers={max_workers})...")

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _worker, n, r, sspikes, original_coord,
                smooth_sigma, seg_len, csv_path, lock_path,
                segment_csv_path, segment_lock_path
            ): (n, r)
            for n, r in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nResults saved to {csv_path}")
    print(f"Segment scores saved to {segment_csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep_neuron_gardner(csv_path: str, save_dir: str | None = None):
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
                    mat[i, j] = df.loc[mask, col].values[-1] # type: ignore
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
          "Global Mismatch vs Neuron Count — Gardner",
          "Mismatch Score",
          os.path.join(save_dir, "Gardner_sweep_neuron_global.png"))
    _plot(_build_matrix("seg_mean"),
          "Segmental Mismatch vs Neuron Count — Gardner",
          "Segmental Mismatch Score (mean)",
          os.path.join(save_dir, "Gardner_sweep_neuron_segmental.png"))


if __name__ == "__main__":
    csv_path = sweep_neuron_gardner(
        n_neurons_list=[10, 20, 40, 60, 80, 100, 111],
        n_repeats=5,
    )
    plot_sweep_neuron_gardner(csv_path)
