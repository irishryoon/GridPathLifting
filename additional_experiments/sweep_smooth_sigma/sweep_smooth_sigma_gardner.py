"""
Sweep smoothing sigma on Gardner (real) data.

For each smooth_sigma value the script:
1. Loads Gardner spikes and trajectory once
2. Extracts toroidal coordinates via Gardner_coord
3. Lifts, smooths with gaussian_filter1d(sigma), and aligns via affine transform
4. Computes global and segmental mismatch scores

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
SAVE_DIR = "revision/sweep_smooth_sigma/sweep_smooth_sigma_gardner_results"


def _worker(
    smooth_sigma: float,
    repeat: int,
    cir_coords: np.ndarray,       # (n_timepoints, 2) — pre-computed toroidal coords
    original_coord: np.ndarray,   # (n_timepoints, 2)
    seg_len: int,
    csv_path: str,
    lock_path: str,
) -> tuple:
    try:
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
        except Exception as e:
            print(f"  sigma={smooth_sigma} repeat={repeat}: Segmental mismatch failed ({e})")
            seg_mean = seg_std = np.nan

    except Exception as e:
        print(f"  sigma={smooth_sigma} repeat={repeat}: failed ({e})")
        row = (smooth_sigma, repeat, np.nan, np.nan, np.nan)
        _append_csv(csv_path, lock_path, row)
        return row

    print(f"  sigma={smooth_sigma} repeat={repeat}: mismatch={ms:.6f} seg_mean={seg_mean}")
    row = (smooth_sigma, repeat, ms, seg_mean, seg_std)
    _append_csv(csv_path, lock_path, row)
    return row


def sweep_smooth_sigma_gardner(
    sigma_list: list[float] | None = None,
    n_repeats: int = 1,
    rat_name: str = RAT_NAME,
    mod_name: str = MOD_NAME,
    day_name: str = DAY_NAME,
    sess_name: str = SESS_NAME,
    seg_len: int = SEG_LEN,
    save_dir: str = SAVE_DIR,
    max_workers: int = 6,
):
    """
    n_repeats defaults to 1 because the smooth sigma sweep is deterministic
    for fixed Gardner data. Increase to average over multiple Gardner_coord
    runs if Gardner_coord has any stochastic component.
    """
    if sigma_list is None:
        sigma_list = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]

    print(f"Loading Gardner data (rat={rat_name}, mod={mod_name}, day={day_name}, sess={sess_name})...")
    sspikes, xx, yy, _, _ = get_spikes(
        rat_name, mod_name, day_name, sess_name,
        bType="pure", bSmooth=True, bSpeed=True,
        folder=GARDNER_DATA_PATH,
    )
    original_coord = np.column_stack((xx, yy))
    print(f"Spike matrix: {sspikes.shape}  trajectory: {original_coord.shape}")

    print("Extracting toroidal coordinates (Gardner_coord)...")
    cir_coords, _, _, _ = Gardner_coord(sspikes)
    print(f"Circular coordinates: {cir_coords.shape}")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "Gardner_sweep_smooth_sigma.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["smooth_sigma", "repeat", "mismatch_score", "seg_mean", "seg_std"])

    completed = _load_completed(csv_path, ["smooth_sigma", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed.")

    jobs = [
        (sigma, r)
        for sigma in sigma_list
        for r in range(n_repeats)
        if (float(sigma), r) not in completed
    ]
    print(f"Launching {len(jobs)} jobs (max_workers={max_workers})...")

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _worker, sigma, r, cir_coords, original_coord,
                seg_len, csv_path, lock_path
            ): (sigma, r)
            for sigma, r in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nResults saved to {csv_path}")
    return csv_path


def plot_sweep_smooth_sigma_gardner(csv_path: str, save_dir: str | None = None):
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    sigma_list = sorted(df["smooth_sigma"].unique())
    repeats = sorted(df["repeat"].unique())
    x = np.array(sigma_list)

    def _build_matrix(col: str) -> np.ndarray:
        mat = np.full((len(sigma_list), len(repeats)), np.nan)
        for i, sigma in enumerate(sigma_list):
            for j, r in enumerate(repeats):
                mask = (df["smooth_sigma"] == sigma) & (df["repeat"] == r)
                if mask.any():
                    mat[i, j] = df.loc[mask, col].values[-1]  # type: ignore
        return mat

    def _plot(mat: np.ndarray, title: str, ylabel: str, fig_path: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        means = np.nanmean(mat, axis=1)
        stds = np.nanstd(mat, axis=1)
        valid = ~np.isnan(means)
        ax.errorbar(x[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
        for j in range(mat.shape[1]):
            ax.scatter(x[valid], mat[valid, j], alpha=0.3, s=15, color="gray")
        ax.set_xlabel("Smoothing Sigma")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.show()

    _plot(
        _build_matrix("mismatch_score"),
        "Global Mismatch vs Smoothing Sigma — Gardner",
        "Mismatch Score",
        os.path.join(save_dir, "Gardner_sweep_smooth_sigma_global.png"),
    )
    seg_mean_mat = _build_matrix("seg_mean")
    seg_std_mat = _build_matrix("seg_std")
    means = np.nanmean(seg_mean_mat, axis=1)
    stds = np.nanmean(seg_std_mat, axis=1)  # average within-run std across repeats
    valid = ~np.isnan(means)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
    for j in range(seg_mean_mat.shape[1]):
        ax.scatter(x[valid], seg_mean_mat[valid, j], alpha=0.3, s=15, color="gray")
    ax.set_xlabel("Smoothing Sigma")
    ax.set_ylabel("Segmental Mismatch Score (mean)")
    ax.set_title("Segmental Mismatch vs Smoothing Sigma — Gardner")
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "Gardner_sweep_smooth_sigma_segmental.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()


def plot_smoothed_reconstruction_examples(
    smoothing_sigmas: list[float] | None = None,
    seg_start: int = 6000,
    seg_len: int = 2000,
    save_dir: str = SAVE_DIR,
    rat_name: str = RAT_NAME,
    mod_name: str = MOD_NAME,
    day_name: str = DAY_NAME,
    sess_name: str = SESS_NAME,
):
    """
    Plot reconstructed path segments at varying smoothing sigmas.

    Panels:
      (1) Original path segment (ground truth)
      (2+) Reconstructed segment smoothed with sigma=0, 10, 20, 30, 40, 50
           Gray background = raw original trajectory segment.
    """
    if smoothing_sigmas is None:
        smoothing_sigmas = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    os.makedirs(save_dir, exist_ok=True)

    print("Loading Gardner data...")
    sspikes, xx, yy, _, _ = get_spikes(
        rat_name, mod_name, day_name, sess_name,
        bType="pure", bSmooth=True, bSpeed=True,
        folder=GARDNER_DATA_PATH,
    )
    original_coord = np.column_stack((xx, yy))
    cir_coords, _, _, _ = Gardner_coord(sspikes)

    lifted_base = toroidal_lifting_distance_upgrade(cir_coords.copy())

    # Compute affine once using sigma=10 (default used by other Gardner scripts)
    _lifted_ref = lifted_base.copy()
    _lifted_ref[:, 0] = gaussian_filter1d(_lifted_ref[:, 0], 10.0)
    _lifted_ref[:, 1] = gaussian_filter1d(_lifted_ref[:, 1], 10.0)
    transform_mat = get_transform_mat(_lifted_ref[seg_start:seg_start + seg_len], original_coord[seg_start:seg_start + seg_len])

    orig_seg = original_coord[seg_start:seg_start + seg_len]

    n_panels = 1 + len(smoothing_sigmas)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    def _draw(ax, xy, title):
        ax.scatter(orig_seg[:, 0], orig_seg[:, 1], c="lightgray", s=1, zorder=0)
        ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="viridis", s=3, alpha=0.8)
        ax.set_title(title, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel 0: original trajectory
    ax0 = axes[0]
    ax0.scatter(orig_seg[:, 0], orig_seg[:, 1], c=np.arange(len(orig_seg)), cmap="viridis", s=3, alpha=0.8)
    ax0.set_title("Original", fontsize=20)
    ax0.set_xticks([])
    ax0.set_yticks([])

    for idx, sigma in enumerate(smoothing_sigmas):
        lifted = lifted_base.copy()
        if sigma > 0:
            lifted[:, 0] = gaussian_filter1d(lifted[:, 0], sigma)
            lifted[:, 1] = gaussian_filter1d(lifted[:, 1], sigma)
        lifted_transformed = apply_transform_mat(lifted, transform_mat)
        seg = lifted_transformed[seg_start:seg_start + seg_len]
        _draw(axes[1 + idx], seg, f"sigma={sigma}")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"Reconstructed path segment (steps {seg_start}-{seg_start + seg_len})", fontsize=20)
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "smoothed_reconstruction_examples.pdf")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    csv_path = sweep_smooth_sigma_gardner(
        sigma_list=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0],
        n_repeats=1,
    )
    plot_sweep_smooth_sigma_gardner(csv_path)
    plot_smoothed_reconstruction_examples()
