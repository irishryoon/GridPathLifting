"""
Sweep Gaussian noise levels on toroidal coordinates and evaluate mismatch score.

For each noise level (sigma), each data point in the toroidal coordinate matrix
is independently shifted by a 2D Gaussian perturbation (wrapped on the torus),
then lifted and compared to the ground-truth trajectory.

The toroidal coords are loaded from a pre-computed .npz cache
(grid_fields/world_1holes/toroidal_coords.npz), which contains:
  - coords : (T, 2)  raw toroidal angles in [0, 2*pi)
  - traj   : (T, 2)  ground-truth trajectory

Results are saved incrementally to a CSV so the run can be resumed after
interruption. A summary figure is also produced.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf

sys.path.append(".")
from pipelines import compare_lifted_pipeline, segmental_compare_pipeline
from toroidal_coordinates.compute_coord import plot_toroidal_coordinates
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance_upgrade
from constants import GRID_FIELDS_PATH
from ROOT_PATH import DATA_ROOT
from revision.csv_helpers import init_csv as _init_csv, append_csv as _append_csv, load_completed as _load_completed


# ---------------------------------------------------------------------------
# Paths and sweep parameters
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(DATA_ROOT, "revision_cache")
NUM_SIMS = 5  # Number of simulations available (indices 0-4)

SAVE_DIR = "revision/sweep_noise/gaussian_noise_toroidal_results"
CSV_NAME = "gaussian_noise_toroidal_sweep_new.csv"
SEG_CSV_NAME = "gaussian_noise_toroidal_sweep_segmental_new.csv"

# Noise sigmas to sweep (in radians; toroidal coords are in [0, 2*pi))
DEFAULT_SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
#DEFAULT_SIGMAS = [0.0]
N_REPEATS = 5
#N_REPEATS = 1
SEG_LEN = 10000  # Segment length for segmental mismatch scores (in timebins)
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------------
# Core perturbation + decode worker
# ---------------------------------------------------------------------------
def _decode_one(
    sigma: float,
    repeat: int,
    coords: np.ndarray,
    traj: np.ndarray,
    csv_path: str,
    seg_csv_path: str,
    lock_path: str,
) -> tuple[float, int, float, float, float]:
    """
    Perturb toroidal coords by adding independent 1D Gaussian noise N(0, sigma^2)
    to each coordinate independently, wrapped to keep coords on the torus.
    """
    rng = np.random.default_rng()

    if sigma > 0.0:
        noise = rng.normal(0.0, sigma, size=coords.shape)
        perturbed = np.mod(coords + noise, TWO_PI)
    else:
        perturbed = coords.copy()

    try:
        lifted = toroidal_lifting_distance_upgrade(perturbed)
        lifted_transformed, ms = compare_lifted_pipeline(lifted, traj)
        seg_scores = segmental_compare_pipeline(lifted_transformed, traj, SEG_LEN)
        seg_mean = float(np.mean(seg_scores))
        seg_std = float(np.std(seg_scores))
    except Exception as e:
        print(f"  sigma={sigma} repeat={repeat}: pipeline failed ({e})")
        ms = seg_mean = seg_std = np.nan
        seg_scores = []

    print(f"  sigma={sigma:.4f} repeat={repeat}: mismatch={ms:.6f}, seg_mean={seg_mean:.6f}"
          if not np.isnan(ms) else f"  sigma={sigma:.4f} repeat={repeat}: mismatch=nan")

    # Write global mismatch row
    _append_csv(csv_path, lock_path, (sigma, repeat, ms))
    # Write one row per segment
    seg_lock_path = seg_csv_path + ".lock"
    for seg_idx, score in enumerate(seg_scores):
        _append_csv(seg_csv_path, seg_lock_path, (sigma, repeat, seg_idx, float(score)))
    return sigma, repeat, ms, seg_mean, seg_std


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_gaussian_noise(
    sigmas: list[float] | None = None,
    n_repeats: int = N_REPEATS,
    save_dir: str = SAVE_DIR,
    n_workers: int = 10,
):
    if sigmas is None:
        sigmas = DEFAULT_SIGMAS

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, CSV_NAME)
    seg_csv_path = os.path.join(save_dir, SEG_CSV_NAME)
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["sigma", "repeat", "mismatch_score"])
    _init_csv(seg_csv_path, ["sigma", "repeat", "segment", "mismatch_score"])

    completed = _load_completed(csv_path, ["sigma", "repeat"])
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed, skipping them.")

    # Load all simulation datasets; repeat index r uses simulation r % NUM_SIMS
    all_coords = [np.load(os.path.join(DATA_PATH, f"toroidal_coords_{i}.npy")) for i in range(NUM_SIMS)]
    all_trajs  = [np.load(os.path.join(DATA_PATH, f"trace_{i}.npy"))           for i in range(NUM_SIMS)]
    print(f"Loaded {NUM_SIMS} simulations; repeat r uses simulation r % {NUM_SIMS}")
    print(f"Sweeping {len(sigmas)} sigma values x {n_repeats} repeats = "
          f"{len(sigmas) * n_repeats} total tasks")

    tasks = [
        (sigma, r)
        for sigma in sigmas
        for r in range(n_repeats)
        if (sigma, r) not in completed
    ]
    print(f"Tasks remaining: {len(tasks)}")

    done_count = 0
    total = len(tasks)

    with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_decode_one, sigma, r,
                            all_coords[r % NUM_SIMS], all_trajs[r % NUM_SIMS],
                            csv_path, seg_csv_path, lock_path): (sigma, r)
            for sigma, r in tasks
        }
        for future in cf.as_completed(futures):
            future.result()
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"  [{done_count}/{total}] tasks completed")

    # Sort and deduplicate global CSV
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["sigma", "repeat"], keep="last")
    df = df.sort_values(["sigma", "repeat"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\nGlobal results saved to {csv_path} ({len(df)} rows)")

    # Sort and deduplicate segmental CSV
    seg_df = pd.read_csv(seg_csv_path)
    seg_df = seg_df.drop_duplicates(subset=["sigma", "repeat", "segment"], keep="last")
    seg_df = seg_df.sort_values(["sigma", "repeat", "segment"]).reset_index(drop=True)
    seg_df.to_csv(seg_csv_path, index=False)
    print(f"Segmental results saved to {seg_csv_path} ({len(seg_df)} rows)")
    return csv_path, seg_csv_path


# ---------------------------------------------------------------------------
# Trajectory colored by toroidal coordinates
# ---------------------------------------------------------------------------
def plot_traj_colored_by_toroidal(
    sigmas: list[float] | None = None,
    save_dir: str = SAVE_DIR,
):
    """
    For each sigma, draw two panels: trajectory scatter colored by theta_1 and
    theta_2. Sigma=0 uses the unperturbed coords; others show a single random
    perturbation for illustration.
    """
    if sigmas is None:
        sigmas = DEFAULT_SIGMAS

    os.makedirs(save_dir, exist_ok=True)

    # Use simulation 0 for illustration plots
    traj   = np.load(os.path.join(DATA_PATH, "trace_0.npy"))           # (T, 2)
    coords = np.load(os.path.join(DATA_PATH, "toroidal_coords_0.npy")) # (T, 2), [0, 2*pi)

    subfolder = os.path.join(save_dir, "noisy_toroidal_examples")
    os.makedirs(subfolder, exist_ok=True)

    rng = np.random.default_rng(0)
    labels = [r"$\theta_1$", r"$\theta_2$"]

    # Truncate trajectory to match coords length if needed
    traj_plot = traj[:len(coords)]

    for sigma in sigmas:
        if sigma > 0.0:
            noise = rng.normal(0.0, sigma, size=coords.shape)
            perturbed = np.mod(coords + noise, TWO_PI)
        else:
            perturbed = coords

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for col in range(2):
            ax = axes[col]
            sc = ax.scatter(
                traj_plot[:, 0], traj_plot[:, 1],
                c=perturbed[:, col],
                cmap="viridis", vmin=0, vmax=TWO_PI,
                s=1, alpha=0.6, rasterized=True,
            )
            fig.colorbar(sc, ax=ax, label=f"{labels[col]} (rad)")
            ax.set_title(f"{labels[col]}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")

        fig.suptitle(f"Trajectory colored by toroidal coords — sigma={sigma}")
        fig.tight_layout()
        sigma_str = f"{sigma:.4f}".rstrip("0").rstrip(".")
        fig_path = os.path.join(subfolder, f"sigma_{sigma_str}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"  Saved {fig_path}")
        plt.close(fig)


def save_noisy_toroidal_coordinate_plots(
    sigmas: list[float] | None = None,
    save_dir: str = SAVE_DIR,
    sim_idx: int = 0,
    seed: int = 0,
    max_points: int = 300000,
):
    """
    Standalone utility: load one toroidal coordinate simulation, add Gaussian
    noise at requested sigma levels, and save coordinate-space plots.

    This function does not run lifting/decoding and does not depend on sweep CSVs.
    """
    if sigmas is None:
        sigmas = DEFAULT_SIGMAS

    coords_path = os.path.join(DATA_PATH, f"toroidal_coords_{sim_idx}.npy")
    traj_path = os.path.join(DATA_PATH, f"trace_{sim_idx}.npy")
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Toroidal coordinate file not found: {coords_path}")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    coords = np.load(coords_path)
    traj = np.load(traj_path)
    rng = np.random.default_rng(seed)

    # Optional deterministic downsampling for faster plotting.
    if max_points is not None and len(coords) > max_points:
        idx = np.linspace(0, len(coords) - 1, max_points, dtype=int)
        base_coords = coords[idx]
        base_traj = traj[:len(coords)][idx]
    else:
        base_coords = coords
        base_traj = traj[:len(coords)]

    out_dir = os.path.join(save_dir, f"noisy_toroidal_coord_space_sim{sim_idx}")
    os.makedirs(out_dir, exist_ok=True)

    for sigma in sigmas:
        if sigma > 0.0:
            noise = rng.normal(0.0, sigma, size=base_coords.shape)
            perturbed = np.mod(base_coords + noise, TWO_PI)
        else:
            perturbed = base_coords.copy()

        times = np.arange(len(perturbed))
        plot_toroidal_coordinates(
            base_traj[:, 0],
            base_traj[:, 1],
            perturbed,
            times,
            subsample_times=True,
            s=5,
            #alpha=0.7,
            rasterized=True,
        )
        #plt.suptitle(f"Toroidal Coordinates (sim={sim_idx}) with sigma={sigma}")
        plt.tight_layout()

        sigma_str = f"{sigma:.4f}".rstrip("0").rstrip(".")
        fig_path = os.path.join(out_dir, f"coord_space_sigma_{sigma_str}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"  Saved {fig_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Mismatch sweep plots
# ---------------------------------------------------------------------------
def _agg_col(df: pd.DataFrame, col: str):
    sigmas = sorted(df["sigma"].unique())
    repeats = sorted(df["repeat"].unique())
    means, stds, all_vals = [], [], []
    for s in sigmas:
        vals = [df.loc[(df["sigma"] == s) & (df["repeat"] == r), col].values
                for r in repeats]
        vals = [v[0] if len(v) else np.nan for v in vals]
        all_vals.append(vals)
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))
    return np.array(sigmas), np.array(means), np.array(stds), all_vals, repeats


def plot_sweep(csv_path: str, save_dir: str | None = None):
    """Plot global mismatch score vs sigma."""
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    sigma_arr, means, stds, all_vals, repeats = _agg_col(df, "mismatch_score")
    valid = ~np.isnan(means)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigma_arr[valid], means[valid], yerr=stds[valid],
                fmt="o-", markersize=5, capsize=4, label="Mean ± Std")
    for r_idx in range(len(repeats)):
        vals = np.array([all_vals[i][r_idx] for i in range(len(sigma_arr))])
        ax.scatter(sigma_arr[valid], vals[valid], alpha=0.3, s=15, color="gray")
    ax.set_xlabel("Gaussian Noise Sigma (radians)")
    ax.set_ylabel("Mismatch Score")
    ax.set_title("Gaussian Noise on Toroidal Coords — world_1holes")
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "gaussian_noise_toroidal_sweep_new.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()


def plot_sweep_segmental(seg_csv_path: str, save_dir: str | None = None):
    """Plot segmental mean mismatch score vs sigma."""
    if save_dir is None:
        save_dir = os.path.dirname(seg_csv_path)

    seg_df = pd.read_csv(seg_csv_path)
    # Aggregate: per (sigma, repeat) mean across all segments
    agg = seg_df.groupby(["sigma", "repeat"])["mismatch_score"].mean().reset_index()
    agg = agg.rename(columns={"mismatch_score": "seg_mean"})
    sigma_arr, means, stds, all_vals, repeats = _agg_col(agg, "seg_mean")
    valid = ~np.isnan(means)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigma_arr[valid], means[valid], yerr=stds[valid],
                fmt="o-", markersize=5, capsize=4, label="Mean ± Std")
    for r_idx in range(len(repeats)):
        vals = np.array([all_vals[i][r_idx] for i in range(len(sigma_arr))])
        ax.scatter(sigma_arr[valid], vals[valid], alpha=0.3, s=15, color="gray")
    ax.set_xlabel("Gaussian Noise Sigma (radians)")
    ax.set_ylabel("Seg Mean Mismatch Score")
    ax.set_title("Segmental Mean Mismatch — Gaussian Noise on Toroidal Coords — world_1holes")
    fig.tight_layout()
    fig_path = os.path.join(save_dir, "gaussian_noise_toroidal_sweep_segmental_new.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    csv_path, seg_csv_path = sweep_gaussian_noise()
    plot_sweep(csv_path)
    plot_sweep_segmental(seg_csv_path)
    plot_traj_colored_by_toroidal()
