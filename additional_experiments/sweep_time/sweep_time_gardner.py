"""
Sweep time length on Gardner (real) data.

For each t_length value the script:
1. Loads Gardner spikes once (shape: n_timepoints × n_neurons)
2. Takes the first t_length time steps (all neurons retained)
3. Extracts toroidal coordinates via Gardner_coord
4. Lifts and aligns via affine transform
5. Computes global and segmental mismatch scores

Results are saved incrementally to CSV (resumable).
"""

import os, sys
import csv
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
SAVE_DIR = "revision/sweep_time/sweep_time_gardner_results"
TOROIDAL_FAIL_MSG = "failed toroidal coordiantes computation"
MAX_T_LENGTH = 40000
TIMEBIN_SECONDS = 0.01


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(
    t_length: int, repeat: int,
    sspikes: np.ndarray,        # (n_timepoints, n_neurons)
    original_coord: np.ndarray, # (n_timepoints, 2)
    smooth_sigma: float,
    seg_len: int,
    csv_path: str, lock_path: str,
    segment_csv_path: str, segment_lock_path: str,
    write_summary_row: bool = True,
) -> tuple:
    n_total_time = sspikes.shape[0]
    t_length_clamped = min(t_length, n_total_time)

    spikes_slice = sspikes[:t_length_clamped, :]
    coord_slice = original_coord[:t_length_clamped]

    try:
        cir_coords, _, _, _ = Gardner_coord(spikes_slice)
    except Exception as e:
        print(f"  t={t_length} repeat={repeat}: Gardner_coord failed ({e})")
        row = (t_length, repeat, TOROIDAL_FAIL_MSG, np.nan, np.nan)
        if write_summary_row:
            _append_csv(csv_path, lock_path, row)
        return row

    lifted = toroidal_lifting_distance_upgrade(cir_coords.copy())
    if smooth_sigma > 0:
        lifted[:, 0] = gaussian_filter1d(lifted[:, 0], smooth_sigma)
        lifted[:, 1] = gaussian_filter1d(lifted[:, 1], smooth_sigma)

    transform_mat = get_transform_mat(lifted, coord_slice)
    lifted_transformed = apply_transform_mat(lifted, transform_mat)
    ms = float(score_mismatch(lifted_transformed, coord_slice))

    if t_length_clamped < seg_len:
        seg_mean = seg_std = np.nan
        print(
            f"  t={t_length} repeat={repeat}: mismatch={ms:.6f} "
            f"segmental skipped (t_length_clamped={t_length_clamped} < seg_len={seg_len})"
        )
        row = (t_length, repeat, ms, seg_mean, seg_std)
        if write_summary_row:
            _append_csv(csv_path, lock_path, row)
        return row

    try:
        seg_scores = segmental_mismatch(lifted_transformed, coord_slice, seg_len)
        if len(seg_scores) > 0:
            seg_mean = float(np.mean(seg_scores))
            seg_std = float(np.std(seg_scores))
        else:
            seg_mean = seg_std = np.nan

        for seg_idx, seg_score in enumerate(seg_scores):
            seg_row = (
                t_length,
                repeat,
                t_length_clamped,
                seg_len,
                seg_idx,
                float(seg_score),
            )
            _append_csv(segment_csv_path, segment_lock_path, seg_row)
    except Exception as e:
        print(f"  t={t_length} repeat={repeat}: Segmental mismatch failed ({e})")
        seg_mean = seg_std = np.nan

    print(f"  t={t_length} repeat={repeat}: mismatch={ms:.6f} seg_mean={seg_mean}")
    row = (t_length, repeat, ms, seg_mean, seg_std)
    if write_summary_row:
        _append_csv(csv_path, lock_path, row)
    return row


def _load_segment_completed(segment_csv_path: str) -> set[tuple[int, int]]:
    """Return (t_length, repeat) pairs already present in segment CSV."""
    if not os.path.exists(segment_csv_path):
        return set()
    df = pd.read_csv(segment_csv_path)
    if df.empty:
        return set()
    return set(zip(df["t_length"].astype(int), df["repeat"].astype(int)))


def _dedupe_summary_csv(csv_path: str) -> None:
    """Keep only the latest row for each (t_length, repeat) pair."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    df = df.drop_duplicates(subset=["t_length", "repeat"], keep="last")
    df = df.sort_values(["t_length", "repeat"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_time_gardner(
    t_length_list: list[int] | None = None,
    n_repeats: int = 1,
    rat_name: str = RAT_NAME,
    mod_name: str = MOD_NAME,
    day_name: str = DAY_NAME,
    sess_name: str = SESS_NAME,
    smooth_sigma: float = SMOOTH_SIGMA,
    seg_len: int = SEG_LEN,
    save_dir: str = SAVE_DIR,
    max_workers: int = 6,
):
    """
    n_repeats defaults to 1 because the time sweep is deterministic
    (first t_length points are always the same). Set higher to average
    over random subsets instead of a contiguous prefix if desired.
    """
    if t_length_list is None:
        t_length_list = [5000, 10000, 20000, 30000, 40000]

    original_t_length_count = len(t_length_list)
    t_length_list = sorted({int(t) for t in t_length_list if int(t) > 0 and int(t) <= MAX_T_LENGTH})
    if len(t_length_list) < original_t_length_count:
        print(f"Capped time lengths at {MAX_T_LENGTH}.")
    if len(t_length_list) == 0:
        raise ValueError("No valid t_length values after applying cap.")

    print(f"Loading Gardner data (rat={rat_name}, mod={mod_name}, day={day_name}, sess={sess_name})...")
    sspikes, xx, yy, _, _ = get_spikes(
        rat_name, mod_name, day_name, sess_name,
        bType="pure", bSmooth=True, bSpeed=True,
        folder=GARDNER_DATA_PATH,
    )
    original_coord = np.column_stack((xx, yy))
    print(f"Spike matrix: {sspikes.shape}  trajectory: {original_coord.shape}")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "Gardner_sweep_time.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["t_length", "repeat", "mismatch_score", "seg_mean", "seg_std"])

    segment_csv_path = os.path.join(save_dir, "Gardner_sweep_time_segment_scores.csv")
    segment_lock_path = segment_csv_path + ".lock"
    if not os.path.exists(segment_csv_path):
        with open(segment_csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "t_length",
                "repeat",
                "t_length_clamped",
                "segment_length",
                "segment_index",
                "segment_mismatch_score",
            ])

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
            executor.submit(
                _worker, t, r, sspikes, original_coord,
                smooth_sigma, seg_len, csv_path, lock_path,
                segment_csv_path, segment_lock_path,
                True,
            ): (t, r)
            for t, r in jobs
        }
        for future in cf.as_completed(futures):
            future.result()

    target_pairs = {(t, r) for t in t_length_list for r in range(n_repeats)}
    segment_completed = _load_segment_completed(segment_csv_path)
    backfill_jobs = [
        (t, r) for (t, r) in sorted(target_pairs)
        if (t, r) in completed and (t, r) not in segment_completed
    ]

    if backfill_jobs:
        print(f"Backfilling segment scores for {len(backfill_jobs)} completed jobs...")
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _worker, t, r, sspikes, original_coord,
                    smooth_sigma, seg_len, csv_path, lock_path,
                    segment_csv_path, segment_lock_path,
                    True,
                ): (t, r)
                for t, r in backfill_jobs
            }
            for future in cf.as_completed(futures):
                future.result()

    _dedupe_summary_csv(csv_path)

    print(f"\nResults saved to {csv_path}")
    print(f"Segment-level scores saved to {segment_csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep_time_gardner(csv_path: str, save_dir: str | None = None):
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    def _label_with_percent(label: str) -> str:
        # When matplotlib uses TeX text rendering, '%' must be escaped.
        return label.replace("%", r"\%") if plt.rcParams.get("text.usetex", False) else label

    df = pd.read_csv(csv_path)
    t_length_list = sorted(df["t_length"].unique())
    repeats = sorted(df["repeat"].unique())
    x = np.array(t_length_list, dtype=float) * TIMEBIN_SECONDS
    t_length_list_global = [t for t in t_length_list if t <= MAX_T_LENGTH]
    x_global = np.array(t_length_list_global, dtype=float) * TIMEBIN_SECONDS

    def _build_matrix(col, t_values):
        mat = np.full((len(t_values), len(repeats)), np.nan)
        for i, t in enumerate(t_values):
            for j, r in enumerate(repeats):
                mask = (df["t_length"] == t) & (df["repeat"] == r)
                if mask.any():
                    val = df.loc[mask, col].values[-1]
                    mat[i, j] = pd.to_numeric(val, errors="coerce")
        return mat

    def _plot(x_vals, mat, title, ylabel, fig_path, use_log_x=True):
        fig, ax = plt.subplots(figsize=(7, 5))
        mat_pct = mat * 100.0
        means = np.nanmean(mat_pct, axis=1)
        stds = np.nanstd(mat_pct, axis=1)
        valid = ~np.isnan(means)
        ax.errorbar(x_vals[valid], means[valid], yerr=stds[valid], fmt="o-", markersize=5, capsize=4)
        for j in range(mat_pct.shape[1]):
            ax.scatter(x_vals[valid], mat_pct[valid, j], s=20, color="gray")
        if use_log_x:
            ax.set_xscale("log")
            ax.set_xlabel("Time (s, log scale)")
        else:
            ax.set_xlabel("Time (s)", fontsize = 20)
        ax.set_ylabel(ylabel, fontsize = 20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.tick_params(axis="both", which="minor", labelsize=20)  # optional    
        ax.set_title(title, fontsize = 20)
        fig.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")
        plt.show()

    _plot(x_global,
          _build_matrix("mismatch_score", t_length_list_global),
          "Global reconstruction error vs duration (2D experimental)",
                    _label_with_percent("Global reconstruction error (%)"),
          os.path.join(save_dir, "Gardner_sweep_time_global.png"),
          use_log_x=False)
    # _plot(x,
    #       _build_matrix("seg_mean", t_length_list),
    #       "Segmental Mismatch vs Time — Gardner",
    #                 _label_with_percent("Local reconstruction error (mean, %)"),
    #       os.path.join(save_dir, "Gardner_sweep_time_segmental.png"))


def plot_segment_score_distribution(segment_csv_path: str, save_dir: str | None = None):
    """Plot per-segment mismatch trends with per-point scatter and error bars."""
    if save_dir is None:
        save_dir = os.path.dirname(segment_csv_path)

    def _label_with_percent(label: str) -> str:
        # When matplotlib uses TeX text rendering, '%' must be escaped.
        return label.replace("%", r"\%") if plt.rcParams.get("text.usetex", False) else label

    if not os.path.exists(segment_csv_path):
        print(f"Segment CSV not found: {segment_csv_path}")
        return None

    df = pd.read_csv(segment_csv_path)
    if df.empty:
        print(f"Segment CSV is empty: {segment_csv_path}")
        return None

    grouped = (
        df.dropna(subset=["segment_mismatch_score"])
        .groupby("t_length")["segment_mismatch_score"]
    )

    if grouped.ngroups == 0:
        print("No valid segment mismatch scores to plot.")
        return None

    x = np.array(sorted(grouped.groups.keys()), dtype=float)
    x_seconds = x * TIMEBIN_SECONDS
    means = grouped.mean().reindex(x).to_numpy(dtype=float) * 100.0
    stds = grouped.std().fillna(0.0).reindex(x).to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x_seconds, means, yerr=stds, fmt="o-", markersize=5, capsize=4)

    # Overlay raw segment-level scores to show distribution at each t_length.
    for t in x:
        scores = (
            df.loc[df["t_length"] == int(t), "segment_mismatch_score"]
            .dropna()
            .to_numpy() * 100.0
        )
        if len(scores) > 0:
            ax.scatter(np.full(len(scores), t * TIMEBIN_SECONDS), scores, s=20, color="gray")

    ax.set_xlabel("Time (s)", fontsize = 20)
    ax.set_ylabel(_label_with_percent("Local reconstruction error (%)"), fontsize = 20)

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=20)  # optional    
    
    ax.set_title("Local reconstruction error vs duration (2D experimental)", fontsize = 20)
    fig.tight_layout()

    fig_path = os.path.join(save_dir, "Gardner_sweep_time_segment_distribution.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()
    return fig_path


if __name__ == "__main__":
    csv_path = sweep_time_gardner(
        t_length_list=[50, 100, 200, 500, 1000, 2000,5000, 10000, 20000, 30000,40000, 60000, 80000, 100000, 126728],
        n_repeats=1,
    )
    plot_sweep_time_gardner(csv_path)
    segment_csv_path = os.path.join(os.path.dirname(csv_path), "Gardner_sweep_time_segment_scores.csv")
    plot_segment_score_distribution(segment_csv_path)
