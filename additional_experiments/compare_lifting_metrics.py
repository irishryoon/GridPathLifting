"""
Compare lifting performance using cosine vs euclidean distance on
simulated and Gardner data.

For simulated data, each repeat generates a fresh random trajectory,
simulates grid-cell activity, extracts toroidal coordinates with the
chosen metric, then lifts and scores the reconstruction.

For Gardner data, the full session is decoded once per metric.

Each dataset has its own runner and plotting function.
"""

import os
import sys
import csv
import concurrent.futures as cf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.append(".")
from constants import GARDNER_DATA_PATH
from helper_scripts.utils import get_spikes
from toroidal_coordinates.Gardner_toroidal import Gardner_coord
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance_upgrade
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from affine_transform.score_mismatch import score_mismatch, segmental_mismatch
from ROOT_PATH import DATA_ROOT
from revision.csv_helpers import (
    init_csv as _init_csv,
    append_csv as _append_csv,
    load_completed as _load_completed,
)


METRICS = ["cosine", "euclidean"]
SAVE_DIR = "revision/metric_lifting_comparison_results"
SIM_CACHE_DIR = os.path.join(DATA_ROOT, "revision_cache")
SIM_N_REPEATS = 10

RAT_NAME = "R"
MOD_NAME = "1"
DAY_NAME = "day2"
SESS_NAME = "OF"

SIMULATED_SEG_LEN = 10000
GARDNER_SEG_LEN = 2000
SIM_SMOOTH_SIGMA = 0.0
GARDNER_SMOOTH_SIGMA = 10.0


def _score_lifted(
    cir_coords: np.ndarray,
    original_coord: np.ndarray,
    smooth_sigma: float,
    seg_len: int,
) -> tuple[float, float, float, list[float]]:
    lifted = toroidal_lifting_distance_upgrade(cir_coords.copy())
    if smooth_sigma > 0:
        lifted[:, 0] = gaussian_filter1d(lifted[:, 0], smooth_sigma)
        lifted[:, 1] = gaussian_filter1d(lifted[:, 1], smooth_sigma)

    transform_mat = get_transform_mat(lifted, original_coord)
    lifted_transformed = apply_transform_mat(lifted, transform_mat)
    mismatch = float(score_mismatch(lifted_transformed, original_coord))

    try:
        seg_scores = segmental_mismatch(lifted_transformed, original_coord, seg_len)
        seg_mean = float(np.mean(seg_scores))
        seg_std = float(np.std(seg_scores))
    except Exception:
        seg_scores = []
        seg_mean = np.nan
        seg_std = np.nan

    return mismatch, seg_mean, seg_std, seg_scores


def _init_simple_csv(csv_path: str, header: list[str]) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def _load_metrics_with_segment_scores(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    if "metric" not in df.columns or "segment_score" not in df.columns:
        return set()

    valid = df["segment_score"].notna()
    return set(df.loc[valid, "metric"].astype(str))


def _run_simulated_worker(
    metric: str,
    repeat: int,
    seg_len: int,
    smooth_sigma: float,
    cache_dir: str,
    csv_path: str,
    lock_path: str,
) -> tuple:
    trace_path = os.path.join(cache_dir, f"trace_{repeat}.npy")
    activity_path = os.path.join(cache_dir, f"grid_activity_{repeat}.npy")
    processed_trace = np.load(trace_path)
    grid_activity = np.load(activity_path)

    try:
        cir_coords, _, _, _ = Gardner_coord(grid_activity.T, metric=metric)
        mismatch, seg_mean, seg_std, _ = _score_lifted(
            cir_coords=cir_coords,
            original_coord=processed_trace,
            smooth_sigma=smooth_sigma,
            seg_len=seg_len,
        )
    except Exception as exc:
        print(f"  [simulated] metric={metric} repeat={repeat}: failed ({exc})")
        row = (metric, repeat, np.nan, np.nan, np.nan)
        _append_csv(csv_path, lock_path, row)
        return row

    print(
        f"  [simulated] metric={metric} repeat={repeat}: "
        f"mismatch={mismatch:.6f} seg_mean={seg_mean:.6f}"
    )
    row = (metric, repeat, mismatch, seg_mean, seg_std)
    _append_csv(csv_path, lock_path, row)
    return row


def compute_single_simulated_segment_scores(
    metrics: list[str] | None = None,
    seg_len: int = SIMULATED_SEG_LEN,
    sim_smooth_sigma: float = SIM_SMOOTH_SIGMA,
    save_dir: str = SAVE_DIR,
    world_size: tuple[int, int] = (100, 100),
    holes: list[tuple[int, int, int, int]] | None = None,
    step_size: int = 5,
    traj_length: int = 10000,
    goal: int = 600000,
    seed: int | None = None,
    output_csv_name: str = "single_simulated_segment_scores.csv",
) -> str:
    """
    Generate one simulated trajectory and save all segmental mismatch scores.

    The output CSV has one row per segment per metric.
    """
    if metrics is None:
        metrics = METRICS.copy()
    if holes is None:
        holes = [(30, 30, 70, 70)]

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, output_csv_name)
    lock_path = csv_path + ".lock"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "metric",
                "segment_idx",
                "segment_score",
                "mismatch_score",
                "seg_mean",
                "seg_std",
            ]
        )

    if seed is not None:
        np.random.seed(seed)

    from trajectory import random_walk, World
    from grid_cell_simulation import preprocess, simulation

    print("Generating one simulated trajectory...")
    world = World(world_size, holes, step_size=step_size)
    trace = random_walk(traj_length, world, save=False, no_warnings=True)
    processed_trace = preprocess(trace, step_size, goal=goal)
    grid_activity = simulation(processed_trace)

    for metric in metrics:
        try:
            cir_coords, _, _, _ = Gardner_coord(grid_activity.T, metric=metric)
            mismatch, seg_mean, seg_std, seg_scores = _score_lifted(
                cir_coords=cir_coords,
                original_coord=processed_trace,
                smooth_sigma=sim_smooth_sigma,
                seg_len=seg_len,
            )
        except Exception as exc:
            print(f"  [single-sim] metric={metric}: failed ({exc})")
            _append_csv(
                csv_path,
                lock_path,
                (metric, -1, np.nan, np.nan, np.nan, np.nan),
            )
            continue

        print(
            f"  [single-sim] metric={metric}: "
            f"mismatch={mismatch:.6f} seg_mean={seg_mean:.6f} n_segments={len(seg_scores)}"
        )

        if not seg_scores:
            _append_csv(
                csv_path,
                lock_path,
                (metric, -1, np.nan, mismatch, seg_mean, seg_std),
            )
            continue

        for idx, score in enumerate(seg_scores):
            _append_csv(
                csv_path,
                lock_path,
                (metric, idx, float(score), mismatch, seg_mean, seg_std),
            )

    print(f"Single simulated segment scores saved to {csv_path}")
    return csv_path


def _run_gardner_metric(
    metric: str,
    sspikes: np.ndarray,
    original_coord: np.ndarray,
    seg_len: int,
    smooth_sigma: float,
    csv_path: str,
    lock_path: str,
    write_summary: bool,
    seg_scores_csv_path: str | None = None,
    seg_scores_lock_path: str | None = None,
    write_segment_scores: bool = False,
) -> tuple:
    try:
        cir_coords, _, _, _ = Gardner_coord(sspikes, metric=metric)
        mismatch, seg_mean, seg_std, seg_scores = _score_lifted(
            cir_coords=cir_coords,
            original_coord=original_coord,
            smooth_sigma=smooth_sigma,
            seg_len=seg_len,
        )
    except Exception as exc:
        print(f"  [Gardner] metric={metric}: failed ({exc})")
        row = (metric, 0, np.nan, np.nan, np.nan)
        if write_summary:
            _append_csv(csv_path, lock_path, row)
        return row

    print(
        f"  [Gardner] metric={metric}: "
        f"mismatch={mismatch:.6f} seg_mean={seg_mean:.6f}"
    )
    row = (metric, 0, mismatch, seg_mean, seg_std)
    if write_summary:
        _append_csv(csv_path, lock_path, row)

    if write_segment_scores and seg_scores_csv_path and seg_scores_lock_path:
        for idx, score in enumerate(seg_scores):
            _append_csv(
                seg_scores_csv_path,
                seg_scores_lock_path,
                (metric, 0, idx, float(score)),
            )
    return row


def compare_lifting_metrics_simulated(
    metrics: list[str] | None = None,
    sim_n_repeats: int = SIM_N_REPEATS,
    seg_len: int = SIMULATED_SEG_LEN,
    sim_smooth_sigma: float = SIM_SMOOTH_SIGMA,
    save_dir: str = SAVE_DIR,
    cache_dir: str = SIM_CACHE_DIR,
    max_workers: int = 4,
) -> str:
    if metrics is None:
        metrics = METRICS.copy()

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "lifting_metric_comparison_simulated.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["metric", "repeat", "mismatch_score", "seg_mean", "seg_std"])
    completed = _load_completed(csv_path, ["metric", "repeat"])
    if completed:
        print(f"Resuming simulated comparison: {len(completed)} tasks already completed.")

    missing = [
        repeat for repeat in range(sim_n_repeats)
        if not os.path.exists(os.path.join(cache_dir, f"trace_{repeat}.npy"))
        or not os.path.exists(os.path.join(cache_dir, f"grid_activity_{repeat}.npy"))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing cached simulation files in {cache_dir} for repeats: {missing}"
        )

    jobs = [
        (metric, repeat)
        for metric in metrics
        for repeat in range(sim_n_repeats)
        if (metric, repeat) not in completed
    ]

    print(f"Launching {len(jobs)} simulated jobs (max_workers={max_workers})...")
    if jobs:
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_simulated_worker,
                    metric,
                    repeat,
                    seg_len,
                    sim_smooth_sigma,
                    cache_dir,
                    csv_path,
                    lock_path,
                ): (metric, repeat)
                for metric, repeat in jobs
            }
            for future in cf.as_completed(futures):
                future.result()

    print(f"\nSimulated results saved to {csv_path}")
    return csv_path


def compare_lifting_metrics_gardner(
    metrics: list[str] | None = None,
    rat_name: str = RAT_NAME,
    mod_name: str = MOD_NAME,
    day_name: str = DAY_NAME,
    sess_name: str = SESS_NAME,
    seg_len: int = GARDNER_SEG_LEN,
    gardner_smooth_sigma: float = GARDNER_SMOOTH_SIGMA,
    save_dir: str = SAVE_DIR,
    save_segment_scores: bool = True,
) -> str:
    if metrics is None:
        metrics = METRICS.copy()

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "lifting_metric_comparison_gardner.csv")
    lock_path = csv_path + ".lock"
    _init_csv(csv_path, ["metric", "repeat", "mismatch_score", "seg_mean", "seg_std"])

    seg_scores_csv_path = os.path.join(save_dir, "lifting_metric_comparison_gardner_segment_scores.csv")
    seg_scores_lock_path = seg_scores_csv_path + ".lock"
    if save_segment_scores:
        _init_simple_csv(seg_scores_csv_path, ["metric", "repeat", "segment_idx", "segment_score"])

    completed = _load_completed(csv_path, ["metric", "repeat"])
    segment_scores_completed = (
        _load_metrics_with_segment_scores(seg_scores_csv_path) if save_segment_scores else set()
    )
    if completed:
        print(f"Resuming Gardner comparison: {len(completed)} tasks already completed.")

    print(f"Loading Gardner data (rat={rat_name}, mod={mod_name}, day={day_name}, sess={sess_name})...")
    sspikes, xx, yy, _, _ = get_spikes(
        rat_name, mod_name, day_name, sess_name,
        bType="pure", bSmooth=True, bSpeed=True,
        folder=GARDNER_DATA_PATH,
    )
    original_coord_gardner = np.column_stack((xx, yy))
    print(f"Gardner spike matrix: {sspikes.shape}  trajectory: {original_coord_gardner.shape}")

    gardner_jobs = []
    for metric in metrics:
        need_summary = (metric, 0) not in completed
        need_segments = save_segment_scores and (metric not in segment_scores_completed)
        if need_summary or need_segments:
            gardner_jobs.append((metric, need_summary, need_segments))

    print(f"Running {len(gardner_jobs)} Gardner jobs...")
    for metric, need_summary, need_segments in gardner_jobs:
        _run_gardner_metric(
            metric=metric,
            sspikes=sspikes,
            original_coord=original_coord_gardner,
            seg_len=seg_len,
            smooth_sigma=gardner_smooth_sigma,
            csv_path=csv_path,
            lock_path=lock_path,
            write_summary=need_summary,
            seg_scores_csv_path=seg_scores_csv_path,
            seg_scores_lock_path=seg_scores_lock_path,
            write_segment_scores=need_segments,
        )

    print(f"\nGardner results saved to {csv_path}")
    return csv_path


def _plot_metric_comparison(
    csv_path: str,
    dataset_label: str,
    fig_name: str,
    save_dir: str | None = None,
    value_col: str = "mismatch_score",
    ylabel: str = "Mismatch Score",
) -> str:
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    color_map = {"cosine": "#1f77b4", "euclidean": "#d62728"}
    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    pivot = (
        df.pivot_table(index="repeat", columns="metric", values=value_col, aggfunc="last")
        .reindex(columns=METRICS)
    )
    paired = pivot.dropna()
    for row in paired.itertuples(index=False):
        ax.plot(
            range(len(METRICS)),
            list(row),
            color="0.7",
            alpha=0.6,
            linewidth=1.0,
            zorder=1,
        )

    for xpos, metric in enumerate(METRICS):
        metric_vals = df.loc[df["metric"] == metric, value_col].dropna().to_numpy()
        if len(metric_vals) == 0:
            continue

        ax.scatter(
            np.full(len(metric_vals), xpos),
            metric_vals,
            s=28,
            alpha=0.65,
            color=color_map[metric],
            edgecolors="none",
            zorder=2,
        )

        mean = float(np.mean(metric_vals))
        std = float(np.std(metric_vals))
        ax.errorbar(
            xpos,
            mean,
            yerr=std,
            fmt="_",
            color="black",
            markersize=18,
            markeredgewidth=2,
            capsize=5,
            linewidth=1.2,
            zorder=3,
        )

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_title(f"Lifting Comparison: {dataset_label}")
    ax.set_xlabel("Distance Metric")
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()
    return fig_path


def plot_lifting_metric_comparison_simulated(csv_path: str, save_dir: str | None = None) -> str:
    return _plot_metric_comparison(
        csv_path=csv_path,
        dataset_label="Simulated",
        fig_name="lifting_metric_comparison_simulated.png",
        save_dir=save_dir,
    )


def plot_lifting_metric_comparison_gardner(csv_path: str, save_dir: str | None = None) -> str:
    return _plot_metric_comparison(
        csv_path=csv_path,
        dataset_label="Gardner",
        fig_name="lifting_metric_comparison_gardner.png",
        save_dir=save_dir,
    )


def plot_segmental_metric_comparison_simulated(csv_path: str, save_dir: str | None = None) -> str:
    return _plot_metric_comparison(
        csv_path=csv_path,
        dataset_label="Simulated (Segmental)",
        fig_name="segmental_metric_comparison_simulated.png",
        save_dir=save_dir,
        value_col="seg_mean",
        ylabel="Segmental Mismatch Score",
    )


def plot_segmental_metric_comparison_gardner(csv_path: str, save_dir: str | None = None) -> str:
    segment_scores_csv_path = csv_path
    if not segment_scores_csv_path.endswith("_segment_scores.csv"):
        segment_scores_csv_path = os.path.join(
            os.path.dirname(csv_path),
            "lifting_metric_comparison_gardner_segment_scores.csv",
        )
    # Pass the summary CSV so overall mismatch_score is overlaid as "x" markers
    summary_csv_path = csv_path if not csv_path.endswith("_segment_scores.csv") else None
    return plot_segmental_scores_comparison_gardner(
        segment_scores_csv_path,
        save_dir=save_dir,
        summary_csv_path=summary_csv_path,
    )


def plot_segmental_scores_comparison_gardner(
    segment_scores_csv_path: str,
    save_dir: str | None = None,
    summary_csv_path: str | None = None,
) -> str:
    """Plot Gardner per-segment mismatch scores for cosine vs euclidean metrics.

    If *summary_csv_path* is provided, the overall mismatch_score from that
    CSV is overlaid on the plot using "x" markers.
    """
    if save_dir is None:
        save_dir = os.path.dirname(segment_scores_csv_path)

    df = pd.read_csv(segment_scores_csv_path)
    required_cols = {"metric", "segment_idx", "segment_score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required_cols)} in {segment_scores_csv_path}, got {list(df.columns)}"
        )

    color_map = {"cosine": "#1f77b4", "euclidean": "#d62728"}
    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    pivot = (
        df.pivot_table(
            index="segment_idx",
            columns="metric",
            values="segment_score",
            aggfunc="last",
        )
        .reindex(columns=METRICS)
    )

    paired = pivot.dropna()
    for row in paired.itertuples(index=False):
        ax.plot(
            range(len(METRICS)),
            list(row),
            color="0.75",
            alpha=0.45,
            linewidth=0.9,
            zorder=1,
        )

    for xpos, metric in enumerate(METRICS):
        metric_vals = df.loc[df["metric"] == metric, "segment_score"].dropna().to_numpy()
        if len(metric_vals) == 0:
            continue

        ax.scatter(
            np.full(len(metric_vals), xpos),
            metric_vals,
            s=18,
            alpha=0.45,
            color=color_map[metric],
            edgecolors="none",
            zorder=2,
        )

        mean = float(np.mean(metric_vals))
        std = float(np.std(metric_vals))
        ax.errorbar(
            xpos,
            mean,
            yerr=std,
            fmt="_",
            color="black",
            markersize=18,
            markeredgewidth=2,
            capsize=5,
            linewidth=1.2,
            zorder=3,
        )

    # Overlay overall mismatch scores from summary CSV as "x" markers
    if summary_csv_path is not None:
        summary_df = pd.read_csv(summary_csv_path)
        for xpos, metric in enumerate(METRICS):
            rows = summary_df.loc[summary_df["metric"] == metric, "mismatch_score"]
            for val in rows:
                ax.scatter(
                    xpos,
                    val,
                    s=80,
                    marker="x",
                    color=color_map.get(metric, "black"),
                    linewidths=1.8,
                    zorder=4,
                )

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_title("Lifting Comparison: Gardner (Per-Segment)")
    ax.set_xlabel("Distance Metric")
    ax.set_ylabel("Segmental Mismatch Score")
    fig.tight_layout()

    fig_path = os.path.join(save_dir, "segmental_metric_comparison_gardner.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()
    return fig_path


def plot_segmental_scores_comparison_simulated_single(
    segment_scores_csv_path: str,
    save_dir: str | None = None,
) -> str:
    """Plot per-segment mismatch scores for a single simulated trajectory (cosine vs euclidean)."""
    if save_dir is None:
        save_dir = os.path.dirname(segment_scores_csv_path)

    df = pd.read_csv(segment_scores_csv_path)
    required_cols = {"metric", "segment_idx", "segment_score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required_cols)} in {segment_scores_csv_path}, got {list(df.columns)}"
        )

    color_map = {"cosine": "#1f77b4", "euclidean": "#d62728"}
    fig, ax = plt.subplots(figsize=(5.2, 4.5))

    pivot = (
        df.pivot_table(
            index="segment_idx",
            columns="metric",
            values="segment_score",
            aggfunc="last",
        )
        .reindex(columns=METRICS)
    )

    paired = pivot.dropna()
    for row in paired.itertuples(index=False):
        ax.plot(
            range(len(METRICS)),
            list(row),
            color="0.75",
            alpha=0.45,
            linewidth=0.9,
            zorder=1,
        )

    for xpos, metric in enumerate(METRICS):
        metric_vals = df.loc[df["metric"] == metric, "segment_score"].dropna().to_numpy()
        if len(metric_vals) == 0:
            continue

        ax.scatter(
            np.full(len(metric_vals), xpos),
            metric_vals,
            s=18,
            alpha=0.45,
            color=color_map[metric],
            edgecolors="none",
            zorder=2,
        )

        mean = float(np.mean(metric_vals))
        std = float(np.std(metric_vals))
        ax.errorbar(
            xpos,
            mean,
            yerr=std,
            fmt="_",
            color="black",
            markersize=18,
            markeredgewidth=2,
            capsize=5,
            linewidth=1.2,
            zorder=3,
        )

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRICS)
    ax.set_title("Lifting Comparison: Simulated (Per-Segment)")
    ax.set_xlabel("Distance Metric")
    ax.set_ylabel("Segmental Mismatch Score")
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(segment_scores_csv_path))[0]
    fig_path = os.path.join(save_dir, f"{stem}_comparison.png")
    plt.savefig(fig_path, dpi=200)
    print(f"Figure saved to {fig_path}")
    plt.show()
    return fig_path


if __name__ == "__main__":
    simulated_csv_path = compare_lifting_metrics_simulated()
    plot_lifting_metric_comparison_simulated(simulated_csv_path)
    plot_segmental_metric_comparison_simulated(simulated_csv_path)

    gardner_csv_path = compare_lifting_metrics_gardner()
    plot_lifting_metric_comparison_gardner(gardner_csv_path)
    plot_segmental_metric_comparison_gardner(gardner_csv_path)
    plot_segmental_scores_by_index_gardner(
        os.path.join(os.path.dirname(gardner_csv_path), "lifting_metric_comparison_gardner_segment_scores.csv")
    )
