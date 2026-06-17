"""
Sweep neuron and time-point subsampling fractions and evaluate torus detection.

Neurons and time points are swept independently:
  - Neuron sweep: vary neuron_fraction over a grid activity chosen at random
    from the cache, keeping all time points.
  - Time sweep:   vary time_fraction over a grid activity chosen at random
    from the cache, keeping all neurons.

Each (fraction, metric, repeat) job randomly picks one cached grid activity
file and subsamples it. Results are saved to a CSV and plotted as line plots.

Results CSV columns: repeat, sweep, fraction, metric, detected, pd_counts
"""

import csv
import filelock
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf

sys.path.append(".")
from persistence_homology import persistence_analysis_spikes
from revision.torus_detection import count_pd_outliers, is_torus
from ROOT_PATH import DATA_ROOT


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(DATA_ROOT, "revision_cache")

DEFAULT_NEURON_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_TIME_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_METRICS = ["cosine", "euclidean"]
N_REPEATS = 10

SAVE_DIR = "revision/sweep_neuron/subsample_torus_detection_results"
CSV_NAME = "subsample_torus_detection.csv"


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _init_csv(csv_path: str) -> None:
    """Writes the CSV header if the file does not already exist.

    Args:
        csv_path: Path to the CSV file.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["repeat", "sweep", "fraction", "metric", "detected", "pd_counts"])


def _append_csv(csv_path: str, row: tuple) -> None:
    """Appends a single result row to the CSV, guarded by a file lock.

    Args:
        csv_path: Path to the CSV file.
        row: Tuple of values to write as a CSV row.
    """
    lock = filelock.FileLock(csv_path + ".lock")
    with lock:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


def _load_completed(csv_path: str) -> set[tuple[int, str, float, str]]:
    """Returns the set of (repeat, sweep, fraction, metric) already saved.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Set of completed (repeat, sweep, fraction, metric) tuples.
    """
    completed: set[tuple[int, str, float, str]] = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                completed.add((int(row[0]), row[1], float(row[2]), row[3]))
    return completed


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def _sweep_neurons(
    repeat: int,
    fraction: float,
    metric: str,
    cache_files: list[str],
    csv_path: str,
) -> tuple[int, str, float, str, bool, str]:
    """Runs one neuron-sweep job.

    Randomly selects a cached grid activity, subsamples neurons at the given
    fraction (keeping all time points), computes persistent homology, and
    tests for torus topology. Writes the result to CSV immediately.

    Args:
        repeat: Repeat index.
        fraction: Fraction of neurons to retain (0, 1].
        metric: Distance metric for persistent homology (e.g. ``"cosine"``).
        cache_files: List of paths to ``grid_activity_*.npy`` files to sample from.
        csv_path: Path to the results CSV; written to immediately on completion.

    Returns:
        Tuple of (repeat, ``"neuron"``, fraction, metric, detected, pd_counts_str).
    """
    rng = np.random.default_rng()

    path = rng.choice(cache_files)
    grid_activity = np.load(path)  # (n_neurons, n_timepoints)
    n_neurons = grid_activity.shape[0]

    n_keep = max(1, int(np.round(fraction * n_neurons)))
    idx = rng.choice(n_neurons, size=n_keep, replace=False)
    idx.sort()
    spikes_sub = grid_activity[idx, :].T  # (n_timepoints, n_keep)

    try:
        persistence = persistence_analysis_spikes(spikes_sub, maxdim=2, metric=metric)
        dgms = persistence["dgms"]
        counts = count_pd_outliers(dgms)
        detected = is_torus(dgms)
    except Exception as e:
        print(f"  [neuron] repeat={repeat} frac={fraction:.2f} metric={metric}: failed ({e})")
        counts = (-1, -1, -1)
        detected = False

    result = (repeat, "neuron", fraction, metric, detected, str(counts))
    _append_csv(csv_path, result)
    print(f"  [neuron] repeat={repeat} frac={fraction:.2f} metric={metric}: "
          f"counts={counts} -> {'TORUS' if detected else 'no'}")
    return result


def _sweep_time(
    repeat: int,
    fraction: float,
    metric: str,
    cache_files: list[str],
    csv_path: str,
) -> tuple[int, str, float, str, bool, str]:
    """Runs one time-sweep job.

    Randomly selects a cached grid activity, subsamples time points at the
    given fraction (keeping all neurons), computes persistent homology, and
    tests for torus topology. Writes the result to CSV immediately.

    Args:
        repeat: Repeat index.
        fraction: Fraction of time points to retain (0, 1].
        metric: Distance metric for persistent homology (e.g. ``"cosine"``).
        cache_files: List of paths to ``grid_activity_*.npy`` files to sample from.
        csv_path: Path to the results CSV; written to immediately on completion.

    Returns:
        Tuple of (repeat, ``"time"``, fraction, metric, detected, pd_counts_str).
    """
    rng = np.random.default_rng()

    path = rng.choice(cache_files)
    grid_activity = np.load(path)  # (n_neurons, n_timepoints)
    n_timepoints = grid_activity.shape[1]

    n_keep = max(1, int(np.round(fraction * n_timepoints)))
    idx = rng.choice(n_timepoints, size=n_keep, replace=False)
    idx.sort()
    spikes_sub = grid_activity[:, idx].T  # (n_keep, n_neurons)

    try:
        persistence = persistence_analysis_spikes(spikes_sub, maxdim=2, metric=metric)
        dgms = persistence["dgms"]
        counts = count_pd_outliers(dgms)
        detected = is_torus(dgms)
    except Exception as e:
        print(f"  [time]   repeat={repeat} frac={fraction:.2f} metric={metric}: failed ({e})")
        counts = (-1, -1, -1)
        detected = False

    result = (repeat, "time", fraction, metric, detected, str(counts))
    _append_csv(csv_path, result)
    print(f"  [time]   repeat={repeat} frac={fraction:.2f} metric={metric}: "
          f"counts={counts} -> {'TORUS' if detected else 'no'}")
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def sweep_subsample_torus_detection(
    neuron_fractions: list[float] | None = None,
    time_fractions: list[float] | None = None,
    metrics: list[str] | None = None,
    n_repeats: int = N_REPEATS,
    cache_dir: str = CACHE_DIR,
    save_dir: str = SAVE_DIR,
    n_workers: int = 4,
) -> str:
    """Sweeps neuron and time subsampling fractions independently.

    For each (fraction, metric, repeat) combination, a cached grid activity
    is selected at random and subsampled along the swept axis. Both sweeps
    run in the same parallel executor and results are written to a shared CSV.

    Args:
        neuron_fractions: Fractions of neurons to retain. Defaults to
            [0.25, 0.5, 0.75, 1.0].
        time_fractions: Fractions of time points to retain. Defaults to
            [0.25, 0.5, 0.75, 1.0].
        metrics: Distance metrics to test. Defaults to
            ``["cosine", "euclidean"]``.
        n_repeats: Number of random repeats per (fraction, metric) pair.
        cache_dir: Directory containing ``grid_activity_*.npy`` files.
        save_dir: Directory where the CSV and figures are saved.
        n_workers: Number of parallel worker processes. Keep low (2–4) to
            avoid OOM — each ripser job is memory-intensive.

    Returns:
        Path to the saved CSV file.
    """
    if neuron_fractions is None:
        neuron_fractions = DEFAULT_NEURON_FRACTIONS
    if time_fractions is None:
        time_fractions = DEFAULT_TIME_FRACTIONS
    if metrics is None:
        metrics = DEFAULT_METRICS

    cache_files = sorted(glob.glob(os.path.join(cache_dir, "grid_activity_*.npy")))
    if not cache_files:
        raise FileNotFoundError(f"No grid_activity_*.npy files found in {cache_dir}")
    print(f"Found {len(cache_files)} cached grid activity files.")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, CSV_NAME)
    _init_csv(csv_path)

    completed = _load_completed(csv_path)
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed.")

    neuron_tasks = [
        (r, nf, m)
        for r in range(n_repeats)
        for nf in neuron_fractions
        for m in metrics
        if (r, "neuron", nf, m) not in completed
    ]
    time_tasks = [
        (r, tf, m)
        for r in range(n_repeats)
        for tf in time_fractions
        for m in metrics
        if (r, "time", tf, m) not in completed
    ]

    n_neuron_total = n_repeats * len(neuron_fractions) * len(metrics)
    n_time_total = n_repeats * len(time_fractions) * len(metrics)
    print(f"Neuron sweep: {n_neuron_total} total ({len(neuron_tasks)} remaining)\n"
          f"Time sweep:   {n_time_total} total ({len(time_tasks)} remaining)")

    total_remaining = len(neuron_tasks) + len(time_tasks)
    done_count = 0

    with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for r, nf, m in neuron_tasks:
            futures[executor.submit(_sweep_neurons, r, nf, m, cache_files, csv_path)] = None
        for r, tf, m in time_tasks:
            futures[executor.submit(_sweep_time, r, tf, m, cache_files, csv_path)] = None

        for future in cf.as_completed(futures):
            future.result()
            done_count += 1
            if done_count % 10 == 0 or done_count == total_remaining:
                print(f"  [{done_count}/{total_remaining}] tasks completed")

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["repeat", "sweep", "fraction", "metric"], keep="last")
    df = df.sort_values(["sweep", "metric", "fraction", "repeat"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path} ({len(df)} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_detection_rates(csv_path: str, save_dir: str | None = None) -> None:
    """Plots torus detection rate vs subsampling fraction for neurons and time.

    Produces two line plots (one per sweep axis), each with one line per
    distance metric. Detection rate is averaged over repeats.

    Args:
        csv_path: Path to the results CSV produced by
            :func:`sweep_subsample_torus_detection`.
        save_dir: Directory to save the figures. Defaults to the CSV's directory.
    """
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)
    metrics = sorted(df["metric"].unique())

    sweep_labels = {
        "neuron": ("Neuron fraction retained", "neuron_sweep"),
        "time": ("Time fraction retained", "time_sweep"),
    }

    for sweep, (xlabel, fname_tag) in sweep_labels.items():
        sub = df[df["sweep"] == sweep]
        if sub.empty:
            continue

        fractions = sorted(sub["fraction"].unique())
        fig, ax = plt.subplots(figsize=(7, 4))

        for metric in metrics:
            msub = sub[sub["metric"] == metric]
            rates = [msub[msub["fraction"] == f]["detected"].mean() for f in fractions]
            ax.plot(fractions, rates, marker="o", label=metric)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Torus detection rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Torus detection — {sweep} subsampling — world_1holes")
        ax.legend()
        fig.tight_layout()

        fig_path = os.path.join(save_dir, f"subsample_torus_detection_{fname_tag}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved to {fig_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = sweep_subsample_torus_detection(
        neuron_fractions=DEFAULT_NEURON_FRACTIONS,
        time_fractions=DEFAULT_TIME_FRACTIONS,
        n_repeats=N_REPEATS,
    )
    plot_detection_rates(csv_path)
