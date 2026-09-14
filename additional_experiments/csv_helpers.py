"""
Shared CSV helpers for incremental sweep result saving.

Usage in sweep scripts:
    import sys; sys.path.append(".")
    from revision.csv_helpers import init_csv, append_csv, load_completed
"""

import csv
import os
import filelock
import pandas as pd


def init_csv(csv_path: str, header: list[str]) -> None:
    """Create CSV with header if it doesn't exist; strip NaN mismatch rows if it does."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    else:
        df = pd.read_csv(csv_path)
        # Keep resumable behavior for sweep summary CSVs while allowing generic CSV headers.
        if "mismatch_score" in df.columns:
            df = df[df["mismatch_score"].notna()]
        df.to_csv(csv_path, index=False)


def append_csv(csv_path: str, lock_path: str, row: tuple) -> None:
    """Append a single result row, guarded by a file lock."""
    with filelock.FileLock(lock_path):
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


def load_completed(csv_path: str, key_cols: list[str]) -> set[tuple]:
    """
    Return set of key tuples for rows that already have a mismatch_score.

    Args:
        csv_path: Path to the CSV file.
        key_cols: Column names that together identify a unique job,
                  e.g. ["n_neurons", "repeat"] or ["downsample_interval", "repeat"].
                  Values are cast to the type inferred by pandas.
    """
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    has_scores = df["mismatch_score"].notna()
    return set(zip(*[df.loc[has_scores, col] for col in key_cols]))
