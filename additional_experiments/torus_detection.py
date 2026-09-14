"""
IQR-based torus detection from persistence diagrams.

A torus T² has Betti numbers (β₀, β₁, β₂) = (1, 2, 1):
  - β₀ = 1  : one connected component   → 1 outlier in PD0
  - β₁ = 2  : two independent loops     → 2 outliers in PD1
  - β₂ = 1  : one void                  → 1 outlier in PD2

For each homological dimension, we compute persistence lifetimes
(death - birth), then flag features whose lifetime exceeds the upper fence
Q3 + 3 * (Q3 - Q1) as "outliers" (i.e., significant features).
If the outlier counts across dimensions equal (1, 2, 1), we declare
reliable torus detection.

Example:
    from revision.torus_detection import is_torus, count_pd_outliers

    # dgms is persistence['dgms'] from ripser, a list of arrays with shape (n, 2)
    counts = count_pd_outliers(dgms)        # e.g. (1, 2, 1)
    detected = is_torus(dgms)               # True / False
"""

import numpy as np


def _lifetimes(dgm: np.ndarray) -> np.ndarray:
    """Returns finite persistence lifetimes for a single diagram.

    Args:
        dgm: Array of shape (n, 2) with columns [birth, death].

    Returns:
        1-D array of finite lifetimes (death - birth).
    """
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    finite = np.isfinite(deaths)
    return (deaths - births)[finite]


def _iqr_outlier_count(lifetimes: np.ndarray, iqr_factor: float = 3.0) -> int:
    """Counts values above the upper fence Q3 + iqr_factor * (Q3 - Q1).

    Falls back to counting values above the median when fewer than 4
    values are present and the IQR is undefined.

    Args:
        lifetimes: 1-D array of persistence lifetimes.
        iqr_factor: Multiplier applied to (Q3 - Q1). Default is 3.0.

    Returns:
        Number of lifetimes that exceed the upper fence.
    """
    if len(lifetimes) == 0:
        return 0
    if len(lifetimes) < 4:
        threshold = np.median(lifetimes)
        return int(np.sum(lifetimes > threshold))
    q1, q3 = np.percentile(lifetimes, [25, 75])
    threshold = q3 + iqr_factor * (q3 - q1)
    return int(np.sum(lifetimes > threshold))


def count_pd_outliers(
    dgms: list,
    dims: tuple[int, int, int] = (0, 1, 2),
    iqr_factor: float = 3.0,
) -> tuple[int, int, int]:
    """Counts IQR outliers in each persistence diagram dimension.

    For each requested dimension, persistence lifetimes are computed and
    features whose lifetime exceeds Q3 + iqr_factor * (Q3 - Q1) are
    counted as outliers (significant topological features).

    Args:
        dgms: Persistence diagrams as returned by ripser
            (``persistence['dgms']``). Each entry is an array of shape
            (n_features, 2) with columns [birth, death].
        dims: Homological dimensions to inspect. Defaults to (0, 1, 2).
        iqr_factor: Multiplier for (Q3 - Q1) when computing the upper
            fence. Defaults to 3.0.

    Returns:
        Tuple of outlier counts, one per dimension in ``dims``,
        e.g. ``(1, 2, 1)`` for a torus.
    """
    counts = []
    for d in dims:
        if d >= len(dgms) or len(dgms[d]) == 0:
            counts.append(0)
            continue
        lives = _lifetimes(dgms[d])
        counts.append(_iqr_outlier_count(lives, iqr_factor=iqr_factor))
    return tuple(counts)  # type: ignore[return-value]


def is_torus(
    dgms: list,
    expected: tuple[int, int, int] = (1, 2, 1),
    iqr_factor: float = 3.0,
) -> bool:
    """Returns True if the persistence diagrams are consistent with a torus T².

    The detection criterion is that the number of IQR outliers in PD0,
    PD1, and PD2 must equal the Betti numbers of T², i.e. (1, 2, 1).
    Outliers are defined as features with lifetime exceeding
    Q3 + iqr_factor * (Q3 - Q1).

    Args:
        dgms: Persistence diagrams (``persistence['dgms']`` from ripser).
            Must contain at least 3 entries for dimensions 0, 1, 2.
        expected: Expected outlier count per dimension. Defaults to
            ``(1, 2, 1)``, the Betti numbers of T².
        iqr_factor: Multiplier for (Q3 - Q1). Defaults to 3.0.

    Returns:
        True if outlier counts match ``expected``, False otherwise.
    """
    counts = count_pd_outliers(dgms, iqr_factor=iqr_factor)
    return counts == expected
