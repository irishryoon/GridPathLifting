import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # type: ignore

sys.path.append(".")
from constants import GRID_FIELDS_PATH

def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def compute_delta(cir_coords: np.ndarray, alpha: float = 0.99, num_bins: int = 1000, plot: bool = False) -> float:
    """
    Old function that computes delta (used to be called "epsilon"). Delta  = 2 pi - epsilon.
    Compute the delta boundary for given circular coordinates.
    This function calculates the delta boundary by analyzing the disposition 
    of circular coordinates and fitting an exponential curve to the histogram 
    of the largest dispositions. The delta boundary is determined based on 
    the cumulative distribution function (CDF) of the fitted curve.
    Args:
        cir_coords (np.ndarray): Array of circular coordinates.
        alpha (float, optional): Threshold for the CDF to determine the delta boundary. Default is 0.99.
        num_bins (int, optional): Number of bins for the histogram. Default is 1000.
    Returns:
        float: The computed delta boundary.
    """
    fit_range = 1 / 3
    
    disposition = np.abs(cir_coords[1:] - cir_coords[:-1])
    large_disposition = np.max(disposition, axis=1)
    
    bin_edges = np.linspace(0, 2 * np.pi, num_bins+1)
    hist, bin_edges = np.histogram(large_disposition, bins=bin_edges)
    hist = hist[::-1][:-int(num_bins*fit_range)]
    bin_edges = bin_edges[:-int(num_bins*fit_range)]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # params, covariance = curve_fit(exponential, bin_centers, hist, p0=(1, 1, 1))
    
    # y_fitted = exponential(bin_centers, *params)
    # hist = hist / np.max(y_fitted)
    # y_fitted = y_fitted / np.max(y_fitted)
    # y_fitted[y_fitted < 0] = 0
    # y_fitted[y_fitted < 3e-4] = 0
    cdf = np.cumsum(hist) / np.sum(hist)
    hist = hist / np.max(hist)
    
    deri_delta = bin_centers[np.argmax(cdf > alpha)]
    # print(deri_delta)
    
    if plot:
        plt.figure()
        plt.xlim([-0.1, 2*np.pi*(1-fit_range) + 0.1])
        plt.ylim([0, 1.1])
        # plt.plot(bin_centers, y_fitted, 'r-', alpha=0.5)
        plt.plot(bin_centers, cdf, 'g-', alpha=0.5)
        plt.bar(bin_centers, hist, width=2 * np.pi / num_bins)
        plt.vlines(deri_delta, -0.5, 1.5, 'r', '--', alpha=0.5)
    
    return deri_delta + 1


def compute_epsilon(cir_coords, alpha=0.99, fit_range=1/3):
    """
    Percentile-based epsilon with high-displacement trimming.

    Parameters
    ----------
    cir_coords : np.ndarray
        Shape (T, 2) circular coordinates.
    alpha : float
        Quantile level in [0, 1], e.g. 0.99.
    fit_range : float
        Fraction of the low-displacement range to trim away.
        Example: 1/3 keeps the top 2/3 of displacement values.
    """
    # 1) Per-step displacement in each toroidal dimension
    disposition = np.abs(cir_coords[1:] - cir_coords[:-1])      # (T-1, 2)

    # 2) Largest displacement per time step
    large_disposition = np.max(disposition, axis=1)             # (T-1,)

    if large_disposition.size == 0:
        raise ValueError("Need at least 2 time points to compute epsilon.")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if not (0.0 <= fit_range < 1.0):
        raise ValueError("fit_range must be in [0, 1).")

    # 3) Trim lower tail by fit_range, keep upper (1 - fit_range) part
    cutoff = 2 * np.pi * fit_range
    kept = large_disposition[large_disposition >= cutoff]
    #print("cutoff:", cutoff, "kept size:", kept.size)

    if kept.size == 0:
        # Fallback (should be rare with valid fit_range)
        kept = large_disposition

    # 4) Alpha-quantile on trimmed distribution. 
    # Find the delta value at which P(x > delta) = alpha, 
    # which is the same as the (1-alpha) quantile.
    deri_delta = np.percentile(kept, 100 - alpha * 100.0)

    # 5) Match your existing return convention
    return deri_delta - 1

    
if __name__ == "__main__":
    with np.load(os.path.join(GRID_FIELDS_PATH, "world_2holes", "toroidal_coords_dreimac.npz")) as data:
        cir_coords = data["coords"]
    compute_epsilon(cir_coords)
    plt.show()