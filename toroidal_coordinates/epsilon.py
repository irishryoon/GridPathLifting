import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # type: ignore

sys.path.append(".")
from constants import GRID_FIELDS_PATH

def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def compute_epsilon(cir_coords: np.ndarray, alpha: float = 0.99, num_bins: int = 1000, plot: bool = False) -> float:
    """
    Compute the epsilon boundary for given circular coordinates.
    This function calculates the epsilon boundary by analyzing the disposition 
    of circular coordinates and fitting an exponential curve to the histogram 
    of the largest dispositions. The epsilon boundary is determined based on 
    the cumulative distribution function (CDF) of the fitted curve.
    Args:
        cir_coords (np.ndarray): Array of circular coordinates.
        alpha (float, optional): Threshold for the CDF to determine the epsilon boundary. Default is 0.99.
        num_bins (int, optional): Number of bins for the histogram. Default is 1000.
    Returns:
        float: The computed epsilon boundary.
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
    
    deri_epsilon = bin_centers[np.argmax(cdf > alpha)] + 0.2
    # print(deri_epsilon)
    
    if plot:
        plt.figure()
        plt.xlim([-0.1, 2*np.pi*(1-fit_range) + 0.1])
        plt.ylim([0, 1.1])
        # plt.plot(bin_centers, y_fitted, 'r-', alpha=0.5)
        plt.plot(bin_centers, cdf, 'g-', alpha=0.5)
        plt.bar(bin_centers, hist, width=2 * np.pi / num_bins)
        plt.vlines(deri_epsilon, -0.5, 1.5, 'r', '--', alpha=0.5)
    
    return deri_epsilon
    
if __name__ == "__main__":
    with np.load(os.path.join(GRID_FIELDS_PATH, "world_2holes", "toroidal_coords_dreimac.npz")) as data:
        cir_coords = data["coords"]
    compute_epsilon(cir_coords)
    plt.show()