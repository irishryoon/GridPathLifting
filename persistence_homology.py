import os, sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ripser import ripser  # type: ignore
from persim import plot_diagrams  # type: ignore
import pickle
import scipy # type: ignore
from scipy.spatial.distance import pdist, squareform # type: ignore
from sklearn.decomposition import PCA # type: ignore
import sklearn.preprocessing # type: ignore

sys.path.append('.')
from Gardner_data_test.Gardner_utils import (
    sample_denoising,
    smooth_knn_dist,
    compute_membership_strengths,
    plot_barcode,
)
from constants import GRID_FIELDS_PATH, PLACE_FIELDS_PATH

def computeDistanceMatrix(
    spikes: np.ndarray,
    dim: int = 6,
    metric: str = "cosine",
    num_times: int = 5,
    active_times: int = 15000,
    k: int = 1000,
    n_points: int = 1200,
    nbs: int = 800,
):
    """
    Compute the distance matrix based on spike data.

    Parameters:
        spikes (np.ndarray): Array of spike data.
        dim (int): Number of dimensions for PCA.
        metric (str): Distance metric to use.
        num_times (int): Number of time intervals.
        active_times (int): Number of active time intervals.
        k (int): Number of nearest neighbors to consider.
        n_points (int): Number of points to sample.
        nbs (int): Number of nearest neighbors to compute.

    Returns:
        np.ndarray: The computed distance matrix.
    """
    num_neurons = len(spikes[0, :])
    times_cube = np.arange(0, len(spikes[:, 0]), num_times)
    movetimes = np.sort(np.argsort(np.sum(spikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]

    pca = PCA(n_components=dim)
    dim_red_spikes_move_scaled = pca.fit_transform(sklearn.preprocessing.scale(spikes[movetimes, :]))
    indstemp, dd, fs = sample_denoising(dim_red_spikes_move_scaled, k, n_points, 1, metric)
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp, :]

    X = squareform(pdist(dim_red_spikes_move_scaled, metric))
    knn_indices = np.argsort(X)[:, :nbs]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices]
    sigmas, rhos = smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = result + transpose - prod_matrix
    result.eliminate_zeros()
    d = -np.log(result.toarray())
    np.fill_diagonal(d, 0)

    return d

def persistence_analysis_spikes(spikes: np.ndarray, maxdim: int = 2, dim: int = 6):
    distance_matrix = computeDistanceMatrix(spikes, dim=dim)
    persistence = ripser(
        distance_matrix,
        maxdim=maxdim,
        coeff=47,
        do_cocycles=True,
        distance_matrix=True,
    )
    return persistence

def persistence_analysis(result_path: str, maxdim: int = 2, dim: int = 6):
    """
    Calculate persistence homology for a grid cell similation result.
    """
    spikes = pickle.load(open(result_path, "rb")).T
    persistence = persistence_analysis_spikes(spikes, maxdim=maxdim, dim=dim)
    
    return persistence

def trajectory_persistence_analysis(result_path: str, maxdim: int = 1):
    """
    Calculate persistence homology for trajectory coordinates.
    """
    trajectory = pickle.load(open(result_path, "rb"))
    trajectory = trajectory[::100, :]
    
    persistence = ripser(
        trajectory,
        maxdim=maxdim,
        coeff=47,
        do_cocycles=True,
        distance_matrix=False,
    )
    
    return persistence


if __name__ == "__main__":
    # Grid Cells
    # for i in range(3):
    #     result_path = os.path.join(GRID_FIELDS_PATH, f'world_{i}holes', 'simulation_result.pkl')
    #     persistence_analysis(result_path)

    # Place Cells
    # for i in range(3):
    #     result_path = os.path.join(PLACE_FIELDS_PATH, f'world_{i}holes', 'simulation_result.pkl')
    #     persistence_analysis(result_path, maxdim=1, dim=500)
    
    # Trajectory
    for i in range(3):
        trajectory_path = os.path.join(GRID_FIELDS_PATH, f'world_{i}holes', 'toroidal_coords_lifted.pkl')
        trajectory_persistence_analysis(trajectory_path)
    
    plt.show()