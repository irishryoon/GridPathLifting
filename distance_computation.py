import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy # type: ignore
from scipy.spatial.distance import pdist, squareform # type: ignore
from sklearn.decomposition import PCA # type: ignore
import sklearn.preprocessing # type: ignore

from itertools import combinations
from scipy.signal import correlate # type: ignore

from itertools import combinations
from scipy.signal import correlate

sys.path.append(".")
from trajectory import World # need for loading trajectories
from Gardner_data_test.Gardner_utils import (
    sample_denoising,
    smooth_knn_dist,
    compute_membership_strengths,
)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # otherwise, get latex error when visualizing plots

def computeDistanceMatrix_Gardner(
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
    Copied from Gardner et al 2022 "Toroidal Topology" paper. 

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

def preprocess_Gardner(
    spikes: np.ndarray,
    dim: int = 6,
    metric: str = "cosine",
    num_times: int = 5,
    active_times: int = 15000,
    k: int = 1000,
    n_points: int = 1200,
):
    """
    Perform the preprocessing steps of the "computeDistance_matrix" function.
    Parameters:
        spikes (np.ndarray): Array of spike data.
        dim (int): Number of dimensions for PCA.
        metric (str): Distance metric to use.
        num_times (int): Number of time intervals.
        active_times (int): Number of active time intervals.
        k (int): Number of nearest neighbors to consider.
        n_points (int): Number of points to sample.
    """
    num_neurons = len(spikes[0, :])

    # create a list of form [0, num_times, num_times * 2, num_times * 3, ...]
    # Allows user to select every "num_times" timebins
    times_cube = np.arange(0, len(spikes[:, 0]), num_times)


    # compute total population activity (by summing firing rates of the entire population) at every time
    population_activity = np.sum(spikes[times_cube,:], 1)

    # select "active_times" number of times (from the times_cube) with highest level population activities
    movetimes = np.sort(np.argsort(population_activity)[-active_times:])
    movetimes = times_cube[movetimes]

    # reduce popoulation vector dimensions
    pca = PCA(n_components=dim)
    dim_red_spikes_move_scaled = pca.fit_transform(sklearn.preprocessing.scale(spikes[movetimes, :]))

    # indstemp further decreases the number of times 
    indstemp, dd, fs = sample_denoising(dim_red_spikes_move_scaled, k, n_points, 1, metric)
    dim_red_spikes_move_scaled2 = dim_red_spikes_move_scaled[indstemp, :]

    return times_cube, population_activity, movetimes, indstemp, dim_red_spikes_move_scaled, dim_red_spikes_move_scaled2

def computeDistance_from_preprocessed_Gardner(
    preprocessed_rates: np.ndarray,
    metric: str = "cosine",
    nbs: int = 800,
):
    """
    Compute the distance matrix based on spike data.

    Parameters:
        preprocessed_rates (np.ndarray): Some preprocessed array. 
                                        Output of function 'preprocess_rates'
        metric (str): Distance metric to use.
        nbs (int): Number of nearest neighbors to compute.

    Returns:
        np.ndarray: The computed distance matrix.
    """
   
    X = squareform(pdist(preprocessed_rates, metric))
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

def preprocess(
    rates: np.ndarray, 
    num_times: int = 250
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:    
    """
    A simplified preprocessing of firing rates array. 
    Selects time bins at regular intervals and reduces the dimension of popoulation vectors. 

    Parameters:
        rates (np.ndarray): Array of firing rates. Each row corresponds to time and each column corresponds to a neuron.
        num_times (int): Select every "num_times" intervals

    Returns:
        np.ndarray: The selected time bins
        np.ndarray: The selected firing rates of the time bins
        np.ndarray: The reduced firing rates
    """

    # select every "num_times" times
    times_cube = np.arange(0, rates.shape[0], num_times)
    rates_small = rates[times_cube, :]

    # reduce popoulation vectors 
    dim = 12
    pca = PCA(n_components=dim)
    dim_red_rates = pca.fit_transform(sklearn.preprocessing.scale(rates_small))


    return times_cube, rates_small, dim_red_rates
    
    

def pair_similarity(raster: np.ndarray, neuron1: int, neuron2: int, limit_len: int) -> float:
    """
    Compute similarity between two neurons in a raster.
    
    Args:
        raster (ndarray): The raster containing spike data.
        neuron1 (int): The index of the first neuron.
        neuron2 (int): The index of the second neuron.
        limit_len (int): The length of the limit.
    
    Returns:
        float: The similarity score between the two neurons.
    """
    n_bins = raster.shape[1]
    
    spikes1 = raster[neuron1,:]
    spikes2 = raster[neuron2,:]
    norm_factor = np.sqrt(np.dot(spikes1, spikes1) * np.dot(spikes2, spikes2))
    correlation = correlate(spikes1, spikes2, mode = 'same')
    score =sum(correlation[n_bins//2-limit_len:n_bins//2+limit_len])/norm_factor
    return score

def compute_similarity(raster: np.ndarray, limit_len: int) -> list[float]:
    """
    Compute similarity among all neurons in a raster.
    
    Args:
        raster (numpy.ndarray): The raster containing neuron data.
        limit_len (int): The maximum length for computing similarity.
        
    Returns:
        list: A list of similarity scores between neuron pairs.
    """
    similarity = []
    n_neurons = raster.shape[0]
    for (i,j) in combinations(range(n_neurons),2):
        score = pair_similarity(raster, i, j, limit_len)
        similarity.append(score)
    
    return similarity