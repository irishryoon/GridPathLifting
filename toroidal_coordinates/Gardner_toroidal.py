import os, sys
import matplotlib.pyplot as plt
import numpy as np 
from ripser import ripser # type: ignore
from sklearn import preprocessing # type: ignore
from scipy.spatial.distance import pdist, squareform # type: ignore
from scipy.sparse import coo_matrix # type: ignore

sys.path.append('.')
from Gardner_data_test.Gardner_utils import *
from distance_computation import preprocess_Gardner, preprocess
from constants import GARDNER_DATA_PATH

def Gardner_persistence(sspikes: np.ndarray, maxdim = 1) -> dict:
    """
    Computes the persistent cohomology of the given spike train data using Gardner's method.
    
    Args:
        sspikes : np.ndarray
            A 2D numpy array where each row represents a time point and each column represents a neuron.
        maxdim : int, optional
            The maximum dimension of homology to compute. Default is 1.

    Returns:
        dict
            A dictionary containing the persistence diagrams and cocycles computed by the Ripser library.
    """
    dim = 6
    ph_classes = [0,1] # Decode the ith most persistent cohomology class
    num_circ = len(ph_classes)
    dec_tresh = 0.99
    metric = 'cosine'
    maxdim = maxdim
    coeff = 47
    active_times = 15000
    k = 1000
    num_times = 5
    n_points = 1200
    nbs = 800
    # sigma = 1500
            
    num_neurons = len(sspikes[0,:])
            
    times_cube = np.arange(0,len(sspikes[:,0]),num_times)
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube,:],1))[-active_times:])
    movetimes = times_cube[movetimes]

    dim_red_spikes_move_scaled,__,__ = pca(preprocessing.scale(sspikes[movetimes,:]), dim = dim)
    indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                        n_points, 1, metric)
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
    X = squareform(pdist(dim_red_spikes_move_scaled, metric))
    knn_indices = np.argsort(X)[:, :nbs]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    sigmas, rhos = smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    d = result.toarray()
    d = -np.log(d)
    np.fill_diagonal(d,0)

    persistence = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles= True, distance_matrix = True)    
    
    return persistence

def Gardner_coord(sspikes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the toroidal coordinates using the Gardner method.

    Args:
        sspikes (numpy.ndarray): Spike data matrix of shape (num_times, num_neurons).

    Returns:
        numpy.ndarray: The toroidal coordinates.
        numpy.ndarray: The corresponding times.
        numpy.ndarray: The centcosall matrix.
        numpy.ndarray: The centsinall matrix.
    """
    persistence = Gardner_persistence(sspikes)
    
    ############ Decode cocycles ################
    diagrams = persistence["dgms"] # the multiset describing the lives of the persistence classes
    cocycles = persistence["cocycles"][1] # the cocycle representatives for the 1-dim classes
    dists_land = persistence["dperm2all"] # the pairwise distance between the points 
    births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1-births1 # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len(indstemp)))
    threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]])*dec_tresh
    for c in ph_classes:
        cocycle = cocycles[iMax[-(c+1)]]
        coords1[c,:],inds = get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

    num_neurons = len(sspikes[0,:])
    centcosall = np.zeros((num_neurons, 2, n_points))
    centsinall = np.zeros((num_neurons, 2, n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp],:])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),spktemp)
        centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),spktemp)

    times = np.where(np.sum(sspikes>0, 1)>=1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times,:]
    dspk = dspk[times,:]

    a = np.zeros((len(sspikes[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

    cc = np.zeros((len(sspikes[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        cc[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

    mtot2 = np.sum(cc,2)
    mtot1 = np.sum(a,2)
    coords = np.arctan2(mtot2,mtot1)%(2*np.pi)
    coordsbox = coords.copy()
    times_box = times.copy()

    return coordsbox, times_box, centcosall, centsinall

if __name__ == '__main__':
    folder = GARDNER_DATA_PATH
    rat_name, mod_name, sess_name, day_name = ('R', '1', 'OF', 'day2')
    sspikes,xx,yy,__,__ = get_spikes(rat_name, mod_name, day_name, sess_name, bType = 'pure',
                                            bSmooth = True, bSpeed = True, folder = folder )
    # Gardner_coord(sspikes)
    persistence = Gardner_persistence(sspikes)
    import pdb; pdb.set_trace()
    plt.show()