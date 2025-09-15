import itertools
from scipy.signal import butter, lfilter, hilbert
from scipy.cluster.hierarchy import dendrogram
from scipy import signal
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import binned_statistic_2d, pearsonr, binned_statistic
from scipy.sparse import coo_matrix
from scipy.interpolate import CubicSpline
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation, RotationSpline
import scipy.optimize as opt
from scipy.sparse.linalg import lsqr, lsmr
from scipy.linalg import eigh
import scipy.misc
import scipy.sparse as sparse
from scipy import stats
import scipy
import scipy.sparse.linalg
import scipy.io as sio
from scipy.special import factorial
from matplotlib import animation, cm, transforms, pyplot as plt, gridspec as grd, gridspec as gridspec
from matplotlib.collections import PathCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.image as mpimg
from sklearn.cluster import AgglomerativeClustering,DBSCAN,KMeans
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KDTree
import sklearn.decomposition as dimred
from mpl_toolkits.mplot3d import Axes3D, proj3d
import psutil
import numpy as np
import glob 
import pickle
import cv2
import os
import sys
import glob
import h5py
from ripser import Rips, ripser
import time
import numba
import functools
import shutil
import traceback
import pandas as pd
from datetime import datetime 
from gtda.diagrams import PairwiseDistance 
from gtda.homology import VietorisRipsPersistence
import os
from code import interact
from re import I
import subprocess
import numpy as np
import warnings
from persim import plot_diagrams
from numba import jit

#from geomtools import *
#from toroidalcoords import *


cs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
ks = np.array([[0,0], [1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
combs_all = {}
combs_all[0] = [[0, 0, 1], 
                [1, 0, 1],
                [0, 0, 3],
                [1, 0, 3],
                [0, 1, 1],
                [1, 1, 1],
                [0, 1, 3],
                [1, 1, 3],                
                [0, 2, 0],
                [0, 3, 0],
                [1, 2, 0],
                [1, 3, 0]]

combs_all[1] = [[0, 0, 2], 
                [1, 0, 4], 
                [0, 1, 2], 
                [1, 1, 4],
                [1, 0, 2],
                [0, 0, 4], 
                [1, 1, 2], 
                [0, 1, 4],                
                [0, 2, 0],
                [0, 3, 0],
                [1, 2, 0],
                [1, 3, 0]]

combs_all[2] = [[0, 2, 2],
                [1, 2, 2], 
                [0, 2, 4], 
                [1, 2, 4], 
                [0, 3, 2], 
                [1, 3, 2], 
                [0, 3, 4], 
                [1, 3, 4],                
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0]]

combs_all[3] = [[0, 2, 1], 
                [1, 2, 3], 
                [0, 3, 1], 
                [1, 3, 3],
                [1, 2, 1], 
                [0, 2, 3], 
                [1, 3, 1], 
                [0, 3, 3],
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0]]

def loadmat(filename):
    '''
    this function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

    
def normalize(M):
    return (M-np.min(M))/(np.max(M)-np.min(M))



def get_coord_distribution(coords_mod1, numbins = 50,epsilon = 0.1, metric = 'euclidean', startindex = -1,
                          bWrap = True):    
    coords = coords_mod1.copy()
    n = coords.shape[0]
    inds_orig = np.arange(n)
    if bWrap:
        coords = np.concatenate((coords, coords, coords, 
                                 coords, coords, coords, 
                                 coords, coords, coords))
        inds_orig = np.concatenate((inds_orig,inds_orig,inds_orig,
                                    inds_orig,inds_orig,inds_orig,
                                    inds_orig,inds_orig,inds_orig))
        coords[1*n:2*n,0] += 2*np.pi
        coords[2*n:3*n,1] += 2*np.pi
        coords[3*n:4*n,:] += 2*np.pi
        coords[4*n:5*n,0] -= 2*np.pi
        coords[5*n:6*n,1] -= 2*np.pi
        coords[6*n:7*n,:] -= 2*np.pi
        coords[7*n:8*n,0] -= 2*np.pi
        coords[7*n:8*n,1] += 2*np.pi
        coords[8*n:9*n,0] += 2*np.pi
        coords[8*n:9*n,1] -= 2*np.pi
        coordsrel = ~((coords[:,0]>2*np.pi + epsilon) | 
                     (coords[:,1]>2*np.pi + epsilon) |
                     (coords[:,0]<-epsilon) | 
                     (coords[:,1]<-epsilon) )
        inds_orig = inds_orig[coordsrel]
        coords = coords[coordsrel]
    n = coords.shape[0]
    inds = np.zeros((n, ), dtype=int)
    inds_label = [[] for i in range(n)]
    if epsilon > 0:            
        n = coords.shape[0]
        if startindex == -1:
            np.random.seed(0) 
            startindex = np.random.randint(n)
        i = startindex
        j = 1
        inds_res = np.arange(n, dtype=int)
        dists = np.zeros((n, ))
        while j < n+1:
            disttemp = (cdist(coords[i, :].reshape(1, -1), coords[:, :], metric=metric) - epsilon)[0]  
            inds_label[i] = inds_orig[np.where(disttemp<=0)[0]]
            dists[inds_res] = np.min(np.concatenate((dists[inds_res][:,np.newaxis], 
                                                     disttemp[inds_res,np.newaxis]),1),1)
            inds[i] = j
            inds_res = inds_res[disttemp[inds_res]>0]
            j = j+1
            if len(inds_res)>0:
                i = inds_res[np.argmax(dists[inds_res])]
            else:
                break
    else:
        inds = np.ones(range(np.shape(coords)[0]))
    inds = np.where(inds[:len(coords_mod1)])[0]
    inds_label = inds_label[:len(coords_mod1)]
    print(len(inds))
    return inds, inds_label

def get_phases(sspikes, coords_mod1, inds, inds_label):    
    dspk = sspikes.copy() - np.mean(sspikes,0)
    num_neurons = len(dspk[0,:])
    masscenters_1 = np.zeros((num_neurons, 2))
    for neurid in range(num_neurons):
        centcosall = np.zeros(2)
        centsinall = np.zeros(2)
        for i in inds:
            centcosall += np.mean(np.multiply(np.cos(coords_mod1[i:i+1, :].T),
                                              dspk[inds_label[i], neurid]),
                                  axis = 1)
            centsinall += np.mean(np.multiply(np.sin(coords_mod1[i:i+1, :].T),
                                              dspk[inds_label[i], neurid]),
                                  axis = 1)
        masscenters_1[neurid] = np.arctan2(centsinall,centcosall)%(2*np.pi)
    return masscenters_1

def get_phases_binned(sspikes, coords_bin, inds_label):    
    dspk = sspikes.copy() - np.mean(sspikes,0)
    cosbin = np.cos(coords_bin[:,:].T)
    sinbin = np.sin(coords_bin[:,:].T)
    return comp_mean(dspk, cosbin, sinbin, inds_label)

@numba.njit(fastmath=True, parallel = False)  # benchmarking `parallel=True` shows it to *decrease* performance
def comp_mean(dspk, cosbin, sinbin, inds_label):
    num_label = len(inds_label)
    num_neurons = len(dspk[0,:])
    masscenters_1 = np.zeros((num_neurons, 2))
    centcosall = np.zeros(2)
    centsinall = np.zeros(2)
    lens = np.zeros(num_label)
    for ii, i in enumerate(inds_label):
        lens[ii] = len(i)
    for neurid in range(num_neurons):
        for ii, i in enumerate(inds_label):
            if lens[ii]>0:
                cc1 = np.multiply(cosbin[:, ii:ii+1],dspk[i, neurid])/lens[ii]
                centcosall[0] += np.sum(cc1[0])
                centcosall[1] += np.sum(cc1[1])
                cc2 = np.multiply(sinbin[:, ii:ii+1],dspk[i, neurid])/lens[ii]
                centsinall[0] += np.sum(cc2[0])
                centsinall[1] += np.sum(cc2[1])
        masscenters_1[neurid] = np.arctan2(centsinall,centcosall)%(2*np.pi)
        centcosall[:] = 0
        centsinall[:] = 0
    return masscenters_1





def match_phases(coords1, sspikes, mc, times = [],numbins = 10, lentmp = 0, t = 0.1, nums = 1, bPlot = False, bSqr = False):
    if len(times) == 0:
        times = np.arange(len(coords1))
    num_neurons = len(sspikes[0,:])
    coords = coords1[times,:]
    
    if lentmp == 0:
        lentmp = len(coords)
    coords_tmp = np.random.rand(lentmp,2)*2*np.pi
    if bSqr:
        spk_sim = simulate_spk_sqr(coords_tmp, mc, t = t, nums = nums)
    else:          
        spk_sim = simulate_spk_hex(coords_tmp, mc, t = t, nums = nums)
    pcorr = np.zeros(num_neurons)

    for i in range(num_neurons):
        
        mtot1 = binned_statistic_2d(coords1[times,0], coords1[times,1], sspikes[times,i], bins = numbins)[0]
        nans = np.isnan(mtot1)
        mtot1[nans] = np.mean(mtot1[~nans])
        
        mtot2 = binned_statistic_2d(coords_tmp[:,0], coords_tmp[:,1], spk_sim[:,i], bins = numbins)[0]
        nans = np.isnan(mtot2)
        mtot2[nans] = np.mean(mtot2[~nans])
        mtot2 = gaussian_filter(mtot2, sigma = 1)                
        pcorr[i] = pearsonr(mtot1.flatten(), mtot2.flatten())[0]    
        if bPlot:
            if i<10:
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(mtot1.T, origin = 'lower', extent = [0,2*np.pi, 0, 2*np.pi])
                ax[1].imshow(mtot2.T, origin = 'lower', extent = [0,2*np.pi, 0, 2*np.pi])
                ax[0].scatter(mc[i,0],mc[i,1], c = 'r', s = 100, marker = 'X')
                ax[1].scatter(mc[i,0],mc[i,1], c = 'r', s = 100, marker = 'X')
                print(pcorr[i])
                plt.show()            
    return pcorr 

def match_phases3(coords1, sspikes, mc, times = [],numbins = 10, lentmp = 0, t = 0.1, nums = 1, 
                 bPlot = False, bSqr = False):
    if len(times) == 0:
        times = np.arange(len(coords1))
    num_neurons = len(sspikes[0,:])
    coords = coords1[times,:]
    
    if lentmp == 0:
        lentmp = len(coords)
    coords_tmp = np.random.rand(lentmp,2)*2*np.pi
    t0 = time.time()
    if bSqr:
        spk_sim = simulate_spk_sqr(coords_tmp, mc, t = t, nums = nums)
    else:          
        spk_sim = simulate_spk_hex(coords_tmp, mc, t = t, nums = nums)
    t1 = time.time()
    pcorr = np.zeros(num_neurons)
    spk = sspikes[times,:]
    for i in range(num_neurons):
        mtot1 = binned_statistic_2d(coords[:,0], coords[:,1], spk[:,i], bins = numbins)[0]
        nans = np.isnan(mtot1)
        mtot1[nans] = np.mean(mtot1[~nans])
        mtot1 = gaussian_filter(mtot1, sigma = 1)                
        
        mtot2 = binned_statistic_2d(coords_tmp[:,0], coords_tmp[:,1], spk_sim[:,i], bins = numbins)[0]
        nans = np.isnan(mtot2)
        mtot2[nans] = np.mean(mtot2[~nans])
        mtot2 = gaussian_filter(mtot2, sigma = 1)                
        pcorr[i] = pearsonr(mtot1.flatten(), mtot2.flatten())[0]    
    if bPlot:
        if i<10:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(mtot1.T, origin = 'lower', extent = [0,2*np.pi, 0, 2*np.pi])
            ax[1].imshow(mtot2.T, origin = 'lower', extent = [0,2*np.pi, 0, 2*np.pi])
            ax[0].scatter(mc[i,0],mc[i,1], c = 'r', s = 100, marker = 'X')
            ax[1].scatter(mc[i,0],mc[i,1], c = 'r', s = 100, marker = 'X')
            print(pcorr[i])
            plt.show()     
    t2 = time.time()
    return pcorr 

    
def get_phases_binned1(sspikes, coords_bin, inds_label):    
    dspk = sspikes.copy() - np.mean(sspikes,0)
    num_neurons = len(dspk[0,:])
    masscenters_1 = np.zeros((num_neurons, 2))
    for neurid in range(num_neurons):
        centcosall = np.zeros(2)
        centsinall = np.zeros(2)
        for ii, i in enumerate(inds_label):
            if len(i)>0:
                centcosall += np.mean(np.multiply(np.cos(coords_bin[ii:ii+1, :].T),
                                                  dspk[i, neurid]),
                                      axis = 1)
                centsinall += np.mean(np.multiply(np.sin(coords_bin[ii:ii+1, :].T),
                                                  dspk[i, neurid]),
                                     axis = 1)
        masscenters_1[neurid] = np.arctan2(centsinall,centcosall)%(2*np.pi)
    return masscenters_1


def get_coord_distribution_binned(coords_mod1, numbins = 50, overlap = 0., bWrap = True):    
    coords = coords_mod1.copy()
    n = coords.shape[0]
    inds_orig = np.arange(n)
    clin = np.linspace(0,2*np.pi, numbins +1)
    cdiff = (clin[1]-clin[0])/2
    clin = clin[1:]-cdiff
    cdiff *= 1 + overlap

    if bWrap:
        coords = np.concatenate((coords, coords, coords, 
                                 coords, coords, coords, 
                                 coords, coords, coords))
        inds_orig = np.concatenate((inds_orig,inds_orig,inds_orig,
                                    inds_orig,inds_orig,inds_orig,
                                    inds_orig,inds_orig,inds_orig))
        coords[1*n:2*n,0] += 2*np.pi
        coords[2*n:3*n,1] += 2*np.pi
        coords[3*n:4*n,:] += 2*np.pi
        coords[4*n:5*n,0] -= 2*np.pi
        coords[5*n:6*n,1] -= 2*np.pi
        coords[6*n:7*n,:] -= 2*np.pi
        coords[7*n:8*n,0] -= 2*np.pi
        coords[7*n:8*n,1] += 2*np.pi
        coords[8*n:9*n,0] += 2*np.pi
        coords[8*n:9*n,1] -= 2*np.pi
        coordsrel = ~((coords[:,0]>2*np.pi + cdiff) | 
                     (coords[:,1]>2*np.pi + cdiff) |
                     (coords[:,0]<-cdiff) | 
                     (coords[:,1]<-cdiff) )
        inds_orig = inds_orig[coordsrel]
        coords = coords[coordsrel]
        
    
    x,y = np.meshgrid(clin, clin)
    coords_bin = np.concatenate((x.flatten()[:,np.newaxis],
                                y.flatten()[:,np.newaxis]), 1)

    inds_label = [[] for i in range(numbins**2)]
    for ii, i in enumerate(coords_bin):    
        inds_label[ii] = inds_orig[np.where((np.abs(i[0]-coords[:,0])<=cdiff).astype(int) +
                                  (np.abs(i[1]-coords[:,1])<=cdiff).astype(int) == 2)[0]]    
#        inds_label[ii] = inds_orig[np.where((np.sqrt(np.square(i[0]-coords[:,0])+ 
#                                                     np.square(i[1]-coords[:,1]))<=cdiff))[0]]    
    
    return coords_bin, inds_label


@numba.njit(parallel=False, fastmath=True)
def radial_downsampling(data_in, epsilon = 0.1, startindex = -1):    
    n = data_in.shape[0]
    epsilon *= epsilon
    if startindex == -1:
        np.random.seed(0) 
        startindex = np.random.randint(n)
    i = startindex
    j = 1
    inds = np.zeros(n,)
    inds1 = np.arange(n,)
    dists = np.zeros(n, )
    dists[i] = -1
    while (len(inds1)>0):        
        i = inds1[np.argmin(dists[inds1])]
        disttemp = np.sum(np.square(data_in[i,:] - data_in[inds1, :]), 1)  - epsilon
        d1 = dists[inds1]
        dw = d1-disttemp<0
        d1[dw] = disttemp[dw]
        dists[inds1] = d1 
        inds[i] = 1
        inds1 = inds1[disttemp>0]
    inds = np.where(inds)[0]
    return inds

def radial_downsampling_metric(data_in, epsilon = 0.1,metric = 'euclidean', startindex = -1):    
    n = data_in.shape[0]
    epsilon *= epsilon
    if startindex == -1:
        np.random.seed(0) 
        startindex = np.random.randint(n)
    i = startindex
    j = 1
    inds = np.zeros(n,)
    inds1 = np.arange(n,)
    dists = np.zeros(n, )
    dists[i] = -1
    while (len(inds1)>0):        
        i = inds1[np.argmin(dists[inds1])]
        disttemp = cdist(data_in[i:i+1,:], data_in[inds1, :], metric = metric)[0,:]  - epsilon
        d1 = dists[inds1]
        dw = d1-disttemp<0
        d1[dw] = disttemp[dw]
        dists[inds1] = d1 
        inds[i] = 1
        inds1 = inds1[disttemp>0]
    inds = np.where(inds)[0]
    return inds




def radial_downsampling3(data_in, epsilon = 0.1, metric = 'euclidean', startindex = -1):    
    n = data_in.shape[0]
    np.random.seed(0) 
    if epsilon > 0:
        n = data_in.shape[0]
        if startindex == -1:
            startindex = np.random.randint(n)
        inds = rad_ds(n, data_in, startindex, epsilon**2)
    else:
        inds = np.ones(range(np.shape(data_in)[0]))
    inds = np.where(inds)[0]
    return inds

@numba.njit(parallel=False, fastmath=True)
def rad_ds(n, data_in, startindex, epsilon):
    i = startindex
    j = 1
    inds = np.zeros(n,)
    inds1 = np.arange(n,)
    dists = np.zeros(n, )
    dists[i] = -1
    while (len(inds1)>0):        
        i = inds1[np.argmin(dists[inds1])]
        disttemp = np.sum(np.square(data_in[i,:] - data_in[inds1, :]), 1)  - epsilon
        d1 = dists[inds1]
        dw = d1-disttemp<0
        d1[dw] = disttemp[dw]
        dists[inds1] = d1 
        inds[i] = 1
        inds1 = inds1[disttemp>0]
    return inds

def radial_downsampling2(data_in, epsilon = 0.1, metric = 'euclidean', startindex = -1):    
    n = data_in.shape[0]
    np.random.seed(0) 
    if epsilon > 0:
        n = data_in.shape[0]
        if startindex == -1:
            startindex = np.random.randint(n)
        i = startindex
        j = 1
        inds = np.zeros((n, ), dtype=int)
        inds1 = np.arange(n, dtype=int)
        dists = np.zeros((n, ))
        while j < n+1:
            disttemp = (cdist(data_in[i, :].reshape(1, -1), data_in[inds1, :], metric=metric) - epsilon)[0]                        
            dists[inds1] = np.max(np.concatenate((dists[inds1][:,np.newaxis], disttemp[:,np.newaxis]),1),1)
            inds[i] = j
            inds1 = inds1[disttemp>0]
            j = j+1
            if len(inds1)>0:
                i = inds1[np.argmin(dists[inds1])]
            else:
                break
    else:
        inds = np.ones(range(np.shape(data_in)[0]))
    inds = np.where(inds)[0]
    return inds




def run_GLM(mouse_sess, cmod, data_dir, 
            num_bins_all = [15], LAM_all = [0], 
            bTor = True, cv_folds = 3, 
            GoGaussian = False, bSess = False, files = [],
            files_dec = [], gain_score = 1, contrast_score = 100):

    spk1, posxx, indsnull, e1, coords_all, gain, contrast, postrial = load_glm_data(mouse_sess, cmod, data_dir, bTor, bSess, files)
    files_dec
    GLMscores = {}
    t0 = 0
    if len(files_dec) == 0:
        files_dec = files
    elif len(files_dec)<len(files):
        files_temp = []
        for fi in files:
            if fi in files_dec:
                files_temp.extend([fi])
            else:
                files_temp.extend([''])
        files_dec = files_temp

    for fi in files:
        times = np.arange(t0, t0+len(posxx[fi]))
        t0 += len(posxx[fi])
        if (fi.find('baseline')==-1) & (fi.find('gain')==-1) & (fi.find('contrast')==-1):
            continue
        poscurr = (gain[fi][postrial[fi]-1] == gain_score) & (contrast[fi][postrial[fi]-1] == contrast_score)
        times = times[poscurr]
        spk = spk1[fi][:,indsnull][:,e1][poscurr,:].copy()
        __, num_neurons = np.shape(spk)

        if bSess:
            GLMscores[fi] = np.zeros((num_neurons, len(files), len(num_bins_all), len(LAM_all)))
        else:
            GLMscores[fi] = np.zeros((num_neurons, len(num_bins_all), len(LAM_all)))
            
        for i1, num_bins in enumerate(num_bins_all):
            for i2, LAM in enumerate(LAM_all):            
                for n in np.arange(0, num_neurons, 1): 
                    if bTor:
                        if bSess:
                            for itfi2, fi2 in enumerate(files_dec):
                                if (fi2 != fi) & (len(fi2)>0):
                                    (__, __, GLMscores[fi][n, itfi2, i1, i2]) = glm(coords_all[fi + str(itfi2)][n][poscurr, :2].copy(), 
                                                                        spk[:,n], 
                                                                        num_bins, 
                                                                        GoGaussian, 
                                                                        cv_folds, LAM)
                        else:                            
                            (__, __, GLMscores[fi][n, i1, i2]) = glm(coords_all[n][times, :2], 
                                                                spk[:,n], 
                                                                num_bins, 
                                                                GoGaussian, 
                                                                cv_folds, LAM)
                    else:
                         (__, __, GLMscores[fi][n, i1, i2]) = glm(posxx[fi][poscurr, np.newaxis], 
                                                           spk[:,n], 
                                                           num_bins,
                                                           GoGaussian, 
                                                           cv_folds, LAM)
    return GLMscores



#@numba.njit(parallel=True, fastmath=True)
def firing_rate(spikeTimes, res = 100000, 
                dt_orig = 0.01,
                sigma = 30, min_time = None, max_time = None):
    """
    Compute firing rate matrix
    
    res = time resolution 1/res seconds
    dt_orig = bin size in seconds 
    sigma = smoothing width in bins
    
    """
    dt = int(dt_orig*res)
    if min_time<0:
        print('hai')
        min_time = np.floor(spikeTimes[0][0])*res-dt
    else:
        min_time*=res
    if not max_time:
        max_time = np.ceil(spikeTimes[0][-1])*res+dt
    else:
        max_time*=res

    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res
    print()
    sigma *= dt
    thresh = int(sigma)*6
    num_thresh = int(thresh/dt)
    num2_thresh = int(2*num_thresh)
    sig2 = 1/(2*(sigma/res)**2)
    
    ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
    kerwhere = np.arange(-num_thresh,num_thresh, dtype = int)*dt
    sspikes = np.zeros((1,len(spikeTimes)))
    sspikes = np.zeros((len(tt)+num2_thresh, len(spikeTimes)))    
    for n, spk in enumerate(spikeTimes):
        spk = spikeTimes[spk]
        spike_times = np.array(spk*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            sspikes[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
    sspikes = sspikes[num_thresh-1:-(num_thresh+1),:]
    sspikes *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
    return sspikes, tt




def get_coords_all(sspikes2, coords1, times_cube, indstemp, dim = 7, spk2 = [], 
                   bPred = False, bPCA = False, bScale = True, k = 30, metric = 'euclidean'):
    num_circ = len(coords1)
    if bScale:
        spkmean = np.mean(sspikes2[times_cube,:], axis = 0)
        spkstd = np.std(sspikes2[times_cube,:], axis = 0)
        if np.sum(spkstd==0)>0:
            spkstd[spkstd==0] = 1
            print('Null-neurons')
        spkscale = (sspikes2-spkmean)/spkstd
    else:
        spkscale = sspikes2.copy()
    dspk1 = spkscale.copy()
    if bPCA:
        __, e1, e2,__ = pca(spkscale[times_cube,:], dim = dim)
        dspk1 = np.dot(e1.T, spkscale.T).T    
        dspk1 /= np.sqrt(e2)    
        dspk = dspk1[times_cube[indstemp],:]
    else:
        dspk = dspk1[times_cube[indstemp],:]
    if len(spk2)>0:
        if bScale:
            dspk1 = preprocessing.scale(spk2,axis = 0)
#            dspk1 = (spk2-spkmean)/spkstd
        else:
            dspk1 = spk2.copy()
            
        if bPCA:
            dspk1 = np.dot(e1.T, dspk1.T).T    
            dspk1 /= np.sqrt(e2)    


    if bPred:
        coords_mod1 = np.zeros((len(dspk1), num_circ))
        for c in range(num_circ):
            coords_mod1[:,c] = predict_color(coords1[c,:], dspk1, dspk, 
                                         dist_measure=metric,  k = k)
    else:
        num_neurons = len(dspk[0,:])
        centcosall = np.zeros((num_neurons, num_circ, len(indstemp)))
        centsinall = np.zeros((num_neurons, num_circ, len(indstemp)))    
        for neurid in range(num_neurons):
            spktemp = dspk[:, neurid].copy()
            centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),spktemp)
            centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),spktemp)

        a = np.zeros((len(dspk1), num_circ, num_neurons))
        for n in range(num_neurons):
            a[:,:,n] = np.multiply(dspk1[:,n:n+1],np.sum(centcosall[n,:,:],1))

        c = np.zeros((len(dspk1), num_circ, num_neurons))
        for n in range(num_neurons):
            c[:,:,n] = np.multiply(dspk1[:,n:n+1],np.sum(centsinall[n,:,:],1))

        mtot2 = np.sum(c,2)
        mtot1 = np.sum(a,2)
        coords_mod1 = np.arctan2(mtot2,mtot1)%(2*np.pi)
    return coords_mod1


def predict_color2(circ_coord_sampled, data, sampled_data, dist_measure='euclidean', num_batch =20000, k = 10):
    num_tot = len(data)
    num_dim = len(circ_coord_sampled[0,:])
#    zero_spikes = np.where(np.sum(data,1) == 0)[0]
#    if len(zero_spikes):
#       data[zero_spikes,:] += 1e-10 
    circ_coord_tot = np.zeros((num_tot, num_dim))
    circ_coord_dist = np.zeros((num_tot, num_dim))
    j = -1
    for j in range(int(num_tot/num_batch)):
        dist_landmarks = cdist(data[j*num_batch:(j+1)*num_batch, :], sampled_data, metric = dist_measure)
        closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
        weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,-1]] for i in range(num_batch)])
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 
        circ_coord_tot[j*num_batch:(j+1)*num_batch] = [np.dot(circ_coord_sampled[closest_landmark[i,:]].T, weights[i,:]) for i in range(num_batch)]
    dist_landmarks = cdist(data[(j+1)*num_batch:, :], sampled_data, metric = dist_measure)
    closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
    lenrest = len(closest_landmark[:,0])
    weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,k-1:k]] for i in range(lenrest)])
    
    if np.shape(weights)[0] == 0:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1 
        weights /= np.sum(weights)
    else:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 
    circ_coord_tot[(j+1)*num_batch:] = [np.dot(circ_coord_sampled[closest_landmark[i,:]].T, weights[i,:]) for i in range(lenrest)]
    return circ_coord_tot



    


def get_ang_hist(c11all, c12all,xx, yy, numangsint = 101):
    binsx = np.linspace(np.min(xx)+ (np.max(xx) - np.min(xx))*0.0025,np.max(xx)- (np.max(xx) - np.min(xx))*0.0025, numangsint)
    binsy = np.linspace(np.min(yy)+ (np.max(yy) - np.min(yy))*0.0025,np.max(yy)- (np.max(yy) - np.min(yy))*0.0025, numangsint)

    nnans = ~np.isnan(c11all)

    mtot, x_edge, y_edge, circ = binned_statistic_2d(xx[nnans],yy[nnans], c11all[nnans], 
        statistic=circmean, bins=(binsx,binsy), range=None, expand_binnumbers=True)
    nans = np.isnan(mtot)
    sintot = np.sin(mtot)
    costot = np.cos(mtot)
    sintot[nans] = 0
    costot[nans] = 0
    sig = 1
#    sintot = smooth_tuning_map(sintot, numangsint, sig) #preprocessing.minmax_scale(gaussian_filter(sintot,1))*2-1#*np.pi
#    costot = smooth_tuning_map(costot, numangsint, sig)#preprocessing.minmax_scale(gaussian_filter(costot,1))*2-1#*np.pi
    #sintot = preprocessing.minmax_scale(gaussian_filter(sintot,1))*2-1#*np.pi
    #costot = preprocessing.minmax_scale(gaussian_filter(costot,1))*2-1#*np.pi
    sintot = gaussian_filter(sintot,sig)#*np.pi
    costot = gaussian_filter(costot,sig)#*np.pi

    mtot = np.arctan2(sintot, costot)
    mtot[nans] = np.nan


    nnans = ~np.isnan(c12all)
    mtot1, x_edge, y_edge, circ = binned_statistic_2d(xx[nnans],yy[nnans], c12all[nnans], 
        statistic=circmean, bins=(binsx,binsy), range=None, expand_binnumbers=True)
    nans = np.isnan(mtot1)
    sintot = np.sin(mtot1)
    costot = np.cos(mtot1)
    sintot[nans] = 0
    costot[nans] = 0
    #sintot = smooth_tuning_map(sintot, numangsint, sig) #preprocessing.minmax_scale(gaussian_filter(sintot,1))*2-1#*np.pi
    #costot = smooth_tuning_map(costot, numangsint, sig)#preprocessing.minmax_scale(gaussian_filter(costot,1))*2-1#*np.pi
    #sintot = preprocessing.minmax_scale(gaussian_filter(sintot,1))*2-1#*np.pi
    #costot = preprocessing.minmax_scale(gaussian_filter(costot,1))*2-1#*np.pi
    sintot = gaussian_filter(sintot,sig)#*np.pi
    costot = gaussian_filter(costot,sig)#*np.pi
    mtot1 = np.arctan2(sintot, costot)    
    mtot1[nans] = np.nan
    return mtot, mtot1, x_edge, y_edge


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



def preprocess(data,nth_bin = 10):
    track_start = 0
    track_end = 400
    dx=5
    dt=0.2
    every_nth_time_bin = nth_bin
    numposbins = np.floor((track_end-track_start)/dx)
    posx_edges = np.linspace(track_start,track_end,numposbins+1)
    posx_centers = 0.5 * posx_edges[0:-1] + 0.5*posx_edges[1::]
    data['posx_centers']=posx_centers
    data['posx_edges']=posx_edges
    posx=data['posx']
    post=data['post']
    trial = data['trial']
    sp =  data['sp']
    
    # resample post, posx, and trial according to desired dt
    post_resampled = post[0::every_nth_time_bin]
    posx_resampled=posx
    posx_resampled[posx_resampled<track_start]=track_start
    posx_resampled[posx_resampled>=track_end]=track_end-0.001 #now happening further down
    #posx_resampled = posx[0::every_nth_time_bin]
    trial_resampled = trial[0::every_nth_time_bin]

    # get cell ids of "good" units
    good_cells = sp['cids'][sp['cgs']==2]

    # time bins for position decoding
    numtimebins = len(post_resampled)
    post_edges = np.squeeze(np.linspace(min(post)-dt/2,max(post)+dt/2,numtimebins+1))
    post_centers = post_edges[range(0,len(post_edges)-1)]+dt/2

    # posx categories for position decoding (binned)
    posx_bin = np.digitize(posx_resampled,posx_edges)
    posx_bin = posx_bin[0::every_nth_time_bin]
    posx_resampled = posx_resampled[0::every_nth_time_bin]

    #speed
    speed = calcSpeed(data['posx'])
    speed_resampled = speed[0::every_nth_time_bin]

    # count spikes in each time bin for each cell
    spikecount = np.empty((len(good_cells),len(post_resampled),))
    spikecount[:] = np.nan
    for cell_idx in range(len(good_cells)):   
        spike_t = sp['st'][sp['clu']==good_cells[cell_idx]]
        spikecount[cell_idx,:] = np.histogram(spike_t,bins=post_edges)[0]
    data['spikecount']=np.transpose(spikecount)
    data['posx_bin']=posx_bin
    data['trial_resampled']=trial_resampled
    data['posx_resampled']=posx_resampled
    data['speed_resampled']=speed_resampled
    return data

def calcSpeed(posx):
    speed = np.diff(posx)/0.02
    speed = np.hstack((0,speed))
    speed[speed>150]=np.nan
    speed[speed<-5]=np.nan
    idx_v = np.flatnonzero(np.logical_not(np.isnan(speed)))
    idx_n = np.flatnonzero(np.isnan(speed))
    speed[idx_n]=np.interp(idx_n,idx_v,speed[~np.isnan(speed)])
    speed = gaussian_filter1d(speed,10)
    return speed

def _fast_occ(occupancy,trials,bins):

    for i,j in zip(trials,bins):
        if (j<0) or j>=occupancy.shape[0] or i>=occupancy.shape[1]:
            pass
        else:
            occupancy[j,i]+=1

def _fast_bin(counts, trials, bins, neurons):
    """
    Given coordinates of spikes, compile binned spike counts. Throw away
    spikes that are outside of tmin and tmax.
    Turns into a matrix neurons x bins x trials
    """
    for i, j, k in zip(trials, bins, neurons):
        if (j < 0) or (int(j) >= counts.shape[1]) or i>=counts.shape[2]:
            pass
        else:
            counts[k, int(j), i] += 1
            
class options:
    
    def __init__(self):
        self.speed_t=0.05;
        self.extract_win = [-2,3];
        self.aux_win = [-50,50];
        self.TimeBin = 0.02;
        self.time_bins =np.arange(-2,3,0.02);
        self.extract_win = [-2,3]
        self.speedSigma = 10;
        self.smoothSigma_time = 0.2; # in sec; for smoothing fr vs time
        self.smoothSigma_dist = 2; # in cm; for smoothing fr vs distance
        self.SpatialBin = 2;
        self.TrackStart = 0
        self.TrackEnd = 400
        self.SpeedCutof = 2
        self.stab_thresh = 0.5
        self.max_lag = 30
                
    @property            
    def time_vecs(self):
        return self.time_bins[0:-1]*0.5 + self.time_bins[1:]*0.5
    @property
    def xbinedges(self):
        return np.arange(self.TrackStart,self.TrackEnd+self.SpatialBin,self.SpatialBin)
    @property
    def xbincent(self):
        return self.xbinedges[0:-1]+self.SpatialBin/2


def calculateFiringRateMap(data,trials2extract=None,good_cells = None,ops=None):
    posx=np.mod(data['posx'],ops.TrackEnd)
    post=data['post']
    trial = data['trial'] 
    sp = data['sp']
    if good_cells is None:
        good_cells = sp['cids'][sp['cgs']==2]
    if trials2extract is None:
        trials2extract = np.arange(trial.min(),trial.max()+1)
    
    
    posx_bin = np.digitize(posx,ops.xbinedges)
    validSpikes = np.in1d(data['sp']['clu'],good_cells)
    spike_clu = data['sp']['clu'][validSpikes]
    (bla,spike_idx) = np.unique(spike_clu,return_inverse=True)
    spiketimes = np.digitize(data['sp']['st'][validSpikes],data['post'])
    spikelocations = posx_bin[spiketimes]-1 # to start at 0
    spiketrials = data['trial'][spiketimes] # to start at 0
    
    valid_trialsSpike = np.in1d(spiketrials,trials2extract)
    spiketimes = spiketimes[valid_trialsSpike]
    spikelocations = spikelocations[valid_trialsSpike]
    spiketrials = spiketrials[valid_trialsSpike]
    spike_idx=spike_idx[valid_trialsSpike] 
    
    valid_trials = np.in1d(trial,trials2extract)
    occupancy = np.zeros((len(ops.xbinedges)-1,len(trials2extract)),dtype = float)
    
    bintrials = trial[valid_trials]
    for i,j in enumerate(np.unique(bintrials)):
        bintrials[bintrials==j] = i
    
    _fast_occ(occupancy,bintrials,posx_bin[valid_trials]-1)
    occupancy *=ops.TimeBin
    
    n_cells = len(good_cells)
    shape = (n_cells, len(ops.xbinedges)-1, len(trials2extract))
    for i,j in enumerate(np.unique(spiketrials)):
        spiketrials[spiketrials==j] = i

    counts = np.zeros(shape, dtype=float)
    _fast_bin(counts,spiketrials,spikelocations,spike_idx)
    spMapN = np.zeros(counts.shape)
    stab =np.zeros(n_cells)
    for iC in range(n_cells):
        tmp = np.divide(counts[iC,:,:],occupancy)
        df = pd.DataFrame(tmp)
        df.interpolate(method='pchip', axis=0, limit=None, inplace=True)
        tmp = df.values
        tmp_f = gaussian_filter1d(tmp,ops.smoothSigma_dist, axis=0,mode='wrap')
        spMapN[iC]=tmp_f
        cc=np.corrcoef(np.transpose(tmp_f))
        

        stab[iC]=np.nanmean(cc[np.triu(np.full(cc.shape,True),1)])
    
    return counts,spMapN,stab



gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)

def read_numerical_file(path, data_type, list_type):
    '''
    Reads in a file consisting of UTF-8 encoded lists of numbers with single or 
    multiple observations per line.

    Parameters
    ----------
    path: str or Path object
        file to be read
    data_type: int or float
        data type of the observations in the file
    list_type: str
        'single'
            single observations per line
        'multiple'
            multiple observations per line
    
    Returns
    -------
    data_list: list
        Simple list of single values, or if 'multiple' data type then nested lists for each
        line in input file
    '''
    if data_type not in ('float', 'int'):
        raise ValueError('Must specify either \'float\' or \'int\' as data_type')
    if list_type not in ('single', 'multiple'):
        raise ValueError('list type must be \'single\' or \'multiple\'')
    fr = open(path, 'r')
    if data_type == 'int':
        d_type = int
    elif data_type == 'float':
        d_type = float
    
    if list_type == 'single':
        data_list = [d_type(line.rstrip()) for line in fr]
    elif list_type == 'multiple':
        data_list = [[d_type(y) for y in line.split()] for line in fr]
    fr.close()
    return data_list


def match_spikes_to_cells(cluster_file_path, timing_file_path, verbose=True):
    '''
    Uses .clu and .res files to associate spike times with putative cells.
    
    Parameters
    ----------
    cluster_file_path: str
        path to .clu file associated with a session and shank
    timing_file_path: str
        path to .res file associated with same session and shank
    
    Returns
    -------
    nCells: int or nan
        number of cells on a given shank,  nan if no cells present
    spike_times: list
        list of floats indicating times where a spike occurred for a cluster
    '''
    print(cluster_file_path)
    print(timing_file_path)

    tmp_clusters = read_numerical_file(cluster_file_path, 'int', 'single')
    tmp_spikes = read_numerical_file(timing_file_path, 'float', 'single')
#    print('clu',cluster_file_path, tmp_clusters)
#    print('clu',timing_file_path, tmp_spikes)
    
    if verbose:
        print( 'Cluster file:', cluster_file_path, 'Timing file ', timing_file_path)

    # First line in cluster file is number of cells (with 0 corresponding to
    # artifacts and 1 to noise)
    nClusters = tmp_clusters[0]
    cluster_ids = list(tmp_clusters[1:])
    if nClusters <= 2:  # ony clusters are 0 and 1; so no cells
        print( 'No cells found')
        nCells = 0
        spike_times = []
        return nCells, spike_times
    if np.max(cluster_ids) != (nClusters - 1):  
        print( 'Clusters listed at beginning of file do not agree')
        nCells = np.nan
        spike_times = []
        return nCells, spike_times

    # Now break this up in various cells
    spike_time_list = [[] for i in range(nClusters)]

    for i in range(len(cluster_ids)):
        spike_time_list[cluster_ids[i]].append(tmp_spikes[i])

    spike_times_incl_noise = [np.array(x) for x in spike_time_list]
    spike_times = spike_times_incl_noise[2:]
    nCells = nClusters - 2  # since 0/1 are noise; subtract from nCluster

    return nCells, spike_times

def gather_session_spike_info(params, verbose=True):
    '''Gather data from the downloaded files.
    For each session we have (a) State information (Wake, REM, SWS), 
    (b) Position and angle info
    and (c) Spike info.
    '''

    session = params['session']
    curr_data_path = params['data_path']
    file_tag = curr_data_path + session
    if verbose:
        print( 'Session: ', session)
        print( curr_data_path)
    # First store the times the animal was in each state in state_times
    state_file_base = file_tag + '.states.'
    state_names = ['Wake', 'REM', 'SWS']

    state_times = {st: read_numerical_file(state_file_base + st, 'float', 'multiple') 
        for st in state_names}

    # Store head direction in angle_list along with the corresponding times recorded.
    angle_list_orig = []#read_numerical_file(file_tag + '.ang', 'float', 'single')
    print(angle_list_orig)
    # When angle couldn't be sampled, these files have -1. But this could mess up 
    # averaging if we're careless, so replace it with NaNs.
    angle_list = np.array(angle_list_orig)
    angle_list[angle_list < -0.5] = np.nan

    # Tag angles with the times at which they were sampled.
    pos_sampling_rate = params['eeg_sampling_rate'] / 32.  # Hz
    angle_times = np.arange(len(angle_list)) / pos_sampling_rate

    # Spike times.
    # There is a .res and .clu file for each shank.
    # .res file contains spike times. .clu contains putative cell identities.
    # Files are of the form session.clu.dd and session.res.dd, where dd is the shank number.
    # The first line in each clu file is the number of clusters for that shank, with clusters
    # 0 and 1 indicating artefacts and noise. So ignore those clusters and start with cluster number
    # 2 as cell number 0.
    # The length of the cluster file should be 1 entry more than the length of the spike 
    # timing files (because the first line is the number of clusters).
    # Note that there are occasionally extra .clu files, from previous rounds of sorting
    # so we want to exclude them. 
    print(curr_data_path + session)
    nShanks = len(
        [fname for fname in glob.glob(curr_data_path + session + '.clu.*') #if re.match(
    #    file_tag + '.clu.\\d+$', fname)
        ])
    print(nShanks)
    if verbose:
        print( 'Number of shanks =', nShanks)
    nCells_per_shank = np.zeros(nShanks)
    
    # Store spike times as dict where keys are (shank, cell) and the values
    # are spike times for that cell. We index shanks starting at 0 but these files 
    # are stored starting from 1, so make sure to subtract 1 where relevant.
    spike_times = {}

    for pyth_shank_idx in range(nShanks):
        data_shank_idx = pyth_shank_idx + 1
        print( '\nAnalyzing shank', data_shank_idx)
        cluster_file = file_tag + '.clu.' + str(data_shank_idx)
        timing_file = file_tag + '.res.' + str(data_shank_idx)

        nCells_per_shank[pyth_shank_idx], tmp_spike_list = match_spikes_to_cells(
            cluster_file, timing_file)
        
        nCells_current = nCells_per_shank[pyth_shank_idx]
        if nCells_current > 0:  # if the shank has actual cells
            # loops over each cell
            for curr_cell in range(int(nCells_current)):
                # Multiply by spike sampling interval to get into units of time
                # For reference this is 1.0/(20e3)
                spike_times[(pyth_shank_idx, curr_cell)] = params[
                    'spike_sampling_interval'] * tmp_spike_list[curr_cell]

    # Check for shanks where the number of clusters doesn't equal number of listed cells
    wrong_count_shanks = np.sum(np.isnan(nCells_per_shank))
    if wrong_count_shanks and verbose:
        print( '\nThe number of shanks with wrong number of cells listed is', wrong_count_shanks)

    # Gather up stuff and return it
    data_to_return = {'session' : session, 'state_times' : state_times, 'angle_list' : 
        np.array(angle_list), 'pos_sampling_rate': pos_sampling_rate, 'angle_times' : 
        np.array(angle_times), 'nShanks': nShanks, 'nCells': nCells_per_shank, 
        'spike_times': spike_times, 'cells_with_weird_clustering': wrong_count_shanks}

    return data_to_return

def load_file_from_pattern(file_pattern):
    file_matches = glob.glob(file_pattern)
    if len(file_matches)>1:
        print('Multiple matches. Using the first one')
    if len(file_matches)==0:
        print('No file found')
        return
    fname = file_matches[0]
    data = load_pickle_file(fname)
    return data, fname

def load_pickle_file(filename):
    fr = open(filename, 'rb')
    data = pickle.load(fr)#, encoding='Latin-1',  errors='ignore')
    fr.close()
    return data

def save_pickle_file(data, filename):
    fw = open(filename, 'wb')
    pickle.dump(data, fw)
    fw.close()
    return 1

def return_dir(input_dir):
    '''Makes the directory input_dir if it doesn't exist.
    Return input_dir.'''
    if not os.path.exists(input_dir):
        print('Making %s'%input_dir)
        os.makedirs(input_dir)
    return input_dir


def load_data_all(mouse_sess, cmod, data_dir, files, data_dir1 = 'giocomo_analyses_250722'):
    f = np.load(data_dir1 + '/' + mouse_sess + '_mods.npz',allow_pickle = True)
    ind = f['ind']
    f.close()
    e1 = ind == cmod
    print('')
    print(mouse_sess, 'ind ' + str(cmod), sum(e1))
    
    ff = glob.glob(data_dir1 + '/' + mouse_sess + '_data.npz')

    if len(ff) == 0:
        (sspikes1, speed1, sspk1, spk1, good_cells, indsnull, speed, 
           pos_trial,data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx) =  get_data(files)
        np.savez(data_dir + '/' + mouse_sess + '_data', sspikes1 = sspikes1, speed1 = speed1, spk1 = spk1, good_cells = good_cells, indsnull = indsnull, 
                 speed = speed, pos_trial = pos_trial, data_pos = data_pos, posx = posx, post = post, posxx = posxx, postt = postt, postrial = postrial, gain = gain, contrast = contrast, lickt = lickt, lickx = lickx)
    else:
        f = np.load(ff[0], allow_pickle = True)
        sspikes1 = f['sspikes1']
        speed1 = f['speed1']
        spk1 = f['spk1'][()]
        good_cells = f['good_cells']
        indsnull = f['indsnull']
        speed = f['speed'][()]
#        pos_trial = f['pos_trial'][()]
        data_pos = f['data_pos'][()]
        posx = f['posx'][()]
        post = f['post'][()]
        posxx = f['posxx'][()]
        postt = f['postt'][()]
        postrial = f['postrial'][()]
        gain = f['gain'][()]
        contrast = f['contrast'][()]
        lickt = f['lickt'][()]
        lickx = f['lickx'][()]
        f.close()
    t0 = 0
    f = np.load(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '.npz', allow_pickle = True)
    dgms = f['dgms_all'][()]
    coords_ds = f['coords_ds_all'][()]
    indstemp = f['indstemp_all'][()]
    f.close()
    coords1 = {}
    data_ensemble = {}
    times_cube = {}
    for it, fi in enumerate(files):
        times = np.arange(t0,t0+len(posxx[fi]))
        data_ensemble[fi] = np.sqrt(sspikes1[times,:][:,ind ==cmod])
        t0 += len(posxx[fi])
        times_cube[fi] = np.where(speed[fi]>10)[0]
        coords1[fi] = get_coords_all(data_ensemble[fi], coords_ds[it], times_cube[fi], indstemp[it], 
            dim = 7, bPred = False, bPCA = True)
    return (files, data_ensemble, speed1, spk1, good_cells, indsnull, speed, 
           data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx, dgms, 
            coords_ds, indstemp, times_cube, coords1, e1)


def load_data(session, data_dir = 'Data'):
    ff = glob.glob(data_dir + '/' + session + '_25.npz')
    if len(ff)>0:
        f = np.load(ff[0], allow_pickle = True)
        ccg = f['ccg'] 
        spikes_bin = f['spikes_bin'][()]
        hdd = f['hdd']
        xx = f['xx']
        yy = f['yy']    
        ind = f['ind']
        f.close()
    else:
        data_path = data_dir + '/' + session + '/'
        f = np.load(data_path + 'data.npz', allow_pickle = True)
        data = f['data'][()]
        f.close()        
        angle_list_orig = np.array(read_numerical_file(data_dir + '/' + session + '/' + session + '.ang', 'float', 'multiple'))
        angle_list = np.array(angle_list_orig[:,1])
        angle_list[angle_list < -0.5] = np.nan
        angle_times  =angle_list_orig[:,0]
        data['angle_list'] = angle_list
        data['angle_times'] = angle_times
        spike_times =data['spike_times']
        hd = data['angle_list']
        samp_rate = data['pos_sampling_rate']

        area_info = load_pickle_file(data_dir + '/area_shank_info.p')
        relevant_shanks = area_info[session]['ADn']
        spike_times1 = {}
        spikes_bin = {}
        
        it = 0
        for spk in spike_times:
            numspk = sum((spike_times[spk]>=data['state_times']['Wake'][0][0]) & 
                   (spike_times[spk]<=data['state_times']['Wake'][0][1]))
            lensess = data['state_times']['Wake'][0][1]-data['state_times']['Wake'][0][0]
            fr = numspk/lensess
    #        if spk[0] in relevant_shanks:
            if (fr>=0.05) | (fr<10):
                spike_times1[it] = spike_times[spk]
                it += 1
            else:
                print('fr ', fr)
            
        for brain_state in ['Wake', 'REM', 'SWS']:
            t_curr =  data['state_times'][brain_state]
            res = 100000
            if brain_state == 'SWS':
                dt = 200000
                sigma = 5000
            else:
                dt = 25000
                sigma = 25000
            thresh = 250000
            num_thresh = int(thresh/dt)
            num2_thresh = int(2*num_thresh)
            sig2 = 1/(2*(sigma/res)**2)
            ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
            kerwhere = np.arange(-num_thresh,num_thresh)*dt

            spikes_bin_tmp = np.zeros((1,len(spike_times1)))
            for ttmp in t_curr:
                min_time = ttmp[0]*res
                max_time = ttmp[-1]*res
                tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)

                spikes_temp = np.zeros((len(tt)+num2_thresh, len(spike_times1)))
                for n, spk in enumerate(spike_times1):
                    spk = spike_times1[spk]
                    spikes = np.array(spk*res-min_time, dtype = int)
                    spikes = spikes[(spikes < (max_time-min_time)) & (spikes > 0)]
                    spikes_mod = dt-spikes%dt
                    spikes = np.array(spikes/dt, int)
                    for m, j in enumerate(spikes):
                        spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
                spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
                spikes_bin_tmp = np.concatenate((spikes_bin_tmp, spikes_temp),0)
            spikes_bin[brain_state] = spikes_bin_tmp[1:,:]
            spikes_bin[brain_state] *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
            if brain_state == 'Wake':
                pos_list_orig = np.array(read_numerical_file(data_dir + '/'+ session + '/' + session + '.pos', 'float', 'multiple'))
                t = pos_list_orig[:,0]
                x = pos_list_orig[:,1]
                y = pos_list_orig[:,2]
                tt/=res
                nnans = ~np.isnan(x)
                xspline = CubicSpline(t[nnans], x[nnans]) 
                nnans = ~np.isnan(y)
                yspline = CubicSpline(t[nnans], y[nnans]) 
                nnans = ~np.isnan(hd)
                hdsplinecos = CubicSpline(t[nnans], np.cos(hd[nnans]))
                hdsplinesin = CubicSpline(t[nnans], np.sin(hd[nnans]))
                xx = xspline(tt)
                yy = yspline(tt)
                hdd = np.arctan2(hdsplinesin(tt), hdsplinecos(tt))%(2*np.pi)   

        ccg = get_cross(session, data, spike_times1, data_dir = data_dir, spike_times = [], files = ['Wake', 'REM', 'SWS'])


def get_cross_corr(sspikes, lencorr = 30, bNorm = False):
    """
    Compute cross correlation across 'lencorr' time lags between columns of 'sspikes'    
    """
    lenspk,num_neurons = np.shape(sspikes)
    crosscorrs = np.zeros((num_neurons, num_neurons, lencorr))
    norm_spk = np.ones((num_neurons))
    if bNorm:
        for i in range(num_neurons):
            norm_spk[i] = np.sum(np.square(sspikes[:,i]))
        for i in range(num_neurons):
            spk_tmp0 = np.concatenate((sspikes[:,i].copy(), np.zeros(lencorr)))
            spk_tmp = np.array([np.roll(spk_tmp0, t1)[:lenspk] for t1 in range(lencorr)])
            crosscorrs[i,:,:] = np.dot(spk_tmp, sspikes).T/(norm_spk[i]*norm_spk[:, np.newaxis])
    else:
        for i in range(num_neurons):
            spk_tmp0 = np.concatenate((sspikes[:,i].copy(), np.zeros(lencorr)))
            spk_tmp = np.array([np.roll(spk_tmp0, t1)[:lenspk] for t1 in range(lencorr)])
            crosscorrs[i,:,:] = np.dot(spk_tmp, sspikes).T   
    return crosscorrs

#    ind = get_ind(session, ccg, data_dir = data_dir, files = ['Wake', 'REM', 'SWS'], nbs = 0.6)
#    np.savez(data_dir + '/' + session + '_25', ind = ind, ccg = ccg, spikes_bin = spikes_bin, hdd = hdd, xx = xx, yy = yy)
#    return ind, ccg, spikes_bin, hdd, xx, yy


def get_crossmat(crosscorr_train, lenspk, lencorr = -1):
    num_neurons = len(crosscorr_train)
    crosscorrs = np.zeros((num_neurons,num_neurons))
    num_neurons = len(crosscorr_train[:,0,0])
    lenorig = int(len(crosscorr_train[0,0,:])/2)
    if lencorr==-1:
        lencorr = lenorig
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            c = crosscorr_train[i,j,lenorig-lencorr:lenorig+lencorr+1].copy()*lenspk
#            print(c)

#            c = crosscorr_train[i,j,:].copy()*lenspk
            if np.min(c)>0:
                crosscorrs[i,j] +=  1-np.exp(-np.square(np.min(c)/np.max(c)))
                #crosscorrs[i,j] +=  1/(np.max(c)-np.min(c))#np.mean(np.exp(-np.square(c)))#-np.min(c))#np.exp(-np.sum(c))#-np.min(c)
            crosscorrs[j,i] = crosscorrs[i,j]
    print(np.unique(crosscorrs))
    crosscorrs[np.isnan(crosscorrs)] = 1
    crosscorrs[np.isinf(crosscorrs)] = 1
    return crosscorrs


def calculateFiringRate(data,good_cells=None,t_edges = None):
    if good_cells is None:
        good_cells = data['sp']['cids'][data['sp']['cgs']==2]
    if t_edges is None:
        dt= 0.2;
        t_edges = np.arange(0,data['sp']['st'].max()+dt,dt)
    else:
        dt = np.mean(np.diff(t_edges))
    # count spikes in each time bin for each cell
    
    spikecount = np.full((len(good_cells),len(t_edges)-1),np.nan)
    
    for cell_idx in range(len(good_cells)):   
        spike_t = data['sp']['st'][data['sp']['clu']==good_cells[cell_idx]]
        spikecount[cell_idx,:] = np.histogram(spike_t,bins=t_edges)[0]

      
    spikecount = np.hstack((spikecount,np.zeros((spikecount.shape[0],1))))  
    spikerate = spikecount/dt
    spikes = np.transpose(spikerate)
    X = gaussian_filter1d(spikes, 2, axis=0)
    return spikes,X,t_edges


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2



def get_sspikes(sspk1, speed, sp, fnames, ind, mod_ind, sig = 6):
    inds_torus = np.where((ind==mod_ind))[0]
    num_neurons = len(inds_torus)
    ################### smooth spikes ####################
    sspikes1 = np.zeros((1,num_neurons))
    for fi in fnames:
        sspikes = np.sqrt(sspk1[fi][:, inds_torus])
        sspikes = gaussian_filter1d(sspikes[:, :],sig, axis = 0)
        sspikes = sspikes[speed[fi]>sp, :]
        sspikes1 = np.concatenate((sspikes1, sspikes),0)            
    sspikes1 = sspikes1[1:,:]
    return sspikes1



def addonein(iii, jjj, pairs):
    if(len(pairs)==0):
        return [iii, jjj]
    for i in range(len(pairs)):
        if ((iii==pairs[i][0] and jjj==pairs[i][1]) or 
            (iii==pairs[i][1] and jjj==pairs[i][0])):
            return
    return [iii, jjj]


def get1dspatialpriorpairs(nn, periodicprior):
    pairs = []
    for i in range(nn):
        for j in range(nn):
            p = None
            if periodicprior and (abs(i-j)==nn-1):
                p = addonein(i, j, pairs)
            elif (abs(i-j)==1):
                p = addonein(i, j, pairs)
            if p:
                pairs.append(p) 
                    
    pairs = np.array(pairs)
    sortedpairs = []
    for i in range(nn):
        kpairs = []
        for j in range(len(pairs[:,0])):
            ii, jj = pairs[j,:]
            if(i == ii or i == jj):
                kpairs.append(pairs[j,:])
        kpairs = np.array(kpairs)
        sortedpairs.append(kpairs)
    return(pairs, sortedpairs)

def get2dspatialpriorpairs(nn, periodicprior):
    pairs = []
    for i in range(nn):
        for j in range(nn):
            for m in range(nn):
                for n in range(nn):
                    if periodicprior:
                        if ((abs(i-m)==nn-1 and (j-n)==0) or 
                            (abs(i-m)==0 and abs(j-n)==nn-1) or
                            (abs(i-m)==nn-1 and abs(j-n)==nn-1) or  
                            (abs(i-m)==1 and abs(j-n)==nn-1) or
                            (abs(i-m)==nn-1 and abs(j-n)==1)):
                            p = addonein(i*nn+j, m*nn+n, pairs)
                            if p:
                                pairs.append(p)  
                            continue
                    if ((abs(i-m)==1 and (j-n)==0) or  
                        (abs(i-m)==0 and abs(j-n)==1) or 
                        (abs(i-m)==1 and abs(j-n)==1)):
                        p = addonein(i*nn+j, m*nn+n, pairs)
                        if p:
                            pairs.append(p) 
                    
    pairs = np.array(pairs)
    sortedpairs = []
    for i in range(nn*nn):
        kpairs = []
        for j in range(len(pairs[:,0])):
            ii, jj = pairs[j,:]
            if(i == ii or i == jj):
                kpairs.append(pairs[j,:])
        kpairs = np.array(kpairs)
        sortedpairs.append(kpairs)
    return(pairs, sortedpairs)

def getpoissonsaturatedloglike(S):
    Sguys = S[S>0.00000000000001] ## note this is same equation as below with Sguy = exp(H)
    return np.sum(np.ravel(Sguys*np.log(Sguys) - Sguys)) - np.sum(np.ravel(np.log(factorial(S))))

def getbernoullisaturatedloglike(S):
    Sguys = S[S>0.00000000000001] ## note this is same equation as below with Sguy = exp(H)
    return np.sum(np.ravel(Sguys*np.log(Sguys) - np.log(1+ Sguys))) 


def singleiter(vals, covariates, GoGaussian, GoBernoulli, finthechat, BC, y, LAM, sortedpairs):
    P = np.ravel(vals)
    H = np.dot(P,covariates)
    num_cov, T = np.shape(covariates)

    if GoGaussian:
        guyH = H
    elif GoBernoulli:
        expH = np.exp(H) 
        guyH = expH / (1.+ expH)   
    else:
        guyH = np.exp(H)

    dP = np.zeros(num_cov)
    for j in range(num_cov):
        pp = 0.
        if LAM > 0:
            if(len(np.ravel(sortedpairs))>0):
                kpairs = sortedpairs[j]
                if(len(np.ravel(kpairs))>0):
                    for k in range(len(kpairs[:,0])):
                        ii, jj = kpairs[k,:]
                        if(j == ii):
                            pp += LAM*(P[ii] - P[jj])
                        if(j == jj):
                            pp += -1.*LAM*(P[ii] - P[jj])
                            
        dP[j] = BC[j] - np.mean(guyH*covariates[j,:]) - pp/T

    if GoGaussian:
        L = -np.sum( (y-guyH)**2 ) 
    elif GoBernoulli:
        L = np.sum(np.ravel(y*H - np.log(1. + expH)))
    else:
        L = np.sum(np.ravel(y*H - guyH)) - finthechat 
    return -L, -dP


def simplegradientdescent(vals, numiters, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs):
    P = vals
    for i in range(0,numiters,1):
        L, dvals = singleiter(vals, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs)
        P -= 0.8 * dvals
    return P, L

def fitmodel(y, covariates, GoGaussian = False,  GoBernoulli = False, LAM = 0, sortedpairs = []):
    num_cov = np.shape(covariates)[0]
    T = len(y)
    BC = np.zeros(num_cov)
    for j in range(num_cov):
        BC[j] = np.mean(y * covariates[j,:])
    if GoGaussian:
        finthechat = 0
    elif GoBernoulli:
        finthechat = 0
    else:
        finthechat = np.sum(np.ravel(np.log(factorial(y))))

    vals, Lmod = simplegradientdescent(np.zeros(num_cov), 2, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs)
    res = opt.minimize(singleiter, vals, (covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs), 
        method='L-BFGS-B', jac = True, options={'ftol' : 1e-5, 'disp': False})
    vals = res.x + 0.
    vals, Lmod = simplegradientdescent(vals, 2, covariates, GoGaussian,  GoBernoulli, finthechat, BC, y, LAM, sortedpairs)
    return vals


def preprocess_dataX(Xin, num_bins):
    Xin = np.transpose(Xin)
    num_dim, num_times = np.shape(Xin)

    tmp = np.linspace(-0.001, 1.001, num_bins+1)
    if num_dim == 1: 
        dig = (np.digitize(np.array(Xin), tmp)-1)
    elif num_dim == 2:
        dig = (np.digitize(np.array(Xin[0,:]), tmp)-1)*num_bins
        dig += (np.digitize(np.array(Xin[1,:]), tmp)-1)
   
    X = np.zeros((num_times, np.power(num_bins,num_dim)))
    X[range(num_times), dig] = 1
    return np.transpose(X)

def glm(xxss, ys, num_bins, GoGaussian,  GoBernoulli, cv_folds, LAM = 0, periodicprior = False):
    T, dim = np.shape(xxss)
    tmp = np.floor(T/cv_folds)
    
    xxss = preprocessing.minmax_scale(xxss,axis =0)
    xvalscores = np.zeros(cv_folds)
    P = np.zeros((np.power(num_bins, dim), cv_folds))
    LL = np.zeros(T)
    yt = np.zeros(T)
    tmp = np.floor(T/cv_folds)
    if(LAM==0):
        sortedpairs = []
    else:
        if dim == 1:
            pairs, sortedpairs = get1dspatialpriorpairs(num_bins, periodicprior)
        elif dim == 2:
            pairs, sortedpairs = get2dspatialpriorpairs(num_bins, periodicprior)
    for i in range(cv_folds):
        fg = np.ones(T)
        if cv_folds == 1:
            fg = fg==1
            nonfg = fg
        else:
            if(i<cv_folds):
                fg[int(tmp*i):int(tmp*(i+1))] = 0
            else:
                fg[-(int(tmp)):] = 0
            fg = fg==1
            nonfg = ~fg
        X_space = preprocess_dataX(xxss[fg,:], num_bins)

        P[:, i] = fitmodel(ys[fg], X_space, GoGaussian,  GoBernoulli, LAM, sortedpairs)
        X_test = preprocess_dataX(xxss[nonfg,:], num_bins)
        H = np.dot(P[:, i], X_test)
        if(GoGaussian):
            yt[nonfg] = H
            LL[nonfg] = -np.sum( (ys[nonfg]-yt[nonfg])**2 )             
        elif(GoBernoulli):
            expH = np.exp(H)
            yt[nonfg] = np.log(1. + expH)
            LL[nonfg] = np.sum(np.ravel(ys[nonfg]*H - np.log(1. + expH)))
        else:
            expH = np.exp(H)
            yt[nonfg] = expH
            finthechat = (np.ravel(np.log(factorial(ys[nonfg]))))
            LL[nonfg] = (np.ravel(ys[nonfg]*H - expH)) - finthechat
    if GoGaussian:
        leastsq = np.sum( (ys-yt)**2)
        ym = np.mean(ys)
        expl_deviance =  (1. - leastsq/np.sum((ys-ym)**2))
    else:
        LLnull = np.zeros(T)
        P_null = np.zeros((1, cv_folds))
        for i in range(cv_folds):
            fg = np.ones(T)
            if cv_folds == 1:
                fg = fg==1
                nonfg = fg
            else:
                if(i<cv_folds):
                    fg[int(tmp*i):int(tmp*(i+1))] = 0
                else:
                    fg[-(int(tmp)):] = 0
                fg = fg==1
                nonfg = ~fg
            
            X_space = np.transpose(np.ones((sum(fg),1)))
            X_test = np.transpose(np.ones((sum(nonfg),1)))
            P_null[:, i] = fitmodel(ys[fg], X_space, GoGaussian, GoBernoulli)
            H = np.dot(P_null[:, i], X_test)
            expH = np.exp(H)
            if GoBernoulli:
                LLnull[nonfg] = np.sum(np.ravel(ys[nonfg]*H - np.log(1. + expH)))
            else:
                finthechat = (np.ravel(np.log(factorial(ys[nonfg]))))
                LLnull[nonfg] = (np.ravel(ys[nonfg]*H - expH)) - finthechat        
        if GoBernoulli:
            LS = getbernoullisaturatedloglike(ys[~np.isinf(LL)]) 
        else:        
            LS = getpoissonsaturatedloglike(ys[~np.isinf(LL)]) 
        expl_deviance = 1 - (np.sum(LS) - np.sum(LL[~np.isinf(LL)]))/(np.sum(LS) - np.sum(LLnull[~np.isinf(LL)]))
    return P, yt, LL, expl_deviance

def load_glm_data(mouse_sess, cmod, data_dir, bTor, bSess, files):
    data_dir1 = 'giocomo_analyses_250722'
    f = np.load(data_dir1 + '/' + mouse_sess + '_mods.npz',allow_pickle = True)
    ind = f['ind']
    f.close()
    e1 = ind == cmod
    print('')
    print(mouse_sess, 'ind ' + str(cmod), sum(e1))
    
    ff = glob.glob(data_dir1 + '/' + mouse_sess + '_data.npz')
    coords_all = {}

    if len(ff) == 0:
        (sspikes1, speed1, sspk1, spk1, good_cells, indsnull, speed, 
           pos_trial,data_pos, posx, post, posxx, postt, 
           postrial, gain, contrast, lickt, lickx) =  get_data(files)
        np.savez(data_dir + '/' + mouse_sess + '_data', sspikes1 = sspikes1, speed1 = speed1, spk1 = spk1, good_cells = good_cells, indsnull = indsnull, 
                 speed = speed, pos_trial = pos_trial, data_pos = data_pos, posx = posx, post = post, posxx = posxx, postt = postt, postrial = postrial, gain = gain, contrast = contrast, lickt = lickt, lickx = lickx)
    else:
        f = np.load(ff[0], allow_pickle = True)
        sspikes1 = f['sspikes1']
        speed1 = f['speed1']
        speed = f['speed'][()]
        spk1 = f['spk1'][()]
        indsnull = f['indsnull']
        posxx = f['posxx'][()]
        postrial = f['postrial'][()]
        gain = f['gain'][()]
        contrast = f['contrast'][()]

        f.close()
    if bTor:
        if bSess:
            coords_f = glob.glob(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all_sess.npz')
            if len(coords_f)>0:
                f = np.load(coords_f[0], allow_pickle = True)
                coords_all = f['coords_all'][()]
                f.close()
            else:
                f = np.load(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '.npz', allow_pickle = True)
                coords_ds = f['coords_ds_all'][()]
                indstemp = f['indstemp_all'][()]
                f.close()
                coords1 = {}
                data_ensemble = {}
                times_cube = {}
                t0 = 0
                for it, fi in enumerate(files):
                    times = np.arange(t0,t0+len(posxx[fi]))
                    data_ensemble[fi] = np.sqrt(sspikes1[times,:][:,ind ==cmod])
                    t0 += len(posxx[fi])
                    times_cube[fi] = np.where(speed[fi]>10)[0]
                    num_neurons = len(data_ensemble[fi][0,:])
                for fi in files:
                    for it2, fi2 in enumerate(files):
                        if fi2 != fi:
                            coords_all[fi + str(it2)] = {}
                            for nn in range(num_neurons):
                                inds = np.ones(num_neurons, dtype = bool)
                                inds[nn] = False
                                coords_all[fi + str(it2)][nn] = get_coords_all(data_ensemble[fi2][:,inds], 
                                                                                coords_ds[it2], 
                                                                                times_cube[fi2], 
                                                                                indstemp[it2], 
                                                                                spk2 = data_ensemble[fi][:,inds],  
                                                                                bPred = False, bPCA = False)

                np.savez(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all_sess.npz', coords_all = coords_all)
        else:   
            coords_f = glob.glob(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all.npz')
            if len(coords_f)>0:
                f = np.load(coords_f[0], allow_pickle = True)
                coords_all = f['coords_all'][()]
                f.close()
            else:
                data_ensemble = np.sqrt(sspikes1[:,ind ==cmod])
                f = np.load(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '.npz', allow_pickle = True)
                coords_ds = f['coords_ds_all'][()][0]
                indstemp = f['indstemp_all'][()][0]
                f.close()
                times_cube = np.where(speed1>10)[0]
                num_neurons = len(data_ensemble[0,:])
                for nn in range(num_neurons):
                    inds = np.ones(num_neurons, dtype = bool)
                    inds[nn] = False
                    coords_all[nn] = get_coords_all(data_ensemble[:,inds], coords_ds, 
                                                    times_cube, indstemp, dim = sum(inds), 
                                                    bPred = False, bPCA = False)
                np.savez(data_dir + '/' + mouse_sess + '_ind' + str(cmod) + '_coords_all.npz', coords_all = coords_all)
    return spk1, posxx, indsnull, e1, coords_all, gain, contrast, postrial



def compute_toroidal_coords(inds_mod, 
                            ind, #indices of clustering 
                            sspk1, #spike train
                            speed,
                            fnames, # folders
                            dim1 = 15, # PCA dimensions to plot
                            k = 1000, # Number of neighbours for "fuzzy downsampling"
                            nbs = 1000, # Number of neighbours in final fuzzy metric computation for homology computation 
                            maxdim = 1, # Number of homology dimensions
                            coeff = 47, # Homology coefficients, random prime number chosen, relevant if one suspects "torsion" group in homology
                            hom_dims = [0, 1],
                            dec_tresh = 0.99, # ratio of cocycle life  determining the "size"/scale of Vietoris-rips complex higher percentage = more edges/simplices
                            ph_classes = [0,1], # compute circular coordinates for top ph_classes cocycles
                            sig = 6, # smoothing width in bins
                            dim = 6, # PCA-dimensions to keep
                            num_times = 10, # temporal downsampling distance (bins)
                            active_times = 12000, # number of points to keep based on highest summed activity
                            n_points = 1600, # Number of points to fuzzy downsample to 
                            sp = 2.5):
    num_circ = len(ph_classes)
    coords_all = {}
    for i in inds_mod:        
        sspikes1 = get_sspikes(sspk1, speed, sp, fnames, ind, i, sig)

        ################### Downsample 1 ####################
        times_cube = np.arange(0,len(sspikes1[:,0]),num_times)
        movetimes = np.sort(np.argsort(np.sum(sspikes1[times_cube,:],1))[-active_times:])
        movetimes = times_cube[movetimes]

        ################### Dimension reduce ####################
        dim_red_spikes_move_scaled,__,e1 = pca(preprocessing.scale(sspikes1[movetimes,:]), dim = dim1)
        plt.figure()
        plt.plot(e1) 
        paras = str(sp)[0] + '_' + str(sig) + '_' + str(dim) + '_' + str(num_times) + '_' + str(active_times) 
        dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[:,:dim]

        ################### Downsample 2 ####################
        indstemp = []
        paras = str(sp)[0] + '_' + str(sig) + '_' + str(dim) + '_' + str(num_times) + '_' + str(active_times) + '_' + str(n_points) 

        if len(indstemp) == 0:
            indstemp,dd,fs  = sample_denoising(dim_red_spikes_move_scaled,  k, 
                                                n_points, metric = 'cosine')
            dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
        else:
            indstemp = indstemp[:n_points]
            dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[:n_points,:]
        ################### Compute distance matrix ####################
        X = squareform(pdist(dim_red_spikes_move_scaled, 'cosine'))
        knn_indices = np.argsort(X)[:, :nbs]
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        sigmas, rhos = smooth_knn_dist(knn_dists, nbs, 
                                       local_connectivity=0) # It's possible to force connectivity between neighbours 
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

        ################### Compute persistence ####################
        thresh = np.max(d[~np.isinf(d)])
        rips_real = ripser(d, maxdim=maxdim, coeff=coeff, 
                           do_cocycles=True, distance_matrix = True, thresh = thresh)
        plt.figure()
        plot_diagrams(
            rips_real['dgms'],
            plot_only=np.arange(1+1),
            lifetime = True)

        ################### Decode coordinates ####################
        # ! Calculation of toroidal coordinates
        diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
        cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
        dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
        births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
        deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
        deaths1[np.isinf(deaths1)] = 0
        lives1 = deaths1-births1 # the lifetime for the 1-dim classes
        iMax = np.argsort(lives1)
        coords1 = np.zeros((num_circ, len(dim_red_spikes_move_scaled[:,0])))
        for j,c in enumerate(ph_classes):
            cocycle = cocycles[iMax[-(c+1)]]
            threshold = births1[iMax[-(c+1)]] + (deaths1[iMax[-(c+1)]] - births1[iMax[-(c+1)]])*dec_tresh
            coordstemp,inds = get_coords(cocycle, threshold, len(dim_red_spikes_move_scaled[:,0]), d, coeff)
            coords1[j,inds] = coordstemp
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(dim_red_spikes_move_scaled[:,0],dim_red_spikes_move_scaled[:,1],dim_red_spikes_move_scaled[:,2], c = np.cos(2*np.pi*coords1[j,:]))
        num_neurons = len(sspikes1[0,:])
        centcosall = np.zeros((num_neurons, num_circ, len(indstemp)))
        centsinall = np.zeros((num_neurons, num_circ, len(indstemp)))
        dspk = preprocessing.scale(sspikes1[movetimes[indstemp],:])
        for neurid in range(num_neurons):
            spktemp = dspk[:, neurid].copy()
            centcosall[neurid,:,:] = np.multiply(np.cos(coords1[:, :]*2*np.pi),spktemp)
            centsinall[neurid,:,:] = np.multiply(np.sin(coords1[:, :]*2*np.pi),spktemp)
        plot_times = np.arange(0,len(sspikes1[:,0]),1)
        dspk1 = preprocessing.scale(sspikes1[plot_times,:])
        a = np.zeros((len(plot_times), num_circ, num_neurons))
        for n in range(num_neurons):
            a[:,:,n] = np.multiply(dspk1[:,n:n+1],np.sum(centcosall[n,:,:],1))

        c = np.zeros((len(plot_times), num_circ, num_neurons))
        for n in range(num_neurons):
            c[:,:,n] = np.multiply(dspk1[:,n:n+1],np.sum(centsinall[n,:,:],1))
        mtot2 = np.sum(c,2)
        mtot1 = np.sum(a,2)
        coords_mod1 = np.arctan2(mtot2,mtot1)%(2*np.pi)
        
        coords_mod2 = np.zeros_like(coords_mod1[:5000,:])
        coords_mod2[:,0] = predict_color(coords1[0,:], preprocessing.scale(sspikes1[:5000,:]), preprocessing.scale(sspikes1[movetimes[indstemp],:]), 
                                     dist_measure='cosine',  k = 30)
        coords_mod2[:,1] = predict_color(coords1[1,:], preprocessing.scale(sspikes1[:5000,:]), preprocessing.scale(sspikes1[movetimes[indstemp],:]), 
                                         dist_measure='cosine',  k = 30)
        plt.figure()
        plt.plot(coords_mod1[:5000,0])
        plt.plot(coords_mod2[:5000,0])
        plt.figure()
        plt.plot(coords_mod1[:5000,1])
        plt.plot(coords_mod2[:5000,1])

        coords_all[i] = coords_mod1.copy()
        plt.show()
    return coords_all



def predict_color(circ_coord_sampled, data, sampled_data, dist_measure='euclidean', num_batch =20000, k = 10):
    num_tot = len(data)
#    zero_spikes = np.where(np.sum(data,1) == 0)[0]
#    if len(zero_spikes):
#       data[zero_spikes,:] += 1e-10 
    circ_coord_tot = np.zeros(num_tot)
    circ_coord_dist = np.zeros(num_tot)
    circ_coord_tmp = circ_coord_sampled*2*np.pi
    j = -1
    for j in range(int(num_tot/num_batch)):
        dist_landmarks = cdist(data[j*num_batch:(j+1)*num_batch, :], sampled_data, metric = dist_measure)
        closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
        weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,-1]] for i in range(num_batch)])
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 
        
        sincirc = [np.dot(np.sin(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
        coscirc = [np.dot(np.cos(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(num_batch)]
        circ_coord_tot[j*num_batch:(j+1)*num_batch] = np.arctan2(sincirc, coscirc)%(2*np.pi)
    
    dist_landmarks = cdist(data[(j+1)*num_batch:, :], sampled_data, metric = dist_measure)
    closest_landmark = np.argsort(dist_landmarks, 1)[:,:k]
    lenrest = len(closest_landmark[:,0])
    weights = np.array([1-dist_landmarks[i,closest_landmark[i,:]]/dist_landmarks[i,closest_landmark[i,k-1:k]] for i in range(lenrest)])
    if np.shape(weights)[0] == 0:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1 
        weights /= np.sum(weights)
    else:
        nans = np.where(np.sum(weights,1)==0)[0]
        if len(nans)>0:
            weights[nans,:] = 1
        weights /= np.sum(weights, 1)[:,np.newaxis] 
    sincirc = [np.dot(np.sin(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(lenrest)]
    coscirc = [np.dot(np.cos(circ_coord_tmp[closest_landmark[i,:]]), weights[i,:]) for i in range(lenrest)]
    circ_coord_tot[(j+1)*num_batch:] = np.arctan2(sincirc, coscirc)%(2*np.pi)
    return circ_coord_tot


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_traj_dist(data, sp, sp_sess, coords_smooth_sess, c, 
                      contrast = 100, gain = 1, numbins = 500, bCenter = False, bFillNans = True ):
    post = data['post'][sp_sess>sp]
    posx = data['posx'][sp_sess>sp]
    coords = coords_smooth_sess[c]
    traj_mean = np.zeros((numbins,4))
    trial_range = np.where((data['trial_contrast']== contrast) & ((data['trial_gain'] == gain)))[0]+1     
    valid_trialsSpike1 = sp_sess>sp
    spiketrials = data['trial'][valid_trialsSpike1]
    traj_all = np.zeros((len(trial_range), numbins, 2))
    for i, trial in enumerate(trial_range):
        trial_range1 = np.array([trial])
        valid_trialsSpike = np.in1d(spiketrials,trial_range1) 
        posxx = np.digitize(posx[valid_trialsSpike], np.linspace(0,np.max(posx)+0.001,numbins))-1
        coords_trial = coords[valid_trialsSpike,:].copy()
        if bCenter:
            coords_trial -= coords_trial[0,:]
            coords_trial += (np.pi, np.pi)
            coords_trial = coords_trial%(2*np.pi)
        traj_temp = binned_statistic(posxx, coords_trial[:,0], 
                                          statistic=circmean, bins=np.linspace(0,numbins,numbins+1))[0]
        traj_all[i,:,0] = traj_temp.copy()
        traj_temp = binned_statistic(posxx, coords_trial[:,1], 
                                          statistic=circmean, bins=np.linspace(0,numbins,numbins+1))[0]
        traj_all[i,:,1] = traj_temp.copy()

    if bFillNans:
        for i in range(len(traj_all[:,0,0])):
            nans = np.where(np.isnan(traj_all[i, :,0]))[0]
            while len(nans)>0:
                traj_all[i,nans,:] = traj_all[i,nans-1,:]  
                nans = np.where(np.isnan(traj_all[i, :,0]))[0]
    traj_dist = np.zeros((len(trial_range),len(trial_range)))
    for i in np.arange(len(trial_range)):
        trajtemp1 = traj_all[i,:,:]
        nans1 = np.isnan(trajtemp1[:,0]) | np.isnan(trajtemp1[:,1])
        for j in np.arange(i+1, len(trial_range)):
            trajtemp2 = traj_all[j,:,:]
            nans2 = np.isnan(trajtemp2[:,0]) | np.isnan(trajtemp2[:,1])
            nansboth = nans1 | nans2
            traj_dist[i,j] = circdist(trajtemp1[~nansboth,:], trajtemp2[~nansboth,:])/np.sum(~nansboth)
            traj_dist[j,i] = traj_dist[i,j]
    return traj_dist, traj_all, trial_range

def compute_traj_dist1(traj_curr):
    thresh = 0.1
    dtrajs = {}
    ntrials, lentrials,__ = traj_curr.shape 
    currtimes = np.arange(lentrials)
    for i in np.arange(ntrials):
        currtimes1 = np.where(~np.isnan(traj_curr[i,:,0]))[0]
        cs11 = CubicSpline(currtimes1, traj_curr[i,currtimes1,:])
        dcs11 = cs11.derivative(1)
        angular_rate1 = np.arctan2(dcs11(currtimes)[:,1],dcs11(currtimes)[:,0])
        cs12 = CubicSpline(currtimes, np.cos(angular_rate1)).derivative(1)
        cs13 = CubicSpline(currtimes, np.sin(angular_rate1)).derivative(1)
        dtrajs[i] = np.concatenate((cs12(currtimes)[:,np.newaxis], cs13(currtimes)[:,np.newaxis]),1)
        dtrajs[i][dtrajs[i]>thresh]  = thresh
        dtrajs[i][dtrajs[i]<-thresh]  = -thresh
        dtrajs[i] = gaussian_filter1d(dtrajs[i], 10,axis = 0)
        dtrajs[i] = preprocessing.scale(dtrajs[i])
        
    for i in range(ntrials):
        nans = np.where(np.isnan(traj_curr[i, :,0]))[0]
        while len(nans)>0:
            traj_curr[i,nans,:] = traj_curr[i,nans-1,:]  
            nans = np.where(np.isnan(traj_curr[i, :,0]))[0]
    angconst = np.sqrt(np.square(2)+ np.square(2))
    traj_dist = np.zeros((ntrials,ntrials))
    for i in np.arange(ntrials):
        trajtemp1 = traj_curr[i,:,:]
        dang1 = np.arctan2(trajtemp1[-1, 0]-trajtemp1[0, 0],
                                         trajtemp1[-1, 1]-trajtemp1[0, 1])
        ttemp1 = dtrajs[i]
        #ttemp1 -= np.mean(ttemp1)
        Ltemp = np.sqrt(np.sum(np.square(ttemp1),0))
        for j in np.arange(i+1, ntrials):
            trajtemp2 = traj_curr[j,:,:]
#            traj_dist[i,j] = np.sqrt(np.sum(np.square((trajtemp1 - trajtemp2))))
            
            ttemp2 = dtrajs[j] 
            #ttemp2 -= np.mean(ttemp2)
            traj_dist[i,j] = 0.25*np.sum(1- np.diagonal(np.matmul(ttemp1.T, ttemp2))/np.multiply(Ltemp,np.sqrt(np.sum(np.square(ttemp2)))))

#            traj_dist[i,j] = np.sqrt(np.sum(np.square((ttemp1 - ttemp2))))
            dang2 = np.arctan2(trajtemp2[-1, 0]-trajtemp2[0, 0],
                                             trajtemp2[-1, 1]-trajtemp2[0, 1])
            traj_dist[i,j] += 0.5*np.sqrt(np.square(np.sin(dang1)-np.sin(dang2))+ np.square(np.cos(dang1)-np.cos(dang2)))/angconst
            #traj_dist[i,j] += 0.5*np.sqrt(np.sum(np.square((trajtemp1[1,:] - trajtemp2[1,:]))))
            traj_dist[j,i] = traj_dist[i,j]
    return traj_dist, dtrajs


def circmean(x):
    return np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x)))%(2*np.pi)

def circdist(traj1, traj2):
    return np.sum(np.abs(np.arctan2(np.sin(traj1-traj2), np.cos(traj1-traj2))))

def cluster_umap(traj_dist, n_neighbours = 15, min_dist = 0.1,
                       spread = 1, repulsion_strength = 1, num_cluster = 10, dist_thresh = 100):
    import umap
    reducer = umap.UMAP(n_components=2,metric='precomputed',random_state=42, n_neighbors = n_neighbours,
                       min_dist = min_dist, spread = spread, repulsion_strength = repulsion_strength)
    X_um = reducer.fit_transform(traj_dist)

    agg = AgglomerativeClustering(affinity='precomputed', n_clusters = None, linkage='average', distance_threshold=dist_thresh)
    lbls = agg.fit(traj_dist).labels_
    print(np.bincount(lbls))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    j = 0
    for i in np.unique(lbls):
        if sum(lbls==i)>num_cluster:
            ax.scatter(X_um[lbls==i,0],X_um[lbls==i,1], c = cs[j], s = 5, alpha = 1,)
            j += 1
    plt.legend()
    return X_um, lbls


def compute_trial_gain(angspeed, runspeed, trials_all, trial_range, gain_trials, 
                       bSaveFigs = False, ax1 = None, folder = ''):
    
    gains = np.unique(gain_trials)
    cc_gains_all = {}
    for i, gain in enumerate(gains):
        trial_curr = trial_range[gain_trials==gain]
        cc_gains = np.zeros(len(trial_curr))            
        for j, cctrial in enumerate(trial_curr):       
            cc_gains[j] = np.sum(runspeed[trials_all == cctrial])/np.sum(angspeed[trials_all == cctrial])
        cc_gains_all[i] = cc_gains
    
    for i in cc_gains_all:
        cc_gains_all[i] /= np.mean(cc_gains_all[len(gains)-1])
        
    gains_mean = [cc_gains_all[i].mean() for i in cc_gains_all]
    gains_std = [cc_gains_all[i].std()/np.sqrt(len(cc_gains_all[i])) for i in cc_gains_all]
    return gains_mean, gains_std




def plot_gain(coords1, postt, posxx, postrial, speed, gain, contrast, sp = -np.inf, folder = '', files = []):
    cs =   {}
    t0 = 0
    gains_means = {}
    gains_stds = {}
    for fi in files:
        times = np.arange(t0,t0+sum(speed[fi]>sp))
        cs[fi] = coords1[times,:]
        t0+=sum(speed[fi]>sp)

        if fi.find('gain')>-1:
            lens_temp = [0,]
            sp = -np.inf

            fig1 = plt.figure()
            contrast_curr = 100

            times = np.arange(len(cs[fi]))
            cs11 = CubicSpline(times, gaussian_filter1d(np.sin(cs[fi]),axis = 0, sigma = 10))
            cc11 = CubicSpline(times,  gaussian_filter1d(np.cos(cs[fi]), axis = 0, sigma = 10))
            angular_rate1 = np.sum(np.sqrt(np.square(cs11(times,1)) + np.square(cc11(times,1))),1)

            traj_keep = (contrast[fi] ==  contrast_curr)        
            trial_range = np.unique(postrial[fi])[traj_keep]

            gain_trials = gain[fi][traj_keep]
            sp_sess = speed[fi][speed[fi]>sp].copy()
            if len(np.unique(gain_trials))>1:
                ax1 = fig1.add_subplot(111)
                gains = np.unique(gain_trials)
                gains_mean, gains_std = compute_trial_gain(angular_rate1, sp_sess, 
                                   postrial[fi][speed[fi]>sp], 
                                   trial_range, gain_trials,
                                   bSaveFigs = False, ax1 = ax1)
                gains_means[fi] = gains_mean.copy()
                gains_stds[fi] = gains_std.copy()
                ax1.scatter(gains, gains_mean, marker = 'o', s = 100, lw = 0.1)
                ax1.set_xticks(gains)  
                ax1.set_ylim([0.25,1.1])
                ax1.set_xlim([gains[0]-0.1,gains[-1]+0.1])
                ax1.plot(plt.gca().get_xlim(), np.ones(2), c = 'k', ls = '--')
                ax1.errorbar(gains, gains_mean, gains_std,0, ls = ':')


                fig1.tight_layout()
                finame = fi.replace('giocomo_data/', '').replace('.mat', '')
                fig1.savefig(folder + '/gain' +  finame, bbox_inches='tight', pad_inches=0.1, transparent = True)
                plt.close()
    return gains_means, gains_stds






def compute_trial_gain0(traj_curr, trial_range, gain_trials, 
                       bSaveFigs = False, folder = ''):
    cc_all_lens = {}
    cc_lens_i = {}
    numbins = len(traj_curr[0,:,0])
    scale = (numbins-2)/400
    lm_inds = np.array([0, int(scale*80), int(scale*160), int(scale*240), int(scale*320), int(scale*400)])

    gains = np.unique(gain_trials)
    gains_mean = np.zeros(len(gains))
    gains_std = np.zeros(len(gains))
    
    gains_mean_lm = np.zeros((len(gains), 5))
    gains_std_lm = np.zeros((len(gains), 5))
    ntrials = len(traj_curr)
    for i in range(ntrials):
        nans = np.where(np.isnan(traj_curr[i, :,0]))[0]
        while len(nans)>0:
            traj_curr[i,nans,:] = traj_curr[i,nans-1,:]  
            nans = np.where(np.isnan(traj_curr[i, :,0]))[0]
    print(gains_std_lm.shape)
    for i, gain in enumerate(gains):
        traj_curr_temp = traj_curr[gain_trials==gain,:,:]
        print(np.shape(traj_curr_temp))
        cc_gains = np.zeros(len(traj_curr_temp))            
        cc_lens_i_gain = np.zeros((len(traj_curr_temp), 5))    
        for j, cctrial in enumerate(traj_curr_temp):       
            for k in range(len(lm_inds[:-1])):
                cc_lens_i_gain[j,k] += np.sqrt(np.sum(np.square(cctrial[lm_inds[k+1],:] - cctrial[lm_inds[k],:])))
            cc_gains[j] = np.sqrt(np.sum(np.square(cctrial[0,:] - cctrial[-1,:])))
        gains_mean[i] = np.mean(cc_gains)
        gains_std[i] = np.std(cc_gains)
        gains_mean_lm[i, :] = np.mean(cc_lens_i_gain,0)
        gains_std_lm[i, :] = np.std(cc_lens_i_gain,0)
    print(gains_mean)
    gains_std /= gains_mean[-1]
    gains_mean /= gains_mean[-1]
    print('sdt', gains_std_lm.shape, gains_mean_lm[-1,:].shape)
    gains_std_lm = np.divide(gains_std_lm,gains_mean_lm[-1,:])
    gains_mean_lm = np.divide(gains_mean_lm,gains_mean_lm[-1,:])

    plt.figure()
    plt.scatter(gains, gains_mean, marker = 'o', s = 100, lw = 0.1)
    plt.xticks(gains)
    plt.xlabel('Gain')
    plt.ylabel('Relative toroidal trajectory length')

    plt.ylim([0.5,3.5])
    plt.xlim([gains[0]-0.1,gains[-1]+0.1])
    plt.plot(plt.gca().get_xlim(), np.ones(2), c = 'k', ls = '--')
    plt.errorbar(gains, gains_mean, gains_std,0)
    if bSaveFigs:            
        if len(folder)== 0:
            folder = str(np.random.randint(1e10))
            os.mkdir(folder)
            print('folder name', folder)
        plt.savefig(folder + '/gain_plot_all', transparent = True)
    plt.figure()
    for k in range(5):
        plt.scatter(gains, gains_mean_lm[:,k], marker = 'o', s = 100, lw = 0.1)
        plt.xticks(gains)
        plt.xlabel('Gain')
        plt.ylabel('Relative toroidal trajectory length')

        plt.ylim([0.5,3.5])
        plt.xlim([gains[0]-0.1,gains[-1]+0.1])
        plt.plot(plt.gca().get_xlim(), np.ones(2), c = 'k', ls = '--')
        plt.errorbar(gains, gains_mean_lm[:,k], gains_std_lm[:,k],0)
    if bSaveFigs:            
        if len(folder)== 0:
            folder = str(np.random.randint(1e10))
            os.mkdir(folder)
            print('folder name', folder)
        plt.savefig(folder + '/gain_plot_segments', transparent = True)


def compute_trial_gain1(coords_smooth_sess, data, gains, contrast, sp_sess, 
                       bSaveFigs = False, folder = '', sp = 2.5):
    cc_all_lens = {}
    valid_trialsSpike1 = sp_sess>sp
    spiketrials = data['trial'][valid_trialsSpike1] # to start at 0p
    posx = data['posx']
    for ccurr in coords_smooth_sess:
        coords = coords_smooth_sess[ccurr]
        gains_mean = np.zeros(len(gains))
        gains_std = np.zeros(len(gains))
        cc_lens_i = {}
        for i,gain in enumerate(gains):
            trial_range = np.where((data['trial_gain']==gain) & (data['trial_contrast']==contrast ))[0]+1            
            cc_gains = np.zeros(len(trial_range))
            cc_lens_i_gain = np.zeros((len(trial_range), 5))
            for j, trial in enumerate(trial_range[:]):
                valid_trialsSpike = spiketrials==trial  
                cctrial_temp = coords[valid_trialsSpike,:]
                cctrial = np.zeros_like(cctrial_temp)
                cctrial[0,:] = cctrial_temp[0,:].copy() 
                k1, k2 = 0, 0            
                for cn  in range(len(cctrial_temp)-1):
                    c1 = cctrial_temp[cn+1]
                    c_temp = [c1 + (k1*2*np.pi, k2*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                              c1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                              c1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                             ]  
                    cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                    cctrial[cn+1,:] = c_temp[cmin]
                    k1 += ks[cmin][0]
                    k2 += ks[cmin][1]            

                posx_trial = posx[valid_trialsSpike]
                lm_inds = np.concatenate(([np.argmin(np.abs(posx_trial-0))],
                          [np.argmin(np.abs(posx_trial-80))],
                          [np.argmin(np.abs(posx_trial-160))],
                          [np.argmin(np.abs(posx_trial-240))],
                          [np.argmin(np.abs(posx_trial-320))],
                          [np.argmin(np.abs(posx_trial-400))],))
                for k in range(len(lm_inds[:-1])):
                    cc_lens_i_gain[j,k] = np.sqrt(np.sum(np.square(cctrial[lm_inds[k+1],:] - cctrial[lm_inds[k],:])))
                    cc_gains[j] += cc_lens_i_gain[j,k]
            gains_mean[i] = np.mean(cc_gains)
            gains_std[i] = np.std(cc_gains)
            cc_lens_i[gain] = cc_lens_i_gain
        cc_all_lens[ccurr] = cc_lens_i
        gains_std /= gains_mean[-1]
        gains_mean /= gains_mean[-1]
        plt.figure()
        plt.scatter(gains, gains_mean, marker = 'o', s = 100, lw = 0.1)
        plt.xticks(gains)
        plt.xlabel('Gain')
        plt.ylabel('Relative toroidal trajectory length')

        plt.ylim([0.5,2.5])
        plt.xlim([gains[0]-0.1,gains[-1]+0.1])
        plt.plot(plt.gca().get_xlim(), np.ones(2), c = 'k', ls = '--')
        plt.errorbar(gains, gains_mean, gains_std,0)
        if bSaveFigs:            
            if len(folder)== 0:
                folder = str(np.random.randint(1e10))
                os.mkdir(folder)
                print('folder name', folder)
            fig.savefig(folder + '/gain_plot', transparent = True)
            plt.close()

def get_inds_mod(mouse_sess):
    if mouse_sess == 'npJ5_0504':
        inds_mod  = [12,]
    elif mouse_sess == 'npJ5_0506':
        #inds_mod  = [3, 10]
        inds_mod  = [1,3, 10]
    elif mouse_sess == 'npI1_0417':
        inds_mod  = [0,3, 6]
    elif mouse_sess == 'npI3_0421':
        inds_mod  = [0,]
    elif mouse_sess == 'npI4_0424':
        inds_mod  = [26]
    elif mouse_sess == 'npI5_0414':
        inds_mod  = [4,]
    elif mouse_sess == 'npF2_1016':
        inds_mod = [2,]
    elif mouse_sess == 'npF4_1025':
        inds_mod = [0,]
    elif mouse_sess == 'npG2_1211':
        inds_mod = [2]
    elif mouse_sess == 'npG2_1214':
        inds_mod = [1, 7]
    elif mouse_sess == 'npH3_0401': 
        inds_mod = [16]
    elif mouse_sess == 'npH3_0402':
        inds_mod = [0]
    elif mouse_sess == 'npH5_0326':
        inds_mod = [2]
    elif mouse_sess == 'npI3_0420':
        inds_mod = [0]
    elif mouse_sess == 'AA2_190809':
        inds_mod = [0,]
    elif mouse_sess == 'AA45_19092':
        inds_mod = [0,]
    elif mouse_sess == 'AA1_190727':
        inds_mod = [1]
    else:
        inds_mod = []
    return inds_mod
        



def plot_single_trials_unbinned(data, spk, ind, inds_mod, coords_smooth_sess, sp_sess,sp = 2.5,
                bSaveFigs = False, folder = '', num_trials = -1, spk_thresh = 1):
    cc_means = {}
    cc_all = {}
    trial_range = np.unique(data['trial'])[data['trial_contrast']==100]
    print(trial_range)
    if num_trials >-1:
        np.random.shuffle(trial_range)
        trial_range = trial_range[:num_trials]
    cc_all_1 = {}

    valid_trialsSpike1 = sp_sess>sp
    posx = data['posx'][valid_trialsSpike1]
    spiketrials = data['trial'][valid_trialsSpike1] # to start at 0p
    for i, trial in enumerate(trial_range):
        print(i)
        if i >-1:
            valid_trialsSpike0 = np.in1d(spiketrials,np.array([trial]))
            
            currplot = 0
            
            gs1 = gridspec.GridSpec(1, 3)
            gs1.update(left=0.0, right=1,wspace=-0.4, top = 1, bottom = 0. )
            
            fig = plt.figure()    
            for ii in inds_mod:
                ax = fig.add_subplot(gs1[0, currplot])
                currplot += 1

                valid_trialsSpike2 = np.sum(spk[valid_trialsSpike1,:][:, ind==ii].astype(bool),1)>spk_thresh
                valid_trialsSpike = valid_trialsSpike0 & valid_trialsSpike2

                cctrial_temp = coords_smooth_sess[ii][valid_trialsSpike,:]
                cctrial = np.zeros_like(cctrial_temp)
                cctrial[0,:] = cctrial_temp[0,:].copy()            
                k1, k2 = 0, 0
                for cn  in range(len(cctrial_temp)-1):
                    c1 = cctrial_temp[cn+1]
                    c_temp = [c1 + (k1*2*np.pi, k2*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                              c1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                              c1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                             ]  
                    cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                    cctrial[cn+1,:] = c_temp[cmin]
                    k1 += ks[cmin][0]
                    k2 += ks[cmin][1]

                ax.plot([-2*np.pi,-2*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
                ax.plot([-2*np.pi,4*np.pi],[4*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
                ax.plot([-2*np.pi,4*np.pi],[-2*np.pi,-2*np.pi], c = 'k', ls = '--', lw = 1)
                ax.plot([4*np.pi,4*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)        
                ax.plot([0,0],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
                ax.plot([-2*np.pi,4*np.pi],[2*np.pi,2*np.pi], c = 'k', ls = '--', lw = 1)
                ax.plot([-2*np.pi,4*np.pi],[0,0], c = 'k', ls = '--', lw = 1)
                ax.plot([2*np.pi,2*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)

                posx_trial = posx[valid_trialsSpike]
                lm_inds = np.concatenate(([np.argmin(np.abs(posx_trial-0))],
                                          [np.argmin(np.abs(posx_trial-80))],
                                          [np.argmin(np.abs(posx_trial-160))],
                                          [np.argmin(np.abs(posx_trial-240))],
                                          [np.argmin(np.abs(posx_trial-320))],
                                          [np.argmin(np.abs(posx_trial-400))],))
            
                ax.scatter(cctrial[:,0], cctrial[:,1], 
                           s = 1, alpha = 0.7, c = preprocessing.minmax_scale(range(len(cctrial[:,0]))), 
                           vmin = 0, vmax = 1, zorder = -3)

                ax.scatter(cctrial[lm_inds,0], cctrial[lm_inds,1], 
                           marker = 'X', lw = 0.5, s = 10, c =cs[:len(lm_inds)], zorder = -1)
                ax.set_aspect(1/ax.get_data_ratio())
                r_box = transforms.Affine2D().skew_deg(15,15)
                for x in ax.images + ax.lines + ax.collections:
                    trans = x.get_transform()
                    x.set_transform(r_box+trans) 
                    if isinstance(x, PathCollection):
                        transoff = x.get_offset_transform()
                        x._transOffset = r_box+transoff     
#                ax.set_xlim(-3.6*np.pi, 4*np.pi )
#                ax.set_ylim(-3.6*np.pi, 4*np.pi )
                ax.set_xlim(-3.6*np.pi, 4*np.pi + 5*3*np.pi/5)
                ax.set_ylim(-3.6*np.pi, 4*np.pi + 5*3*np.pi/5)
                ax.set_aspect('equal', 'box') 
                ax.axis('off')
#                ax.set_title(' spk: ' + str(np.sum(valid_trialsSpike)))

#            ax.set_title('Trial: ' + str(trial) + ' gain: ' + str(data['trial_gain'][trial-1]) + ' spk: ' + str(np.sum(valid_trialsSpike)))
#            fig.tight_layout()
        if bSaveFigs:            
            if len(folder)== 0:
                folder = str(np.random.randint(1e10))
                os.mkdir(folder)
                print('folder name', folder)
            fig.savefig(folder + '/T' + str(trial), transparent = True)
            plt.close()
        else:
            plt.show()








def get_coords_distribution1(spk1, spk2, coords1, indstemp,
                           starttime, num_frames, sig = 3, num_bins = 50):
    _2_PI = 2*np.pi

    cc2 = coords1.T.copy()
        
    num_neurons = len(spk1[0,:])
    centcosall = np.zeros((num_neurons, 2, len(indstemp)))
    centsinall = np.zeros((num_neurons, 2, len(indstemp)))
    dspk = preprocessing.scale(spk1[indstemp,:])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid,:,:] = np.multiply(np.cos(cc2[:, :].T*2*np.pi),spktemp)
        centsinall[neurid,:,:] = np.multiply(np.sin(cc2[:, :].T*2*np.pi),spktemp)

    dspk = preprocessing.scale(spk2)
    a = np.zeros((len(spk2[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

    c = np.zeros((len(spk2[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

    mtot2 = np.sum(c,2)
    mtot1 = np.sum(a,2)
    coordsbox = np.arctan2(mtot2,mtot1)%(2*np.pi)
    coordsbox/=_2_PI
    
    centall = np.zeros((num_neurons, num_bins,num_bins))
    bins = np.linspace(0,1,num_bins+1)
    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        mtot_tmp = binned_statistic_2d(coordsbox[:,0], coordsbox[:,1],
                                                   spktemp, bins = bins)[0] 
        nans = np.isnan(mtot_tmp)
        mtot_tmp[np.isnan(mtot_tmp)] = np.mean(mtot_tmp[~np.isnan(mtot_tmp)])
        mtot_tmp = np.rot90(smooth_tuning_map(np.rot90(mtot_tmp,1), num_bins+1, sig, bClose = False) ,3)
        centall[neurid,:] = mtot_tmp.copy()
        
        
    centall = centall.T
    coordsnew = np.zeros((num_frames, len(centall[:,0,0]),len(centall[:,0,0])))
    coordsnew = np.zeros((num_frames, len(centall[:,0,0]),len(centall[:,0,0])))
    for i, n in enumerate(np.arange(starttime,starttime+num_frames)):
        coordsnewtmp = np.dot(centall[:,:,:],dspk[n,:])
        coordsnew[i,:,:] = coordsnewtmp
    return coordsnew, coordsbox
    

from scipy.stats import binned_statistic
def get_coords_distribution_1d(spk1, spk2, coords1, indstemp,
                           starttime, num_frames,
                            num_bins = 50,
                               sig = 3):
    _2_PI = 2*np.pi

    cc2 = coords1.T.copy()
        
    num_neurons = len(spk1[0,:])
    centcosall = np.zeros((num_neurons, 2, len(indstemp)))
    centsinall = np.zeros((num_neurons, 2, len(indstemp)))
    dspk = preprocessing.scale(spk1[indstemp,:])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid,:,:] = np.multiply(np.cos(cc2[:, :].T*2*np.pi),spktemp)
        centsinall[neurid,:,:] = np.multiply(np.sin(cc2[:, :].T*2*np.pi),spktemp)

    dspk = preprocessing.scale(spk2)
    a = np.zeros((len(spk2[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

    c = np.zeros((len(spk2[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

    mtot2 = np.sum(c,2)
    mtot1 = np.sum(a,2)
    coordsbox = np.arctan2(mtot2,mtot1)%(2*np.pi)
    coordsbox/=_2_PI
    
    centall = np.zeros((num_neurons, num_bins))
    bins = np.linspace(0,1,num_bins+1)
    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        mtot_tmp = binned_statistic(coordsbox[:,0],
                                                   spktemp, bins = bins)[0] 
        nans = np.isnan(mtot_tmp)
        mtot_tmp[np.isnan(mtot_tmp)] = np.mean(mtot_tmp[~np.isnan(mtot_tmp)])
        mtot_tmp = gaussian_filter1d(mtot_tmp, mode = 'wrap', sigma = np.sqrt(sig))#np.rot90(smooth_tuning_map(np.rot90(mtot_tmp,1), num_bins+1, sig, bClose = False) ,3)
        centall[neurid,:] = mtot_tmp.copy()
        
        
    centall = centall.T
    coordsnew = np.zeros((num_frames, len(centall[:,0])))
    coordsnew = np.zeros((num_frames, len(centall[:,0])))
    for i, n in enumerate(np.arange(starttime,starttime+num_frames)):
        coordsnewtmp = np.dot(centall[:,:],dspk[n,:])
        coordsnew[i,:] = coordsnewtmp
    return coordsnew, coordsbox
    


from scipy.stats import binned_statistic
def get_coords_distribution_1d(spk1, spk2, coords1, indstemp,
                           starttime, num_frames,
                            num_bins = 50,
                               sig = 3):
    _2_PI = 2*np.pi

    cc2 = coords1.T.copy()
        
    num_neurons = len(spk1[0,:])
    centcosall = np.zeros((num_neurons, 2, len(indstemp)))
    centsinall = np.zeros((num_neurons, 2, len(indstemp)))
    dspk = preprocessing.scale(spk1[indstemp,:])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid,:,:] = np.multiply(np.cos(cc2[:, :].T*2*np.pi),spktemp)
        centsinall[neurid,:,:] = np.multiply(np.sin(cc2[:, :].T*2*np.pi),spktemp)

    dspk = preprocessing.scale(spk2)
    a = np.zeros((len(spk2[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centcosall[n,:,:],1))

    c = np.zeros((len(spk2[:,0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:,:,n] = np.multiply(dspk[:,n:n+1],np.sum(centsinall[n,:,:],1))

    mtot2 = np.sum(c,2)
    mtot1 = np.sum(a,2)
    coordsbox = np.arctan2(mtot2,mtot1)%(2*np.pi)
    coordsbox/=_2_PI
    
    centall = np.zeros((num_neurons, num_bins))
    bins = np.linspace(0,1,num_bins+1)
    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        mtot_tmp = binned_statistic(coordsbox[:,0],
                                                   spktemp, bins = bins)[0] 
        nans = np.isnan(mtot_tmp)
        mtot_tmp[np.isnan(mtot_tmp)] = np.mean(mtot_tmp[~np.isnan(mtot_tmp)])
        mtot_tmp = gaussian_filter1d(mtot_tmp, mode = 'wrap', sigma = np.sqrt(sig))#np.rot90(smooth_tuning_map(np.rot90(mtot_tmp,1), num_bins+1, sig, bClose = False) ,3)
        centall[neurid,:] = mtot_tmp.copy()
        
        
    centall = centall.T
    coordsnew = np.zeros((num_frames, len(centall[:,0])))
    coordsnew = np.zeros((num_frames, len(centall[:,0])))
    for i, n in enumerate(np.arange(starttime,starttime+num_frames)):
        coordsnewtmp = np.dot(centall[:,:],dspk[n,:])
#        coordsnewtmp[np.isnan(coordsnewtmp)] = np.mean(coordsnewtmp[~np.isnan(coordsnewtmp)])
#        coordsnewtmp = smooth_tuning_map(coordsnewtmp, len(coordsnewtmp)+1, sig, False)
        coordsnew[i,:] = coordsnewtmp
    return coordsnew, coordsbox
    

def get_coord_distribution1d(coords_mod1, numbins = 50,epsilon = 0.1, metric = 'euclidean', startindex = -1,
                          bWrap = True):    
    coords = coords_mod1[:,:1].copy()
    n = coords.shape[0]
    inds_orig = np.arange(n)
    if bWrap:
        coords = np.concatenate((coords, coords, coords,))
        inds_orig = np.concatenate((inds_orig,inds_orig,inds_orig,))
        coords[1*n:2*n] += 2*np.pi
        coords[2*n:3*n] -= 2*np.pi
        coordsrel = ~((coords[:,0]>2*np.pi + epsilon) | 
                     (coords[:,0]<-epsilon))
        inds_orig = inds_orig[coordsrel]
        coords = coords[coordsrel]
    n = coords.shape[0]
    inds = np.zeros((n, ), dtype=int)
    inds_label = [[] for i in range(n)]
    if epsilon > 0:            
        n = coords.shape[0]
        if startindex == -1:
            np.random.seed(0) 
            startindex = np.random.randint(n)
        i = startindex
        j = 1
        inds_res = np.arange(n, dtype=int)
        dists = np.zeros((n, ))
        while j < n+1:
            disttemp = (cdist(coords[i, :].reshape(1, -1), coords[:, :], metric=metric) - epsilon)[0]  
            inds_label[i] = inds_orig[np.where(disttemp<=0)[0]]
            dists[inds_res] = np.min(np.concatenate((dists[inds_res][:,np.newaxis], 
                                                     disttemp[inds_res,np.newaxis]),1),1)
            inds[i] = j
            inds_res = inds_res[disttemp[inds_res]>0]
            j = j+1
            if len(inds_res)>0:
                i = inds_res[np.argmax(dists[inds_res])]
            else:
                break
    else:
        inds = np.ones(range(np.shape(coords)[0]))
    inds = np.where(inds[:len(coords_mod1)])[0]
    inds_label = inds_label[:len(coords_mod1)]
    print(len(inds))
    return inds, inds_label

def get_phases1d(sspikes, coords_mod1, inds, inds_label):    
    dspk = sspikes.copy() - np.mean(sspikes,0)
    num_neurons = len(dspk[0,:])
    masscenters_1 = np.zeros((num_neurons,))
    for neurid in range(num_neurons):
        centcosall = 0
        centsinall = 0
        for i in inds:
            centcosall += np.mean(np.multiply(np.cos(coords_mod1[i:i+1].T),
                                              dspk[inds_label[i], neurid]),)
            centsinall += np.mean(np.multiply(np.sin(coords_mod1[i:i+1].T),
                                              dspk[inds_label[i], neurid]),)
            
        masscenters_1[neurid] = np.arctan2(centsinall,centcosall)%(2*np.pi)
    return masscenters_1


def match_phases1d(coords1, sspikes, mc, times = [],numbins = 20, lentmp = 0, t = 0.1, nums = 1, bPlot = False, bSqr = False):
    if len(times) == 0:
        times = np.arange(len(coords1))
    num_neurons = len(sspikes[0,:])
    coords = coords1[times,:]    
    if lentmp == 0:
        lentmp = len(coords)
    coords_tmp = np.random.rand(lentmp)*2*np.pi

    _2_PI = 2*np.pi
    spk_sim = np.zeros((len(coords1), num_neurons))
    numsall = np.arange(-nums,nums+1)
    for i in range(num_neurons):
        cctmp = ((coords_tmp - mc[i])%(_2_PI))/(_2_PI)
        for k in numsall:
            spk_sim[:,i] += np.exp(-np.pi/t*(k+cctmp)**2)
    pcorr = np.zeros(num_neurons)

    for i in range(num_neurons):
        mtot1 = binned_statistic(coords1[times,0], sspikes[times,i], bins = numbins)[0]
        nans = np.isnan(mtot1)
        mtot1[nans] = np.mean(mtot1[~nans])
        
        mtot2 = binned_statistic(coords_tmp, spk_sim[:,i], bins = numbins)[0]
        nans = np.isnan(mtot2)
        mtot2[nans] = np.mean(mtot2[~nans])
        mtot2 = gaussian_filter(mtot2, sigma = 1)                        
        pcorr[i] = pearsonr(mtot1.flatten(), mtot2.flatten())[0]    
    return pcorr 



def plot_phase_distribution(masscenters_1, masscenters_2, SourceName = '', dpi = 300):
    r_box = transforms.Affine2D().skew_deg(15,15)
    fig = plt.figure(dpi = dpi)
    ax = fig.add_subplot(111)
    plt.axis('off')
    num_neurons = len(masscenters_1[:,0])
    for i in np.arange(num_neurons):
        ax.scatter(masscenters_1[i,0], masscenters_1[i,1], s = 10, c = 'r', transform=r_box + ax.transData)
        if len(masscenters_2)>0:
            ax.scatter(masscenters_2[i,0], masscenters_2[i,1], s = 10, c ='k', transform=r_box + ax.transData)
            line = masscenters_1[i,:] - masscenters_2[i,:]
            dline = line[1]/line[0]
            if line[0]< - np.pi and line[1] < -np.pi:
                line = (-2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
                dline = line[1]/line[0]
                if (masscenters_1[i,1] + (- masscenters_1[i,0])*dline)>0:
                    ax.plot([masscenters_1[i,0], 0],
                            [masscenters_1[i,1], masscenters_1[i,1] + (- masscenters_1[i,0])*dline],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([2*np.pi,2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline], 
                            [masscenters_1[i,1] + (- masscenters_1[i,0])*dline, 0],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([2*np.pi + -(masscenters_1[i,1] + (- masscenters_1[i,0])*dline)/dline, 
                             masscenters_2[i,0]], 
                            [2*np.pi,masscenters_2[i,1]],
                            c = 'k', lw = 1, alpha = 0.5)
                else:
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                            [masscenters_1[i,1], 0],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 0],
                            [2*np.pi, 2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline], 
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([2*np.pi, 
                             masscenters_2[i,0]], 
                            [2*np.pi + -(masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,
                            masscenters_2[i,1]],
                            c = 'k', lw = 1, alpha = 0.5)
            elif line[0]> np.pi and line[1] >np.pi:
                line = (2*np.pi + masscenters_2[i,:]) - masscenters_1[i,:]
                dline = line[1]/line[0]
                if (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline)<2*np.pi:
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline],

                           [masscenters_1[i,1],2*np.pi],
                           c = 'k', lw = 1, alpha = 0.5)
                    ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline, 2*np.pi],
                           [0,(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline], 
                           c = 'k', lw = 1, alpha = 0.5)
                    ax.plot([0,masscenters_2[i,0]],
                           [(2*np.pi - (masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline))*dline, 
                            masscenters_2[i,1]], 
                           c = 'k', lw = 1, alpha = 0.5)          
                else:
                    ax.plot([masscenters_1[i,0],2*np.pi],
                            [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([0,(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                            [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 2*np.pi],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([(2*np.pi - (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                             masscenters_2[i,0]], 
                            [0,masscenters_2[i,1]],
                            c = 'k', lw = 1, alpha = 0.5)#
            elif line[0]>np.pi and line[1] <-np.pi:  
                line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]            
                if (masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline)>0:
                    ax.plot([masscenters_1[i,0],2*np.pi],
                            [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([0,(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline], 
                            [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, 0],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([(-(masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline))/dline, 
                             masscenters_2[i,0]], 
                            [2*np.pi,masscenters_2[i,1]],
                            c = 'k', lw = 1, alpha = 0.5)

                else:
                    line = [2*np.pi + masscenters_2[i,0], -2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                    dline = line[1]/line[0]
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (- masscenters_1[i,1])/dline],
                            [masscenters_1[i,1], 0],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([masscenters_1[i,0] + (- masscenters_1[i,1])/dline, 2*np.pi], 
                            [2*np.pi, 2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([0, masscenters_2[i,0]], 
                            [2*np.pi + (2*np.pi- masscenters_1[i,0] + (- masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                            c = 'k', lw = 1, alpha = 0.5)
            elif line[0]<-np.pi and line[1] >np.pi:
                line = [-2*np.pi + masscenters_2[i,0], 2*np.pi + masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]
                if ((masscenters_1[i,1] + -(masscenters_1[i,0])*dline)<2*np.pi):

                    ax.plot([masscenters_1[i,0],0],
                            [masscenters_1[i,1], masscenters_1[i,1] + -(masscenters_1[i,0])*dline],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([2*np.pi, 2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline], 
                            [masscenters_1[i,1] + -(masscenters_1[i,0])*dline, 2*np.pi],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([2*np.pi + (2*np.pi - (masscenters_1[i,1] + -(masscenters_1[i,0])*dline))/dline, 
                             masscenters_2[i,0]], 
                            [0,masscenters_2[i,1]],
                            c = 'k', lw = 1, alpha = 0.5)
                else:
                    ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline],
                            [masscenters_1[i,1], 2*np.pi],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline, 0], 
                            [0, 0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline],
                            c = 'k', lw = 1, alpha = 0.5)

                    ax.plot([2*np.pi, masscenters_2[i,0]], 
                            [0 + -(masscenters_1[i,0] + (2*np.pi-masscenters_1[i,1])/dline)*dline,masscenters_2[i,1]],
                            c = 'k', lw = 1, alpha = 0.5)

            elif line[0]< -np.pi:
                line = [(2*np.pi + masscenters_1[i,0]), masscenters_1[i,1]] - masscenters_2[i,:]
                dline = line[1]/line[0]
                ax.plot([masscenters_2[i,0],2*np.pi],
                        [masscenters_2[i,1], masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline], 
                        alpha = 0.5, c = 'k', lw = 1)            
                ax.plot([0,masscenters_1[i,0]],
                        [masscenters_2[i,1] + (2*np.pi - masscenters_2[i,0])*dline, masscenters_1[i,1]], 
                        alpha = 0.5, c = 'k', lw = 1)
            elif line[0]> np.pi:
                line = [ masscenters_2[i,0]+ 2*np.pi, masscenters_2[i,1]] - masscenters_1[i,:]
                dline = line[1]/line[0]


                ax.plot([masscenters_1[i,0],2*np.pi],
                        [masscenters_1[i,1], masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline],
                        c = 'k', lw = 1, alpha = 0.5)
                ax.plot([0,masscenters_2[i,0]],
                        [masscenters_1[i,1] + (2*np.pi - masscenters_1[i,0])*dline, masscenters_2[i,1]], 
                        alpha = 0.5, c = 'k', lw = 1)
            elif line[1]< -np.pi:
                line = [ masscenters_1[i,0], (2*np.pi + masscenters_1[i,1])] - masscenters_2[i,:]
                dline = line[1]/line[0]

                ax.plot([masscenters_2[i,0], masscenters_2[i,0] + (2*np.pi - masscenters_2[i,1])/dline], 
                        [masscenters_2[i,1],2*np.pi], alpha = 0.5, c = 'k', lw = 1),
                ax.plot([masscenters_1[i,0] - masscenters_1[i,1]/dline,masscenters_1[i,0]],
                        [0, masscenters_1[i,1]], 
                        alpha = 0.5, c = 'k', lw = 1)
            elif line[1]> np.pi:
                line = [ masscenters_2[i,0], masscenters_2[i,1]+ 2*np.pi] - masscenters_1[i,:]
                dline = line[1]/line[0]

                ax.plot([masscenters_1[i,0], masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline], 
                        [masscenters_1[i,1], 2*np.pi], alpha = 0.5, c = 'k', lw = 1),

                ax.plot([masscenters_1[i,0] + (2*np.pi - masscenters_1[i,1])/dline,masscenters_2[i,0]],
                        [0, masscenters_2[i,1]], 
                        alpha = 0.5, c = 'k', lw = 1)
            else:
                ax.plot([masscenters_1[i,0],masscenters_2[i,0]],
                        [masscenters_1[i,1],masscenters_2[i,1]], 
                        alpha = 0.5, c = 'k', lw = 1)

    ax.plot([0,0], [0,2*np.pi], c = 'k')
    ax.plot([0,2*np.pi], [0,0], c = 'k')
    ax.plot([2*np.pi,2*np.pi], [0,2*np.pi], c = 'k')
    ax.plot([0,2*np.pi], [2*np.pi,2*np.pi], c = 'k')

    for x in ax.images + ax.lines + ax.collections + ax.get_xticklabels() + ax.get_yticklabels():
        trans = x.get_transform()
        x.set_transform(r_box+trans) 
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r_box+transoff 
    ax.set_xlim([0,2*np.pi + 3/5*np.pi])
    ax.set_ylim([0,2*np.pi + 3/5*np.pi])
    ax.set_aspect('equal', 'box')
    if len(SourceName)>0:
        data = []
        data_names = []
        data.append(pd.Series(masscenters_1[:,0]))
        data_names.extend(['center1_x'])
        data.append(pd.Series(masscenters_1[:,1]))
        data_names.extend(['center1_y'])
        data.append(pd.Series(masscenters_2[:,0]))
        data_names.extend(['center2_x'])
        data.append(pd.Series(masscenters_2[:,1]))
        data_names.extend(['center2_y'])
        
        df = pd.concat(data, ignore_index=True, axis=1)            
        df.columns = data_names
        df.to_excel('Source_data/'+ SourceName +'.xlsx', sheet_name=SourceName)  
        
        plt.savefig('Figures/'+ SourceName + '.png', transparent = True, bbox_inches='tight', pad_inches=0.2)
        plt.savefig('Figures/'+ SourceName + '.pdf', transparent = True, bbox_inches='tight', pad_inches=0.2)
    return ax

def plot_cumulative_stat(stat, stat_shuffle, stat_range, stat_scale, xs, ys, xlim, ylim, 
                         cs = [0,0.4,0.4], ax = None, SourceName = '', dpi = 300):
    num_neurons = len(stat)
    num_shuffle = len(stat_shuffle)

    if not ax:
        fig = plt.figure(dpi = dpi)
        ax = plt.axes()
    numbins = 30
    meantemp = np.zeros(numbins)
    stat_all = np.array([])
    mean_stat_all = np.zeros(num_shuffle)
    for i in range(num_shuffle):
        stat_all = np.concatenate((stat_all, stat_shuffle[i]))
        mean_stat_all[i] = np.mean(stat_shuffle[i])

    meantemp1 = np.histogram(stat_all, range = stat_range, bins = numbins)[0]
    meantemp = np.cumsum(meantemp1)
    meantemp = np.divide(meantemp, num_shuffle)
    meantemp = np.divide(meantemp, num_neurons)
    y,x = np.histogram(stat, range = stat_range, bins = numbins)
    y = np.cumsum(y)
    y = np.divide(y, num_neurons)
    x = x[1:]-(x[1]-x[0])/2
    x *= stat_scale
    
    ax.plot(x, meantemp, c ='k', ls = ':', alpha = 0.7, lw = 4)
    ax.plot(x,y, c = cs,  alpha = 0.8, lw = 6)
    ax.set_xticks(xs)
    ax.set_xticklabels(np.zeros(len(xs),dtype=str))
    ax.xaxis.set_tick_params(width=1, length =5)
    ax.set_yticks(ys)
    ax.set_yticklabels(np.zeros(len(ys),dtype=str))
    ax.yaxis.set_tick_params(width=1, length =5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()

    ax.set_aspect(abs(x1-x0)/(abs(y1-y0)*1.4))

    plt.gca().axes.spines['top'].set_visible(False)
    plt.gca().axes.spines['right'].set_visible(False)
    if len(SourceName)>0:
        data = []
        data_names = []
        data.append(pd.Series(x))
        data_names.extend(['x'])
        data.append(pd.Series(y))
        data_names.extend(['y'])
        data.append(pd.Series(meantemp))
        data_names.extend(['shuffle'])
        df = pd.concat(data, ignore_index=True, axis=1)            
        df.columns = data_names
        df.to_excel('Source_data/'+ SourceName +'.xlsx', sheet_name=SourceName)  
        
        plt.savefig('Figures/'+ SourceName + '.png', transparent = True, bbox_inches='tight', pad_inches=0.2)
        plt.savefig('Figures/'+ SourceName + '.pdf', transparent = True, bbox_inches='tight', pad_inches=0.2)




def plot_cluster_repeat(traj_all, trial_range, lbls, bCenter = False,
                        bSaveFigs = False, folder = '', bRight = True, numbins = 500): 
    traj = {}
    scale = (numbins-2)/400
    lm_inds = np.array([0, int(scale*80), int(scale*160), int(scale*240), int(scale*320), int(scale*400)])
    cs = [[1,0,0,],
          [0,1,0,],
          [0,0,1,],
          [1,1,0,],
          [0,1,1,],
          [1,0,1,]]
    for lbl in np.unique(lbls):
        ltraj = np.where(lbls ==lbl)[0]
        if len(ltraj)>2:
            traj_mean = np.zeros((500,4))
            trial_range1 = trial_range[ltraj]
            print('label', lbl)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot([0,0],[0,2*np.pi], c = 'k')
            ax.plot([0,2*np.pi],[2*np.pi,2*np.pi], c = 'k')
            ax.plot([0,2*np.pi],[0,0], c = 'k')
            ax.plot([2*np.pi,2*np.pi],[0,2*np.pi], c = 'k')
            for i, trial in enumerate(trial_range1):
                cctrial = traj_all[ltraj[i],:,:].copy()
                if bCenter:
                    cctrial-= cctrial[0,:]
                    cctrial += (np.pi,np.pi)
                ax.scatter(cctrial[:,0], cctrial[:,1], s = 1, alpha = 0.3, c = preprocessing.minmax_scale(np.arange(numbins)), vmin = 0, vmax = 1, zorder = -2)
                ax.scatter(cctrial[lm_inds,0], cctrial[lm_inds,1], marker = 'X', lw = 0.1, s = 10, c = cs, zorder = -1)

                ax.scatter(cctrial[:,0]+2*np.pi, cctrial[:,1]+2*np.pi, s = 1, alpha = 0.3, c = preprocessing.minmax_scale(np.arange(numbins)), vmin = 0, vmax = 1, zorder = -2)
                ax.scatter(cctrial[lm_inds,0]+2*np.pi, cctrial[lm_inds,1]+2*np.pi, marker = 'X', lw = 0.1, s = 10, c = cs, zorder = -1)

                ax.scatter(cctrial[:,0], cctrial[:,1]+2*np.pi, s = 1, alpha = 0.3, c = preprocessing.minmax_scale(np.arange(numbins)), vmin = 0, vmax = 1, zorder = -2)
                ax.scatter(cctrial[lm_inds,0], cctrial[lm_inds,1]+2*np.pi, marker = 'X', lw = 0.1, s = 10, c = cs, zorder = -1)

                ax.scatter(cctrial[:,0]+2*np.pi, cctrial[:,1], s = 1, alpha = 0.3, c = preprocessing.minmax_scale(np.arange(numbins)), vmin = 0, vmax = 1, zorder = -2)
                ax.scatter(cctrial[lm_inds,0]+2*np.pi, cctrial[lm_inds,1], marker = 'X', lw = 0.1, s = 10, c = cs, zorder = -1)
            ax.set_aspect(1/ax.get_data_ratio())

            #ax.imshow(mtot.T, origin = 'lower', extent = [0,2*np.pi, 0, 2*np.pi], vmin = 0, vmax = np.max(mtot)*0.975)
            if bRight:
                r_box = transforms.Affine2D().skew_deg(15,15)
            else:
                r_box = transforms.Affine2D().skew_deg(-15,-15)
                
            for x in ax.images + ax.lines + ax.collections:
                trans = x.get_transform()
                x.set_transform(r_box+trans) 
                if isinstance(x, PathCollection):
                    transoff = x.get_offset_transform()
                    x._transOffset = r_box+transoff     
            
            if bRight:
                ax.set_xlim(0, 4*np.pi + 5*3*np.pi/5)
                ax.set_ylim(0, 4*np.pi + 5*3*np.pi/5)
            else:
                ax.set_xlim(0 - 5*3*np.pi/5, 4*np.pi )
                ax.set_ylim(0 -.5*3*np.pi/5, 4*np.pi )
            ax.set_aspect('equal', 'box') 
            ax.axis('off')

            print('trial', trial_range1)
            print('')
            traj[lbl] = traj_mean
            plt.show()
            
def plot_cluster_trials(traj_all, trial_range, lbls, bCenter = False,
                        bSaveFigs = False, folder = '', bRight = True, numbins = 500): 
    traj = {}
    scale = (numbins-2)/400
    lm_inds = np.array([0, int(scale*80), int(scale*160), int(scale*240), int(scale*320), int(scale*400)])
    for lbl in np.unique(lbls):
        ltraj = np.where(lbls ==lbl)[0]
        if len(ltraj)>2:
            trial_range1 = trial_range[ltraj]
            print('label', lbl)
                    
            fig = plt.figure()    
            ax = fig.add_subplot(111)
            for i, trial in enumerate(trial_range1):
                cctrial_temp = cctrial = traj_all[ltraj[i],:,:].copy()
                cctrial = np.zeros_like(cctrial_temp)
                cctrial[0,:] = cctrial_temp[0,:].copy()            
                k1, k2 = 0, 0
                for cn  in range(len(cctrial_temp)-1):
                    c1 = cctrial_temp[cn+1]
                    c_temp = [c1 + (k1*2*np.pi, k2*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                              c1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                              c1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                              c1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                             ]  
                    cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                    cctrial[cn+1,:] = c_temp[cmin]
                    k1 += ks[cmin][0]
                    k2 += ks[cmin][1]
                if bCenter:
                    cctrial-= cctrial[0,:]
                    cctrial += (np.pi,np.pi)

                ax.scatter(cctrial[:,0], cctrial[:,1], 
                           s = 0.5, alpha = 0.7, c = preprocessing.minmax_scale(range(len(cctrial[:,0]))), 
                           vmin = 0, vmax = 1, zorder = -3)

                ax.scatter(cctrial[lm_inds,0], cctrial[lm_inds,1], 
                           marker = 'X', lw = 0.5, s = 10, c =cs[:len(lm_inds)], zorder = -1)

            ax.plot([-2*np.pi,-2*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
            ax.plot([-2*np.pi,4*np.pi],[4*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
            ax.plot([-2*np.pi,4*np.pi],[-2*np.pi,-2*np.pi], c = 'k', ls = '--', lw = 1)
            ax.plot([4*np.pi,4*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)        
            ax.plot([0,0],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
            ax.plot([-2*np.pi,4*np.pi],[2*np.pi,2*np.pi], c = 'k', ls = '--', lw = 1)
            ax.plot([-2*np.pi,4*np.pi],[0,0], c = 'k', ls = '--', lw = 1)
            ax.plot([2*np.pi,2*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)

            ax.set_aspect(1/ax.get_data_ratio())
            r_box = transforms.Affine2D().skew_deg(15,15)
            for x in ax.images + ax.lines + ax.collections:
                trans = x.get_transform()
                x.set_transform(r_box+trans) 
                if isinstance(x, PathCollection):
                    transoff = x.get_offset_transform()
                    x._transOffset = r_box+transoff     
            ax.set_xlim(-3.6*np.pi, 4*np.pi + 5*3*np.pi/5)
            ax.set_ylim(-3.6*np.pi, 4*np.pi + 5*3*np.pi/5)
            ax.set_aspect('equal', 'box') 
            ax.axis('off')            
            fig.tight_layout()
            if bSaveFigs:            
                if len(folder)== 0:
                    folder = str(np.random.randint(1e10))
                    os.mkdir(folder)
                    print('folder name', folder)
                fig.savefig(folder + '/clust_' + str(lbl), transparent = True)
                plt.close()
            else:
                plt.title(folder + '/clust_' + str(lbl))
                plt.show()
            
            
def plot_single_trial(ax, traj):
    ax.plot([-2*np.pi,-2*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
    ax.plot([-2*np.pi,4*np.pi],[4*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
    ax.plot([-2*np.pi,4*np.pi],[-2*np.pi,-2*np.pi], c = 'k', ls = '--', lw = 1)
    ax.plot([4*np.pi,4*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)        
    ax.plot([0,0],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
    ax.plot([-2*np.pi,4*np.pi],[2*np.pi,2*np.pi], c = 'k', ls = '--', lw = 1)
    ax.plot([-2*np.pi,4*np.pi],[0,0], c = 'k', ls = '--', lw = 1)
    ax.plot([2*np.pi,2*np.pi],[-2*np.pi,4*np.pi], c = 'k', ls = '--', lw = 1)
    
    
    numbins = len(traj)    
    scale = (numbins-1)/400
    lm_inds = np.array([0, int(scale*80), int(scale*160), int(scale*240), int(scale*320), int(scale*400)])

    ax.scatter(traj[:,0], traj[:,1], 
               s = 1, alpha = 0.7, c = preprocessing.minmax_scale(range(len(traj[:,0]))), 
               vmin = 0, vmax = 1, zorder = -3)

    ax.scatter(traj[lm_inds,0], traj[lm_inds,1], 
               marker = 'X', lw = 0.5, s = 10, c =cs[:len(lm_inds)], zorder = -1)
    ax.set_aspect(1/ax.get_data_ratio())
    r_box = transforms.Affine2D().skew_deg(15,15)
    for x in ax.images + ax.lines + ax.collections:
        trans = x.get_transform()
        x.set_transform(r_box+trans) 
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r_box+transoff     
    ax.set_xlim(-3.6*np.pi, 4*np.pi + 5*3*np.pi/5)
    ax.set_ylim(-3.6*np.pi, 4*np.pi + 5*3*np.pi/5)
    ax.set_aspect('equal', 'box') 
    ax.axis('off')
    return ax

def get_comb_bu(coords1,coords2, spk, acorr):    
#    acorr_obj = preprocessing.scale(acorr)
    acorr_obj = acorr.flatten()
    acorr_obj -= acorr_obj.mean()
    Lacorr = np.sqrt(np.sum(np.square(acorr_obj)))
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()
    it = 0    
    res = []
    combs = []
    pshifts = []
    rots = []
    for i in [0,1]:
        if i == 0:
            c11 = c2.copy()
            c12 = c1.copy()
        else:   
            c11 = c1.copy()
            c12 = c2.copy()
        for j in [0,1]:
            if j == 0:
                c21 = c11.copy()
                c22 = (2*np.pi-c12)
            else:
                c21 = c11.copy()
                c22 = c12.copy()
            for k in [0,1]:
                if k == 0:
                    c31 = (2*np.pi-c21)
                    c32 = c22.copy()
                else:
                    c31 = c21.copy()
                    c32 = c22.copy()
                for m in [0,1,2,3,4,5,6,7,8]:        
                    if m == 0:
                        c41 = (c31 + np.pi/3*c32)
                        c42 = c32.copy()
                    elif m== 1:
                        c41 = (c31 - np.pi/3*c32)
                        c42 = c32.copy()
                    elif m == 2:
                        c41 = c31.copy() 
                        c42 = c32 + np.pi/3*c31                   
                    elif m == 3:
                        c41 = c31.copy() 
                        c42 = c32 - np.pi/3*c31                    
                    elif m == 4:
                        c41 = (2*np.pi-c31 + np.pi/3*c32)
                        c42 = c32.copy()
                    elif m== 5:
                        c41 = (2*np.pi-c31 - np.pi/3*c32)
                        c42 = c32.copy()
                    elif m == 6:
                        c41 = c31.copy() 
                        c42 = 2*np.pi-c32 + np.pi/3*c31                    
                    elif m == 7:
                        c41 = c31.copy() 
                        c42 = 2*np.pi-c32 - np.pi/3*c31                    
                    else:
                        c41 = c31.copy()
                        c42 = c32.copy()
                    combs.append([i,j,k,m])
                    coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
                    pshift = np.arctan2(np.mean(np.sin(coords_2_1-coords2),0), np.mean(np.cos(coords_2_1 - coords2),0))
                    coords_mod_1_2 = (coords_2_1 - pshift)%(2*np.pi)
                    
                    pshifts.append(pshift)
                    
                    nums = 10000
                    lenseg = nums#2000
                    times = []
                    j1 = 0
                    c = 0
                    seg = int(len(coords_mod_1_2[:,0])/lenseg)
                    nnseg = int(nums/lenseg)
                    nseg = int(seg/nnseg)

                    for j2 in range(nnseg):
                        times.extend(np.arange(j1*lenseg, (j1+1)*lenseg))
                        j1 +=  nseg

                    if 1== 0:
                        cs11 = CubicSpline(times, np.cos(coords2[:nums,0]))
                        dcs11 = cs11.derivative(1)
                        cs12 = CubicSpline(times, np.sin(coords2[:nums,0]))
                        dcs12 = cs12.derivative(1)
                        angular_rate11 = np.arctan2(dcs12(times), dcs11(times))

                        cs11 = CubicSpline(times, np.cos(coords2[:nums,1]))
                        dcs11 = cs11.derivative(1)
                        cs12 = CubicSpline(times, np.sin(coords2[:nums,1]))
                        dcs12 = cs12.derivative(1)
                        angular_rate12 = np.arctan2(dcs12(times), dcs11(times))


                        cs11 = CubicSpline(times, np.cos(coords_mod_1_2[:nums,0]))
                        dcs11 = cs11.derivative(1)
                        cs12 = CubicSpline(times, np.sin(coords_mod_1_2[:nums,0]))
                        dcs12 = cs12.derivative(1)
                        angular_rate21 = np.arctan2(dcs12(times), dcs11(times))

                        cs11 = CubicSpline(times, np.cos(coords_mod_1_2[:nums,1]))
                        dcs11 = cs11.derivative(1)
                        cs12 = CubicSpline(times, np.sin(coords_mod_1_2[:nums,1]))
                        dcs12 = cs12.derivative(1)
                        angular_rate22 = np.arctan2(dcs12(times), dcs11(times))                    
                        res.append(np.sum(np.square(angular_rate11-angular_rate21)+
                            np.square(angular_rate12-angular_rate22)))   

                    elif 1==2:
                        nsum = 0
                        cctrial_temp1 = coords2[times,:]
                        cctrial_temp2 = coords_mod_1_2[times,:]
                        currtimes = range(lenseg)
                        
                        for j1 in range(nnseg):
                            cctrial_temp = cctrial_temp1[j1*lenseg:(j1+1)*lenseg,:].copy()
                            cctrial = np.zeros_like(cctrial_temp)
                            cctrial[0,:] = cctrial_temp[0,:].copy()            
                            k1, k2 = 0, 0
                            for cn  in range(len(cctrial_temp)-1):
                                ctmp1 = cctrial_temp[cn+1]
                                c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                                          ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                                          ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                                          ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                                          ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                                          ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                                          ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                                          ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                                          ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                                         ]  
                                cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                                cctrial[cn+1,:] = c_temp[cmin]
                                k1 += ks[cmin][0]
                                k2 += ks[cmin][1]
                            cs11 = CubicSpline(currtimes, cctrial)
                            dcs11 = cs11.derivative(1)
                            angular_rate1 = np.arctan2(dcs11(currtimes)[:,1],dcs11(currtimes)[:,0])
                            
                            cctrial_temp = cctrial_temp2[j1*lenseg:(j1+1)*lenseg,:].copy()
                            cctrial = np.zeros_like(cctrial_temp)
                            cctrial[0,:] = cctrial_temp[0,:].copy()            
                            k1, k2 = 0, 0
                            for cn  in range(len(cctrial_temp)-1):
                                ctmp1 = cctrial_temp[cn+1]
                                c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                                          ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                                          ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                                          ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                                          ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                                          ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                                          ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                                          ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                                          ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                                         ]  
                                cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                                cctrial[cn+1,:] = c_temp[cmin]
                                k1 += ks[cmin][0]
                                k2 += ks[cmin][1]
                            cs11 = CubicSpline(currtimes, cctrial)
                            dcs11 = cs11.derivative(1)
                            angular_rate2 = np.arctan2(dcs11(currtimes)[:,1],dcs11(currtimes)[:,0])
                            nsum += np.sum(np.square(np.sin(angular_rate1)-np.sin(angular_rate2)) + 
                                           np.square(np.cos(angular_rate1)-np.cos(angular_rate2)))
                        res.append(nsum)
                    elif 1==1:
                        acorr3 = get_mean_acorr(coords_mod_1_2, spk[:,:])
                        acorr3 -= np.mean(acorr3)
                        res.extend([1- np.dot(acorr_obj, acorr3.flatten())/(Lacorr*np.sqrt(np.sum(np.square(acorr3))))])

    comb = combs[np.argmin(res)]
    pshift = pshifts[np.argmin(res)]
    return comb, pshift,res



def get_comb_bu1(coords1,coords2, spk, acorr, times = []): 
    acorr_obj = acorr.flatten()
    acorr_obj -= acorr_obj.mean()
    Lacorr = np.sqrt(np.sum(np.square(acorr_obj)))
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()
    it = 0    
    res1 = []
    combs = []
    pshifts = []
    rots = []
    for i in [0,1]:
        if i == 0:
            c11 = c2.copy()
            c12 = c1.copy()
        else:   
            c11 = c1.copy()
            c12 = c2.copy()
        for j in [0,1]:
            if j == 0:
                c31 = c11.copy()
                c32 = (2*np.pi-c12)
            else:
                c31 = c11.copy()
                c32 = c12.copy()
            for m in [0,1,2,3,4,5,6,7,8]:        
                if m == 0:
                    c41 = (c31 + np.pi/3*c32)
                    c42 = c32.copy()
                elif m== 1:
                    c41 = (c31 - np.pi/3*c32)
                    c42 = c32.copy()
                elif m == 2:
                    c41 = c31.copy() 
                    c42 = c32 + np.pi/3*c31                   
                elif m == 3:
                    c41 = c31.copy() 
                    c42 = c32 - np.pi/3*c31                    
                elif m == 4:
                    c41 = (2*np.pi-c31 + np.pi/3*c32)
                    c42 = c32.copy()
                elif m== 5:
                    c41 = (2*np.pi-c31 - np.pi/3*c32)
                    c42 = c32.copy()
                elif m == 6:
                    c41 = c31.copy() 
                    c42 = 2*np.pi-c32 + np.pi/3*c31                    
                elif m == 7:
                    c41 = c31.copy() 
                    c42 = 2*np.pi-c32 - np.pi/3*c31                    
                else:
                    c41 = c31.copy()
                    c42 = c32.copy()
                combs.append([i,j,m])
                coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
                
                acorr3 = get_mean_acorr(coords_2_1, spk[:,:], times = times)
                
                acorr3 -= np.mean(acorr3)
                res1.extend([1- np.dot(acorr_obj, acorr3.flatten())/(Lacorr*np.sqrt(np.sum(np.square(acorr3))))])
    combcurr = combs[np.argmin(res1)]
    
    coords_2_2 = align(coords1, combcurr, [0,0], 0) 
    res2 = []
    for k in [0,1]:
        if k == 1:                        
            coords_2_2 = 2*np.pi-coords_2_2.copy()                        
        pshift = np.arctan2(np.mean(np.sin(coords_2_2-coords2),0), np.mean(np.cos(coords_2_2 - coords2),0))
        coords_mod_1_2 = (coords_2_2 - pshift)%(2*np.pi)
        pshifts.append(pshift)

        nums = 10000
        lenseg = nums#2000
        times = []
        j1 = 0
        c = 0
        seg = int(len(coords_mod_1_2[:,0])/lenseg)
        nnseg = int(nums/lenseg)
        nseg = int(seg/nnseg)

        for j2 in range(nnseg):
            times.extend(np.arange(j1*lenseg, (j1+1)*lenseg))
            j1 +=  nseg

        nsum = 0
        cctrial_temp1 = coords2[times,:]
        cctrial_temp2 = coords_mod_1_2[times,:]
        currtimes = range(lenseg)

        for j1 in range(nnseg):
            cctrial_temp = cctrial_temp1[j1*lenseg:(j1+1)*lenseg,:].copy()
            cctrial = np.zeros_like(cctrial_temp)
            cctrial[0,:] = cctrial_temp[0,:].copy()            
            k1, k2 = 0, 0
            for cn  in range(len(cctrial_temp)-1):
                ctmp1 = cctrial_temp[cn+1]
                c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                          ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                          ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                          ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                          ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                          ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                          ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                          ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                          ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                         ]  
                cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                cctrial[cn+1,:] = c_temp[cmin]
                k1 += ks[cmin][0]
                k2 += ks[cmin][1]
            cs11 = CubicSpline(currtimes, cctrial)
            dcs11 = cs11.derivative(1)
            angular_rate1 = np.arctan2(dcs11(currtimes)[:,1],dcs11(currtimes)[:,0])

            cctrial_temp = cctrial_temp2[j1*lenseg:(j1+1)*lenseg,:].copy()
            cctrial = np.zeros_like(cctrial_temp)
            cctrial[0,:] = cctrial_temp[0,:].copy()            
            k1, k2 = 0, 0
            for cn  in range(len(cctrial_temp)-1):
                ctmp1 = cctrial_temp[cn+1]
                c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                          ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                          ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                          ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                          ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                          ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                          ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                          ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                          ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                         ]  
                cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
                cctrial[cn+1,:] = c_temp[cmin]
                k1 += ks[cmin][0]
                k2 += ks[cmin][1]
            cs11 = CubicSpline(currtimes, cctrial)
            dcs11 = cs11.derivative(1)
            angular_rate2 = np.arctan2(dcs11(currtimes)[:,1],dcs11(currtimes)[:,0])
            nsum += np.sum(np.square(np.sin(angular_rate1)-np.sin(angular_rate2)) + 
                           np.square(np.cos(angular_rate1)-np.cos(angular_rate2)))
        res2.append(nsum)
    return combs, pshifts, res1, res2



def plot_single_trials_binned(trajs, trial_range, inds_mod, bSaveFigs = False, folder = '', num_trials = -1):
    klist = list(trajs.keys())
    keysname = []
    for i in klist:
        if int(i[-1]) in inds_mod:
            keysname.extend([i])
    for ii, c in enumerate(keysname):
        if (ii%len(inds_mod)> 0):
            continue
        trial_range_curr = []
        for ii in inds_mod:
            trial_range_curr.extend(trial_range[c[:-1] + str(ii)])
        trial_count = np.bincount(trial_range_curr)
        trial_range_curr = np.where(trial_count == len(inds_mod))[0]
        if num_trials >-1:
            np.random.shuffle(trial_range_curr)
            trial_range_curr = trial_range_curr[:num_trials]
        for i, trial in enumerate(trial_range_curr):
            currplot = 0
            fig = plt.figure()    
            gs1 = gridspec.GridSpec(1, len(inds_mod))
            for jj in inds_mod:
                ax = fig.add_subplot(gs1[0, currplot])
                plot_single_trial(ax, trajs[c[:-1] + str(jj)][i,:])
                currplot += 1

            gs1.update(left=0.0, right=1,wspace=-0.4, top = 1, bottom = 0. )            
            if bSaveFigs: 
                fig.savefig(folder + '/' +  c[c.find('\\')+1:c.find('.')] + '_T' + str(trial), transparent = True)
                plt.close()
            else:
                plt.title(folder + '/' + c[c.find('\\')+1:c.find('.')] + '_T' + str(trial))
                plt.show()


def get_traj(data, spk, sp, sp_sess, coords_smooth, 
                      contrast = 100, gain = 1, numbins = 500, bCenter = False, bFillNans = True ):
    post = data['post'][sp_sess>sp]
    posx = data['posx'][sp_sess>sp]
    
    ss11 = CubicSpline(post, np.sin(coords_smooth[:,:]))
    cs11 = CubicSpline(post, np.cos(coords_smooth[:,:]))
    dss11 = ss11.derivative(2)
    dcs11 = cs11.derivative(2)
    xs11 = CubicSpline(post, posx)        
    
    
    valid_trialsSpike1 = sp_sess>sp
    spiketrials = data['trial'][valid_trialsSpike1]
    trial_range = np.unique(spiketrials)
    traj_all = np.zeros((len(trial_range), 2, 2))
    traj_curr = np.zeros((len(trial_range), numbins, 2))
    trial_spk = np.ones(len(trial_range))
    trial_der = np.ones(len(trial_range))
    for i, trial in enumerate(trial_range):
        trial_range1 = np.array([trial])
        valid_trialsSpike = np.in1d(spiketrials,trial_range1) 
        spk_trial = spk[valid_trialsSpike,:]
        post_trial = post[valid_trialsSpike]
        trial_spk[i] = np.sum(np.sum(spk_trial>0,1)>1)/len(spk_trial[:,0])
        posxx = np.digitize(posx[valid_trialsSpike], np.linspace(0,np.max(posx)+0.001,numbins))-1

        cctrial_temp = coords_smooth[valid_trialsSpike,:]
        cctrial = np.zeros_like(cctrial_temp)
        cctrial[0,:] = cctrial_temp[0,:].copy()            
        k1, k2 = 0, 0
        for cn  in range(len(cctrial_temp)-1):
            c1 = cctrial_temp[cn+1]
            c_temp = [c1 + (k1*2*np.pi, k2*2*np.pi), 
                      c1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                      c1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                      c1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                      c1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                      c1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                      c1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                      c1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                      c1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                     ]  
            cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
            cctrial[cn+1,:] = c_temp[cmin]
            k1 += ks[cmin][0]
            k2 += ks[cmin][1]
            
        trial_der[i] = (np.sum(np.abs(dss11(post_trial)) + np.abs(dcs11(post_trial))))/len(post_trial)
        traj_temp = binned_statistic(posxx, cctrial[:,0], 
                              statistic='mean', bins=np.linspace(0,numbins,numbins+1))[0]
        traj_curr[i,:,0] = traj_temp.copy()
        
        traj_temp = binned_statistic(posxx, cctrial[:,1], 
                                          statistic='mean', bins=np.linspace(0,numbins,numbins+1))[0]
        traj_curr[i,:,1] = traj_temp.copy()

        traj_all[i,0, :] = cctrial[0,:]
        traj_all[i,1, :] = cctrial[-1,:]
    return traj_all, traj_curr, trial_der, trial_spk


def twoD_Gaussian(p,  sigma_x, sigma_y, theta):
    (x,y) = p    
    amplitude = 1
    xo = 25
    yo = 25
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def fit_gauss_acorr(acorr, bSaveFigs = False, figname = ''):
    data = acorr.copy().ravel()
    initial_guess = (20,20, 0)
    x = np.arange(51)
    y = np.arange(51)
    x,y = np.meshgrid(x, y)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), normit(data), p0=initial_guess, )
    if popt[0]<popt[1]:
        popt_temp = popt[0]
        popt[0] =popt[1]
        popt[1] =popt_temp        
        popt[-1] = (popt[-1]+np.pi/2)%(2*np.pi)
    else:
        popt[-1] = (popt[-1])%(2*np.pi)
    data_fitted = twoD_Gaussian((x, y), *popt)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(data_fitted.reshape(51, 51), cmap=plt.cm.jet, origin = 'lower', 
        extent=(x.min(), x.max(), y.min(), y.max()))
    plt.axis('off')
    ax = fig.add_subplot(122)
    ax.imshow(data.reshape(51, 51), cmap=plt.cm.jet, origin = 'lower', 
        extent=(x.min(), x.max(), y.min(), y.max()))
    plt.axis('off')
    if bSaveFigs:
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
    return popt[-1]

def get_combs1(theta1, theta, thresh = 0.1):     
    if ((theta1>np.pi) & (theta1<=3/2*np.pi)) | ((theta1>0) & (theta1<=np.pi/2)):
        if ((theta>np.pi) & (theta<=3/2*np.pi)) | ((theta>0) & (theta<=np.pi/2)):
            combs1 = combs_all[3]
        else:
            combs1 = combs_all[1]
    else:
        if ((theta>np.pi) & (theta<=3/2*np.pi)) | ((theta>0) & (theta<=np.pi/2)):
            combs1 = combs_all[0]
        else:
            combs1 = combs_all[2]   
    return combs1


def get_pshift(coords1, comb, coords2):
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()
    for j in comb[1:2]:
        if j == 0:
            c21 = c1.copy()
            c22 = c2.copy()
        elif j == 1:
            c21 = 2*np.pi-c1.copy()
            c22 = 2*np.pi-c2.copy()
            
        elif j == 2:
            c21 = c1.copy()
            c22 = 2*np.pi-c2.copy()
        elif j == 3:
            c21 = 2*np.pi-c1.copy()
            c22 = c2.copy()           
        for k in comb[2:3]:
            if k == 0:
                c31 = c21.copy()
                c32 = c22.copy()
            elif k == 1:
                c31 = c21.copy() - np.pi/3*c22
                c32 = c22.copy()
            elif k == 2:
                c31 = c21.copy() + np.pi/3*c22
                c32 = c22.copy()
            elif k == 3:
                c31 = c21.copy()
                c32 = c22.copy() - np.pi/3*c21
            elif k == 4:
                c31 = c21.copy()
                c32 = c22.copy() + np.pi/3*c21  
                
            for i in comb[0:1]:
                if i == 0:
                    c41 = c31.copy()
                    c42 = c32.copy()
                else:
                    c41 = c32.copy()
                    c42 = c31.copy()
                coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
                pshift = np.arctan2(np.mean(np.sin(coords_2_1-coords2),0), np.mean(np.cos(coords_2_1 - coords2),0))
    return pshift



#@numba.njit(
#    fastmath=True#, parallel = True
#) 
def unwrap_coords(cctrial_temp, ):
    _2_PI = 2*np.pi
    cctrial = np.zeros(cctrial_temp.shape)
    cctrial[0,:] = cctrial_temp[0,:]            
    k1, k2 = 0, 0
    for cn  in range(len(cctrial_temp)-1):
        c1 = cctrial_temp[cn+1][0]-cctrial[cn][0]
        c2 = cctrial_temp[cn+1][1]-cctrial[cn][1]
        c_temp = [[c1 + k1*_2_PI, c2 + k2*_2_PI],
                  [c1 + (k1+1)*_2_PI,  c2 + k2*_2_PI],
                  [c1 + k1*_2_PI, c2 + (k2+1)*_2_PI],
                  [c1 + (k1+1)*_2_PI, c2 + (k2+1)*_2_PI],
                  [c1 + (k1-1)*_2_PI, c2 + k2*_2_PI],
                  [c1 + k1*_2_PI, c2 + (k2-1)*_2_PI],
                  [c1 + (k1-1)*_2_PI, c2 + (k2-1)*_2_PI],
                  [c1 + (k1+1)*_2_PI, c2 + (k2-1)*_2_PI],
                  [c1 + (k1-1)*_2_PI, c2 + (k2+1)*_2_PI]]
        cc = np.array([[c[0]*c[0]+c[1]*c[1]] for c in c_temp])
        cmin = np.argmin(cc)
        cctrial[cn+1,:] = c_temp[cmin]+cctrial[cn]
        k1 = k1 + ks[cmin][0]
        k2 = k2 + ks[cmin][1]
    return cctrial



def get_cross_corr(sspikes, lencorr = 30, bNorm = False):
    """
    Compute cross correlation across 'lencorr' time bins between columns of 'sspikes' 
    Normalize if necessary    
    """
    lenspk,num_neurons = np.shape(sspikes)
    crosscorrs = np.zeros((num_neurons, num_neurons, lencorr))
    for i in range(num_neurons):
        spk_tmp0 = np.concatenate((sspikes[:,i].copy(), np.zeros(lencorr)))
        spk_tmp = np.array([np.roll(spk_tmp0, t1)[:lenspk] for t1 in range(lencorr)])
        crosscorrs[i,:,:] = np.dot(spk_tmp, sspikes).T   
    if bNorm:
        norm_spk = np.ones((num_neurons))
        for i in range(num_neurons):
            norm_spk[i] = np.sum(np.square(sspikes[:,i]))
        for i in range(num_neurons):
            crosscorrs[i,:,:] /= (norm_spk[i]*norm_spk[:, np.newaxis])
    return crosscorrs

def get_xcorr_dist(crosscorrs, bPlot = True):
    num_neurons = len(crosscorrs)
    crosscorrs1 = np.zeros((num_neurons,num_neurons))    
    for i1 in range(num_neurons):
        for j1 in np.arange(i1+1, num_neurons):
            a = crosscorrs[i1,j1,:]
            b = crosscorrs[j1,i1,:]
            c = np.concatenate((a,b))                     
            crosscorrs1[i1,j1] =  (np.min(c)/np.max(c))
            crosscorrs1[j1,i1] = crosscorrs1[i1,j1]
    vals = np.unique(crosscorrs1)
    if bPlot:
        plt.figure()
        plt.imshow(crosscorrs1, vmin = vals[int(0.05*len(vals))], vmax = vals[int(0.95*len(vals))])
        plt.colorbar()
    return crosscorrs1


def align(coords1, comb,):
#    addconst = np.pi/3
#    addconst = np.sqrt(3)/2
#    addconst = 2/np.sqrt(3)
    addconst = 1
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()
    for j in comb[1:2]:
        if j == 0:
            c21 = c1.copy()
            c22 = c2.copy()
        elif j == 1:
            c21 = 2*np.pi-c1.copy()
            c22 = 2*np.pi-c2.copy()
            
        elif j == 2:
            c21 = c1.copy()
            c22 = 2*np.pi-c2.copy()
        elif j == 3:
            c21 = 2*np.pi-c1.copy()
            c22 = c2.copy()           
        for k in comb[2:3]:
            if k == 0:
                c31 = c21.copy()
                c32 = c22.copy()
            elif k == 1:
                c31 = c21.copy() - addconst*c22
                c32 = c22.copy()
            elif k == 2:
                c31 = c21.copy() + addconst*c22
                c32 = c22.copy()
            elif k == 3:
                c31 = c21.copy()
                c32 = c22.copy() - addconst*c21
            elif k == 4:
                c31 = c21.copy()
                c32 = c22.copy() + addconst*c21  
                
            for i in comb[0:1]:
                if i == 0:
                    c41 = c31.copy()
                    c42 = c32.copy()
                else:
                    c41 = c32.copy()
                    c42 = c31.copy()
                coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
    return coords_2_1


def get_combs(theta1, theta):     
#    if ((theta1>np.pi) & (theta1<=3/2*np.pi)) | ((theta1>0) & (theta1<=np.pi/2)):
#        if ((theta>np.pi) & (theta<=3/2*np.pi)) | ((theta>0) & (theta<=np.pi/2)):
    if theta1:
        if theta:
            combs1 = combs_all[3]
        else:
            combs1 = combs_all[1]
    else:
        if theta:
            combs1 = combs_all[0]
        else:
            combs1 = combs_all[2]   
    return combs1

def fit_coords(coords1, coords2, combs1, times = [], thresh = 0.1): 
    res2 = []        
    for (i,j,k) in combs1:
        coords_2_1 = align(coords1, [i,j,k])
        pshift = np.arctan2(np.mean(np.sin(coords_2_1-coords2),0), np.mean(np.cos(coords_2_1 - coords2),0))
        coords_2_1 = (coords_2_1-pshift)%(2*np.pi)
        res2.extend([np.mean(np.abs(np.arctan2(np.sin(coords_2_1 - coords2), np.cos(coords_2_1 - coords2))),0)])
    return res2

def fit_derivative2(coords1, coords2, combs1, times = [], thresh = 0.1): 
    res2 = []    
    cctrial = unwrap_coords(coords2.copy())
    cs11 = CubicSpline(times, cctrial[times,:])
    dcs11 = cs11.derivative(1)
    angular_rate1 = np.arctan2(dcs11(times)[:,1],dcs11(times)[:,0])
    cs12 = CubicSpline(times, np.cos(angular_rate1)).derivative(1)
    cs13 = CubicSpline(times, np.sin(angular_rate1)).derivative(1)
    dtrajs2 = np.concatenate((cs12(times)[:,np.newaxis], cs13(times)[:,np.newaxis]),1)
    dtrajs2[dtrajs2>thresh]  = thresh
    dtrajs2[dtrajs2<-thresh]  = -thresh
    dtrajs2 = preprocessing.scale(dtrajs2)
    Ltemp = np.sqrt(np.sum(np.square(dtrajs2),0))
    
    for (i,j,k) in combs1:
        coords_2_1 = align(coords1, [i,j,k])
        cctrial = unwrap_coords(coords_2_1.copy())
        cs11 = CubicSpline(times, cctrial[times,:])
        dcs11 = cs11.derivative(1)
        angular_rate1 = np.arctan2(dcs11(times)[:,1],dcs11(times)[:,0])
        cs12 = CubicSpline(times, np.cos(angular_rate1)).derivative(1)
        cs13 = CubicSpline(times, np.sin(angular_rate1)).derivative(1)
        dtrajs1 = np.concatenate((cs12(times)[:,np.newaxis], cs13(times)[:,np.newaxis]),1)
        dtrajs1[dtrajs1>thresh]  = thresh
        dtrajs1[dtrajs1<-thresh]  = -thresh
        dtrajs1 = preprocessing.scale(dtrajs1)        
        res2.extend([np.sum(1- np.diagonal(np.matmul(dtrajs1.T, dtrajs2))/np.multiply(Ltemp,np.sqrt(np.sum(np.square(dtrajs1)))))])
    return res2

def fit_derivative1(coords1, coords2, combs1, times = [], thresh = 0.1): 
    res2 = []    
    cctrial_temp = coords2.copy()
    cctrial = np.zeros_like(cctrial_temp)
    cctrial[0,:] = cctrial_temp[0,:].copy()            
    k1, k2 = 0, 0
    for cn  in range(len(cctrial_temp)-1):
        ctmp1 = cctrial_temp[cn+1]
        c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                  ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                  ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                  ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                  ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                  ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                  ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                  ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                  ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                 ]  
        cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
        cctrial[cn+1,:] = c_temp[cmin]
        k1 += ks[cmin][0]
        k2 += ks[cmin][1]          
    cs11 = CubicSpline(times, cctrial[times,:])
    dcs11 = cs11.derivative(1)
    angular_rate1 = np.arctan2(dcs11(times)[:,1],dcs11(times)[:,0])
    cs12 = CubicSpline(times, np.cos(angular_rate1)).derivative(1)
    cs13 = CubicSpline(times, np.sin(angular_rate1)).derivative(1)
    dtrajs2 = np.concatenate((cs12(times)[:,np.newaxis], cs13(times)[:,np.newaxis]),1)
    dtrajs2[dtrajs2>thresh]  = thresh
    dtrajs2[dtrajs2<-thresh]  = -thresh
    dtrajs2 = preprocessing.scale(dtrajs2)
    Ltemp = np.sqrt(np.sum(np.square(dtrajs2),0))
    
    for (i,j,k) in combs1:
        coords_2_1 = align(coords1, [i,j,k], coords2)
        cctrial_temp = coords_2_1
        cctrial = np.zeros_like(cctrial_temp)
        cctrial[0,:] = cctrial_temp[0,:].copy()            
        k1, k2 = 0, 0
        for cn  in range(len(cctrial_temp)-1):
            ctmp1 = cctrial_temp[cn+1]
            c_temp = [ctmp1 + (k1*2*np.pi, k2*2*np.pi), 
                      ctmp1 + ((k1+1)*2*np.pi, k2*2*np.pi), 
                      ctmp1 + (k1*2*np.pi, (k2+1)*2*np.pi), 
                      ctmp1 + ((k1+1)*2*np.pi, (k2+1)*2*np.pi), 
                      ctmp1 + ((k1-1)*2*np.pi, k2*2*np.pi), 
                      ctmp1 + (k1*2*np.pi, (k2-1)*2*np.pi), 
                      ctmp1 + ((k1-1)*2*np.pi, (k2-1)*2*np.pi), 
                      ctmp1 + ((k1+1)*2*np.pi, (k2-1)*2*np.pi), 
                      ctmp1 + ((k1-1)*2*np.pi, (k2+1)*2*np.pi), 
                     ]  
            cmin = np.argmin(np.sum(np.square(c_temp-cctrial[cn]),1))
            cctrial[cn+1,:] = c_temp[cmin]
            k1 += ks[cmin][0]
            k2 += ks[cmin][1]
        cs11 = CubicSpline(times, cctrial[times,:])
        dcs11 = cs11.derivative(1)
        angular_rate1 = np.arctan2(dcs11(times)[:,1],dcs11(times)[:,0])
        cs12 = CubicSpline(times, np.cos(angular_rate1)).derivative(1)
        cs13 = CubicSpline(times, np.sin(angular_rate1)).derivative(1)
        dtrajs1 = np.concatenate((cs12(times)[:,np.newaxis], cs13(times)[:,np.newaxis]),1)
        dtrajs1[dtrajs1>thresh]  = thresh
        dtrajs1[dtrajs1<-thresh]  = -thresh
        dtrajs1 = preprocessing.scale(dtrajs1)        
        res2.extend([np.sum(1- np.diagonal(np.matmul(dtrajs1.T, dtrajs2))/np.multiply(Ltemp,np.sqrt(np.sum(np.square(dtrajs1)))))])
    return res2
        


def align_bu1(coords1, comb, pshift,k):
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()

    for i in [comb[0],]:
        if i == 0:
            c11 = c2.copy()
            c12 = c1.copy()
        else:   
            c11 = c1.copy()
            c12 = c2.copy()
        for j in [comb[1],]:
            if j == 0:
                c31 = c11.copy()
                c32 = (2*np.pi-c12)
            else:
                c31 = c11.copy()
                c32 = c12.copy() 
            for m in [comb[2],]:       
                if m == 0:
                    c41 = (c31 + np.pi/3*c32)
                    c42 = c32.copy()
                elif m== 1:
                    c41 = (c31 - np.pi/3*c32)
                    c42 = c32.copy()
                elif m == 2:
                    c41 = c31.copy() 
                    c42 = c32 + np.pi/3*c31
                elif m == 3:
                    c41 = c31.copy() 
                    c42 = c32 - np.pi/3*c31
                elif m == 4:
                    c41 = (2*np.pi-c31 + np.pi/3*c32)
                    c42 = c32.copy()
                elif m== 5:
                    c41 = (2*np.pi-c31 - np.pi/3*c32)
                    c42 = c32.copy()
                elif m == 6:
                    c41 = c31.copy() 
                    c42 = 2*np.pi-c32 + np.pi/3*c31
                elif m == 7:
                    c41 = c31.copy() 
                    c42 = 2*np.pi-c32 - np.pi/3*c31
                else:
                    c41 = c31.copy()
                    c42 = c32.copy()
                coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
                if k == 1:
                    coords_2_1 = 2*np.pi-coords_2_1
                coords_mod_1_2 = (coords_2_1 - pshift)%(2*np.pi)
    return coords_mod_1_2

def align_bu(coords1, comb, pshift):
    c1 = coords1[:,0].copy()
    c2 = coords1[:,1].copy()

    for i in [comb[0],]:
        if i == 0:
            c11 = c2.copy()
            c12 = c1.copy()
        else:   
            c11 = c1.copy()
            c12 = c2.copy()
        for j in [comb[1],]:
            if j == 0:
                c21 = c11.copy()
                c22 = (2*np.pi-c12)
            else:
                c21 = c11.copy()
                c22 = c12.copy()            
            for k in [comb[2],]:
                if k == 0:
                    c31 = (2*np.pi-c21)
                    c32 = c22.copy()
                else:
                    c31 = c21.copy()
                    c32 = c22.copy()
                for m in [comb[3],]:       
                    if m == 0:
                        c41 = (c31 + np.pi/3*c32)
                        c42 = c32.copy()
                    elif m== 1:
                        c41 = (c31 - np.pi/3*c32)
                        c42 = c32.copy()
                    elif m == 2:
                        c41 = c31.copy() 
                        c42 = c32 + np.pi/3*c31
                    elif m == 3:
                        c41 = c31.copy() 
                        c42 = c32 - np.pi/3*c31
                    elif m == 4:
                        c41 = (2*np.pi-c31 + np.pi/3*c32)
                        c42 = c32.copy()
                    elif m== 5:
                        c41 = (2*np.pi-c31 - np.pi/3*c32)
                        c42 = c32.copy()
                    elif m == 6:
                        c41 = c31.copy() 
                        c42 = 2*np.pi-c32 + np.pi/3*c31
                    elif m == 7:
                        c41 = c31.copy() 
                        c42 = 2*np.pi-c32 - np.pi/3*c31
                    else:
                        c41 = c31.copy()
                        c42 = c32.copy()
                    coords_2_1 = np.concatenate((c41[:,np.newaxis], c42[:,np.newaxis]),1)%(2*np.pi)
                    coords_mod_1_2 = (coords_2_1 - pshift)%(2*np.pi)
    return coords_mod_1_2

def get_mean_acorr(coords, spk, bSaveFig = False, bins = 50, times = []):
    numbins = 51
    numangsint = 52
    numangsint_1 = numangsint-1
    mid = int((numangsint_1)/2)
    bins = np.linspace(0,2*np.pi, numbins+1)
    num_neurons = len(spk[0,:])
    acorr0 = {}
    for k in range(num_neurons):
        mtot = binned_statistic_2d(coords[:, 0], coords[:,1], spk[:,k],
                                   bins = bins, range = None, expand_binnumbers = True)[0]
        mtot[np.isnan(mtot)] = np.mean(mtot[~np.isnan(mtot)])
        acorr0[k] = pearson_correlate2d(mtot, mtot)
        acorr0[k][mid,mid] = -np.inf
        acorr0[k][mid,mid] = np.max(acorr0[k])
    acorrall0 = np.zeros_like(acorr0[0])
    for j in acorr0:
        acorrall0 += (acorr0[j]-np.mean(acorr0[j]))/np.std(acorr0[j])#acorr0[j]
    return acorrall0/num_neurons

def plot_mean_acorr(coords, spk, bSaveFig = False, bins = 50, times = [], fname = ''):
    if len(times) == 0:
        times = np.arange(len(coords[:,0]))
    acorrall0 = get_mean_acorr(coords, spk, bSaveFig, bins, times)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(acorrall0.T, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = 0, vmax = np.max(acorrall0) *0.975)
    ax.set_aspect('equal', 'box') 
    ax.axis('off')
    if bSaveFig:
        plt.savefig(fname)
        plt.close()
    return acorrall0

def get_dgms(sspikes2, speed1,num_times  = 1, lencube = 50000, maxdim = 1,     
             omega = 1, k  = 1000, n_points = 800, sp = 10,
             dim = 7, nbs = 800, eps = 1, metric = 'cosine',
             indstemp = [], speed_times = []):
    
    dgms_all = {}
    speed_times = np.where(speed1>sp)[0]    
    sp1 = np.arange(0,len(speed_times),num_times)
    speed_times = speed_times[sp1]

    dim_red_spikes_move_scaled = preprocessing.scale(sspikes2[speed_times,:], axis = 0)
    dim_red_spikes_move_scaled, e1, e2 = pca(dim_red_spikes_move_scaled, dim = dim)
    dim_red_spikes_move_scaled /= np.sqrt(e2[:dim])
    
    if len(indstemp)==0:
        startindex = np.argmax(np.sum(np.abs(dim_red_spikes_move_scaled),1))
        movetimes1 = radial_downsampling(dim_red_spikes_move_scaled, epsilon = eps, 
            startindex = startindex)
        indstemp,__  = sample_denoising(dim_red_spikes_move_scaled[movetimes1,:],  k, 
                                           n_points, omega, metric)
        indstemp = movetimes1[indstemp]
    else:
        movetimes1 = []
    dim_red_spikes_move_scaled = dim_red_spikes_move_scaled[indstemp,:]
    X = squareform(pdist(dim_red_spikes_move_scaled[:,:], metric))
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
    thresh = np.max(d[~np.isinf(d)])
    hom_dims = list(range(maxdim+1))
    VR = VietorisRipsPersistence(
    homology_dimensions=hom_dims,
    metric='precomputed',
    coeff=47,
    max_edge_length= thresh,
    collapse_edges=False,  # True faster?
    n_jobs=None  # -1 faster?
    )
    diagrams = VR.fit_transform([d])
    dgms_all[0] = from_giotto_to_ripser(diagrams[0])
    persistence = ripser(d, maxdim=maxdim, coeff=47, do_cocycles= True, distance_matrix = True, thresh = thresh)    
    return dgms_all, persistence, indstemp, movetimes1, speed_times



def get_data(files, sigma = 6000):
    ################### Get good cells ####################    
    good_cells = []
    for fi in files:
        data = loadmat(fi)
        anatomy = data['anatomy']
        if 'parent_shifted' in anatomy:
            group = anatomy['parent_shifted']
        else:
            group = anatomy['cluster_parent']
        regions = ('MEC',)#'VISp','RS')

        idx = [str(ss).startswith(regions) for ss in group]
        idxagl = [str(ss).startswith('RSPagl') for ss in group]
        region_idx = np.array(idx) & np.logical_not(np.array(idxagl))
        _,sn = os.path.split(fi)
        good_cells.extend(data['sp']['cids'][(data['sp']['cgs']==2) & region_idx])
    good_cells = np.where(np.bincount(good_cells)==len(files))[0]

    data_trials = {}
    data_pos = {}
    sspk1 = {}
    spk1 = {}
    speed = {}
    posx = {}
    post = {}
    posxx = {}
    postt = {}
    postrial = {}
    gain = {}
    contrast = {}
    lickt = {}
    lickx = {}
    res = 100000
    dt = 1000

    thresh = sigma*5
    num_thresh = int(thresh/dt)
    num2_thresh = int(2*num_thresh)
    sig2 = 1/(2*(sigma/res)**2)
    ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
    kerwhere = np.arange(-num_thresh,num_thresh)*dt

    for fi in files:
        data = loadmat(fi)
        spikes = {}
        for cell_idx in range(len(good_cells)):   
            spikes[cell_idx] = data['sp']['st'][data['sp']['clu']==good_cells[cell_idx]]        
        sspikes = np.zeros((1,len(spikes)))
        min_time = 0
        max_time = data['sp']['st'].max()*res+dt
        tt = np.arange(np.floor(min_time), np.ceil(max_time), dt)

        spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes)))
        for n, spk in enumerate(spikes):
            spk = spikes[spk]
            spikes1 = np.array(spk*res-min_time, dtype = int)
            spikes1 = spikes1[(spikes1 < (max_time-min_time)) & (spikes1 > 0)]
            spikes_mod = dt-spikes1%dt
            spikes1 = np.array(spikes1/dt, int)
            for m, j in enumerate(spikes1):
                spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
        spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
        sspikes = np.concatenate((sspikes, spikes_temp),0)
        sspikes = sspikes[1:,:]
        sspikes *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
        sspk1[fi] = sspikes.copy()

        print(sspikes.shape)
        spikes_bin = np.zeros((len(tt), len(spikes)), dtype = int)    
        for n in spikes:
            spike_times = np.array(spikes[n]*res-min_time, dtype = int)
            spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
            spikes_mod = dt-spike_times%dt
            spike_times= np.array(spike_times/dt, int)
            for m, j in enumerate(spike_times):
                spikes_bin[j, n] += 1

        spk1[fi] = spikes_bin.copy()

        tt /= res

        xtmp = data['posx'].copy()
        xtmp += (data['trial']-1)*400
        xtmp = gaussian_filter1d(xtmp, sigma = 5)        
        xx_spline = CubicSpline(data['post'], xtmp)
        speed0 = xx_spline(tt,1)
        
        speed[fi] = speed0.copy()

        xx_spline = CubicSpline(data['post'], data['posx'])

        posxx[fi] = xx_spline(tt).copy()%400
        postt[fi] = tt.copy()

        posx[fi] = data['posx']
        post[fi] = data['post']
        postrial[fi] = data['trial']
        gain[fi] = data['trial_gain']
        contrast[fi] = data['trial_contrast']
        lickt[fi] = data['lickt']        
        lickx[fi] = data['lickx']
        tt_dig = np.digitize(tt, data['post'])-1    
        postrial[fi] = data['trial'][tt_dig]
        data_trials[fi] = np.unique(data['trial'])

    ################### Filter firing rate ####################    
    num_neurons = len(good_cells)
    indsnull = np.ones(num_neurons,dtype =bool)
    for fi in spk1:
        spksum = np.sum(spk1[fi], 0)/len(spk1[fi])*100
        indsnull[(spksum<0.05) | (spksum>10)] = False
    ################### concatenate spikes ####################
    sspikes1 = np.zeros((1,sum(indsnull)))
    for fi in files:
        sspikes1 = np.concatenate((sspikes1, sspk1[fi][:,indsnull]),0)            
    sspikes1 = sspikes1[1:,:]
    speed1 = []
    for fi in files:
        speed1.extend(speed[fi])
    speed1 = np.array(speed1)
    return sspikes1, speed1, sspk1, spk1, good_cells, indsnull, speed, data_trials, data_pos, posx, post, posxx, postt, postrial, gain, contrast, lickt, lickx


def get_cross(mouse_sess, data_dir = 'giocomo_figures0/', files = []):
    crosscorrfile = glob.glob(data_dir + '/' + mouse_sess + '_crosscorr_train2.npz')
    if len(crosscorrfile)>-10:
        f = np.load(crosscorrfile[0], allow_pickle = True)
        crosscorr_train = f['crosscorr_train'][()]
        f.close()
    else:
        sspk1_cross = {}
        res = 100000
        sigma = 30000
        dt = 3000
        thresh = sigma*5
        num_thresh = int(thresh/dt)
        num2_thresh = int(2*num_thresh)
        sig2 = 1/(2*(sigma/res)**2)
        ker = np.exp(-np.power(np.arange(thresh+1)/res, 2)*sig2)
        kerwhere = np.arange(-num_thresh,num_thresh)*dt

        for fi in files:
            data = loadmat(fi)
            spikes = {}
            for cell_idx in range(len(good_cells)):   
                spikes[cell_idx] = data['sp']['st'][data['sp']['clu']==good_cells[cell_idx]]        
            sspikes = np.zeros((1,len(spikes)))
            min_time = 0
            max_time = data['sp']['st'].max()*res+dt
            tt = np.arange(np.floor(min_time), np.ceil(max_time), dt)

            spikes_temp = np.zeros((len(tt)+num2_thresh, len(spikes)))
            for n, spk in enumerate(spikes):
                spk = spikes[spk]
                spikes1 = np.array(spk*res-min_time, dtype = int)
                spikes1 = spikes1[(spikes1 < (max_time-min_time)) & (spikes1 > 0)]
                spikes_mod = dt-spikes1%dt
                spikes1 = np.array(spikes1/dt, int)
                for m, j in enumerate(spikes1):
                    spikes_temp[j:j+num2_thresh, n] += ker[np.abs(kerwhere+spikes_mod[m])]
            spikes_temp = spikes_temp[num_thresh-1:-(num_thresh+1),:]
            sspikes = np.concatenate((sspikes, spikes_temp),0)
            sspikes = sspikes[1:,:]
            sspikes *= 1/np.sqrt(2*np.pi*(sigma/res)**2)
            sspk1_cross[fi] = sspikes.copy()
            print(sspikes.shape)

        ################### Get crosscorrs ####################    
        crosscorr_train = {}
        for fi in files[:]:
            sspk1_d = sspk1_cross[fi].copy().astype(float)        
            num_neurons = len(sspk1_d[0,:])
            crosscorrs = np.zeros((len(sspk1_d[0,:]), len(sspk1_d[0,:]), lencorr))
            spksum = np.sum(sspk1_d,0)
            for i in np.where(indsnull)[0]:
                spktemp = np.concatenate((sspk1_d[:,i], np.zeros(lencorr)))
                lenspk = len(sspk1_d[:,i])
                for j in np.where(indsnull)[0]:
                    if j == i:
                        continue
                    for t1 in range(lencorr):
                        crosscorrs[i,j,t1] = np.dot(sspk1_d[:,j],np.roll(spktemp, t1)[:lenspk])
                    crosscorrs[i,j,:] /= (spksum[j]*spksum[i])
            crosscorr_train[fi] = crosscorrs.copy()
        np.savez(data_dir + '/' + mouse_sess + '_crosscorr_train2', crosscorr_train = crosscorr_train)
    return crosscorr_train



def get_ind1(mouse_sess, data_dir = 'giocomo_figures_final12', indsnull = [], posxx = [], nbs = -1, lentmp = 0):
    indfile = glob.glob(data_dir + '/' + mouse_sess + '_mods.npz')
    if len(indfile)>0:
        f = np.load(indfile[0], allow_pickle = True)
        ind = f['ind'][()]
        f.close()
    else:
        ################### Get crosscorr stats ####################  
        crosscorr_train = get_cross(mouse_sess)
        num_neurons = len(good_cells)
        crosscorrs = np.zeros((num_neurons,num_neurons))
        for fi in files: 
            fi1 = fi.replace('\\', '/')
            crosscorrs_tmp = crosscorr_train[fi1].copy()
            num_neurons = len(crosscorrs_tmp[:,0,0])
            for i in range(num_neurons):
                for j in np.arange(i+1, num_neurons):
                    a = crosscorrs_tmp[i,j,:lentmp]
                    b = crosscorrs_tmp[j,i,:lentmp]
                    c = np.concatenate((a,b))
#                    crosscorrs[i,j] +=  (1-np.exp(-np.square(np.min(c)/np.max(c))))/len(files)
                    crosscorrs[i,j] +=  np.square(np.min(c)/np.max(c))/len(files)
                    crosscorrs[j,i] = crosscorrs[i,j]

        num_neurons = np.sum(indsnull)
        crosscorr_tmp = crosscorrs[indsnull,:]
        crosscorr_tmp = crosscorr_tmp[:,indsnull]
        X1  = crosscorr_tmp
        X1[np.isnan(X1)] = 1
        X1[np.isinf(X1)] = 1
        agg = AgglomerativeClustering(n_clusters=None,affinity='precomputed', linkage='average', 
                                      distance_threshold=nbs)
        ind = agg.fit(X1).labels_
        np.savez(data_dir + '/' + mouse_sess + '_mods', ind = ind)
    return ind




def get_ind(crosscorrs, nbs = -1, linkage = 'average', aff = 'precomputed', bPlot = True, bClass = False):
    """
    Cluster correlations (or any matrix) by average linkage clustering
    
    nbs = threshold for average value in each cluster
    linkage = {'average', 'complete' (maximum), 'single', 'ward' (variance, affinity needs to be 'euclidean')}
    aff = metric/affinity
    """
    crosscorrs[np.isinf(crosscorrs)] = 2*np.max(crosscorrs[~np.isinf(crosscorrs)])
    agg = AgglomerativeClustering(n_clusters=None, linkage=linkage, 
                                  distance_threshold=nbs)
    ind = agg.fit(crosscorrs).labels_
    if bPlot:
        fig = plt.figure()
        crosscorrs = crosscorrs[np.argsort(ind), :]
        crosscorrs = crosscorrs[:, np.argsort(ind)]    
        vals = np.unique(crosscorrs)
        plt.imshow(crosscorrs, 
                   vmin = vals[int(len(vals)*0.05)],
                   vmax = vals[int(len(vals)*0.95)],
                  )
        bin_ind = np.bincount(ind)
        numneuronsind = np.flip(np.argsort(bin_ind))
        print('num: ', bin_ind[numneuronsind[:10]])
        print('ind: ', numneuronsind[:10])
    if bClass:
        return ind, agg
    else:
        return ind


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)




def cross_corr_dist(sspikes, lencorr = 30, numcorr = -1):
    """
    Compute cross correlation across 'lencorr' time lags between columns of 'sspikes'    
    """
    
    lenspk,num_neurons = np.shape(sspikes)
    crosscorrs = np.zeros((num_neurons, num_neurons, lencorr))
    if numcorr>0:
        num_times = np.arange(0, lenspk, numcorr)
        sspikes_tmp = sspikes[num_times,:]
        for i in range(num_neurons):
            spk_tmp0 = np.concatenate((sspikes[:,i].copy(), np.zeros(lencorr)))
            spk_tmp = np.array([np.roll(spk_tmp0, t1)[:lenspk][num_times] for t1 in range(lencorr)])
            crosscorrs[i,:,:] = np.dot(spk_tmp, sspikes_tmp).T
    else:
        for i in range(num_neurons):
            spk_tmp0 = np.concatenate((sspikes[:,i].copy(), np.zeros(lencorr)))
            spk_tmp = np.array([np.roll(spk_tmp0, t1)[:lenspk] for t1 in range(lencorr)])
            crosscorrs[i,:,:] = np.dot(spk_tmp, sspikes).T

    crosscorrs1 = np.zeros((num_neurons,num_neurons))    
    for i1 in range(num_neurons):
        for j1 in np.arange(i1+1, num_neurons):
            a = crosscorrs[i1,j1,:]
            b = crosscorrs[j1,i1,:]
            c = np.concatenate((a,b))         
            crosscorrs1[i1,j1] =  (np.min(c)/np.max(c))
            crosscorrs1[j1,i1] = crosscorrs1[i1,j1]
    vals = np.unique(crosscorrs1)
    plt.figure()
    plt.imshow(crosscorrs1, vmin = vals[int(0.05*len(vals))], vmax = vals[int(0.95*len(vals))])
    plt.colorbar()
    return crosscorrs1

def spk_count(spikeTimes, res = 100000, dt_orig = 0.01, min_time = -1, max_time = None):
    """
    Compute spike count matrix
    
    res = time resolution 1/res seconds
    dt_orig = bin size in seconds 
    
    """
    dt = dt_orig*res
    if min_time<0:
        min_time = np.floor(spikeTimes[0][0])*res-dt
    else:
        min_time*=res

    if not max_time:
        max_time = np.ceil(spikeTimes[0][-1])*res+dt
    else:
        max_time*=res
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res
    spikes = np.zeros((len(tt), len(spikeTimes)))    
    for n, spk in enumerate(spikeTimes):
        spk = spikeTimes[spk]
        spike_times = np.array(spk*res-min_time, dtype = int)
        spike_times = spike_times[(spike_times < (max_time-min_time)) & (spike_times > 0)]
        spikes_mod = dt-spike_times%dt
        spike_times= np.array(spike_times/dt, int)
        for m, j in enumerate(spike_times):
            spikes[j, n] += 1
    return spikes, tt


def scatter_coords(coords,xx,yy):
    plt.hsv()
    fig, ax = plt.subplots(1,2) 
    ax[0].scatter(xx, yy, c =  coords[:,0], s = 10)
    ax[1].scatter(xx, yy, c =  coords[:,1], s = 10)
    for a in ax:
        a.axis('off')
        a.set_aspect(1/a.get_data_ratio())
    plt.show()
        

def scores_cluster(sspikes, scores, inds, xx= [],yy = [], spk2 = [], num_example = 3, 
    dim = 10, bSave = False, fname = '', bUMAP = True, xkey = []):
    import umap
    dim_red_spikes_move_scaled, e1, e2, var_exp = pca(preprocessing.scale(sspikes,axis = 0), dim = dim)    
    times = np.arange(0,len(dim_red_spikes_move_scaled), int(np.ceil(len(dim_red_spikes_move_scaled)/4000)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(dim_red_spikes_move_scaled[times,0],dim_red_spikes_move_scaled[times,1],dim_red_spikes_move_scaled[times,2],
        c = np.arange(len(times)), s = 10,alpha = 0.9)
    if bUMAP:
        reducer = umap.UMAP(n_components=3,metric='cosine',random_state=42, n_neighbors = 100,
                           min_dist = 0.1, spread = 1)
        X_um = reducer.fit_transform(dim_red_spikes_move_scaled[times,:])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X_um[:,0], X_um[:,1], X_um[:,2],
            c = np.arange(len(times)), s = 10,alpha = 0.9)


    plt.figure()
    plt.plot(var_exp[:15])
    plt.xlabel('Principal component',fontsize = 16)
    plt.ylabel('Variance explained',fontsize = 16)
    plt.title('Num neurons: ' + str(len(inds)))
    if bSave:
        fig.savefig(fname + '_pca_var')

    fig, axs = plt.subplots(1,dim, figsize= (10,5), dpi = 120)
    axs[2].set_title(str(dim) + ' first principal components')
    plt.axis('off')
    if (len(xx)>0):
        if len(yy)>0:
            for c in range(dim):
                mtot, __, __, circ  = binned_statistic_2d(xx,
                                                          yy,
                                                          dim_red_spikes_move_scaled[:,c], 
                                                          statistic = 'mean', 
                                                          bins = 30,
                                                          expand_binnumbers = True)

                nans = np.isnan(mtot)
                mtot[nans] = np.percentile(mtot[~nans], 50)
                mtot = gaussian_filter(mtot, 1)
                plt.viridis()
                mtot[nans] = np.nan
                axs[c].imshow(mtot, 
                              vmin = np.percentile(mtot[~nans], 5), 
                              vmax = np.percentile(mtot[~nans], 95))
                axs[c].axis('off')
                axs[c].set_aspect(1/axs[c].get_data_ratio())
        else:
            for c in range(dim):
                mtot, __, circ  = binned_statistic(xx,dim_red_spikes_move_scaled[:,c], 
                                                          statistic = 'mean', 
                                                          bins = 30,)

                nans = np.isnan(mtot)
                mtot[nans] = np.percentile(mtot[~nans], 50)
                mtot = gaussian_filter(mtot, 1)
                plt.viridis()
                axs[c].plot(mtot,)
                axs[c].set_aspect(1/axs[c].get_data_ratio())
                axs[c].axis('off')
    else:
        for c in range(dim):
            axs[c].plot(dim_red_spikes_move_scaled[:,c])
            axs[c].axis('off')
            axs[c].set_aspect(1/axs[c].get_data_ratio())

    if bSave:
        fig.savefig(fname + '_pca_tuning')
    if len(spk2)>0:
        dim_red_spikes_move_scaled, e1, e2, var_exp = pca(preprocessing.scale(spk2,axis = 0), dim = dim)    
        plt.figure()
        plt.plot(var_exp[:15])
        plt.xlabel('Principal component',fontsize = 16)
        plt.ylabel('Variance explained',fontsize = 16)
        plt.title('Num neurons: ' + str(len(inds)))
        fig, axs = plt.subplots(1,dim, figsize= (10,5), dpi = 120)
        for c in range(dim):
            axs[c].plot(dim_red_spikes_move_scaled[:,c])
            axs[c].axis('off')
            axs[c].set_aspect(1/axs[c].get_data_ratio())

    names = []
    for name, score in scores:
        names.extend(name)
        currscore = score[inds].copy()
        nans = np.isnan(currscore)
        currscore[nans] = 0
        if name in ('rmap','acorr2d'):
            nans = ~(np.isnan(currscore) | np.isinf(currscore))
            currscore -= np.nanpercentile(currscore,50, axis = (1,2))[:,np.newaxis,np.newaxis]
            currscore1 = np.sum(currscore,0)
            nans = ~(np.isnan(currscore1) | np.isinf(currscore1))
            fig = plt.figure()
            plt.imshow(currscore1, 
                       vmin = np.percentile(currscore1[nans].flatten(), 5), 
                       vmax = np.percentile(currscore1[nans].flatten(), 95))
            plt.title('Stacked ' + name)
            fig, ax = plt.subplots(1,num_example)
            ax[0].set_title('example neurons')
            indstemp = np.arange(len(inds))
            np.random.shuffle(indstemp)
            for i, ii in enumerate(indstemp[:num_example]):
                vals = np.unique(currscore[ii,:,:])
                nans = ~(np.isnan(currscore[ii,:,:]) | np.isinf(currscore[ii,:,:]))
                ax[i].imshow(currscore[ii,:,:], 
                             vmin = np.percentile(currscore[ii,:,:][nans].flatten(), 5), 
                             vmax = np.percentile(currscore[ii,:,:][nans].flatten(), 95))
                ax[i].axis('off')
        elif name in ('acorr1d'):
            
            currscore -= np.percentile(currscore[~np.isnan(currscore)],50,axis = 1)[:,np.newaxis]
            currmean = currscore.mean(0)
            currstd = currscore.std(0)#/np.sqrt(len(currtmp[:,0]))
            fig, ax = plt.subplots(1,1)
            ax.plot(currmean, lw = 2, c= '#1f77b4', alpha = 1,)
            ax.fill_between(np.arange(len(currmean)),currmean, currmean + currstd,
                            lw = 0, alpha = 0.3, color = '#1f77b4')
            ax.fill_between(np.arange(len(currmean)),currmean, currmean - currstd,
                            lw = 0, alpha = 0.3, color= '#1f77b4')
            ax.set_aspect(1/ax.get_data_ratio())
            ax.set_title('Mean ' + name)            
            if bSave:
                fig.savefig(fname + '_acorr_mean')
                    
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(currscore, vmin = 0., vmax = 0.1)
            ax.set_aspect(1/ax.get_data_ratio())
            if bSave:
                fig.savefig(fname + '_acorr_all')

        elif name in ('tacorrs',):
            currscore -= np.mean(currscore,(1))[:,np.newaxis]
            currmean = currscore.mean(0)
            currstd = currscore.std(0)#/np.sqrt(len(currtmp[:,0]))
            fig, ax = plt.subplots(1,1)
            ax.plot(currmean, lw = 2, c= '#1f77b4', alpha = 1,)
            ax.fill_between(np.arange(len(currmean)),currmean, currmean + currstd,
                            lw = 0, alpha = 0.3, color = '#1f77b4')
            ax.fill_between(np.arange(len(currmean)),currmean, currmean - currstd,
                            lw = 0, alpha = 0.3, color= '#1f77b4')
            ax.set_aspect(1/ax.get_data_ratio())
            ax.set_title('Mean ' + name)
            if bSave:
                fig.savefig(fname + '_tacorr_mean')

        else:
            if np.sum(nans)>0:
                print('nans ', sum(nans))
                currscore = currscore[~nans]
            plt.figure()
            plt.boxplot((score[~np.isnan(score)], currscore))
            plt.xticks([1,2], ['All', 'Cluster'], fontsize = 16)
            plt.title(name + ' mean: ' + str(np.round(np.mean(currscore[~np.isnan(currscore)]),2)) +
                       ' std: ' + str(np.round(np.std(currscore[~np.isnan(currscore)]),2)))
    hasbeen = []
    if len(xkey)==0:
        for name1, score1 in scores:
            xkey.extend([name1])
    for name1, score1 in scores:
        if name1 in xkey:
            for name2, score2 in scores:
                if ((name1 == name2) |  
                    (name1 in ('rmap', 'acorr','acorr2d', 'tacorrs')) |  
                    (name2 in ('rmap', 'acorr', 'acorr2d','tacorrs')) | 
                    ((name1 + name2) in hasbeen)):
                    continue
                hasbeen.extend([name1 + name2])
                hasbeen.extend([name2 + name1])
                plt.figure()
                nans = ~(np.isnan(score1) | np.isnan(score2))
                plt.scatter(score1[nans], score2[nans])
                nans = ~(np.isnan(score1[inds]) | np.isnan(score2[inds]))
                plt.scatter(score1[inds][nans], score2[inds][nans])
                plt.xlabel(name1, fontsize = 16)
                plt.ylabel(name2, fontsize = 16)
    print('')

def get_corr_dist(masscenters_1,masscenters_2, mtot_1, mtot_2, sig = 2.75, num_shuffle = 1000):
    numangsint = len(mtot_1[0,:,0])+1
    num_neurons = len(masscenters_1[:,0])
    cells_all = np.arange(num_neurons)
    corr = np.zeros(num_neurons)
    corr1 = np.zeros(num_neurons)
    corr2 = np.zeros(num_neurons)
    mtot_1_shuf = np.zeros_like(mtot_1)
    mtot_2_shuf = np.zeros_like(mtot_2)
    for n in cells_all:
        m1 = mtot_1[n,:,:].copy()
        m1[np.isnan(m1)] = np.min(m1[~np.isnan(m1)])
        m2 = mtot_2[n,:,:].copy()
        m2[np.isnan(m2)] = np.min(m2[~np.isnan(m2)])
        m1 = smooth_tuning_map(m1, numangsint, sig, bClose = False)
        m2 = smooth_tuning_map(m2, numangsint, sig, bClose = False)
        corr[n] = pearsonr(m1.flatten(), m2.flatten())[0]
        mtot_1_shuf[n,:,:]= m1
        mtot_2_shuf[n,:,:]= m2

    dist =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2),
                                  np.cos(masscenters_1 - masscenters_2))),1))
    dist_shuffle = np.zeros((num_shuffle, num_neurons))
    corr_shuffle = np.zeros((num_shuffle, num_neurons))
    np.random.seed(47)
    for i in range(num_shuffle):
        inds = np.arange(num_neurons)
        np.random.shuffle(inds)
        for n in cells_all:
            corr_shuffle[i,n] = pearsonr(mtot_1_shuf[n,:,:].flatten(), mtot_2_shuf[inds[n],:,:].flatten())[0]
        dist_shuffle[i,:] =  np.sqrt(np.sum(np.square(np.arctan2(np.sin(masscenters_1 - masscenters_2[inds,:]),
                np.cos(masscenters_1 - masscenters_2[inds,:]))),1))
    return corr, dist, corr_shuffle, dist_shuffle

          


def get_coords_consistent(rips_real, coeff,  ph_classes = [0,1], cthr = 0.99, bConsistent = False, bTogether = False):    
    n_landmarks = len(rips_real['dperm2all'])
    edges1 = []
    if bConsistent & ~bTogether & (len(ph_classes)>1):
        bTogether = True 
        print('bTogether = True')
    if bTogether:
        d = rips_real['dperm2all']
        cpick = np.max(ph_classes)+1
        lives1 = np.diff(rips_real['dgms'][1],axis = 1)[:,0]
        maxb = 0
        mind = np.inf
        for cc in (np.argsort(lives1)[-cpick:]):
            maxb = max(maxb, rips_real['dgms'][1][cc][0])
            mind = min(mind, rips_real['dgms'][1][cc][1])
        eps = (maxb + (mind-maxb)*cthr)
        if eps <0:
            return 0,0
        if np.isinf(eps):
            eps = np.max(d[~np.isinf(d)])
        d[np.tril_indices(n_landmarks)] = np.inf
        edges1 = np.array(np.where(d<= eps)).T
        verts = np.array(np.unique(edges1))
    num_edges1 = len(edges1)
    coeff = 47
    num_circ = len(ph_classes)    
    diagrams = rips_real["dgms"] # the multiset describing the lives of the rips_real classes
    cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
    births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
    lives1 = deaths1-births1 # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    Ys = []    
    coords1 = np.zeros((num_circ, n_landmarks))
    for it,c in enumerate(ph_classes):
        cocycle = cocycles[iMax[-(c+1)]].copy()
        zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
        cocycle[zint, 2] = cocycle[zint, 2] - coeff
        if not bTogether:
            d = rips_real['dperm2all'].copy()
            eps = (births1[iMax[-(c+1)]] + (deaths1[iMax[-(c+1)]]-births1[iMax[-(c+1)]])*cthr)
            if eps <0:
                return 0,0
            if np.isinf(eps):
                eps = np.max(d[~np.isinf(d)])
            d[np.tril_indices(n_landmarks)] = np.inf
            edges1 = np.array(np.where(d<= eps)).T
            num_edges1 = len(edges1)
            verts = np.array(np.unique(edges1))
        values = np.zeros(num_edges1)
        for i in cocycle:
            ii = np.where((edges1[:,0]==i[0]) & (edges1[:,1]==i[1]))[0]
            if len(ii) == 0:
                ii = np.where((edges1[:,1]==i[0]) & (edges1[:,0]==i[1]))[0]
                if len(ii) == 0:
                    continue
                values[ii[0]] = -i[2]
            else:
                values[ii[0]] = i[2]
        col = np.zeros(num_edges1*2, dtype=int)
        row = np.zeros(num_edges1*2, dtype=int)
        val = np.zeros(num_edges1*2, dtype=int)

        for i in range(num_edges1):
            j = i*2
            col[j] = np.where(verts == edges1[i,0])[0]
            col[j+1] = np.where(verts == edges1[i,1])[0]
            row[j:j+2] = i
            val[j] = -1
            val[j+1] = 1
        A = coo_matrix((val, (row, col)), shape=(num_edges1, n_landmarks))
        coords1[it,:] = lsqr(A,values)[0]
        Ys.append(values - A.dot(coords1[it,:]))
        coords1[it,:] = (coords1[it,:]%1)#*2*np.pi

    if bConsistent & bTogether:
        W = coo_matrix((np.ones(num_edges1), (np.arange(num_edges1), np.arange(num_edges1))), 
                                    shape=(num_edges1,num_edges1)).tocsr()
        gram_matrix = np.zeros((len(Ys),len(Ys)))
        for i in range(len(Ys)):
            for j in range(len(Ys)):
                gram_matrix[i,j] = Ys[i].T @ W @ Ys[j]
        circ_coords_consistent, ngm_consistent, cb = reduce_circular_coordinates(coords1, gram_matrix)
        return coords1, circ_coords_consistent
    else:
        return coords1

def plot_barcode(persistence, diagrams_roll = {}, file_name = '', disp = True, 
                 num_bars = 15, figsize = (5,10), dpi = 120, percshuf = 99, bMax = False, xmax = -1,
                SaveSourceDataName = ''):
    if 1 == 2:
        filenames=glob.glob('Results/Roll/' + file_name + '_H2_roll_*')
        for i, fname in enumerate(filenames): 
            f = np.load(fname, allow_pickle = True)
            diagrams_roll[i] = list(f['diagrams'])
            f.close() 
    if len(file_name)>0:
        f = np.load(file_name, allow_pickle = True)
        diagrams_roll = f['dgms'][()]
        f.close()
    maxdim = len(persistence)-1
    dims =np.arange(maxdim+1)
    cs = np.repeat([[0,0.55,0.2]],maxdim+1).reshape(3,maxdim+1).T
    alpha=1
    inf_delta=0.1
    legend=True
    colormap=cs
    num_rolls = len(diagrams_roll)
    data, data_names = [], []
    if num_rolls>0:
#        diagrams_all = diagrams_roll[0].copy()
        diagrams_all = [[[0,0]] for dim in dims]
        for i in np.arange(1,num_rolls):
            for d in dims:
                dgmstmp = diagrams_roll[i][d].copy()
                dgmstmp[np.isinf(dgmstmp)]  = 0  
                if bMax :
                    diagrams_all[d] = np.concatenate((diagrams_all[d], 
                                                      dgmstmp[np.argmax(np.abs(np.diff(dgmstmp,1)))][np.newaxis,:]),0)
                else:
                    diagrams_all[d] = np.concatenate((diagrams_all[d], diagrams_roll[i][d]),0)
        for d in dims:
            diagrams_all[d] = diagrams_all[d][1:]
#        infs = np.isinf(diagrams_all[0])
#        diagrams_all[0][infs] = 0
        #diagrams_all[0][infs] = np.max(diagrams_all[0])
    min_birth, max_death = 0,0            
    for dim in dims:
        persistence_dim = persistence[dim][~np.isinf(persistence[dim][:,1]),:]
        dlife = (persistence_dim[:,1] - persistence_dim[:,0])
        dinds = np.argsort(dlife)[-num_bars:]
        if len(dinds)==0:
            min_birth = min(min_birth, persistence_dim[dinds,0])
            max_death = max(max_death, persistence_dim[dinds,1])
        else:
            min_birth = min(min_birth, np.min(persistence_dim[dinds,0]))
            max_death = max(max_death, np.max(persistence_dim[dinds,1]))
    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    print(infinity)
    if xmax > -1:
        infinity = xmax
    axis_start = min_birth - delta            
    plotind = (dims[-1]+1)*100 + 10 +1
    fig = plt.figure(figsize = figsize, dpi = dpi)
    gs = grd.GridSpec(len(dims),1)
    
    indsall =  0
    labels = ["$H^0$", "$H^1$", "$H^2$", "$H^3$", "$H^4$"]
    for dit, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis('off')
        d = np.copy(persistence[dim])
        d[np.isinf(d[:,1]),1] = infinity
        dlife = (d[:,1] - d[:,0])
        dinds = np.argsort(dlife)[-num_bars:]
        if dim>0:
            dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
        axes.barh(
            0.5+np.arange(len(dinds)),
            dlife[dinds],
            height=0.8,
            left=d[dinds,0],
            alpha=alpha,
            color=colormap[dim],
            linewidth=0,
        )
        indsall = len(dinds)
        data.append(pd.Series(dlife[dinds]))
        data_names.extend(['Dim_' + str(dim) + '_lifetime'])
        data.append(pd.Series(d[dinds,0]))
        data_names.extend(['Dim_' + str(dim) + '_births'])

        if num_rolls>0:
            bins = 50
            #cs = np.flip([[0.4,0.4,0.4], [0.6,0.6,0.6], [0.8, 0.8,0.8]])
            #cs = np.repeat([[1,0.55,0.1]],maxdim+1).reshape(3,maxdim+1).T
            cs = np.repeat([[0.5,0.5,0.5]],maxdim+1).reshape(3,maxdim+1).T
            cc = 0
            lives1_all = diagrams_all[dim][:,1] - diagrams_all[dim][:,0]
            x1 = np.linspace(diagrams_all[dim][:,0].min()-1e-5, diagrams_all[dim][:,0].max()+1e-5, bins-2)
            dx1 = (x1[1] - x1[0])
            x1 = np.concatenate(([x1[0]-dx1], x1, [x1[-1]+dx1]))
            dx = x1[:-1] + dx1/2
            ytemp = np.zeros((bins-1))
            binned_birth = np.digitize(diagrams_all[dim][:,0], x1)-1
            x1  = d[dinds,0]
            ytemp =x1 + np.percentile(lives1_all,percshuf)#np.max(lives1_all)
            x1 = np.concatenate(([x1[0]], x1))
            ytemp = np.concatenate(([ytemp[0]], ytemp))
            axes.fill_betweenx(np.arange(len(dinds)+1), x1, ytemp, color = cs[(dim)], zorder = -2, alpha = 0.3)

            data.append(pd.Series(ytemp))
            data_names.extend(['Dim_' + str(dim) + '_shuffled_line'])

        axes.plot([0,0], [0, indsall], c = 'k', linestyle = '-', lw = 3)
        axes.plot([0,infinity],[0,0], c = 'k', linestyle = '-', lw = 2)
        axes.set_xlim([0, infinity])
    if len(SaveSourceDataName)>0:
        print('')
        df = pd.concat(data, ignore_index=True, axis=1)            
        df.columns = data_names
        df.to_excel('Source_data/' + SaveSourceDataName + '.xlsx', sheet_name=SaveSourceDataName)  

#        if disp:
#            print('')
#            axes.text(-infinity*0.075, indsall*0.45, labels[dit], fontsize = 15)
#            axes.text(-infinity*0.0075,-indsall*0.15,'0', fontsize = 10)
#            axes.text(infinity*0.975,-indsall*0.15,str(int(np.round(infinity,0))), fontsize = 10)
#            axes.text(infinity*0.45,-indsall*0.15,'Radius', fontsize = 10)




def fill_nans(arr):
    nans = np.where(np.isnan(arr))[0]
    while len(nans)>0:
        nansnext = nans+1
        nansnext[nansnext>=len(arr)] -= 2
        nansnext[np.isnan(arr[nansnext])] -= 2
        arr[nans] = arr[nansnext]
        nans = np.where(np.isnan(arr))[0]
    return arr

def pearson_correlate2d(in1, in2, mode='same', fft=True, nan_to_zero=True):
    """
    Pearson cross-correlation of two 2-dimensional arrays.

    Cross correlate `in1` and `in2` with output size determined by `mode`.
    NB: `in1` is kept still and `in2` is moved.

    Array in2 is shifted with respect to in1 and for each possible shift
    the Pearson correlation coefficient for the overlapping part of
    the two arrays is determined and written in the output rate.
    For in1 = in2 this results in the  Pearson auto-correlogram.
    Note that erratic values (for example seeking the correlation in
    array of only zeros) result in np.nan values which are by default set to 0.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
        If operating in 'valid' mode, either `in1` or `in2` must be
        at least as large as the other in every dimension.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    Returns
    -------
    pearson_corr : ndarray
        A 2-dimensional array containing a subset of the discrete pearson
        cross-correlation of `in1` with `in2`.
    """
    kwargs = dict(mode=mode, fft=fft, normalize=True,
                  set_small_values_zero=1e-10)
    corr = functools.partial(correlate2d, **kwargs)
    ones = np.ones_like(in1)
    pearson_corr = (
        (corr(in1, in2) - corr(ones, in2) * corr(in1, ones))
        / (
            np.sqrt(corr(in1 ** 2, ones) - corr(in1, ones) ** 2)
            * np.sqrt(corr(ones, in2 ** 2) - corr(ones, in2) ** 2)
        )
    )
    if nan_to_zero:
        pearson_corr[np.isnan(pearson_corr)] = 0.
    return pearson_corr


def correlate2d(in1, in2, mode, fft, normalize=True,
                set_small_values_zero=None):
    """
    Correlate two 2-dimensional arrays using FFT and possibly normalize

    NB: `in1` is kept still and `in2` is moved.

    Convenience function. See signal.convolve2d or signal.correlate2d
    for documenation.
    Parameters
    ----------
    normalize : Bool
        Decide wether or not to normalize each element by the
        number of overlapping elements for the associated displacement
    set_small_values_zero : float, optional
        Sometimes very small number occur. In particular FFT can lead to
        very small negative numbers.
        If specified, all entries with absolute value smalle than
        `set_small_values_zero` will be set to 0.
    Returns
    -------

    """
    if normalize:
        ones = np.ones_like(in1)
        n = signal.fftconvolve(ones, ones, mode=mode)
    else:
        n = 1
    if fft:
        # Turn the second array to make it a correlation
        ret = signal.fftconvolve(in1, in2[::-1, ::-1], mode=mode) / n
        if set_small_values_zero:
            condition = (np.abs(ret) < set_small_values_zero)
            if condition.any():
                ret[condition] = 0.
        return ret
    else:
        return signal.correlate2d(in1, in2, mode=mode) / n
    
def pca(data, dim=2):
    if dim < 1:
        return data, [0]
    m, n = data.shape
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dim]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    tot = np.sum(evals)
    var_exp = [(i / tot) * 100 for i in sorted(evals[:], reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    components = np.dot(evecs.T, data.T).T
    return components, evecs, evals[:dim], var_exp


def sample_denoising(data,  k = 10, num_sample = 500, omega = 1, metric = 'euclidean'):    
    n = data.shape[0]
    leftinds = np.arange(n)
    F_D = np.zeros(n)

    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)

    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X,1)
    print(np.mean(F),np.median(F))
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    inds_all = np.arange(n)
    inds_left = inds_all>-1
    inds_left[i] = False
    inds = [i]
    j = 0
    for j in np.arange(1,num_sample):
        F -= omega*X[i,:]
        Fmax = np.argmax(F[inds_left])
        Fs[j] = F[inds_left][Fmax]
        i = inds_all[inds_left][Fmax]
        
        inds_left[i] = False   
        inds.extend([i])
    return inds, Fs


@numba.njit(parallel=True, fastmath=True)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                #val = ((knn_dists[i, j] - rhos[i]) / (sigmas[i]))
            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
    return rows, cols, vals


@numba.njit(
    fastmath=True
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0, bandwidth=1.0):
    target = np.log2(k) * bandwidth
#    target = np.log(k) * bandwidth
#    target = k
    
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
#                    psum += d / mid
 
                else:
                    psum += 1.0
#                    psum += 0

            if np.fabs(psum - target) < 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < 1e-3 * mean_ith_distances:
                result[i] = 1e-3 * mean_ith_distances
        else:
            if result[i] < 1e-3 * mean_distances:
                result[i] = 1e-3 * mean_distances

    return result, rho




def plot_distance_cells(mouse_sess, indsnull, e1, folder): 
    dist_name = "GiocomoLab-Campbell_Attinger_CellReports-d825378\\intermediate_data/dark_distance\\dist_tun.mat"
    dist_tuning = sio.loadmat(dist_name)
    names = np.array(['Mouse', 'Date', 'MouseDate', 'Session', 'CellID', 'UniqueID', 'BrainRegion', 'InsertionDepth', 'DepthOnProbe',
             'DepthFromSurface', 'peak', 'period', 'prom', 'pval', 'peak_shuf', 'DepthAdjusted'])
    dist_tun = {}
    for i, names in enumerate(names):
        dist_tun[names] = dist_tuning['b']['data'][0,0][0,i]

    pval_cutoff = 0.01
    min_prom = 0.1
    if len(np.where((dist_tun['MouseDate'] == mouse_sess))[0]) == 0:
        print('Campbell not computed')
    else:
        cIdx_dist = np.where((dist_tun['MouseDate'] == mouse_sess) & (dist_tun['BrainRegion']=='MEC'))[0]
        ids = (dist_tun['pval'][cIdx_dist]<pval_cutoff) & (dist_tun['prom'][cIdx_dist]>min_prom)
        if len(ids)>len(indsnull):
            print('Campbell ids larger')
        else:
            print(np.sum(ids[indsnull,0][e1]), sum(e1))

            fig, ax = plt.subplots(1,1)
            ax.scatter(dist_tun['pval'][cIdx_dist],dist_tun['prom'][cIdx_dist])
            ax.scatter(dist_tun['pval'][cIdx_dist][indsnull][e1],dist_tun['prom'][cIdx_dist][indsnull][e1])
            ax.set_xlim([-0.05,1.05])
            ax.set_ylim([-0.03,0.7])
            ax.set_aspect(1/ax.get_data_ratio())
            fig.savefig(folder + '/distance_scores')
            plt.close()



def plot_diagrams1(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    colormap1 = "default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    lifetime=False,
    rel_life= False,
    legend=True,
    show=False,
    ax=None,
    torus_colors = [],
    lw = 2.5,
):


    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = [
            "$H_0$",
            "$H_1$",
            "$H_2$",
            "$H_3$",
            "$H_4$",
            "$H_5$",
            "$H_6$",
            "$H_7$",
            "$H_8$",
        ]

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if len(plot_only)>0:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]
    
    if lifetime:
        ylabel = "Lifetime"

        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]
    elif rel_life:
        ylabel = "Relative Lifetime"

        for dgm in diagrams:
            dgm[dgm[:,0]>0, 1] /= dgm[dgm[:,0]>0, 0]
    aspect = 'equal'
    # find min and max of all visible diagrams
    concat_dgms_b = np.concatenate(diagrams)[:,0]#
    finite_dgms_b = concat_dgms_b[np.isfinite(concat_dgms_b)]
    concat_dgms_d = np.concatenate(diagrams)[:,1]#
    finite_dgms_d = concat_dgms_d[np.isfinite(concat_dgms_d)]
    has_inf = np.any(np.isinf(concat_dgms_d))
    
    if not xy_range:
        ax_min, ax_max = np.min(finite_dgms_b), np.max(finite_dgms_b)
        x_r = ax_max - ax_min
        buffer = 1 if xy_range == 0 else x_r / 5
        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        ax_min, ax_max = np.min(finite_dgms_d), np.max(finite_dgms_d)
        y_r = ax_max - ax_min
        buffer = 1 if xy_range == 0 else x_r / 5
        y_down = ax_min - buffer / 2
        y_up = ax_max + buffer
        
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down
    if lifetime | rel_life:
        y_down = -yr * 0.05
        y_up = yr - y_down
        

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    i = 0
    for dgm, label in zip(diagrams, labels):
        c = cs[plot_only[i]]
        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none", c = c)
        i += 1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    if len(torus_colors)>0:
        births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
        deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
        deaths1[np.isinf(deaths1)] = 0
        #lives1 = deaths1-births1
        #inds1 = np.argsort(lives1)
        inds1 = np.argsort(deaths1)
        ax.scatter(diagrams[1][inds1[-1], 0], diagrams[1][inds1[-1], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[0], facecolor = "none")
        ax.scatter(diagrams[1][inds1[-2], 0], diagrams[1][inds1[-2], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[1], facecolor = "none")
        
        
        births2 = diagrams[2][:, ] #the time of birth for the 1-dim classes
        deaths2 = diagrams[2][:, 1] #the time of death for the 1-dim classes
        deaths2[np.isinf(deaths2)] = 0
        #lives2 = deaths2-births2
        #inds2 = np.argsort(lives2)
        inds2 = np.argsort(deaths2)
#        print(lives2, births2[inds2[-1]],deaths2[inds2[-1]], diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1])
        ax.scatter(diagrams[2][inds2[-1], 0], diagrams[2][inds2[-1], 1], 
                   10*size, linewidth =lw, edgecolor=torus_colors[2], facecolor = "none")
        
        
    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect(1/ax.get_data_ratio())

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="upper right")

    if show is True:
        plt.show()
    return



def from_ripser_to_giotto(dgm, infmax= np.inf):
    dgm_ret = []
    for dim, i in enumerate(dgm):
        i[np.isinf(i)] = infmax
        for j in i:
            dgm_ret.append([j[0],j[1], dim])
    return dgm_ret

def from_giotto_to_ripser(dgm):
    dgm_ret = []
    for i in range(int(np.max(dgm[:, 2]))+1):
        dgm_ret.append(dgm[dgm[:,2]==i, :2])
    if np.sum(np.isinf(dgm_ret[0]))==0:
        dgm_ret[0] = np.concatenate((dgm_ret[0], np.array([0,np.inf])[np.newaxis,:]))
    return dgm_ret

def get_coords(cocycle, threshold, num_sampled, dists, coeff):
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.NaN
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.NaN
    d[dists == 0] = np.NaN
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        v1[i, 0] = i
        v2[i, 0] = i
        v1[i, 1] = np.where(verts == edges[0][i])[0]
        v2[i, 1] = np.where(verts == edges[1][i])[0]

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1
  
    f = lsmr(A.astype(float), values.astype(float))[0]%1
    return f, verts


def information_score_2d(mtemp, circ, mu):
    numangsint = mtemp.shape[0]+1
    circ = np.ravel_multi_index(circ-1, np.shape(mtemp))
    mtemp = mtemp.flatten() 
    p = np.bincount(circ, minlength = (numangsint-1)**2)/len(circ)
    logtemp = np.log2(mtemp/mu)
    mtemp = np.multiply(np.multiply(mtemp,p), logtemp)
    return np.sum(mtemp[~np.isnan(mtemp)])

        
def information_score_1d(mtemp, circ, mu):
    numangsint = mtemp.shape[0]
    p = np.bincount(circ, minlength = (numangsint))/len(circ)
    logtemp = np.log2(mtemp/mu)
    mtemp = np.multiply(np.multiply(mtemp,p), logtemp)
    return np.sum(mtemp[~np.isnan(mtemp)])


def get_coords_ds(rips_real, len_indstemp, ph_classes = [0,1], dec_thresh = 0.99, coeff = 47):
    num_circ = len(ph_classes)    
    ################### Decode coordinates ####################
    diagrams = rips_real["dgms"] # the multiset describing the lives of the persistence classes
    cocycles = rips_real["cocycles"][1] # the cocycle representatives for the 1-dim classes
    dists_land = rips_real["dperm2all"] # the pairwise distance between the points 
    births1 = diagrams[1][:, 0] #the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1] #the time of death for the 1-dim classes
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1-births1 # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len_indstemp))
    for j,c in enumerate(ph_classes):
        cocycle = cocycles[iMax[-(c+1)]]
        threshold = births1[iMax[-(c+1)]] + (deaths1[iMax[-(c+1)]] - births1[iMax[-(c+1)]])*dec_thresh
        coordstemp,inds = get_coords(cocycle, threshold, len_indstemp, dists_land, coeff)
        coords1[j,inds] = coordstemp
    return coords1





def normit(xxx):
    xx = xxx-np.min(xxx)
    xx = xx/np.max(xx)
    return(xx)



def plot_sim_acorr(data_ensemble, coords, posxx, pos_trial, files, folder, speed = [],sp = -np.inf, theta = False, numbins = 10):
    coords0 = coords.copy()
    simtype = 'hex' 
    if theta:
        coords0[:,0] = 2*np.pi-coords0[:,0]     
        simtype = 'hex0'    
    
    mcstemp, mtot_all = get_ratemaps_center(coords0, data_ensemble[:,:], numbins = numbins)    
    spk_sim = get_sim(coords, mcstemp, numbins = numbins, simtype = simtype) 

    ################### Get acorrs ####################    
    acorr_sig = 2
    data_dir = 'giocomo_data'
    num_neurons = len(data_ensemble[0,:])
    acorr_real = {}
    acorr_sim = {}
    acorr_corr = {}
    t0 = 0    
    cs = ['#1f77b4', '#ff7f0e', '#2ca02c']
    if len(speed)== 0:
        speed = {}
        for fi in files:
            speed[fi] = np.ones(len(posxx[fi]))
            
    for fi in files[:]:
        finame = fi.replace('giocomo_data/', '').replace('.mat', '')
        
        times = np.arange(t0,t0+len(posxx[fi]))
        times = times[speed[fi]>sp]
        t0 +=len(posxx[fi])
        
        acorr_real[fi] = get_acorrs(data_ensemble[times,:], pos_trial[fi][speed[fi]>sp], posxx[fi][speed[fi]>sp])
        acorr_sim[fi] = get_acorrs(spk_sim[times,:], pos_trial[fi][speed[fi]>sp], posxx[fi][speed[fi]>sp])        

        acorr_corr[fi] = np.zeros(num_neurons)
        for i in range(num_neurons):
            fi1 = fi.replace('\\', '/')
            acorr_corr[fi][i] = pearsonr(acorr_real[fi][i,1:], acorr_sim[fi][i,1:])[0] 

        print(np.mean(acorr_corr[fi]), np.std(acorr_corr[fi])/np.sqrt(len(acorr_corr[fi])))
        
        fig, ax = plt.subplots(1,1)        
        acorr_mean_real = acorr_real[fi] - acorr_real[fi].mean(1)[:,np.newaxis]
        acorrmean = acorr_mean_real.mean(0)
        acorrstd = 1*acorr_real[fi].std(0)/np.sqrt(len(acorr_real[fi][:,0]))
        ax.plot(acorrmean, lw = 2, c= cs[0])
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean + acorrstd,
                        lw = 0, color= cs[0], alpha = 0.3)
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean - acorrstd,
                        lw = 0, color= cs[0], alpha = 0.3)        

        acorr_mean_sim = acorr_sim[fi] - acorr_sim[fi].mean(1)[:,np.newaxis]
        acorrmean = acorr_mean_sim.mean(0)
        acorrstd = 1*acorr_sim[fi].std(0)/np.sqrt(len(acorr_real[fi][:,0]))
        ax.plot(acorrmean, lw = 2, c= cs[1])
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean + acorrstd,
                        lw = 0, color= cs[1], alpha = 0.3)
        ax.fill_between(np.arange(len(acorrmean)),acorrmean, acorrmean - acorrstd,
                        lw = 0, color= cs[1], alpha = 0.3)        
        ax.set_aspect(1/ax.get_data_ratio())
        #plt.xticks([0,50,100,150,200], ['', '', '', '',''])
        #plt.yticks(np.arange(0,3,1)/100, ['','',''])
        plt.gca().axes.spines['top'].set_visible(False)
        plt.gca().axes.spines['right'].set_visible(False)
        plt.savefig(folder + '/acorr_simreal_' + finame, bbox_inches='tight', pad_inches=0.1, transparent = True)
#        plt.savefig('acorr_classes' + str(fi) + '.pdf', bbox_inches='tight', pad_inches=0.1, transparent = True)
        plt.ylim([-0.2, 0.5])
        plt.close()        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(gaussian_filter1d(acorr_mean_real, axis = 1, sigma = 1),vmin = 0.0, vmax = 0.1)
        ax.set_aspect(1/ax.get_data_ratio())
        fig.tight_layout()
        fig.savefig(folder + '/_acorr_real' + finame, transparent = True)
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(acorr_mean_sim,vmin = 0, vmax = 0.1)
        ax.set_aspect(1/ax.get_data_ratio())
        fig.tight_layout()
        fig.savefig(folder + '/_acorr_sum' + finame, transparent = True)
        plt.close()
    return acorr_real, acorr_sim, acorr_corr


def get_centered_ratemaps(coords, spk, mcstemp, numbins = 15,):
    mid = int((numbins)/2)
    bins = np.linspace(0,2*np.pi, numbins+1)
    num_neurons = len(spk[0,:])    
    mtot_all = {}
    for n in range(num_neurons):        
        coords_temp = (coords.copy() - mcstemp[n,:])%(2*np.pi)
        coords_temp = (coords_temp + (np.pi, np.pi))%(2*np.pi)

        mtot = binned_statistic_2d(coords_temp[:, 0], coords_temp[:,1], spk[:,n],
                                   bins = bins, range = None, expand_binnumbers = True)[0]
        mtot_all[n] = mtot.copy()
    return mtot_all

def get_ratemaps_center(coords, spk, numbins = 15,bMcs = True):
    mid = int((numbins)/2)
    bins = np.linspace(0,2*np.pi, numbins+1)
    num_neurons = len(spk[0,:])    
    mcstemp = np.zeros((num_neurons,2))
    mtot_all = {}
    for n in range(num_neurons):
        mtot = binned_statistic_2d(coords[:, 0], coords[:,1], spk[:,n],
                                   bins = bins, range = None, expand_binnumbers = True)[0]
        mtot_all[n] = mtot.copy()
        
    if bMcs:
        xv, yv = np.meshgrid(bins[0:-1] + (bins[1:] -bins[:-1])/2, 
                     bins[0:-1] + (bins[1:] -bins[:-1])/2)
        pos  = np.concatenate((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis]),1)
        ccos = np.cos(pos)
        csin = np.sin(pos)
        for n in range(num_neurons):
            mtot = mtot_all[n].T.flatten()
            nans  = ~np.isnan(mtot) 
            centcos = np.sum(np.multiply(ccos[nans,:],mtot[nans,np.newaxis]),0)
            centsin = np.sum(np.multiply(csin[nans,:],mtot[nans,np.newaxis]),0)
            mcstemp[n,:] = np.arctan2(centsin,centcos)%(2*np.pi)
    return mcstemp, mtot_all


def plot_centered_ratemaps(coords, data, mcstemp, numbins, ax = None, SourceName = ''):
    mtot_temp = get_centered_ratemaps(coords, data, mcstemp, numbins = numbins)
    mtot_mean = np.zeros_like(mtot_temp[0])
    num_neurons = len(mtot_temp)
    for n in range(num_neurons):
        nans = np.isnan(mtot_temp[n])
        mmean = np.mean(mtot_temp[n][~nans])
        mstd = np.std(mtot_temp[n][~nans])
        mtot_temp[n][nans] = mmean
        mtot_mean += (mtot_temp[n] - mmean)/mstd
    if not ax:
        fig,ax = plt.subplots(1,1)

    ax.imshow(mtot_mean, vmin = 0, vmax = mtot_mean.flatten()[np.argsort(mtot_mean.flatten())[int(len(mtot_mean.flatten())*0.99)]])
    ax.axis('off')

    if len(SourceName)>0:
        data = []
        data_names = []
        for i in range(len(mtot_mean)):
            data.append(pd.Series(mtot_mean[:,i]))
            data_names.extend(['col' + str(i)])

        df = pd.concat(data, ignore_index=True, axis=1)            
        df.columns = data_names
        df.to_excel('Source_data/'+ SourceName +'.xlsx', sheet_name=SourceName)  
        
        plt.savefig('Figures/'+ SourceName + '.png', transparent = True, bbox_inches='tight', pad_inches=0.2)
        plt.savefig('Figures/'+ SourceName + '.pdf', transparent = True, bbox_inches='tight', pad_inches=0.2)



def get_score(dgms, dim = 1, dd = 1):
    births = dgms[dim][:, 0] #the time of birth for the 1-dim classes
    deaths = dgms[dim][:, 1] #the time of death for the 1-dim classes
    deaths[np.isinf(deaths)] = 0
    lives = deaths-births
    lives_sort = np.argsort(lives)
    lives = lives[lives_sort]
    births = births[lives_sort]
    deaths = deaths[lives_sort]
    if dd == 0:
        gaps = np.diff(np.diff(lives[-25:]))
    else:
        gaps = np.diff(lives[-25:])-1
    gapsmax = np.argmax(np.flip(gaps))
    return gapsmax 


def get_sim(coords, mcstemp, numbins = 10, t = 0.1, nums = 1, simtype = ''): 
    """Simulate activity """
    if simtype == 'hex0':
        coords0 = coords.copy()
        coords0[:,0] = 2*np.pi - coords0[:,0]
        spk_sim = simulate_spk_hex(coords0, mcstemp, t = t, nums = nums)
    elif simtype == 'hex' :
        spk_sim = simulate_spk_hex(coords, mcstemp, t = t, nums = nums)
    else:
        spk_sim = simulate_spk_sqr(coords, mcstemp, t = t, nums = nums)
    return spk_sim


def get_sim_hex_corr(coords, data, numbins = 10, folder = '', fname = ''):    
    num_neurons = len(data[0,:])
    coords0 = coords.copy()
    coords0[:,0] = 2*np.pi-coords0[:,0]     
    mcstemp, mtot_all = get_ratemaps_center(coords, data, numbins = numbins,)
    mcstemp0, mtot_all0 = get_ratemaps_center(coords0, data, numbins = numbins,)    
    intervals = np.linspace(0, 2*np.pi, 100)
    intervals = intervals[1:] - (intervals[1]-intervals[0])/2
    c1,c2 = np.meshgrid(intervals,intervals)
    coords_uniform = np.concatenate((c1.flatten()[:,np.newaxis], c2.flatten()[:,np.newaxis]), 1)
    
    spk_sim_hex = get_sim(coords_uniform, mcstemp, numbins = numbins, simtype = 'hex')
    spk_sim_hex0 = get_sim(coords_uniform, mcstemp0, numbins = numbins, simtype = 'hex0')
        
    ps1_hex = np.zeros(num_neurons)
    ps1_hex0 = np.zeros(num_neurons)
    

    __, mtot_all_hex = get_ratemaps_center(coords_uniform, spk_sim_hex, numbins = numbins, bMcs = False)
    __, mtot_all_hex0 = get_ratemaps_center(coords_uniform, spk_sim_hex0, numbins = numbins, bMcs = False)

    for n in range(num_neurons):        
        mtot_curr = mtot_all[n].copy()
        mtot_curr[np.isnan(mtot_curr)] = np.mean(mtot_curr[~np.isnan(mtot_curr)])
        
        mtot_hex = mtot_all_hex[n].copy()
        mtot_hex[np.isnan(mtot_hex)] = np.mean(mtot_hex[~np.isnan(mtot_hex)])
        
        mtot_hex0 = mtot_all_hex0[n].copy()
        mtot_hex0[np.isnan(mtot_hex0)] = np.mean(mtot_hex0[~np.isnan(mtot_hex0)])
        
        ps1_hex[n] = pearsonr(mtot_hex.flatten(), mtot_curr.flatten())[0]        
        ps1_hex0[n] = pearsonr(mtot_hex0.flatten(), mtot_curr.flatten())[0]
        
    theta = np.median(ps1_hex0)>np.median(ps1_hex)
    print( np.median(ps1_hex), np.median(ps1_hex0))
    if theta:
        return ps1_hex0
    else:
        return ps1_hex

def get_sim_corr(coords, data, numbins = 10, bSqr = True, folder = '', fname = ''):    
    num_neurons = len(data[0,:])
    coords0 = coords.copy()
    coords0[:,0] = 2*np.pi-coords0[:,0]     
    mcstemp, mtot_all = get_ratemaps_center(coords, data, numbins = numbins,)
    mcstemp0, mtot_all0 = get_ratemaps_center(coords0, data, numbins = numbins,)    
    intervals = np.linspace(0, 2*np.pi, 100)
    intervals = intervals[1:] - (intervals[1]-intervals[0])/2
    c1,c2 = np.meshgrid(intervals,intervals)
#    coords_uniform = np.random.rand(len(coords),2)*2*np.pi
    coords_uniform = np.concatenate((c1.flatten()[:,np.newaxis], c2.flatten()[:,np.newaxis]), 1)
    
    spk_sim_hex = get_sim(coords_uniform, mcstemp, numbins = numbins, simtype = 'hex')
    spk_sim_hex0 = get_sim(coords_uniform, mcstemp0, numbins = numbins, simtype = 'hex0')
        
    ps1_hex = np.zeros(num_neurons)
    ps1_hex0 = np.zeros(num_neurons)
    ps1_sqr = np.zeros(num_neurons)
    

    __, mtot_all_hex = get_ratemaps_center(coords_uniform, spk_sim_hex, numbins = numbins, bMcs = False)
    __, mtot_all_hex0 = get_ratemaps_center(coords_uniform, spk_sim_hex0, numbins = numbins, bMcs = False)
    if bSqr:
        spk_sim_sqr = get_sim(coords_uniform, mcstemp, numbins = numbins, simtype = 'sqr')
        __, mtot_all_sqr = get_ratemaps_center(coords_uniform, spk_sim_sqr, numbins = numbins, bMcs = False)

    for n in range(num_neurons):        
        mtot_curr = mtot_all[n].copy()
        mtot_curr[np.isnan(mtot_curr)] = np.mean(mtot_curr[~np.isnan(mtot_curr)])
        
        mtot_hex = mtot_all_hex[n].copy()
        mtot_hex[np.isnan(mtot_hex)] = np.mean(mtot_hex[~np.isnan(mtot_hex)])
        
        mtot_hex0 = mtot_all_hex0[n].copy()
        mtot_hex0[np.isnan(mtot_hex0)] = np.mean(mtot_hex0[~np.isnan(mtot_hex0)])
        
        ps1_hex[n] = pearsonr(mtot_hex.flatten(), mtot_curr.flatten())[0]        
        ps1_hex0[n] = pearsonr(mtot_hex0.flatten(), mtot_curr.flatten())[0]
        
        if bSqr:
            mtot_sqr = mtot_all_sqr[n].copy()
            mtot_sqr[np.isnan(mtot_sqr)] = np.mean(mtot_sqr[~np.isnan(mtot_sqr)])
            ps1_sqr[n] = pearsonr(mtot_sqr.flatten(), mtot_curr.flatten())[0]
    theta = np.median(ps1_hex0)>np.median(ps1_hex)
    if len(folder) >0:
        fig,ax = plt.subplots(1,2)
        print(theta)
        plot_centered_ratemaps(coords, data, mcstemp, numbins, ax[0])
        if theta:
    #        plot_centered_ratemaps(coords0, data, mcstemp0, numbins, ax[0])
            plot_centered_ratemaps(coords_uniform, spk_sim_hex0, mcstemp, numbins, ax[1])
        else:
            plot_centered_ratemaps(coords_uniform, spk_sim_hex, mcstemp, numbins, ax[1])
            
        fig.tight_layout()
        fig.savefig(folder + '/stacked_ratemap' + fname, transparent = True)
        plt.close()
    
    print(np.median(ps1_sqr), np.median(ps1_hex), np.median(ps1_hex0))


    return (ps1_sqr, ps1_hex, ps1_hex0, theta)



@numba.njit(fastmath=True, parallel = True) 
def simulate_spk_hex(cc, mcstemp, t = 0.1, nums = 4):
    _2_PI = 2*np.pi
    _t_pi_sqrt_3_2 = -np.pi/t*2/np.sqrt(3)
    num_neurons = len(mcstemp)
    spk_sim = np.zeros((len(cc), num_neurons))
    nums_all = np.arange(-nums,nums+1)
    num_numsall = nums*2+1
    for i in range(num_neurons):
        cctmp = ((cc - mcstemp[i,:])%(_2_PI))/(_2_PI)
        k_all = [(k+cctmp[:,0]) for k in nums_all]
        l_all = [(l+cctmp[:,1]) for l in nums_all]        
        for k in range(num_numsall):
            for l in range(num_numsall):
                spk_sim[:,i] += np.exp(_t_pi_sqrt_3_2*(k_all[k]**2 + k_all[k]*l_all[l] + l_all[l]**2))
    return spk_sim


@numba.njit(fastmath=True, parallel = True)  # benchmarking `parallel=True` shows it to *decrease* performance
def simulate_spk_sqr(cc, mcstemp, t = 0.1, nums = 4):
    num_neurons = len(mcstemp)
    spk_sim = np.zeros((len(cc), num_neurons))
    numsall = np.arange(-nums,nums+1)
    for i in range(num_neurons):
        cctmp = ((cc -mcstemp[i,:])%(2*np.pi))/(2*np.pi)
#        cctmp = (cc -mcstemp[i,:])/(2*np.pi)
        for k in numsall:
            for l in numsall:
                spk_sim[:,i] += np.exp(-np.pi/t*((k+cctmp[:,0])**2 + 
                                                              (l +cctmp[:,1])**2))
    return spk_sim


@numba.njit(fastmath=True, parallel = True)  # benchmarking `parallel=True` shows it to *decrease* performance
def simulate_spk(cc, num_neurons, mcstemp, t = 0.1, nums = 4):
    spk_sim = np.zeros((len(cc), num_neurons))
    numsall = np.arange(-nums,nums+1)
    for i in range(num_neurons):
        cctmp = ((cc -mcstemp[i,:])%(2*np.pi))/(2*np.pi)
        for k in numsall:
            for l in numsall:
                spk_sim[:,i] += np.exp(-np.pi/t*2/np.sqrt(3)*((k+cctmp[:,0])**2 + 
                                                              (k+cctmp[:,0])*(l+cctmp[:,1]) + 
                                                              (l +cctmp[:,1])**2))
    return spk_sim

def plot_toroidal_ratemaps(mouse_sess, data_ensemble, files, e1, coords1, speed, 
                           pos_trial, posxx, theta, folder_curr, sp = 10, deg = 15):    
    num_neurons = len(e1)
    r_box = transforms.Affine2D().skew_deg(deg,deg)
    numangsint = 51
    sig = 2.75
    numbins = 50
    bins = np.linspace(0,2*np.pi, numbins+1)
    plt.viridis()
    numfigs = len(files)
    numw = 4
    numh = int(np.ceil(num_neurons/numw))
    outer1 = gridspec.GridSpec(1, numw)
    fig = plt.figure(figsize=(np.ceil((numw*numfigs+numw-1)*1.05), np.ceil(numh*1.1)))
    len_acorr = 500 #len(acorr_sess[fi][0,:])
    nw = 0
    mtots =  {}
    files1 = glob.glob('giocomo_data/' + mouse_sess + '*.mat')
    for fi in files1:
        if fi.find('dark')>-1:
            acorr_real = get_acorrs(data_ensemble[fi][:,e1][speed[fi]>sp,:], pos_trial[fi][speed[fi]>sp], posxx[fi][speed[fi]>sp])

    
    for fi in files:
        mtots[fi] = np.zeros((num_neurons, numbins, numbins))
    for nn, n in enumerate(range(num_neurons)):
        nnn = nn%numh
        if nnn == 0:
            outer2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer1[nw], wspace = .1)
            gs2 = gridspec.GridSpecFromSubplotSpec(numh, len(files)+1, subplot_spec = outer2[0], wspace = .1)
            nw += 1
        posnum = 0 
        xnum = 0
        t0 = 0
        for fi in files[:]:
            if len(coords1)== len(files):
                cc1 = coords1[fi]
                spk = data_ensemble[fi][:,e1[n]]
            else:
                times = np.arange(t0,t0+len(speed[fi]))
                t0+=len(speed[fi])
                spk = data_ensemble[:,n][times]
                cc1 = coords1[times,:]
            if xnum == 0:
                ax = plt.subplot(gs2[nnn,xnum]) 

                ax.bar(np.arange(300),acorr_real[n,:], width = 1, color = 'k')
                ax.set_xlim([0,300])
                ax.set_ylim([0, 0.4])

                ax.set_xticks([0, 100, 200, 300])
                ax.set_yticks([0.0, 0.2, 0.4])
                ax.set_yticklabels('')
                ax.set_xticklabels('')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_aspect(1/ax.get_data_ratio())

                xnum += 1
            ax = plt.subplot(gs2[nnn,xnum])
            mtot_tmp, x_edge, y_edge,c2 = binned_statistic_2d(cc1[:,0],cc1[:,1], 
                                                 spk, statistic='mean', 
                                                 bins=bins, range=None, expand_binnumbers=True)
            mtots[fi][nn, :,:] = mtot_tmp.copy()
            nans = np.isnan(mtot_tmp)
            mtot_tmp[np.isnan(mtot_tmp)] = np.mean(mtot_tmp[~np.isnan(mtot_tmp)])
            if theta:
                mtot_tmp = np.rot90(mtot_tmp,1)
          
            mtot_tmp = smooth_tuning_map(mtot_tmp, numangsint, sig, bClose = True) 
            mtot_tmp[nans] = -np.inf
            ax.imshow(mtot_tmp, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = 0, vmax = np.max(mtot_tmp) *0.975)
            ax.set_xticks([])
            ax.set_yticks([])
            for x in ax.images + ax.lines + ax.collections:
                trans = x.get_transform()
                x.set_transform(r_box+trans) 
                if isinstance(x, PathCollection):
                    transoff = x.get_offset_transform()
                    x._transOffset = r_box+transoff     
            ax.set_xlim(0, 2*np.pi + 3*np.pi/5)
            ax.set_ylim(0, 2*np.pi + 3*np.pi/5)
            ax.set_aspect('equal', 'box') 
            ax.axis('off')
            xnum += 1            
#    fig.savefig(folder_curr + '/ratemaps', transparent = True)
#    plt.close()


def plot_map(i, num, map_l, cNo):
    '''
    Plot a large figure where each subplot is a ratemap. Titles the indice.
    '''
    snum = np.int(np.sqrt(num)+1)
    # plt.tight_layout(0.1)
    ax = plt.subplot(snum, snum, i)
    ax.imshow(map_l[cNo], interpolation='nearest')
    ax.axis('off')
    ax.set_xlim(0, map_l[cNo].shape[0])
    ax.set_ylim(0, map_l[cNo].shape[1])
    return ax


def get_temporal_acorr(spk, maxspikes = 1000, maxt = 0.2, res = 1e-3, thresh = 0.02, inds = []):
    num_bins = int(2*maxt/res)+1
    bin_times = np.linspace(-maxt,maxt, num_bins)
    num_neurons = len(spk)
    acorr = np.zeros((num_neurons, len(bin_times)-1), dtype = int)        
    maxt-=1e-5
    mint = -maxt
    if len(inds) == 0:
        inds = np.arange(num_neurons)
    for i in inds:
        spk1 = np.array(spk[i][:maxspikes])
        for ss in spk1:
            stemp = spk1[(spk1<ss+maxt) & (spk1>ss+mint)]
            dd = stemp-ss
            acorr[i,:] += np.bincount(np.digitize(dd, bin_times)-1, minlength=num_bins)[:-1]
    return acorr

def get_temporal_acorr1(spk, maxt = 0.2, res = 1e-3, thresh = 0.02, bLog = False, bOne = False):
    if bLog:
        num_bins = 100
        bin_times = np.ones(num_bins+1)*10
        bin_times = np.power(bin_times, np.linspace(np.log10(0.005), log10(maxt), num_bins+1))
        bin_times = np.unique(np.concatenate((-bin_times, bin_times)))
        num_bins = len(bin_times)
    elif bOne:
        num_bins = int(maxt/res)+1
        bin_times = np.linspace(0,maxt, num_bins)
    else:
        num_bins = int(2*maxt/res)+1
        bin_times = np.linspace(-maxt,maxt, num_bins)
    num_neurons = len(spk)
    acorr = np.zeros((num_neurons, len(bin_times)-1), dtype = int)        
    maxt-=1e-5
    mint = -maxt
    if bOne:
        mint = -1e-5
    for i, spk in enumerate(spk):
        for ss in spk:
            stemp = spk[(spk<ss+maxt) & (spk>ss+mint)]
            dd = stemp-ss
            acorr[i,:] += np.bincount(np.digitize(dd, bin_times)-1, minlength=num_bins)[:-1]
    return acorr

def get_acorr(mouse_sess, data_dir = 'giocomo_figures0', good_cells = [], data_trials = [], 
              posxx = [], sspk1 = [], files = [], postrial = []):
    ################### Get acorrs ####################  
    lencorr = 500
    num_neurons = len(good_cells)
    acorr_sess = {}
    for fi in files[:]:
        numtrials = int(len(np.unique(data_trials[fi])))  
        sspk1_d = sspk1[fi].copy().astype(float)
        numbins = max(100, int(1000/numtrials))
        trial_range = data_trials[fi]
        num_neurons = len(sspk1_d[0,:])
        spk = np.zeros((len(trial_range)*numbins, num_neurons))
        binsx = np.linspace(0,np.max(posxx[fi]), numbins+1)
        t0 = time.time()
        for i, trial in enumerate(np.unique(postrial[fi])):
            valid_trialsSpike = postrial[fi]==trial        
            posx_trial = posxx[fi][valid_trialsSpike]
            sspk1_d1 = sspk1_d[valid_trialsSpike].copy() 
            idg = np.digitize(posx_trial, binsx)-1
            for k in np.unique(idg):
                spk[i*numbins+k, :] = np.mean(sspk1_d1[idg==k, :],0) #binned_statistic(posx_trial, sspk1_d1, bins = binsx)[0]        
        print(time.time()-t0)
        t0 = time.time()
        spk[np.isnan(spk)] = 0
        acorrs = np.zeros((len(spk[0,:]), lencorr))
        mid = int(len(spk[:,0])/2)
        lenspk = len(spk[:,0])
        for i in range(num_neurons):
            spk_tmp0 = np.concatenate((spk[:,i].copy(), np.zeros(lencorr)))
            spk_tmp = np.array([np.roll(spk_tmp0, t1)[:lenspk] for t1 in range(lencorr)])
            acorrs[i,:] = np.dot(spk_tmp,spk[:,i])
            acorrs[i,:] /= acorrs[i,0]
            acorrs[i,0] = np.mean(acorrs[i,1:])

        print(time.time()-t0)
        acorrs0 = acorrs.copy()
        acorr_sess[fi] = acorrs0
    return acorr_sess




def get_pos(x, y, t, hd, dt_orig = 0.01, res = 100000, min_time = -1, max_time = None):
    dt = int(dt_orig*res)
    if min_time<0:
        min_time = np.floor(t[0])*res-dt
    else:
        min_time*=res
    if not max_time:
        max_time = np.ceil(t[-1])*res+dt
    else:
        max_time*=res

    for arr in [x,y,hd]:
        if len(arr)>0:
            arr[np.isnan(arr)] = np.mean(arr[~np.isnan(arr)])
#        fill_nans(arr)

    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res
    xspline = CubicSpline(t, x)
    xx = xspline(tt)
    if len(y) > 0:
        yspline = CubicSpline(t, y)
        yy = yspline(tt)
        speed = np.sqrt(np.square(np.diff(gaussian_filter1d(xx, 100))) + 
                        np.square(np.diff(gaussian_filter1d(yy, 100))))
    else:
        yy = []
        speed = np.square(np.diff(xx))
    speed = np.concatenate(([speed[0]],speed))/dt_orig
    
    if len(hd)>0:
        hdcosspline = CubicSpline(t, np.cos(hd))
        hdsinspline = CubicSpline(t, np.sin(hd))
        aa = np.arctan2(hdsinspline(tt),hdcosspline(tt))%(2*np.pi)
    else:
        aa = []
    return tt, xx, yy, speed, aa    
    

def get_pos1(x, y, t, hd, dt_orig = 0.01, res = 100000, min_time = -1, max_time = None):
    dt = int(dt_orig*res)
    if min_time<0:
        min_time = np.floor(t[0])*res-dt
    else:
        min_time*=res
    if not max_time:
        max_time = np.ceil(t[-1])*res+dt
    else:
        max_time*=res

    for arr in [x,y,hd]:
        if len(arr)>0:
            arr[np.isnan(arr)] = np.mean(arr[~np.isnan(arr)])
#        fill_nans(arr)
    tt = np.arange(np.floor(min_time), np.ceil(max_time)+1, dt)/res
    xx = gaussian_filter1d(x, 100)
    xspline = CubicSpline(t, xx)
    xx = xspline(tt)
    if len(y) > 0:
        yy = gaussian_filter1d(y, 100)
        yspline = CubicSpline(t, yy)
        yy = yspline(tt)
        speed = np.sqrt(np.square(np.diff(xx)) + 
                        np.square(np.diff(yy)))
#        speed = np.sqrt(np.square(np.diff(gaussian_filter1d(xx, 100))) + 
#                        np.square(np.diff(gaussian_filter1d(yy, 100))))
    else:
        yy = []
        speed = np.square(np.diff(xx))
    speed = np.concatenate(([speed[0]],speed))/dt_orig
    
    if len(hd)>0:
        hdcosspline = CubicSpline(t, np.cos(hd))
        hdsinspline = CubicSpline(t, np.sin(hd))
        aa = np.arctan2(hdsinspline(tt),hdcosspline(tt))%(2*np.pi)
    else:
        aa = []
    return tt, xx, yy, speed, aa    


    
def get_masscenters(sspikes1, coords_mod1):
    dspk = sspikes1.copy() 
    dspk -= np.mean(dspk,0)
    num_neurons = len(dspk[0,:])
    masscenters_1 = np.zeros((num_neurons, 2))
    for neurid in range(num_neurons):
        centcosall = np.multiply(np.cos(coords_mod1[:, :].T),dspk[:, neurid])
        centsinall = np.multiply(np.sin(coords_mod1[:, :].T),dspk[:, neurid])
        masscenters_1[neurid] = np.arctan2(np.sum(centsinall,1),np.sum(centcosall,1))%(2*np.pi)
    return masscenters_1
        

        
def match_phases4(coords1, sspikes, times = [] ,numbins = 10, t = 0.2):
    if len(times) == 0:
        times = np.arange(len(coords1))
    num_neurons = len(sspikes[0,:])
    mc = get_masscenters(sspikes[times,:], coords1, numbins = 20)
    spk_sim = simulate_spk_hex(coords1, mc, t = t, nums = 1)
    pcorr_hex = np.zeros(num_neurons)            
    num_neurons = len(sspikes[0,:])
    for i in range(num_neurons):
        pcorr_hex[i] = pearsonr(spk_sim[:,i], sspikes[times,i])[0]    
    print(np.mean(pcorr_hex), np.median(pcorr_hex))
    
    coords1[:,0] = 2*np.pi - coords1[:,0]
    
    mc = get_masscenters(sspikes[times,:], coords1, numbins = 20)
    spk_sim = simulate_spk_hex(coords1, mc, t = t, nums = 1)
    pcorr_hex1 = np.zeros(num_neurons)            
    num_neurons = len(sspikes[0,:])
    for i in range(num_neurons):
        pcorr_hex1[i] = pearsonr(spk_sim[:,i], sspikes[times,i])[0]    
    print(np.mean(pcorr_hex1), np.median(pcorr_hex1))    
    
    spk_sim = simulate_spk_sqr(coords1, mc, t = t, nums = 1)
    pcorr_sqr = np.zeros(num_neurons)            
    num_neurons = len(sspikes[0,:])
    for i in range(num_neurons):
        pcorr_sqr[i] = pearsonr(spk_sim[:,i], sspikes[times,i])[0]    
    print(np.mean(pcorr_sqr), np.median(pcorr_sqr))
    print('')
    return pcorr_hex,pcorr_hex1, pcorr_sqr



def match_phases2(coords, sspikes ,numbins = 10, lentmp = 0):
    num_neurons = len(sspikes[0,:])
    mc = get_masscenters(sspikes, coords)    
    
    if lentmp == 0:
        lentmp = len(coords)
    coords_tmp = np.random.rand(lentmp,2)*2*np.pi
    spk_sim = simulate_spk_hex(coords_tmp, mc, t = 0.1, nums = 1)    
    coords2 = coords.copy()
    coords2[:,0] = 2*np.pi-coords2[:,0]
    mc1 = get_masscenters(sspikes, coords2)
    coords_tmp1 = coords_tmp.copy()
    coords_tmp1[:,0] = 2*np.pi-coords_tmp1[:,0]
    spk_sim2 = simulate_spk_hex(coords_tmp1, mc1, t = 0.1, nums = 1)
    
    spk_sqr = simulate_spk_sqr(coords_tmp, mc, t = 0.1, nums = 1)
    pcorr = np.zeros((num_neurons, 3))
 
    for i in range(num_neurons):
        mtot1 = binned_statistic_2d(coords[:,0], coords[:,1], sspikes[:,i], bins = numbins)[0]
        nans = np.isnan(mtot1)
        mtot1[nans] = np.mean(mtot1[~nans])

        mtot2 = binned_statistic_2d(coords_tmp[:,0], coords_tmp[:,1], spk_sim[:,i], bins = numbins)[0]
        nans = np.isnan(mtot2)
        mtot2[nans] = np.mean(mtot2[~nans])
        mtot2 = gaussian_filter(mtot2, sigma = 1) 
        
        mtot22 = binned_statistic_2d(coords_tmp[:,0], coords_tmp[:,1], spk_sim2[:,i], bins = numbins)[0]
        nans = np.isnan(mtot2)
        mtot22[nans] = np.mean(mtot22[~nans])
        mtot22 = gaussian_filter(mtot22, sigma = 1)                
        
        mtot3 = binned_statistic_2d(coords_tmp[:,0], coords_tmp[:,1], spk_sqr[:,i], bins = numbins)[0]
        nans = np.isnan(mtot3)
        mtot3[nans] = np.mean(mtot3[~nans])
        mtot3 = gaussian_filter(mtot3, sigma = 1)                

        pcorr[i,0] = pearsonr(mtot1.flatten(), mtot2.flatten())[0]    
        pcorr[i,1] = pearsonr(mtot1.flatten(), mtot22.flatten())[0]    
        pcorr[i,2] = pearsonr(mtot1.flatten(), mtot3.flatten())[0]        
    return pcorr 


def smooth_tuning_map(mtot, numangsint, sig, bClose = True):
    numangsint_1 = numangsint-1
    mid = int((numangsint_1)/2)
    indstemp1 = np.zeros((numangsint_1,numangsint_1), dtype=int)
    indstemp1[indstemp1==0] = np.arange((numangsint_1)**2)
    indstemp1temp = indstemp1.copy()
    mid = int((numangsint_1)/2)
    mtemp1_3 = mtot.copy()
    for i in range(numangsint_1):
        mtemp1_3[i,:] = np.roll(mtemp1_3[i,:],int(i/2))
    mtot_out = np.zeros_like(mtot)
    mtemp1_4 = np.concatenate((mtemp1_3, mtemp1_3, mtemp1_3),1)
    mtemp1_5 = np.zeros_like(mtemp1_4)
    mtemp1_5[:, :mid] = mtemp1_4[:, (numangsint_1)*3-mid:]  
    mtemp1_5[:, mid:] = mtemp1_4[:,:(numangsint_1)*3-mid]      
    mtemp1_6 = np.concatenate((mtemp1_5,mtemp1_4,mtemp1_5)) 
    if bClose:
        nans = np.isnan(mtemp1_6)
        mtemp1_6[nans] = np.mean(mtemp1_6[~nans])
        mtemp1_6 = gaussian_filter(mtemp1_6,sigma = sig)
        mtemp1_6[nans] = np.nan
        radius = 1
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        kernel = np.array((X ** 2 + Y ** 2) <= radius ** 2).astype(np.uint8)        
        mtemp1_6 = cv2.morphologyEx(mtemp1_6, cv2.MORPH_CLOSE, kernel, iterations = 1)
    else:
        mtemp1_6 = gaussian_filter(mtemp1_6, sigma = sig)
    for i in range(numangsint_1):
        mtot_out[i, :] = mtemp1_6[(numangsint_1)+i, 
                                          (numangsint_1) + (int(i/2) +1):(numangsint_1)*2 + (int(i/2) +1)] 
    return mtot_out




def plot_centers(masscenters_1, ax = None):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    r_box = transforms.Affine2D().skew_deg(15,15)
    ax.scatter(masscenters_1[:,0], masscenters_1[:,1], s = 10, c = 'r', zorder = -1, transform=r_box + ax.transData)
    ax.axis('off')

    ax.plot([0,0], [0,2*np.pi], c = 'k', zorder = -2)
    ax.plot([0,2*np.pi], [0,0], c = 'k', zorder = -2)
    ax.plot([2*np.pi,2*np.pi], [0,2*np.pi], c = 'k', zorder = -2)
    ax.plot([0,2*np.pi], [2*np.pi,2*np.pi], c = 'k', zorder = -2)

    for x in ax.images + ax.lines + ax.collections + ax.get_xticklabels() + ax.get_yticklabels():
        trans = x.get_transform()
        x.set_transform(r_box+trans) 
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r_box+transoff 
    ax.set_xlim([0,2*np.pi + 3/5*np.pi])
    ax.set_ylim([0,2*np.pi + 3/5*np.pi])
    ax.set_aspect('equal', 'box')


    
def plot_ratemaps_all(sspikes1, coords_mod0, xx, yy, 
                      movetimes0 = [], torsort = [], 
                      numfigs = 2, numw = 10, sig1 = 1, numbins1 = 30):  
    
    num_times, num_neurons = np.shape(sspikes1)
    if len(torsort)==0:
        torsort = np.arange(num_neurons)
    if len(movetimes0)==0:
        movetimes0 = np.arange(num_times)
        
    numh = int(np.ceil(num_neurons/numw))
    outer1 = gridspec.GridSpec(1, numw)
    fig = plt.figure(figsize=(np.ceil((numw*numfigs+numw-1)*1.05), np.ceil(numh*1.1)), dpi = 120)
    plt.viridis()
    nw = 0
    for nn, n in enumerate(torsort):
        nnn = nn%numh
        if nnn == 0:
            outer2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer1[nw], wspace = .3)
            gs2 = gridspec.GridSpecFromSubplotSpec(numh, numfigs, subplot_spec = outer2[0], hspace = 0.2,wspace = .0)
            nw += 1
        xnum = 0

        ax = plt.subplot(gs2[nnn,xnum]) 
        xnum += 1

        mtot_tmp, __, __, circ  = binned_statistic_2d(xx[movetimes0],
                                                  yy[movetimes0],
                                                  sspikes1[movetimes0,n], 
                                                  statistic = 'mean', 
                                                  bins = 30,
                                                  expand_binnumbers = True)

        nans = np.isnan(mtot_tmp)
        mtot_tmp[nans] = np.median(mtot_tmp[~nans])
        mtot_tmp = gaussian_filter(mtot_tmp, sig1)
        mintot = np.percentile(mtot_tmp.flatten(), 2.5)
        maxtot = np.percentile(mtot_tmp.flatten(), 97.5)

        ax.imshow(mtot_tmp, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = mintot, vmax = maxtot)
        ax.set_xticks([])
        ax.set_yticks([])


        ax = plt.subplot(gs2[nnn,xnum]) 
        xnum += 1    


        mtot_tmp, __, __, circ  = binned_statistic_2d(coords_mod0[movetimes0,0],
                                                  coords_mod0[movetimes0,1],
                                                  sspikes1[movetimes0,n], 
                                                  statistic = 'mean', 
                                                  bins = 30,
                                                  expand_binnumbers = True)    
        nans = np.isnan(mtot_tmp)
        mtot_tmp[np.isnan(mtot_tmp)] = np.mean(mtot_tmp[~np.isnan(mtot_tmp)])
        maxtot = np.sort(mtot_tmp.flatten())
        mintot = maxtot[int(0.025*len(maxtot))]
        maxtot = maxtot[int(0.975*len(maxtot))]
        mtot_tmp = smooth_tuning_map(mtot_tmp, numbins1+1, sig1, bClose = False) 
        mtot_tmp[nans] = -np.inf
        ax.imshow(mtot_tmp, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = mintot, vmax = maxtot)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1/ax.get_data_ratio())
        r_box = transforms.Affine2D().skew_deg(15,15)    
        maxtot = np.sort(mtot_tmp.flatten())
        mintot = maxtot[int(0.025*len(maxtot))]
        maxtot = maxtot[int(0.975*len(maxtot))]

        ax.imshow(mtot_tmp, origin = 'lower', extent = [0,2*np.pi,0, 2*np.pi], vmin = 0, vmax = maxtot)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1/ax.get_data_ratio())
        r_box = transforms.Affine2D().skew_deg(15,15)

        for x in ax.images + ax.lines + ax.collections:
            trans = x.get_transform()
            x.set_transform(r_box+trans) 
            if isinstance(x, PathCollection):
                transoff = x.get_offset_transform()
                x._transOffset = r_box+transoff     
        ax.set_xlim(0, 2*np.pi + 3*np.pi/5)
        ax.set_ylim(0, 2*np.pi + 3*np.pi/5)
        ax.set_aspect('equal', 'box') 
        ax.axis('off')   




def compute_persistence(X, maxdim, coeff = 47):
    thresh = np.max(X[~np.isinf(X)])
    if maxdim > 1:
        hom_dims = list(range(maxdim+1))
        VR = VietorisRipsPersistence(
        homology_dimensions=hom_dims,
        metric='precomputed',
        coeff=47,
        max_edge_length= thresh,
        collapse_edges=False,  # True faster?
        n_jobs=None  # -1 faster?
        )
        diagrams = VR.fit_transform([X])
        dgms = from_giotto_to_ripser(diagrams[0])
        persistence = ripser(X, maxdim=1, coeff=coeff, do_cocycles= True, distance_matrix = True, thresh = thresh)    
    else:
        persistence = ripser(X, maxdim=1, coeff=coeff, do_cocycles= True, distance_matrix = True, thresh = thresh)    
        dgms = persistence['dgms'] 
    return persistence, dgms 

def get_ratemaps(sspikes1, coords_mod0, movetimes0, inds = []):
    num_times, num_neurons = np.shape(sspikes1)
    if len(inds) == 0:
        inds = np.arange(num_neurons)
    mtot_tmp = {}
    for n in np.arange(num_neurons):
        mtot_tmp[str(n) + '_' + str(inds[n])], __, __, circ  = binned_statistic_2d(coords_mod0[movetimes0,0],
                                                  coords_mod0[movetimes0,1],
                                                  sspikes1[movetimes0,n], 
                                                  statistic = 'mean', 
                                                  bins = 30,
                                                  expand_binnumbers = True)    
    return(mtot_tmp)




def match_phases1(coords1, sspikes, phases, numbins = 6, times = [], adds1 = [], adds2 = []):
    if len(times) == 0:
        times = np.arange(0, len(coords1), 50)
    r_box = transforms.Affine2D().skew_deg(15,15)
    if len(adds1) == 0:
        adds1 = np.linspace(0,2*np.pi, numbins)
        adds2 = np.linspace(0,2*np.pi, numbins)
    medall = np.zeros((len(adds1), len(adds2)))
    meanall = np.zeros((len(adds1), len(adds2)))
    for i1, j in enumerate(adds1):
        for i2, k in enumerate(adds2):
            coords = coords1[times,:]+ np.array([[j,k]])
            coords = coords%(2*np.pi)
            coords_bin, inds_label =  get_coord_distribution_binned(coords, numbins = 50, overlap = 0.)        
            mc = get_phases_binned(sspikes[times,:], coords_bin, inds_label)  
            pcorr = np.sum(np.abs(np.arctan2(np.sin(mc-phases), np.cos(mc-phases))),1)
            medall[i1,i2] = np.median(pcorr)
            meanall[i1,i2] = np.mean(pcorr)
    print('median', np.min(medall), np.max(medall))
    print('meanall', np.min(meanall), np.max(meanall))
    return medall, meanall        

def opt_phase(p, spk, coords1, phases ):
    coords = coords1 + p
    coords = coords%(2*np.pi)
    coords_bin, inds_label =  get_coord_distribution_binned(coords, 
                                                            numbins = 50, 
                                                            overlap = 0.,
                                                            bWrap = False)        
    mc = get_phases_binned(spk, coords_bin, inds_label)  
    pcorr = np.sum(np.abs(np.arctan2(np.sin(mc-phases), 
                                     np.cos(mc-phases))),1)
    #pcorr = np.sum(np.abs(np.arctan2(np.sin(mc-spatial_phases_square), np.cos(mc-spatial_phases_square))),1)
    #print(np.median(pcorr))
    
    return np.median(pcorr)#1./(1+np.exp(-np.median(pcorr)))

def switch_coords(coords_mod0, i, j):
    coords_mod2 = coords_mod0.copy()
    if i == 1:
        coords_mod2[:,0] -= np.sqrt(3)/2*coords_mod2[:,1]
    elif i == 2:
        coords_mod2[:,0] += np.sqrt(3)/2*coords_mod2[:,1]
    coords_mod2 = coords_mod2%(2*np.pi)

    if j == 0:
        coords_mod1 = coords_mod2.copy()
    elif j == 1:
        #-a-b
        coords_mod1 = 2*np.pi-coords_mod2.copy() 
    elif j == 2:
        #ba
        coords_mod1 = coords_mod2.copy() 
        coords_mod1 = np.flip(coords_mod1, axis = 1)
    elif j == 3:
        #-a-b
        coords_mod1 = 2*np.pi-coords_mod2.copy() 
        coords_mod1 = np.flip(coords_mod1, axis = 1)
    elif j == 4:
        # -ab
        coords_mod1 = coords_mod2.copy()
        coords_mod1[:,0] = 2*np.pi - coords_mod1[:,0]
    elif j == 5:
        # a-b
        coords_mod1 = coords_mod2.copy()
        coords_mod1[:,1] = 2*np.pi - coords_mod1[:,1]
    elif j == 6:
        # b-a
        coords_mod1 = coords_mod2.copy()
        coords_mod1[:,0] = 2*np.pi - coords_mod1[:,0]
        coords_mod1 = np.flip(coords_mod1, axis = 1)
    elif j == 7:
        # -ba
        coords_mod1 = coords_mod2.copy()
        coords_mod1[:,1] = 2*np.pi - coords_mod1[:,1]
        coords_mod1 = np.flip(coords_mod1, axis = 1)
    return coords_mod1%(2*np.pi)



def align_coords(coords_mod1,  ds_times1 = [],  ds_times2 = [], indstemp2 = [], coords_ds2 = [], data_ensemble1 = [], data_ensemble2 = [], 
                 bMod = False, coords_mod2 = [], dim = 7):
    ks = np.array([[0,0], [1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
    sig_smooth = 20    

    combs = [[0, 0, 0], 
                [0, 0, 1], 
                [0, 0, 2], 
                [0, 0, 3], 
                [0, 0, 4],
                [0, 1, 0], 
                [0, 1, 1], 
                [0, 1, 2], 
                [0, 1, 3], 
                [0, 1, 4],
                [0, 2, 0], 
                [0, 2, 1], 
                [0, 2, 2], 
                [0, 2, 3], 
                [0, 2, 4],
                [0, 3, 0], 
                [0, 3, 1], 
                [0, 3, 2], 
                [0, 3, 3], 
                [0, 3, 4],
                [1, 0, 0], 
                [1, 0, 1], 
                [1, 0, 2], 
                [1, 0, 3], 
                [1, 0, 4],
                [1, 1, 0], 
                [1, 1, 1], 
                [1, 1, 2], 
                [1, 1, 3], 
                [1, 1, 4],
                [1, 2, 0], 
                [1, 2, 1], 
                [1, 2, 2], 
                [1, 2, 3], 
                [1, 2, 4],
                [1, 3, 0], 
                [1, 3, 1], 
                [1, 3, 2], 
                [1, 3, 3], 
                [1, 3, 4],
                ]

    if not bMod:
        coords_mod2 = get_coords_all(data_ensemble2, 
                                      coords_ds2,   
                                      ds_times2, 
                                      indstemp2, 
                                      spk2 = data_ensemble1,
                                      dim = dim, 
                                      bPCA = True,
                                     )

    coords_smooth_1 = np.arctan2(gaussian_filter1d(np.sin(coords_mod1), sigma = sig_smooth, axis = 0),
        gaussian_filter1d(np.cos(coords_mod1), sigma = sig_smooth, axis = 0))%(2*np.pi)

    coords_smooth_2 = np.arctan2(gaussian_filter1d(np.sin(coords_mod2), sigma = sig_smooth, axis = 0),
                    gaussian_filter1d(np.cos(coords_mod2), sigma = sig_smooth, axis = 0))%(2*np.pi)

    if bMod:
        times = np.arange(0, len(coords_smooth_1), 10)
        res2 = fit_derivative(coords_smooth_1,
                              coords_smooth_2,
                              combs, 
                              times, 
                              thresh = 10
                             )   
        print(res2)
    else:
        res2 = fit_coords(coords_smooth_1[ds_times1,:].copy(),
                          coords_smooth_2[ds_times1,:].copy(),
                          combs, 
                         )
        res2 = np.sum(res2,1)
    comb = combs[np.argmin(res2)]
    coords_smooth_1 = align(coords_smooth_1, comb)
    pshift = np.arctan2(np.mean(np.sin(coords_smooth_1 - coords_smooth_2),0), 
                        np.mean(np.cos(coords_smooth_1 - coords_smooth_2),0))

    coords_ret = align(coords_mod1, comb)
    coords_ret = (coords_ret-pshift)%(2*np.pi)

    return coords_ret, comb, pshift


def fit_derivative(coords1, coords2, combs1, times = [], thresh = 0.1): 
    res2 = np.zeros(len(combs1))
    cctrial = unwrap_coords(coords2.copy())
    times0 = np.arange(len(cctrial))
    cs11 = CubicSpline(times0, cctrial[:,:])
    dcs11 = cs11.derivative(1)(times)
    nans1 = (np.sum(np.abs(dcs11),1)<thresh)
    for ii, (i,j,k) in enumerate(combs1):
        coords_2_1 = align(coords1, [i,j,k])
        cctrial = unwrap_coords(coords_2_1.copy())
        cs22 = CubicSpline(times0, cctrial[:,:])
        dcs22 = cs22.derivative(1)(times)
        nans2 = (np.sum(np.abs(dcs22),1)<thresh)
        nans = nans1 & nans2
#        dcs22 = dcs22[nans] - np.mean(dcs22[nans],0)
        dcs11 = dcs11[nans] - np.mean(dcs11[nans],0)
        Ltemp1 = np.sqrt(np.sum(np.square(dcs11),0))
        Ltemp2 = np.sqrt(np.sum(np.square(dcs22),0))
        res2[ii] = np.sum([1-(np.dot(dcs11[:,i], dcs22[:,i])/np.multiply(Ltemp1[i],Ltemp2[i])) for i in range(2)])
        print(res2[ii])
    return res2





############## GEOMTOOLS ######################
"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To provide tools for quickly computing all pairs self-similarity
and cross-similarity matrices, for doing "greedy permutations," and for
some topological operations like adding cocycles and creating partitions of unity
"""

"""#########################################
   Self-Similarity And Cross-Similarity
#########################################"""

def get_csm(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    
    Parameters
    ----------
    X : ndarray (M, d)
        A matrix holding the coordinates of M points
    Y : ndarray (N, d) 
        A matrix holding the coordinates of N points
    Returns
    ------
    D : ndarray (M, N)
        An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def get_csm_projarc(X, Y):
    """
    Return the projective arc length cross-similarity between two point
    clouds specified as points on the sphere
    Parameters
    ----------
    X : ndarray (M, d)
        A matrix holding the coordinates of M points on RP^{d-1}
    Y : ndarray (N, d) 
        A matrix holding the coordinates of N points on RP^{d-1}
    Returns
    ------
    D : ndarray (M, N)
        An MxN  cross-similarity matrix
    """
    D = np.abs(X.dot(Y.T))
    D[D < -1] = -1
    D[D > 1] = 1
    D = np.arccos(np.abs(D))
    return D

def get_ssm(X):
    return get_csm(X, X)


"""#########################################
         Greedy Permutations
#########################################"""

def get_greedy_perm_pc(X, M, verbose = False, csm_fn = get_csm):
    """
    A Naive O(NM) algorithm to do furthest points sampling, assuming
    the input is a point cloud specified in Euclidean space.  This saves 
    computation over having compute the full distance matrix if the number
    of landmarks M << N
    
    Parameters
    ----------
    X : ndarray (N, d) 
        An Nxd Euclidean point cloud
    M : integer
        Number of landmarks to compute
    verbose: boolean
        Whether to print progress
    csm_fn: function X, Y -> D
        Cross-similarity function (Euclidean by default)

    Return
    ------
    result: Dictionary
        {'Y': An Mxd array of landmarks, 
         'perm': An array of indices into X of the greedy permutation
         'lambdas': Insertion radii of the landmarks
         'D': An MxN array of distances from landmarks to points in X}
    """
    # By default, takes the first point in the permutation to be the
    # first point in the point cloud, but could be random
    N = X.shape[0]
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = csm_fn(X[0, :][None, :], X).flatten()
    D = np.zeros((M, N))
    D[0, :] = ds
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        thisds = csm_fn(X[idx, :][None, :], X).flatten()
        D[i, :] = thisds
        ds = np.minimum(ds, thisds)
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas, 'D':D}

def get_greedy_perm_dm(D, M, verbose = False):
    """
    A Naive O(NM) algorithm to do furthest points sampling, assuming
    the input is a N x N distance matrix
    
    Parameters
    ----------
    D : ndarray (N, N) 
        An N x N distance matrix
    M : integer
        Number of landmarks to compute
    verbose: boolean
        Whether to print progress

    Return
    ------
    result: Dictionary
        {'perm': An array of indices into X of the greedy permutation
         'lambdas': Insertion radii of the landmarks
         'DLandmarks': An MxN array of distances from landmarks to points in the point cloud}
    """
    # By default, takes the first point in the permutation to be the
    # first point in the point cloud, but could be random
    N = D.shape[0]
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = D[0, :]
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    DLandmarks = D[perm, :] 
    return {'perm':perm, 'lambdas':lambdas, 'DLandmarks':DLandmarks}



"""#########################################
        Cohomology Utility Functions
#########################################"""

def add_cocycles(c1, c2, p = 2, real = False):
    S = {}
    c = np.concatenate((c1, c2), 0)
    for k in range(c.shape[0]):
        [i, j, v] = c[k, :]
        i, j = min(i, j), max(i, j)
        if not (i, j) in S:
            S[(i, j)] = v
        else:
            S[(i, j)] += v
    cret = np.zeros((len(S), 3))
    cret[:, 0:2] = np.array([s for s in S])
    cret[:, 2] = np.array([np.mod(S[s], p) for s in S])
    dtype = np.int64
    if real:
        dtype = np.float32
    cret = np.array(cret[cret[:, -1] > 0, :], dtype = dtype)
    return cret

def make_delta0(R):
    """
    Return the delta0 coboundary matrix
    :param R: NEdges x 2 matrix specifying edges, where orientation
    is taken from the first column to the second column
    R specifies the "natural orientation" of the edges, with the
    understanding that the ranking will be specified later
    It is assumed that there is at least one edge incident
    on every vertex
    """
    NVertices = int(np.max(R) + 1)
    NEdges = R.shape[0]
    
    #Two entries per edge
    I = np.zeros((NEdges, 2))
    I[:, 0] = np.arange(NEdges)
    I[:, 1] = np.arange(NEdges)
    I = I.flatten()
    
    J = R[:, 0:2].flatten()
    
    V = np.zeros((NEdges, 2))
    V[:, 0] = -1
    V[:, 1] = 1
    V = V.flatten()
    I = np.array(I, dtype=int)
    J = np.array(J, dtype=int)
    Delta = sparse.coo_matrix((V, (I, J)), shape=(NEdges, NVertices)).tocsr()
    return Delta

def reindex_cocycles(cocycles, idx_land, N):
    """
    Convert the indices of a set of cocycles to be relative
    to a list of indices in a greedy permutation
    Parameters
    ----------
    cocycles: list of list of ndarray
        The cocycles
    idx_land: ndarray(M, dtype=int)
        Indices of the landmarks in the greedy permutation, with
        respect to all points
    N: int
        Number of total points
    """
    idx_map = -1*np.ones(N, dtype=int)
    idx_map[idx_land] = np.arange(idx_land.size)
    for ck in cocycles:
        for c in ck:
            c[:, 0:-1] = idx_map[c[:, 0:-1]]


"""#########################################
        Partition of Unity Functions
#########################################"""

def partunity_linear(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return r_cover - ds

def partunity_quadratic(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return (r_cover - ds)**2

def partunity_exp(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return np.exp(r_cover**2/(ds**2-r_cover**2))

PARTUNITY_FNS = {'linear':partunity_linear, 'quadratic':partunity_quadratic, 'exp':partunity_exp}



################################## EMCOORDS ###############################

"""
A superclass for shared code across all different types of coordinates
"""

"""#########################################
    Some Window Management Utilities
#########################################"""

DREIMAC_FIG_RES = 5 # The resolution of a square cell in inches

def in_notebook(): # pragma: no cover
    """
    Return true if we're in a notebook session, and false otherwise
    with help from https://stackoverflow.com/a/22424821
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except:
        return False
    return True

def compute_dpi(width_cells, height_cells, width_frac=0.5, height_frac=0.65, verbose=False):
    """
    Automatically compute the dpi so that the figure takes
    up some proportion of the available screen width/height
    Parameters
    ----------
    width_cells: float
        The target width of the figure, in units of DREIMAC_FIG_RES
    height_inches: float
        The target height of the figure, in units of DREIMAC_FIG_RES
    width_frac: float
        The fraction of the available width to take up
    height_frac: float
        The fraction of the available height to take up
    verbose: boolean
        Whether to print information about the dpi calculation
    """
    width_inches = DREIMAC_FIG_RES*width_cells
    height_inches = DREIMAC_FIG_RES*height_cells
    # Try to use the screeninfo library to figure out the size of the screen
    width = 1200
    height = 900
    try:
        import screeninfo
        monitor = screeninfo.get_monitors()[0]
        width = monitor.width
        height = monitor.height
    except:
        warnings.warn("Could not accurately determine screen size")
    dpi_width = int(width_frac*width/width_inches)
    dpi_height = int(height_frac*height/height_inches)
    dpi = min(dpi_width, dpi_height)
    if verbose:
        print("width = ", width)
        print("height = ", height)
        print("dpi_width = ", dpi_width)
        print("dpi_height = ", dpi_height)
        print("dpi = ", dpi)
    return dpi

"""#########################################
        Main Circular Coordinates Class
#########################################"""

class EMCoords(object):
    def __init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose):
        """
        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        assert(maxdim >= 1)
        self.verbose = verbose
        if verbose:
            tic = time.time()
            print("Doing TDA...")
        res = ripser(X, distance_matrix=distance_matrix, coeff=prime, maxdim=maxdim, n_perm=n_landmarks, do_cocycles=True)
        if verbose:
            print("Elapsed time persistence: %.3g seconds"%(time.time() - tic))
        self.X_ = X
        self.prime_ = prime
        self.dgms_ = res['dgms']
        self.dist_land_data_ = res['dperm2all']
        self.idx_land_ = res['idx_perm']
        self.dist_land_land_ = self.dist_land_data_[:, self.idx_land_]
        self.cocycles_ = res['cocycles']
        # Sort persistence diagrams in descending order of persistence
        idxs = np.argsort(self.dgms_[1][:, 0]-self.dgms_[1][:, 1])
        self.dgms_[1] = self.dgms_[1][idxs, :]
        self.dgm1_lifetime = np.array(self.dgms_[1])
        self.dgm1_lifetime[:, 1] -= self.dgm1_lifetime[:, 0]
        self.cocycles_[1] = [self.cocycles_[1][idx] for idx in idxs]
        reindex_cocycles(self.cocycles_, self.idx_land_, X.shape[0])
        self.n_landmarks_ = n_landmarks
        self.type_ = "emcoords"

    def setup_ax_persistence(self, y_compress=1):
        """
        Setup the persistence plot in an interactive window
        Parameters
        ----------
        y_compress: float
            The factor by which to compress the y-axis to make room
            for a plot underneath
        """
        dgm = self.dgm1_lifetime
        # Switch to lifetime
        ax_min, ax_max = np.min(dgm), np.max(dgm)
        x_r = ax_max - ax_min
        buffer = x_r / 5
        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer
        y_down, y_up = x_down, x_up
        y_up += (y_up-y_down)*(y_compress-1)
        self.ax_persistence.plot([0, ax_max], [0, 0], "--", c=np.array([0.0, 0.0, 0.0]))
        self.dgmplot, = self.ax_persistence.plot(dgm[:, 0], dgm[:, 1], 'o', c='C0', picker=5)
        self.selected_plot = self.ax_persistence.scatter([], [], 100, c='C1')
        self.ax_persistence.set_xlim([x_down, x_up])
        self.ax_persistence.set_ylim([y_down, y_up])
        self.ax_persistence.set_aspect('equal', 'box')
        self.ax_persistence.set_title("H1 Cocycle Selection")
        self.ax_persistence.set_xlabel("Birth")
        self.ax_persistence.set_ylabel("Lifetime")
        self.persistence_text_labels = [self.ax_persistence.text(dgm[i, 0], dgm[i, 1], '') for i in range(dgm.shape[0])]

    def recompute_coords(self, clicked=[], clear_persistence_text = False):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram
        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        clear_persistence_text: boolean
            Whether to clear all previously labeled dots
        """
        self.selected = self.selected.symmetric_difference(set(clicked))
        idxs = np.array(list(self.selected))
        fmt = "c%i +"*len(idxs)
        fmt = fmt[0:-1]
        self.selected_cocycle_text.set_text(fmt%tuple(idxs))
        if clear_persistence_text:
            for label in self.persistence_text_labels:
                label.set_text("")
        for idx in idxs:
            self.persistence_text_labels[idx].set_text("%i"%idx)
        if idxs.size > 0:
            ## Step 1: Highlight point on persistence diagram
            self.selected_plot.set_offsets(self.dgm1_lifetime[idxs, :])
            ## Step 2: Update coordinates
            perc = self.perc_slider.val
            partunity_fn = PARTUNITY_FNS[self.partunity_selector.value_selected]
            self.coords = self.get_coordinates(cocycle_idx = idxs, perc=perc, partunity_fn = partunity_fn)
        else:
            self.coords = {'X':np.zeros((0, 2))}
            self.selected_plot.set_offsets(np.zeros((0, 2)))

    def setup_param_chooser_gui(self, fig, xstart, ystart, width, height, init_params, button_idx = -1):
        """
        Setup a GUI area 
        Parameters
        ----------
        fig: matplotlib figure handle
            Handle to the interactive figure
        xstart: float
            Where this GUI element is starting along x
        ystart: float
            Where this GUI element is starting along y
        width: float
            Width of GUI element
        height: float
            Height of GUI element
        init_params: dict
            Initial parameters
        button_idx: int
            Index of the circular coordinate to create a press button, or None
            if no button is created
        Returns
        -------
        percslider: matplotlib.widgets.Slider
            Handle to to the slider for choosing coverage
        partunity_selector: matplotlib.widgets.RadioButtons
            Radio buttons for choosing partition of unity type
        """
        # Percent coverage slider
        ax_perc_slider = fig.add_axes([xstart, ystart+height*0.15, 0.5*width, 0.02])
        perc = init_params['perc']
        perc_slider = Slider(ax_perc_slider, "Coverage", valmin=0, valmax=1, valstep=0.01, valinit=perc)
        
        # Partition of unity radio button
        ax_part_unity_label = fig.add_axes([xstart-width*0.175, ystart, 0.3*width, 0.045])
        ax_part_unity_label.text(0.1, 0.3, "Partition\nof Unity")
        ax_part_unity_label.set_axis_off()
        ax_part_unity = fig.add_axes([xstart, ystart, 0.2*width, 0.045])
        active_idx = 0
        partunity_fn = init_params['partunity_fn']
        partunity_keys = tuple(PARTUNITY_FNS.keys())
        for i, key in enumerate(partunity_keys):
            if partunity_fn == PARTUNITY_FNS[key]:
                active_idx = i
        partunity_selector = RadioButtons(ax_part_unity, partunity_keys, active=active_idx)

        # Selected cocycle display
        ax_selected_cocycles_label = fig.add_axes([xstart-width*0.175, ystart-height*0.15, 0.3*width, 0.045])
        ax_selected_cocycles_label.text(0.1, 0.3, "Selected\nCocycle")
        ax_selected_cocycles_label.set_axis_off()
        ax_selected_cocycles = fig.add_axes([xstart, ystart-height*0.15, 0.2*width, 0.045])
        selected_cocycle_text = ax_selected_cocycles.text(0.02, 0.5, "")
        ax_selected_cocycles.set_axis_off()

        # Button to select this particular coordinate
        select_button = None
        if button_idx > -1:
            ax_button_label = fig.add_axes([xstart+width*0.25, ystart, 0.2*width, 0.045])
            select_button = Button(ax_button_label, "Coords {}".format(button_idx))
        return perc_slider, partunity_selector, selected_cocycle_text, select_button

    def get_selected_info(self):
        """
        Return information about what the user selected in
        the interactive plot
        Returns
        -------
        {
            'partunity_fn': (dist_land_data, r_cover) -> phi
                The selected function handle for the partition of unity
            'cocycle_idxs':ndarray(dtype = int)
                Indices of the selected cocycles,
            'perc': float
                The selected percent coverage,
        }
        """
        return {
                'partunity_fn':PARTUNITY_FNS[self.partunity_selector.value_selected], 
                'cocycle_idxs':np.array(list(self.selected)), 
                'perc':self.perc_slider.val,
                }

"""#########################################
        Miscellaneous Utilities
#########################################"""

def callback_factory(callback, k):
    """
    Setup a callback that's linked to a particular
    circular coordinate index.  Having this function takes
    care of scoping issues
    Parameters
    ----------
    callback: function
        The callback to use
    k: int
        The index of the circular coordinate
    """
    return lambda evt: callback(evt, k)

def set_pi_axis_labels(ax, labels):
    """
    Set the axis labels of plots to be the pi symbols
    Parameters
    ----------
    ax: matplotlib handle
        The axes of which to change the symbol labels
    labels: list of string
        The names of the axes
    """
    ax.set_xlabel(labels[0])
    ax.set_xlim([-0.2, 2*np.pi+0.2])
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax.set_ylabel(labels[1])
    ax.set_ylim([-0.2, 2*np.pi+0.2])
    ax.set_yticks([0, np.pi, 2*np.pi])
    ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
    if len(labels) > 2:
        ax.set_zlabel(labels[2])
        ax.set_zlim([-0.2, 2*np.pi+0.2])
        ax.set_zticks([0, np.pi, 2*np.pi])
        ax.set_zticklabels(["0", "$\\pi$", "$2\\pi$"])

def set_3dplot_equalaspect(ax, X, pad=0.1):
    """
    An equal aspect ratio hack for 3D
    Parameters
    ----------
    ax: matplotlib axis
        Handle to the axis to change
    X: ndarray(N, 3)
        Point cloud that's being plotted
    pad: float
        The factor of padding to use around the point cloud
    """
    maxes = np.max(X)
    mins = np.min(X)
    r = maxes - mins
    ax.set_xlim([mins-r*pad, maxes+r*pad])
    ax.set_ylim([mins-r*pad, maxes+r*pad])
    ax.set_zlim([mins-r*pad, maxes+r*pad])

################ TOROIDALCOORDS #######################

# sparse symmetric matrix principal squareroot

sqrt_chebyshev_coeff = np.array([1.800632632, 0.6002108774, -0.1200421755, 0.05144664664, \
-0.02858147035, 0.01818820841, -0.01259183659, 0.009234013499, \
-0.007061304440, 0.005574714032, -0.004512863740, 0.003728017872, \
-0.003131535013, 0.002667603900, -0.002299658534, 0.002002928401, \
-0.001760149201])

def matrix_sqrt(A, degree=8):
    coeff = sqrt_chebyshev_coeff[:degree]
    s = A.shape[0]
    scale = scipy.sparse.linalg.norm(A)
    A = A * (1/scale) - scipy.sparse.identity(s)

    T = 2 * A
    d = scipy.sparse.identity(s) * coeff[-1]
    dd = scipy.sparse.csr_matrix((s,s))
    for n in range(coeff.shape[0]-2,0,-1):
        d, dd = T @ d - dd + coeff[n] * scipy.sparse.identity(s), d

    res = A @ d - dd + 0.5 * coeff[0] * scipy.sparse.identity(s)

    return res * np.sqrt(scale)


"""#########################################
        Main Circular Coordinates Class
#########################################"""
SCATTER_SIZE = 50

class CircularCoords(EMCoords):
    def __init__(self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False):
        """
        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        self.type_ = "circ"

    def get_coordinates(self, perc = 0.5, inner_product = "uniform", cocycle_idxs = [[0]], normalize = True, partunity_fn = partunity_exp, return_gram_matrix=False):
        """
        Perform circular coordinates via persistent cohomology of 
        sparse filtrations (Jose Perea 2018)
        Parameters
        ----------
        perc : float
            Percent coverage
        inner_product : string
            Either 'uniform', 'exponential', or 'consistent'.
        cocycle_idxs : list of lists of integers
            Each list must consist of indices of possible cocycles, and represents the
            cohomology class given by adding the cocycles with the chosen indices
        normalize : bool
            Whether to return circular coordinates between 0 and 1
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        return_gram_matrix : boolean
            Whether to return the gram matrix consisting of the inner products
            between the selected cocycles
        """


        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
        prime = self.prime_

        cocycles = []
        cohomdeaths = []
        cohombirths = []

        for cocycle_idx in cocycle_idxs:
            cohomdeath = -np.inf
            cohombirth = np.inf
            cocycle = np.zeros((0, 3))
            for k in range(len(cocycle_idx)):
                cocycle = add_cocycles(cocycle, self.cocycles_[1][cocycle_idx[k]], p=prime)
                cohomdeath = max(cohomdeath, dgm1[cocycle_idx[k], 0])
                cohombirth = min(cohombirth, dgm1[cocycle_idx[k], 1])
            cocycles.append(cocycle)
            cohomdeaths.append(cohomdeath)
            cohombirths.append(cohombirth)

        cohomdeath = min(cohomdeaths)
        cohombirth = max(cohombirths)


        ## Step 2: Determine radius for balls
        dist_land_data = self.dist_land_data_
        dist_land_land = self.dist_land_land_
        coverage = np.max(np.min(dist_land_data, 1))
        r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
        self.r_cover_ = r_cover # Store covering radius for reference
        if self.verbose:
            print("r_cover = %.3g"%r_cover)


        ## Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        # Compute the partition of unity 
        # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are %i point not covered by a landmark"%nzero)
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]

        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)
        

        threshold = 2*r_cover

        # boundary matrix
        neighbors = { i:set([]) for i in range(n_landmarks) }
        NEdges = n_landmarks**2
        edge_pair_to_index = {}
        l = 0
        row_index = []
        col_index = []
        value = []
        for i in range(n_landmarks):
            for j in range(n_landmarks):
                if i != j and dist_land_land[i,j] < threshold:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    edge_pair_to_index[(i,j)] = l
                    row_index.append(l)
                    col_index.append(i)
                    value.append(-1)
                    row_index.append(l)
                    col_index.append(j)
                    value.append(1)
                l += 1
        delta0 = sparse.coo_matrix((value, (row_index,col_index)),shape=(NEdges, n_landmarks)).tocsr()

        if inner_product=="uniform":
            row_index = []
            col_index = []
            value = []
            for l in edge_pair_to_index.values():
                row_index.append(l)
                col_index.append(l)
                value.append(1)
            WSqrt = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
            W = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
        elif inner_product=="exponential":
            row_index = []
            col_index = []
            value = []
            sqrt_value = []
            for pl in edge_pair_to_index.items():
                p,l = pl
                i,j = p
                val = np.exp(-dist_land_land[i, j]**2/(threshold/2))
                row_index.append(l)
                col_index.append(l)
                value.append(val)
                sqrt_value.append(np.sqrt(val))
            WSqrt = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
            W = scipy.sparse.coo_matrix((sqrt_value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
        elif inner_product=="consistent":
            #print("We assume dataset is Euclidean.")
            #print("start def inner product")
            row_index = []
            col_index = []
            value = []
            for i in range(n_landmarks):
                nn_i = np.argwhere(dist_land_data[i,:] < threshold)[:,0]
                n_nn_i = len(nn_i)
                if n_nn_i == 0:
                    continue
                [aa, bb] = np.meshgrid(nn_i, nn_i)
                aa = aa[np.triu_indices(n_nn_i, 1)]
                bb = bb[np.triu_indices(n_nn_i, 1)]
                how_many = 5
                idx = np.argsort( np.linalg.norm(self.X_[aa] - self.X_[bb], axis=1)  )[:how_many]
                #fraction = 8
                #idx = np.arange(len(aa))
                #idx = idx[np.linalg.norm(self.X_[aa] - self.X_[bb], axis=1) < (r_cover/fraction)]
                aa = aa[idx]
                bb = bb[idx]
                K = lambda x,y : np.exp( - (np.linalg.norm(self.X_[x] - self.X_[y], axis=1) / (r_cover))**2 )
                partial_prod = (varphi[i,aa] + varphi[i,bb]) * K(aa,bb)
                #print(partial_prod[partial_prod>0])
                #print( K(aa,bb).shape )
                for j in neighbors[i]:
                    for k in neighbors[i]:
                        if dist_land_land[j,k] >= threshold :
                            continue
                        a = edge_pair_to_index[(i,j)]
                        b = edge_pair_to_index[(i,k)]
                        row_index.append(a)
                        col_index.append(b)
                        val = 2 * np.sum( partial_prod * (varphi[j,bb] - varphi[j,aa]) * (varphi[k,bb] - varphi[k,aa]))
                        value.append(val)
            W = scipy.sparse.coo_matrix((value, (row_index, col_index)), shape=(NEdges,NEdges)).tocsr()
            #print("defined inner product")

            WSqrt = matrix_sqrt(W, degree=8)
            #print("took square root")
        else:
            raise Exception("inner_product must be uniform, exponential, or consistent!")

        A = WSqrt @ delta0

        all_tau = []

        Ys = []

        for cocycle in cocycles:
            ## Step 3: Setup coboundary matrix, delta_0, for Cech_{r_cover }
            ## and use it to find a projection of the cocycle
            ## onto the image of delta0

            Y = np.zeros((NEdges,))
            for i, j, val in cocycle:
                # lift to integer cocycle
                if val > (prime-1)/2:
                    val -= prime
                if (i,j) in edge_pair_to_index:
                    Y[edge_pair_to_index[(i,j)]] = val
                    Y[edge_pair_to_index[(j,i)]] = -val
 
            b = WSqrt.dot(Y)
            tau = lsqr(A, b)[0]

            Y = Y - delta0.dot(tau)
            Ys.append(Y)

            all_tau.append(tau)

        gram_matrix = np.zeros((len(Ys),len(Ys)))
        for i in range(len(Ys)):
            for j in range(len(Ys)):
                gram_matrix[i,j] = Ys[i].T @ W @ Ys[j]

        circ_coords = []
        for Y, tau in zip(Ys,all_tau):
            ## Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
        
            # compute all transition functions
            theta_matrix = np.zeros((n_landmarks, n_landmarks))

            for pl in edge_pair_to_index.items():
                p,l = pl
                i,j = p
                v = np.mod(Y[l] + 0.5, 1) - 0.5
                theta_matrix[i, j] = v
            class_map = -tau[ball_indx]
            for i in range(n_data):
                class_map[i] += theta_matrix[ball_indx[i], :].dot(varphi[:, i])
            thetas = np.mod(2*np.pi*class_map, 2*np.pi)
            circ_coords.append(thetas)

        if normalize:
            circ_coords = np.array(circ_coords) / (2 * np.pi)

        if return_gram_matrix:
            return circ_coords, gram_matrix
        else:
            return circ_coords

    def update_colors(self):
        if len(self.selected) > 0:
            idxs = np.array(list(self.selected))
            self.selected_plot.set_offsets(self.dgm1_lifetime[idxs, :])
            ## Step 2: Update circular coordinates on point cloud
            thetas = self.coords
            c = plt.get_cmap('magma_r')
            thetas -= np.min(thetas)
            thetas /= np.max(thetas)
            thetas = np.array(np.round(thetas*255), dtype=int)
            C = c(thetas)
            if self.Y.shape[1] == 2:
                self.coords_scatter.set_color(C)
            else:
                self.coords_scatter._facecolor3d = C
                self.coords_scatter._edgecolor3d = C
        else:
            self.selected_plot.set_offsets(np.zeros((0, 2)))
            if self.Y.shape[1] == 2:
                self.coords_scatter.set_color('C0')
            else:
                self.coords_scatter._facecolor3d = 'C0'
                self.coords_scatter._edgecolor3d = 'C0'

    def recompute_coords_dimred(self, clicked = []):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram, and update the circular coordinates
        colors accordingly

        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        """
        EMCoords.recompute_coords(self, clicked)
        self.update_colors()
        
    def onpick_dimred(self, evt):
        if evt.artist == self.dgmplot:
            ## Step 1: Highlight point on persistence diagram
            clicked = set(evt.ind.tolist())
            self.recompute_coords_dimred(clicked)
        self.ax_persistence.figure.canvas.draw()
        self.ax_coords.figure.canvas.draw()
        return True

    def on_perc_slider_move_dimred(self, evt):
        self.recompute_coords_dimred()

    def on_partunity_selector_change_dimred(self, evt):
        self.recompute_coords_dimred()

    def plot_dimreduced(self, Y, using_jupyter = True, init_params = {'cocycle_idxs':[], 'perc':0.99, 'partunity_fn':partunity_linear, 'azim':-60, 'elev':30}, dpi=None):
        """
        Do an interactive plot of circular coordinates, coloring a dimension
        reduced version of the point cloud by the circular coordinates

        Parameters
        ----------
        Y: ndarray(N, d)
            A 2D point cloud with the same number of points as X
        using_jupyter: boolean
            Whether this is an interactive plot in jupyter
        init_params: dict
            The intial parameters.  Optional fields of the dictionary are as follows:
            {
                cocycle_idxs: list of int
                    A list of cocycles to start with
                u: ndarray(3, float)
                    The initial stereographic north pole
                perc: float
                    The percent coverage to start with
                partunity_fn: (dist_land_data, r_cover) -> phi
                    The partition of unity function to start with
                azim: float
                    Initial azimuth for 3d plots
                elev: float
                    Initial elevation for 3d plots
            }
        dpi: int
            Dot pixels per inch
        """
        if Y.shape[1] < 2 or Y.shape[1] > 3:
            raise Exception("Dimension reduced version must be in 2D or 3D")
        self.Y = Y
        if using_jupyter and in_notebook():
            import matplotlib
            matplotlib.use("nbAgg")
        if not dpi:
            dpi = compute_dpi(2, 1)
        fig = plt.figure(figsize=(DREIMAC_FIG_RES*2, DREIMAC_FIG_RES), dpi=dpi)
        ## Step 1: Plot H1
        self.ax_persistence = fig.add_subplot(121)
        self.setup_ax_persistence(y_compress=1.37)
        fig.canvas.mpl_connect('pick_event', self.onpick_dimred)
        self.selected = set([])

        ## Step 2: Setup window for choosing coverage / partition of unity type
        ## and for displaying the chosen cocycle
        self.perc_slider, self.partunity_selector, self.selected_cocycle_text, _ = EMCoords.setup_param_chooser_gui(self, fig, 0.25, 0.75, 0.4, 0.5, init_params)
        self.perc_slider.on_changed(self.on_perc_slider_move_dimred)
        self.partunity_selector.on_clicked(self.on_partunity_selector_change_dimred)

        ## Step 3: Setup axis for coordinates
        if Y.shape[1] == 3:
            self.ax_coords = fig.add_subplot(122, projection='3d')
            self.coords_scatter = self.ax_coords.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=SCATTER_SIZE, cmap='magma_r')
            set_3dplot_equalaspect(self.ax_coords, Y)
            if 'azim' in init_params:
                self.ax_coords.azim = init_params['azim']
            if 'elev' in init_params:
                self.ax_coords.elev = init_params['elev']
        else:
            self.ax_coords = fig.add_subplot(122)
            self.coords_scatter = self.ax_coords.scatter(Y[:, 0], Y[:, 1], s=SCATTER_SIZE, cmap='magma_r')
            self.ax_coords.set_aspect('equal')
        self.ax_coords.set_title("Dimension Reduced Point Cloud")
        if len(init_params['cocycle_idxs']) > 0:
            # If some initial cocycle indices were chosen, update
            # the plot
            self.recompute_coords_dimred(init_params['cocycle_idxs'])
        plt.show()
    
    def get_selected_dimreduced_info(self):
        """
        Return information about what the user selected and their viewpoint in
        the interactive dimension reduced plot

        Returns
        -------
        {
            'partunity_fn': (dist_land_data, r_cover) -> phi
                The selected function handle for the partition of unity
            'cocycle_idxs':ndarray(dtype = int)
                Indices of the selected cocycles,
            'perc': float
                The selected percent coverage,
            'azim':float
                Azumith if viewing in 3D
            'elev':float
                Elevation if viewing in 3D
        }
        """
        ret = EMCoords.get_selected_info(self)
        if self.Y.shape[1] == 3:
            ret['azim'] = self.ax_coords.azim
            ret['elev'] = self.ax_coords.elev
        return ret

    def update_plot_torii(self, circ_idx):
        """
        Update a joint plot of circular coordinates, switching between
        2D and 3D modes if necessary

        Parameters
        ----------
        circ_idx: int
            Index of the circular coordinates that have
            been updated
        """
        N = self.plots_in_one
        n_plots = len(self.plots)
        ## Step 1: Figure out the index of the involved plot
        plot_idx = int(np.floor(circ_idx/N))
        plot = self.plots[plot_idx]

        ## Step 2: Extract the circular coordinates from all
        ## plots that have at least one cochain representative selected
        labels = []
        coords = []
        for i in range(N):
            idx = plot_idx*N + i
            c_info = self.coords_info[idx]
            if len(c_info['selected']) > 0:
                # Only include circular coordinates that have at least
                # one persistence dot selected
                coords.append(c_info['coords'])
                labels.append("Coords {}".format(idx))
        ## Step 3: Adjust the plot accordingly
        if len(labels) > 0:
            X = np.array([])
            if len(labels) == 1:
                # Just a single coordinate; put it on a circle
                coords = np.array(coords).flatten()
                X = np.array([np.cos(coords), np.sin(coords)]).T
            else:
                X = np.array(coords).T
            updating_axes = False
            if X.shape[1] == 3 and plot['axis_2d']:
                # Need to switch from 2D to 3D coordinates
                self.fig.delaxes(plot['ax'])
                plot['axis_2d'] = False
                updating_axes = True
            elif X.shape[1] == 2 and not plot['axis_2d']:
                # Need to switch from 3D to 2D coordinates
                self.fig.delaxes(plot['ax'])
                plot['axis_2d'] = True
                updating_axes = True
            if X.shape[1] == 3:
                if updating_axes:
                    plot['ax'] = self.fig.add_subplot(2, n_plots+1, n_plots+3+plot_idx, projection='3d')
                    plot['coords_scatter'] = plot['ax'].scatter(X[:, 0], X[:, 1], X[:, 2], s=SCATTER_SIZE, c=self.coords_colors)
                    plot['ax'].set_title('Joint 3D Plot')
                else:
                    plot['coords_scatter'].set_offsets(X)
                set_pi_axis_labels(plot['ax'], labels)
            else:
                if updating_axes:
                    plot['ax'] = self.fig.add_subplot(2, n_plots+1, n_plots+3+plot_idx)
                    plot['coords_scatter'] = plot['ax'].scatter(X[:, 0], X[:, 1], s=SCATTER_SIZE, c=self.coords_colors)
                else:
                    plot['coords_scatter'].set_offsets(X)
                if len(labels) > 1:
                    set_pi_axis_labels(plot['ax'], labels)
                    plot['ax'].set_title('Joint 2D Plot')
                else:
                    plot['ax'].set_xlabel('')
                    plot['ax'].set_xlim([-1.1, 1.1])
                    plot['ax'].set_ylabel('')
                    plot['ax'].set_ylim([-1.1, 1.1])
                    plot['ax'].set_title(labels[0])
        else:
            X = np.array([])
            if plot['axis_2d']:
                X = -2*np.ones((self.X_.shape[0], 2))
            else:
                X = -2*np.ones((self.X_.shape[0], 3))
            plot['coords_scatter'].set_offsets(X)
            
    
    def recompute_coords_torii(self, clicked = []):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram, and update the circular coordinates
        joint torii plots accordingly

        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        """
        EMCoords.recompute_coords(self, clicked)
        # Save away circular coordinates
        self.coords_info[self.selected_coord_idx]['selected'] = self.selected
        self.coords_info[self.selected_coord_idx]['coords'] = self.coords
        self.update_plot_torii(self.selected_coord_idx)

    def onpick_torii(self, evt):
        """
        Handle a pick even for the torii plot
        """
        if evt.artist == self.dgmplot:
            ## Step 1: Highlight point on persistence diagram
            clicked = set(evt.ind.tolist())
            self.recompute_coords_torii(clicked)
        self.ax_persistence.figure.canvas.draw()
        self.fig.canvas.draw()
        return True

    def select_torii_coord(self, idx):
        """
        Select a particular circular coordinate plot and un-select others
        
        Parameters
        ----------
        idx: int
            Index of the plot to select
        """
        for i, coordsi in enumerate(self.coords_info):
            if i == idx:
                self.selected_coord_idx = idx
                coordsi = self.coords_info[idx]
                # Swap in the appropriate GUI objects for selection
                self.selected = coordsi['selected']
                self.selected_cocycle_text = coordsi['selected_cocycle_text']
                self.perc_slider = coordsi['perc_slider']
                self.partunity_selector = coordsi['partunity_selector']
                self.persistence_text_labels = coordsi['persistence_text_labels']
                self.coords = coordsi['coords']
                coordsi['button'].color = 'red'
                for j in np.array(list(self.selected)):
                    self.persistence_text_labels[j].set_text("%i"%j)
                idxs = np.array(list(self.selected), dtype=int)
                if idxs.size > 0:
                    self.selected_plot.set_offsets(self.dgm1_lifetime[idxs, :])
                else:
                    self.selected_plot.set_offsets(np.array([[np.nan]*2]))
            else:
                coordsi['button'].color = 'gray'
        self.ax_persistence.set_title("H1 Cocycle Selection: Coordinate {}".format(idx))

    def on_perc_slider_move_torii(self, evt, idx):
        """
        React to a change in coverage
        a particular circular coordinate, and recompute the 
        coordinates if they aren't trivial
        """
        if not self.selected_coord_idx == idx:
            self.select_torii_coord(idx)
        if len(self.selected) > 0:
            self.recompute_coords_torii()

    def on_partunity_selector_change_torii(self, evt, idx):
        """
        React to a change in partition of unity type for 
        a particular circular coordinate, and recompute the 
        coordinates if they aren't trivial
        """
        if not self.selected_coord_idx == idx:
            self.select_torii_coord(idx)
        if len(self.selected) > 0:
            self.recompute_coords_torii()

    def on_click_torii_button(self, evt, idx):
        """
        React to a click event, and change the selected
        circular coordinate if necessary
        """
        if not self.selected_coord_idx == idx:
            self.select_torii_coord(idx)

    def plot_torii(self, f, using_jupyter=True, zoom=1, dpi=None, coords_info=2, plots_in_one = 2, lowerleft_plot = None, lowerleft_3d=False):
        """
        Do an interactive plot of circular coordinates, where points are drawn on S1, 
        on S1 x S1, or S1 x S1 x S1

        Parameters
        ----------
        f: Display information for the points
            On of three options:
            1) A scalar function with which to color the points, represented
               as a 1D array
            2) A list of colors with which to color the points, specified as
               an Nx3 array
            3) A list of images to place at each location
        using_jupyter: boolean
            Whether this is an interactive plot in jupyter
        zoom: int
            If using patches, the factor by which to zoom in on them
        dpi: int
            Dot pixels per inch
        coords_info: Information about how to perform circular coordinates.  There will
            be as many plots as the ceil of the number of circular coordinates, and
            they will be plotted pairwise.
            This parameter is one of two options
            1) An int specifying the number of different circular coordinate
               functions to compute
            2) A list of dictionaries with pre-specified initial parameters for
               each circular coordinate.  Each dictionary has the following keys:
               {
                    'cocycle_reps': dictionary
                        A dictionary of cocycle representatives, with the key
                        as the cocycle index, and the value as the coefficient
                    TODO: Finish update to support this instead of a set
                    'perc': float
                        The percent coverage to start with,
                    'partunity_fn': (dist_land_data, r_cover) -> phi
                        The partition of unity function to start with
               }
        plots_in_one: int
            The max number of circular coordinates to put in one plot
        lowerleft_plot: function(matplotlib axis)
            A function that plots something in the lower left
        lowerleft_3d: boolean
            Whether the lower left plot is 3D
        """
        if plots_in_one < 2 or plots_in_one > 3:
            raise Exception("Cannot be fewer than 2 or more than 3 circular coordinates in one plot")
        self.plots_in_one = plots_in_one
        self.f = f
        ## Step 1: Figure out how many plots are needed to accommodate all
        ## circular coordinates
        n_plots = 1
        if type(coords_info) is int:
            n_plots = int(np.ceil(coords_info/plots_in_one))
            coords_info = []
        else:
            n_plots = int(np.ceil(len(coords_info)/plots_in_one))
        while len(coords_info) < n_plots*plots_in_one:
            coords_info.append({'selected':set([]), 'perc':0.99, 'partunity_fn':partunity_linear})
        self.selecting_idx = 0 # Index of circular coordinate which is currently being selected
        if using_jupyter and in_notebook():
            import matplotlib
            matplotlib.use("nbAgg")
        if not dpi:
            dpi = compute_dpi(n_plots+1, 2)
        fig = plt.figure(figsize=(DREIMAC_FIG_RES*(n_plots+1), DREIMAC_FIG_RES*2), dpi=dpi)
        self.dpi = dpi
        self.fig = fig

        ## Step 2: Setup H1 plot, along with initially empty text labels
        ## for each persistence point
        self.ax_persistence = fig.add_subplot(2, n_plots+1, 1)
        self.setup_ax_persistence()
        fig.canvas.mpl_connect('pick_event', self.onpick_torii)


        ## Step 2: Setup windows for choosing coverage / partition of unity type
        ## and for displaying the chosen cocycle for each circular coordinate.
        ## Also store variables for selecting cocycle representatives
        width = 1/(n_plots+1)
        height = 1/plots_in_one
        partunity_keys = tuple(PARTUNITY_FNS.keys())
        for i in range(n_plots):
            xstart = width*(i+1.4)
            for j in range(plots_in_one):
                idx = i*plots_in_one+j
                # Setup plots and state for a particular circular coordinate
                ystart = 0.8 - 0.4*height*j
                coords_info[idx]['perc_slider'], coords_info[idx]['partunity_selector'], coords_info[idx]['selected_cocycle_text'], coords_info[idx]['button'] = self.setup_param_chooser_gui(fig, xstart, ystart, width, height, coords_info[idx], idx)
                coords_info[idx]['perc_slider'].on_changed(callback_factory(self.on_perc_slider_move_torii, idx))
                coords_info[idx]['partunity_selector'].on_clicked = callback_factory(self.on_partunity_selector_change_torii, idx)
                coords_info[idx]['button'].on_clicked(callback_factory(self.on_click_torii_button, idx))
                dgm = self.dgm1_lifetime
                coords_info[idx]['persistence_text_labels'] = [self.ax_persistence.text(dgm[i, 0], dgm[i, 1], '') for i in range(dgm.shape[0])]
                coords_info[idx]['idx'] = idx
                coords_info[idx]['coords'] = np.zeros(self.X_.shape[0])
        self.coords_info = coords_info

        ## Step 3: Figure out colors of coordinates
        self.coords_colors = None
        if not (type(f) is list):
            # Figure out colormap if images aren't passed along
            self.coords_colors = f
            if f.size == self.X_.shape[0]:
                # Scalar function, so need to apply colormap
                c = plt.get_cmap('magma_r')
                fscaled = f - np.min(f)
                fscaled = fscaled/np.max(fscaled)
                C = c(np.array(np.round(fscaled*255), dtype=np.int32))
                self.coords_colors = C[:, 0:3]
        
        ## Step 4: Setup plots
        plots = []
        self.n_plots = n_plots
        for i in range(n_plots):
            # 2D by default, but can change to 3D later
            ax = fig.add_subplot(2, n_plots+1, n_plots+3+i)
            pix = -2*np.ones(self.X_.shape[0])
            plot = {}
            plot['ax'] = ax
            plot['coords_scatter'] = ax.scatter(pix, pix, s=SCATTER_SIZE, c=self.coords_colors) # Scatterplot for circular coordinates
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            plot['axis_2d'] = True
            plot['patch_boxes'] = [] # Array of image patch display objects
            plots.append(plot)
        self.plots = plots

        ## Step 5: Initialize plots with information passed along
        for i in reversed(range(len(coords_info))):
            self.select_torii_coord(i)
            self.recompute_coords_torii([])
        
        ## Step 6: Plot something in the lower left corner if desired
        if lowerleft_plot:
            if lowerleft_3d:
                ax = fig.add_subplot(2, n_plots+1, n_plots+2, projection='3d')
            else:
                ax = fig.add_subplot(2, n_plots+1, n_plots+2)
            lowerleft_plot(ax)

        plt.show()

def do_two_circle_test():
    """
    Test interactive plotting with two noisy circles of different sizes
    """
    prime = 41
    np.random.seed(2)
    N = 500
    X = np.zeros((N*2, 2))
    t = np.linspace(0, 1, N+1)[0:N]**1.2
    t = 2*np.pi*t
    X[0:N, 0] = np.cos(t)
    X[0:N, 1] = np.sin(t)
    X[N::, 0] = 2*np.cos(t) + 4
    X[N::, 1] = 2*np.sin(t) + 4
    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    X = X + 0.2*np.random.randn(X.shape[0], 2)

    f = np.concatenate((t, t + np.max(t)))
    f = f[perm]
    fscaled = f - np.min(f)
    fscaled = fscaled/np.max(fscaled)
    c = plt.get_cmap('magma_r')
    C = c(np.array(np.round(fscaled*255), dtype=np.int32))[:, 0:3]
    #plt.scatter(X[:, 0], X[:, 1], s=SCATTER_SIZE, c=C)
    
    cc = CircularCoords(X, 100, prime = prime)
    #cc.plot_dimreduced(X, using_jupyter=False)
    cc.plot_torii(f, coords_info=2, plots_in_one=3)

def do_torus_test():
    """
    Test interactive plotting with a torus
    """
    prime = 41
    np.random.seed(2)
    N = 10000
    R = 5
    r = 2
    X = np.zeros((N, 3))
    s = np.random.rand(N)*2*np.pi
    t = np.random.rand(N)*2*np.pi
    X[:, 0] = (R + r*np.cos(s))*np.cos(t)
    X[:, 1] = (R + r*np.cos(s))*np.sin(t)
    X[:, 2] = r*np.sin(s)

    cc = CircularCoords(X, 100, prime=prime)
    f = s
    def plot_torus(ax):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=f, cmap='magma_r')
        set_3dplot_equalaspect(ax, X)

    cc.plot_torii(f, coords_info=2, plots_in_one=2, lowerleft_plot=plot_torus, lowerleft_3d=True)



### Circular Coordinates AUX

# operations with circular coordinates
def center_circ_coord(coord, bins=50):
    coord = coord.copy()
    
    x = np.hstack(coord)
    vals, ticks = np.histogram(x,bins=bins)
    coord = ((coord - ticks[np.argmax(vals)]) + 0.5) % 1
    return coord

def sum_circ_coords(c1, c2):
    return (c1 + c2) % 1

def sub_circ_coords(c1, c2):
    return (c1 - c2) % 1

def offset_circ_coord(c, offset):
    return (c + offset) % 1


# improve circular coordinates with lattice reduction
def reduce_circular_coordinates(circ_coords, gram_matrix):
    lattice_red_input = np.linalg.cholesky(gram_matrix)
    new_vectors, change_basis = LLL(lattice_red_input.T)
    change_basis = change_basis.T
    new_gram_matrix = new_vectors.T @ new_vectors
    new_circ_coords = (change_basis @ circ_coords) % 1
    return new_circ_coords, new_gram_matrix, change_basis

# approximate geodesic distance
def geodesic_distance(X, k = 15):
    iso = Isomap(n_components = 2,n_neighbors=k)
    return iso.fit(X).dist_matrix_


# sliding window embedding
def sw(ts, d, tau):
    emb = []
    last = len(ts) - ((d - 1) * tau)
    for i in range(last):
        emb.append(ts[i:i + d * tau:tau])
    return np.array(emb)


# Dirichlet form between maps to circle
exp_weight = lambda radius : lambda d : np.exp(-d**2/radius**2)
const_weight = lambda d : 1

def differential_circle_valued_map(fi,fj):
    if fi < fj:
        a = fj - fi
        b = (fj - 1) - fi
    else:
        a = fj - fi
        b = (fj + 1) - fi
    if np.abs(a) < np.abs(b):
        return a
    else:
        return b

def dirichlet_form(X,f,g,weight,eps,k,graph_type):
    ##print("We assume dataset is Euclidean.")
    # type may be "k" or "eps"
    # f and g are vectors of numbers between 0 and 1, interpreted as maps to the circle
    # we assume Euclidean distance
    tree = KDTree(X)
    if graph_type=="k":
        _, neighbors = tree.query(X,k=k)
    if graph_type=="eps":
        neighbors = tree.query_radius(X,r=eps)
    checked = set()
    partial_form = 0
    for i in range(X.shape[0]):
        for j in neighbors[i]:
            if (i,j) in checked or (j,i) in checked:
                continue
            dist_ij = np.linalg.norm(X[i] - X[j])
            df_ij = differential_circle_valued_map(f[i], f[j])
            dg_ij = differential_circle_valued_map(g[i], g[j])
            partial_form += weight(dist_ij) * df_ij * dg_ij
    return partial_form

def dirichlet_form_gram_matrix(X,circ_coords,weight,eps=None,k=None,graph_type="k"):
    number_circ_coords = len(circ_coords)
    res = np.zeros((number_circ_coords,number_circ_coords))
    for i in range(number_circ_coords):
        for j in range(number_circ_coords):
            x = dirichlet_form(X,circ_coords[i], circ_coords[j], weight, eps, k, graph_type)
            res[i,j] = x
            res[j,i] = x
    return res


# prominence diagram
def prominence_diagram(dgm, max_proms=10):
    prominences = dgm[:,1] - dgm[:,0]
    how_many = min(max_proms, len(prominences))
    order_by_prominence = np.argsort(-prominences)
    prominences = prominences[order_by_prominence]
    return order_by_prominence


#### Lattice Reduction

# Optimal algorithm for the case of two vectors:

def Lagrange(B):
    B = B.copy()
    change = np.eye(B.shape[0])

    if np.linalg.norm(B[:,0]) < np.linalg.norm(B[:,1]):
        B0 = B[:,0].copy()
        B1 = B[:,1].copy()
        B[:,0], B[:,1] = B1, B0
        change0 = change[:,0].copy()
        change1 = change[:,1].copy()
        change[:,0], change[:,1] = change1, change0

    
    while np.linalg.norm(B[:,1]) <= np.linalg.norm(B[:,0]):
        q = round(np.dot(B[:,0],B[:,1]/np.linalg.norm(B[:,1])**2))
        r = B[:,0] - q * B[:,1]
        c = change[:,0] - q * change[:,1]
        B[:,0] = B[:,1]
        change[:,0] = change[:,1]
        B[:,1] = r
        change[:,1] = c

    return B, change
    

#Gram-Schmidt (without normalization)

def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

# projects v2 onto v1
def proj(v1, v2):
    return gs_cofficient(v1, v2) * v1


def GS(B):
    n = len(B)
    A = np.zeros((n,n))
    A[:,0] = B[:,0]
    for i in range(1,n):
        Ai = B[:, i]
        for j in range(0, i):
            Aj = B[:, j]
            #t = np.dot(B[i],B[j])
            Ai = Ai - proj(Aj,Ai)
        A[:, i] = Ai
    return A

# LLL algorithm 

def LLL(B, delta = 3/4):
    B = B.copy()
    Q = GS(B)
    change = np.eye(B.shape[0])

    def mu(i,j):
        v = B[:,i]
        u = Q[:,j]
        return np.dot(v,u) / np.dot(u,u)   

    n, k = len(B), 1
    while k < n:

        # length reduction step
        for j in reversed(range(k)):
            if abs(mu(k,j)) > .5:
                mu_kj = mu(k,j)
                B[:,k] = B[:,k] - round(mu_kj)*B[:,j]
                change[:,k] = change[:,k] - round(mu_kj)*change[:,j]
                Q = GS(B)

        # swap step
        if np.dot(Q[:,k],Q[:,k]) > (delta - mu(k,k-1)**2)*(np.dot(Q[:,k-1],Q[:,k-1])):
            k = k + 1
        else:
            B_k = B[:,k].copy()
            B_k1 = B[:,k-1].copy()
            B[:,k], B[:,k-1] = B_k1, B_k
            change_k = change[:,k].copy()
            change_k1 = change[:,k-1].copy()
            change[:,k], change[:,k-1] = change_k1, change_k
 
            Q = GS(B)
            k = max(k-1, 1)

    return B, change

############################ END TOROIDALCOORDS ###################################


############################ BEGIN SPHERECOORDS ###################################


import dreimac
from dreimac import GeometryExamples, PlotUtils, ComplexProjectiveCoords, ProjectiveMapUtils, GeometryUtils
from dreimac.utils import PartUnity, CohomologyUtils, EquivariantPCA
from dreimac.combinatorial import (
    combinatorial_number_system_table,
    combinatorial_number_system_d1_forward,
    combinatorial_number_system_d2_forward,
)
from scipy.optimize import LinearConstraint#, milp

def milp(c, *, integrality=None, bounds=None, constraints=None, options=None):
    r"""
    Mixed-integer linear programming

    Solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & b_l \leq A x \leq b_u,\\
        & l \leq x \leq u, \\
        & x_i \in \mathbb{Z}, i \in X_i

    where :math:`x` is a vector of decision variables;
    :math:`c`, :math:`b_l`, :math:`b_u`, :math:`l`, and :math:`u` are vectors;
    :math:`A` is a matrix, and :math:`X_i` is the set of indices of
    decision variables that must be integral. (In this context, a
    variable that can assume only integer values is said to be "integral";
    it has an "integrality" constraint.)

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        b_l <= A @ x <= b_u
        l <= x <= u
        Specified elements of x must be integers

    By default, ``l = 0`` and ``u = np.inf`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1D dense array_like
        The coefficients of the linear objective function to be minimized.
        `c` is converted to a double precision array before the problem is
        solved.
    integrality : 1D dense array_like, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or take value ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or take value ``0``.

        By default, all variables are continuous. `integrality` is converted
        to an array of integers before the problem is solved.

    bounds : scipy.optimize.Bounds, optional
        Bounds on the decision variables. Lower and upper bounds are converted
        to double precision arrays before the problem is solved. The
        ``keep_feasible`` parameter of the `Bounds` object is ignored. If
        not specified, all decision variables are constrained to be
        non-negative.
    constraints : sequence of scipy.optimize.LinearConstraint, optional
        Linear constraints of the optimization problem. Arguments may be
        one of the following:

        1. A single `LinearConstraint` object
        2. A single tuple that can be converted to a `LinearConstraint` object
           as ``LinearConstraint(*constraints)``
        3. A sequence composed entirely of objects of type 1. and 2.

        Before the problem is solved, all values are converted to double
        precision, and the matrices of constraint coefficients are converted to
        instances of `scipy.sparse.csc_array`. The ``keep_feasible`` parameter
        of `LinearConstraint` objects is ignored.
    options : dict, optional
        A dictionary of solver options. The following keys are recognized.

        disp : bool (default: ``False``)
            Set to ``True`` if indicators of optimization status are to be
            printed to the console during optimization.
        node_limit : int, optional
            The maximum number of nodes (linear program relaxations) to solve
            before stopping. Default is no maximum number of nodes.
        presolve : bool (default: ``True``)
            Presolve attempts to identify trivial infeasibilities,
            identify trivial unboundedness, and simplify the problem before
            sending it to the main solver.
        time_limit : float, optional
            The maximum number of seconds allotted to solve the problem.
            Default is no time limit.
        mip_rel_gap : float, optional
            Termination criterion for MIP solver: solver will terminate when
            the gap between the primal objective value and the dual objective
            bound, scaled by the primal objective value, is <= mip_rel_gap.

    Returns
    -------
    res : OptimizeResult
        An instance of :class:`scipy.optimize.OptimizeResult`. The object
        is guaranteed to have the following attributes.

        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimal solution found.

            ``1`` : Iteration or time limit reached.

            ``2`` : Problem is infeasible.

            ``3`` : Problem is unbounded.

            ``4`` : Other; see message for details.

        success : bool
            ``True`` when an optimal solution is found and ``False`` otherwise.

        message : str
            A string descriptor of the exit status of the algorithm.

        The following attributes will also be present, but the values may be
        ``None``, depending on the solution status.

        x : ndarray
            The values of the decision variables that minimize the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        mip_node_count : int
            The number of subproblems or "nodes" solved by the MILP solver.
        mip_dual_bound : float
            The MILP solver's final estimate of the lower bound on the optimal
            solution.
        mip_gap : float
            The difference between the primal objective value and the dual
            objective bound, scaled by the primal objective value.

    Notes
    -----
    `milp` is a wrapper of the HiGHS linear optimization software [1]_. The
    algorithm is deterministic, and it typically finds the global optimum of
    moderately challenging mixed-integer linear programs (when it exists).

    References
    ----------
    .. [1] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           https://highs.dev/
    .. [2] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5

    Examples
    --------
    Consider the problem at
    https://en.wikipedia.org/wiki/Integer_programming#Example, which is
    expressed as a maximization problem of two variables. Since `milp` requires
    that the problem be expressed as a minimization problem, the objective
    function coefficients on the decision variables are:

    >>> import numpy as np
    >>> c = -np.array([0, 1])

    Note the negative sign: we maximize the original objective function
    by minimizing the negative of the objective function.

    We collect the coefficients of the constraints into arrays like:

    >>> A = np.array([[-1, 1], [3, 2], [2, 3]])
    >>> b_u = np.array([1, 12, 12])
    >>> b_l = np.full_like(b_u, -np.inf)

    Because there is no lower limit on these constraints, we have defined a
    variable ``b_l`` full of values representing negative infinity. This may
    be unfamiliar to users of `scipy.optimize.linprog`, which only accepts
    "less than" (or "upper bound") inequality constraints of the form
    ``A_ub @ x <= b_u``. By accepting both ``b_l`` and ``b_u`` of constraints
    ``b_l <= A_ub @ x <= b_u``, `milp` makes it easy to specify "greater than"
    inequality constraints, "less than" inequality constraints, and equality
    constraints concisely.

    These arrays are collected into a single `LinearConstraint` object like:

    >>> from scipy.optimize import LinearConstraint
    >>> constraints = LinearConstraint(A, b_l, b_u)

    The non-negativity bounds on the decision variables are enforced by
    default, so we do not need to provide an argument for `bounds`.

    Finally, the problem states that both decision variables must be integers:

    >>> integrality = np.ones_like(c)

    We solve the problem like:

    >>> from scipy.optimize import milp
    >>> res = milp(c=c, constraints=constraints, integrality=integrality)
    >>> res.x
    [1.0, 2.0]

    Note that had we solved the relaxed problem (without integrality
    constraints):

    >>> res = milp(c=c, constraints=constraints)  # OR:
    >>> # from scipy.optimize import linprog; res = linprog(c, A, b_u)
    >>> res.x
    [1.8, 2.8]

    we would not have obtained the correct solution by rounding to the nearest
    integers.

    Other examples are given :ref:`in the tutorial <tutorial-optimize_milp>`.

    """
    args_iv = _milp_iv(c, integrality, bounds, constraints, options)
    c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options = args_iv

    highs_res = _highs_wrapper(c, indptr, indices, data, b_l, b_u,
                               lb, ub, integrality, options)

    res = {}

    # Convert to scipy-style status and message
    highs_status = highs_res.get('status', None)
    highs_message = highs_res.get('message', None)
    status, message = _highs_to_scipy_status_message(highs_status,
                                                     highs_message)
    res['status'] = status
    res['message'] = message
    res['success'] = (status == 0)
    x = highs_res.get('x', None)
    res['x'] = np.array(x) if x is not None else None
    res['fun'] = highs_res.get('fun', None)
    res['mip_node_count'] = highs_res.get('mip_node_count', None)
    res['mip_dual_bound'] = highs_res.get('mip_dual_bound', None)
    res['mip_gap'] = highs_res.get('mip_gap', None)

    return OptimizeResult(res)

class EMCoords(object):
    def __init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose):
        """
        Perform persistent homology on the landmarks, store distance
        from the landmarks to themselves and to the rest of the points,
        and sort persistence diagrams and cocycles in decreasing order of persistence

        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions, or a matrix of distances from N points to d points.
            See distance_matrix, below, for a description of the second scenario.
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud.
            If X is square, then the i-th row should represent the same point as the i-th column, meaning
            that the matrix should be symmetric.
            If X is not square, then it should have more columns than rows (i.e., N > d).
            Moreover, if i < N, the i-th row should represent the same point as the i-th column.
            When X is not square, the rows of X are interpreted as a subsample and the columns as all available points; thus
            X represents the distance from the points in the subsample to all available points.
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology. 

        """
        assert maxdim >= 1
        self.verbose = verbose
        self.X_ = X
        if verbose:
            tic = time.time()
            print("Doing TDA...")
        if distance_matrix is False:
            ripser_metric_input = X 
        elif X.shape[0] == X.shape[1]:
            ripser_metric_input = X 
        else:
            ripser_metric_input = X[:,:X.shape[0]]
        if np.sum(np.isinf(ripser_metric_input)>0):
            thresh = np.max(ripser_metric_input[~np.isinf(ripser_metric_input)])    
        else:
            thresh = np.percentile(ripser_metric_input.flatten(), 60)
        res = ripser(
            ripser_metric_input,
            distance_matrix=distance_matrix,
            coeff=prime,
            maxdim=maxdim,
            n_perm=n_landmarks,
            do_cocycles=True,
            thresh = thresh
        )
        if verbose:
            print("Elapsed time persistence: %.3g seconds" % (time.time() - tic))
        self.prime_ = prime
        self.dgms_ = res["dgms"]
        self.idx_land_ = res["idx_perm"]
        self.n_landmarks_ = len(self.idx_land_)
        #self.dist_land_data_ = res["dperm2all"]
        if distance_matrix is False:
            self.dist_land_data_ = res["dperm2all"]
        elif X.shape[0] == X.shape[1]:
            self.dist_land_data_ = res["dperm2all"]
        else:
            self.dist_land_data_ = X[self.idx_land_,:]
        self.coverage_ = np.max(np.min(self.dist_land_data_, 1))
        self.dist_land_land_ = self.dist_land_data_[:, self.idx_land_]
        self.cocycles_ = res["cocycles"]
        # Sort persistence diagrams in descending order of persistence
        for i in range(1, maxdim+1):
            idxs = np.argsort(self.dgms_[i][:, 0] - self.dgms_[i][:, 1])
            self.dgms_[i] = self.dgms_[i][idxs, :]
            dgm_lifetime = np.array(self.dgms_[i])
            dgm_lifetime[:, 1] -= dgm_lifetime[:, 0]
            self.cocycles_[i] = [self.cocycles_[i][idx] for idx in idxs]
        CohomologyUtils.reindex_cocycles(self.cocycles_, self.idx_land_, X.shape[0])

        self.type_ = "emcoords"

    def get_representative_cocycle(self, cohomology_class, homological_dimension):
        """
        Compute the representative cocycle, given a list of cohomology classes

        Parameters
        ----------
        cohomology_class : integer
            Integer representing the index of the persistent cohomology class.
            Persistent cohomology classes are ordered by persistence, from largest to smallest.

        Returns
        -------
        cohomdeath: float
            Cohomological death of the linear combination or single cocycle
        cohombirth: float
            Cohomological birth of the linear combination or single cocycle
        cocycle: ndarray(K, homological_dimension+2, dtype=int)
            Representative cocycle. First homological_dimension+1 columns are vertex indices,
            and last column takes values in finite field corresponding to self.prime_
        """

        assert isinstance(cohomology_class, int)

        dgm = self.dgms_[homological_dimension]
        cocycles = self.cocycles_[homological_dimension]
        return (
            dgm[cohomology_class, 0],
            dgm[cohomology_class, 1],
            cocycles[cohomology_class],
        )

    def get_cover_radius(self, perc, cohomdeath_rips, cohombirth_rips, standard_range):
        """
        Determine radius for covering balls

        Parameters
        ----------
        perc : float
            Percent coverage
        cohomdeath: float
            Cohomological death
        cohombirth: float
            Cohomological birth
        standard_range: float
            Whether or not to use the range that guarantees that the cohomology class selected
            is a non-trivial cohomology class in the Cech complex. If False, the class is only
            guaranteed to be non-trivial in the Rips complex.

        Returns
        -------
        r_cover : float
        rips_threshold : float

        """
        start = 2*cohomdeath_rips if standard_range else cohomdeath_rips
        end = cohombirth_rips
        if start > end:
            raise Exception(
                "The cohomology class selected is too short, try setting standard_range to False."
            )
        self.rips_threshold_ = (1 - perc) * start + perc * end
        self.r_cover_ = self.rips_threshold_ / 2

        return self.r_cover_, self.rips_threshold_

    def get_covering_partition(self, r_cover, partunity_fn):
        """
        Create the open covering U = {U_1,..., U_{s+1}} and partition of unity

        Parameters
        ----------
        r_cover: float
            Covering radius
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function

        Returns
        -------
        varphi: ndarray(n_data, dtype=float)
            varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b)),
        ball_indx: ndarray(n_data, dtype=int)
            The index of the first open set each data point belongs to
        """
        dist_land_data = self.dist_land_data_
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        # Compute the partition of unity
        # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are {} point not covered by a landmark".format(nzero))
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]
        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)
        return varphi, ball_indx

class ComplexProjectiveCoords(EMCoords):
    """
    Object that performs multiscale complex projective coordinates via
    persistent cohomology of sparse filtrations (Perea 2018).


    Parameters
    ----------
    X: ndarray(N, d)
        A point cloud with N points in d dimensions
    n_landmarks: int
        Number of landmarks to use
    distance_matrix: boolean
        If true, treat X as a distance matrix instead of a point cloud
    prime : int
        Field coefficient with which to compute rips on landmarks
    maxdim : int
        Maximum dimension of homology.  Only dimension 2 is needed for complex projective coordinates,
        but it may be of interest to see other dimensions
    """

    def __init__(
        self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=2, verbose=False
    ):
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        simplicial_complex_dimension = 3
        self.cns_lookup_table_ = combinatorial_number_system_table(
            n_landmarks, simplicial_complex_dimension
        )
        self.type_ = "complexprojective"

    def get_coordinates(
        self,
        perc=0.5,
        cocycle_idx=0,
        proj_dim=1,
        partunity_fn=PartUnity.linear,
        standard_range=True,
        check_cocycle_condition=True,
        projective_dim_red_mode="exponential",
    ):
        """
        Get complex projective coordinates.


        Parameters
        ----------
        perc : float
            Percent coverage. Must be between 0 and 1.
        cocycle_idx : integer
            Integer representing the index of the persistent cohomology class
            used to construct the Eilenberg-MacLane coordinate. Persistent cohomology
            classes are ordered by persistence, from largest to smallest.
        proj_dim : integer
            Complex dimension down to which to project the data.
        partunity_fn : (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        standard_range : bool
            Whether to use the parameter perc to choose a filtration parameter that guarantees
            that the selected cohomology class represents a non-trivial class in the Cech complex.
        check_cocycle_condition : bool
            Whether to check, and fix if necessary, that the integer cocycle constructed
            using finite field coefficients satisfies the cocycle condition.
        projective_dim_red_mode : string
            Either "one-by-one", "exponential", or "direct". How to perform equivariant
            dimensionality reduction. "exponential" usually works best, being fast
            without compromising quality.

        Returns
        -------
        ndarray(N, proj_dim+1)
            Complex projective coordinates
        """

        homological_dimension = 2
        cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(
            cocycle_idx, homological_dimension
        )

        # determine radius for balls
        r_cover, rips_threshold = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        # compute partition of unity and choose a cover element for each data point
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        # compute boundary matrix
        delta1 = CohomologyUtils.make_delta1(
            self.dist_land_land_, rips_threshold, self.cns_lookup_table_
        )

        # lift to integer cocycles
        integer_cocycle = CohomologyUtils.lift_to_integer_cocycle(
            cocycle, prime=self.prime_
        )

        # go from sparse to dense representation of cocycles
        integer_cocycle_as_vector = CohomologyUtils.sparse_cocycle_to_vector(
            integer_cocycle, self.cns_lookup_table_, self.n_landmarks_, int
        )

        if check_cocycle_condition:
            is_cocycle = _is_two_cocycle(
                integer_cocycle_as_vector,
                self.dist_land_land_,
                rips_threshold,
                self.cns_lookup_table_,
            )
            if not is_cocycle:
                delta2 = CohomologyUtils.make_delta2_compact(
                    self.dist_land_land_, rips_threshold, self.cns_lookup_table_
                )
                d2cocycle = delta2 @ integer_cocycle_as_vector.T

                y = d2cocycle // self.prime_

                constraints = LinearConstraint(delta2, y, y, keep_feasible=True)
                n_edges = delta2.shape[1]
                objective = np.zeros((n_edges), dtype=int)
                integrality = np.ones((n_edges), dtype=int)
                optimizer_solution = milp(
                    objective,
                    integrality=integrality,
                    constraints=constraints,
                )

                if not optimizer_solution["success"]:
                    raise Exception(
                        "The cohomology class at index "
                        + str(cocycle_idx)
                        + " does not have an integral lift."
                    )
                else:
                    solution = optimizer_solution["x"]
                    new_cocycle_as_vector = (
                        integer_cocycle_as_vector
                        - self.prime_ * np.array(np.rint(solution), dtype=int)
                    )
                    integer_cocycle_as_vector = new_cocycle_as_vector

        # compute harmonic representatives of cocycles and their projective-valued integrals
        integral = lsqr(delta1, integer_cocycle_as_vector)[0]
        harmonic_representative = integer_cocycle_as_vector - delta1 @ integral

        # compute complex projective coordinates on data points
        class_map = _sparse_integrate(
            harmonic_representative,
            integral,
            varphi,
            ball_indx,
            self.dist_land_land_,
            rips_threshold,
            self.cns_lookup_table_,
        )

        # reduce dimensionality of complex projective space
        epca = EquivariantPCA.ppca(
            class_map,
            proj_dim,
            projective_dim_red_mode,
            self.verbose,
        )
        self.variance_ = epca["variance"]

        return epca["X"]


# turn cocycle into tensor
def _two_cocycle_to_tensor(
    cocycle: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float,
    lookup_table: np.ndarray,
):
    n_points = dist_mat.shape[0]

    res = np.zeros((n_points, n_points, n_points))

    @jit(fastmath=True, nopython=True)
    def _get_res(
        cocycle: np.ndarray,
        dist_mat: np.ndarray,
        threshold: float,
        lookup_table: np.ndarray,
        n_points: int,
        res: np.ndarray,
    ):
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_mat[i, j] < threshold:
                    for k in range(j + 1, n_points):
                        if dist_mat[i, k] < threshold and dist_mat[j, k] < threshold:
                            flat_index = combinatorial_number_system_d2_forward(
                                i, j, k, lookup_table
                            )
                            val = cocycle[flat_index]
                            # 012
                            res[i, j, k] = val
                            # 021
                            res[i, k, j] = -val
                            # 102
                            res[j, i, k] = -val
                            # 210
                            res[k, j, i] = -val
                            # 201
                            res[k, i, j] = val
                            # 120
                            res[j, k, i] = val

    _get_res(cocycle, dist_mat, threshold, lookup_table, n_points, res)

    return res


def _one_cocycle_to_tensor(
    cocycle: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float,
    lookup_table: np.ndarray,
):
    n_points = dist_mat.shape[0]

    res = np.zeros((n_points, n_points))

    @jit(fastmath=True, nopython=True)
    def _get_res(
        cocycle: np.ndarray,
        dist_mat: np.ndarray,
        threshold: float,
        lookup_table: np.ndarray,
        n_points: int,
        res: np.ndarray,
    ):
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_mat[i, j] < threshold:
                    flat_index = combinatorial_number_system_d1_forward(
                        i, j, lookup_table
                    )
                    val = cocycle[flat_index]
                    res[i, j] = val
                    res[j, i] = -val

    _get_res(cocycle, dist_mat, threshold, lookup_table, n_points, res)

    return res


def _sparse_integrate(
    harm_rep,
    integral,
    part_unity,
    membership_function,
    dist_mat,
    threshold,
    lookup_table,
):
    nu = _one_cocycle_to_tensor(integral, dist_mat, threshold, lookup_table)

    eta = _two_cocycle_to_tensor(
        harm_rep,
        dist_mat,
        threshold,
        lookup_table,
    )

    class_map0 = np.zeros_like(part_unity.T)

    @jit(nopython=True)
    def _assemble(
        class_map: np.ndarray,
        nu: np.ndarray,
        eta: np.ndarray,
        varphi: np.ndarray,
        n_landmarks: int,
        n_data: int,
        ball_indx: np.ndarray,
    ):
        for b in range(n_data):
            for i in range(n_landmarks):
                class_map[b, i] += nu[i, ball_indx[b]]
                for t in range(n_landmarks):
                    class_map[b, i] += varphi[t, b] * eta[i, ball_indx[b], t]
        return np.exp(2 * np.pi * 1j * class_map0) * np.sqrt(varphi.T)

    return _assemble(
        class_map0,
        nu,
        eta,
        part_unity,
        dist_mat.shape[0],
        part_unity.shape[1],
        membership_function,
    )


@jit(fastmath=True, nopython=True)
def _is_two_cocycle(
    cochain: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float,
    lookup_table: np.ndarray,
):
    is_cocycle = True
    n_points = dist_mat.shape[0]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dist_mat[i, j] < threshold:
                for k in range(j + 1, n_points):
                    if dist_mat[i, k] < threshold and dist_mat[j, k] < threshold:
                        for l in range(k + 1, n_points):
                            if (
                                dist_mat[i, l] < threshold
                                and dist_mat[j, l] < threshold
                                and dist_mat[k, l] < threshold
                            ):
                                index_ijk = combinatorial_number_system_d2_forward(
                                    i, j, k, lookup_table
                                )
                                index_ijl = combinatorial_number_system_d2_forward(
                                    i, j, l, lookup_table
                                )
                                index_ikl = combinatorial_number_system_d2_forward(
                                    i, k, l, lookup_table
                                )
                                index_jkl = combinatorial_number_system_d2_forward(
                                    j, k, l, lookup_table
                                )

                                if (
                                    cochain[index_ijk]
                                    - cochain[index_ijl]
                                    + cochain[index_ikl]
                                    - cochain[index_jkl]
                                    != 0
                                ):
                                    is_cocycle = False
    return is_cocycle




############################ END SPHERECOORDS ###################################

def plot_curr(data, cols, r1 = 110, r2 = 40):
    fig = plt.figure(figsize = (10,10), dpi = 120)
    ax = fig.add_subplot(121, projection = '3d', computed_zorder=False)
    ax.scatter(data[:,0], 
               data[:,1], 
               data[:,2],
               c = cols[0,:],
               vmin = -0.,vmax = 1,
               s = 5, alpha = 0.9,
              cmap =cm.hsv, zorder = -2,
              edgecolors = None, depthshade=True)
    ax.view_init(r1,r2)
    ax.axis('off')

    ax = fig.add_subplot(122, projection = '3d', computed_zorder=False)
    ax.scatter(data[:,0], 
               data[:,1], 
               data[:,2],
               c = cols[1,:],
               vmin = -0.,vmax = 1,
               s = 5, alpha = 0.9,
              cmap =cm.hsv, zorder = -2,
              edgecolors = None, depthshade=True)
    ax.view_init(r1,r2)
    ax.axis('off')
    plt.show()
