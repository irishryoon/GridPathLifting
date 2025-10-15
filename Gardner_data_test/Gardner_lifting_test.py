import os, sys
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser  # type: ignore
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

sys.path.append('.')
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance_upgrade
from toroidal_coordinates.Gardner_toroidal import Gardner_coord
from toroidal_coordinates.dreimac_toroidal import compute_toroidal_coords_dreimac
from toroidal_coordinates.compute_coord import visualize_tor_coords_x_traj, interp_arr
from Gardner_data_test.Gardner_utils import get_spikes, fit_para, toroidal_alignment, plot_barcode, get_ratemaps
from persistence_homology import computeDistanceMatrix
from toroidal_coordinates.epsilon import compute_epsilon
from constants import GARDNER_DATA_PATH

def gaussian_smooth_toroidal(data, sigma, degrees=True):
    """
    Applies Gaussian smoothing to toroidal (circular) data.
    
    Args:
        data (numpy.ndarray): 1D array of angular values (in degrees or radians).
        sigma (float): Standard deviation for the Gaussian kernel.
        degrees (bool): If True, input data is in degrees; if False, data is in radians.
    
    Returns:
        numpy.ndarray: Smoothed data, with wrap-around handling.
    """
    # Convert to radians if data is in degrees
    if degrees:
        data = np.deg2rad(data)

    # Convert angular data to complex representation on the unit circle
    complex_data = np.exp(1j * data)

    # Wrap the complex data to handle edges
    pad_width = int(3 * sigma)  # Pad based on sigma for smoothness at edges
    padded_complex_data = np.concatenate([complex_data[-pad_width:], complex_data, complex_data[:pad_width]])

    # Apply Gaussian smoothing in real and imaginary components separately
    real_smoothed = gaussian_filter1d(padded_complex_data.real, sigma=sigma)
    imag_smoothed = gaussian_filter1d(padded_complex_data.imag, sigma=sigma)

    # Recombine smoothed real and imaginary parts and crop to original length
    smoothed_complex = (real_smoothed + 1j * imag_smoothed)[pad_width:-pad_width]

    # Convert back to angles
    smoothed_data = np.angle(smoothed_complex)

    # Convert back to degrees if needed
    if degrees:
        smoothed_data = np.rad2deg(smoothed_data)

    return smoothed_data

def interpolate_2d_along_axis_0(array_2d, new_length):
    """
    Interpolates a 2D array along the 0-axis to the desired new length.

    Parameters:
    - array_2d (numpy.ndarray): The original 2D array.
    - new_length (int): The desired number of rows in the interpolated array.

    Returns:
    - numpy.ndarray: A 2D array with the specified new length along the 0-axis.
    """
    # Get the original length along the 0-axis
    original_length = array_2d.shape[0]

    # Generate original and new x-coordinates along the 0-axis
    x_original = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)

    # Interpolate each column separately along the 0-axis
    interpolated_array = np.array([
        interp1d(x_original, array_2d[:, col], kind='linear')(x_new)
        for col in range(array_2d.shape[1])
    ]).T

    return interpolated_array

        
def gaussian_wind_fn(mu, sigma, x):
    '''Normalized Gaussian'''
    return np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def get_kernel_sum(spike_list, t_points, win_fun):
    '''Sum a bunch of kernels centered at each spike time. This can be slow, but 
    is only run once at the beginning so not optimizing.
    '''
    result = np.zeros_like(t_points)
    for spike in spike_list:
        result = result + win_fun(spike, t_points)
    return result


def get_rates_angles_kernel_dict(inp_data, params, interval):
    '''Convert the set of spike_times into rates using time_bins. Assume that spike_times
    is a dictionary with keys being the cells and values the spike times. 
    Params contains sigma, dt.'''

    if params['method'] == 'gaussian':
        wind_fn = lambda mu, x: gaussian_wind_fn(mu, params['sigma'], x)
    else:
        print ('Unknown windowing function')

    spike_times = inp_data
    samp_rate = 1
    dt = params['dt']

    bin_edges = np.arange(interval[0], interval[1], dt)
    time_vals = bin_edges[:-1] + (dt / 2.)
    cell_IDs = spike_times.keys()

    # For each cell, pull out the appropriate spike times and compute rates
    interval_spike_times = {}
    rates = {}

    for y in cell_IDs:
        print (y)
        interval_spike_times[y] = [x for x in spike_times[y]
                                   if interval[0] <= x < interval[1]]
        rates[y] = get_kernel_sum(interval_spike_times[y], time_vals, wind_fn)

    return time_vals, rates, time_vals


def Gardner_lifting_test():
    folder = GARDNER_DATA_PATH
    
    rat_name, mod_name, sess_name, day_name = ('R', '1', 'OF', 'day2')
    sspikes,xx,yy,__,__ = get_spikes(rat_name, mod_name, day_name, sess_name, bType = 'pure',
                                            bSmooth = True, bSpeed = True, folder = folder )
    
    coords, times, _, _ = Gardner_coord(sspikes)
    
    compute_epsilon(coords)
    plt.show()
    
    xx = interp_arr(xx, coords.shape[0])
    yy = interp_arr(yy, coords.shape[0])
    times = np.arange(xx.shape[0])
    visualize_tor_coords_x_traj(xx, yy, coords, times)
    
    # coordsbox, times_box, centcosall, centsinall = Gardner_coord(sspikes, xx, yy)
    print(coords.shape, times.shape, xx.shape, yy.shape)
    plt.show()
    # import pdb; pdb.set_trace()
    lifted_coords = toroidal_lifting_distance_upgrade(coords)
    
    lifted_coords[:, 0] = gaussian_filter1d(lifted_coords[:, 0], 10)
    lifted_coords[:, 1] = gaussian_filter1d(lifted_coords[:, 1], 10)
    
    lim = 10000
    
    colors = plt.cm.viridis(np.linspace(0, 1, min(xx.shape[0], lim)))
    
    plt.figure()
    plt.scatter(xx[:lim], yy[:lim], marker='.', color=colors)
    plt.figure()
    plt.scatter(lifted_coords[:lim, 0], lifted_coords[:lim, 1], marker='.', color=colors)
    plt.show()

if __name__ == '__main__':
    Gardner_lifting_test()