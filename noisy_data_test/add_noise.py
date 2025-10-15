import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats # type: ignore
import concurrent.futures as cf

sys.path.append('.')
from constants import GRID_FIELDS_PATH

def gaussian(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Generate a Gaussian curve."""
    return stats.norm.pdf(x, loc=mean, scale=std)

def fast_gaussian_curve(std: float, noise_max: float = 0.08, cut_thrsh: float = 1e-4) -> np.ndarray:
    """
    Generate a normalized Gaussian curve centered and truncated at a threshold.

    Args:
        std: Standard deviation of the Gaussian.
        noise_max: Maximum amplitude of the curve. Default 0.08.
        cut_thrsh: Threshold for truncation. Default 1e-4.

    Returns:
        np.ndarray: Centered Gaussian curve normalized to noise_max.
    """
    x = np.linspace(0, 4*std)
    arr = stats.norm.pdf(x, scale=std)
    arr *= noise_max / np.max(arr)
    index = np.where(arr > cut_thrsh)[0].max()
    center = int(x[index])
    xx = np.arange(center*2)
    yy = stats.norm.pdf(xx, loc=center, scale=std)
    yy *= noise_max / np.max(yy)
    return yy

def add_noise_1d(data: np.ndarray, portion: float, s: float = 50, noise_max: float = 0.08) -> np.ndarray:
    """
    Add Gaussian noise to a portion of the data.
    
    Args:
        data (np.ndarray): The input data array to which noise will be added.
        portion (float): The portion of the data to which noise will be added, specified as a fraction (0 < portion <= 1).
        s (float, optional): The standard deviation of the Gaussian noise. Default is 50.
        noise_max (float, optional): The maximum amplitude of the noise to be added. Default is 0.08.
        
    Returns:
        np.ndarray: The data array with added noise.
    """
    max_thrsh = np.max(data)
    num_timepoints = int(data.shape[0] * portion)
    
    ind_noise = np.random.choice(data.shape[0], num_timepoints, replace=False)
    # noise = np.random.normal(0, s)
    
    arr = np.arange(data.shape[0])
    for i in ind_noise:
        curve = gaussian(arr, i, s)
        curve *= noise_max / np.max(curve)
        data += curve
    
    # print(noise)
    
    # data[ind_noise] += noise
    data[data > max_thrsh] = max_thrsh
    data[data < 0] = 0
    
    return data

def add_noise_1d_fast(data: np.ndarray, portion: float, s: float = 50, noise_max: float = 0.08) -> np.ndarray:
    """
    Add Gaussian noise to a 1D data array at random time points.
    
    Args:
        data: Input 1D array.
        portion: Fraction of points to add noise to (0.0-1.0).
        s: Standard deviation of Gaussian noise. Default 50.
        noise_max: Maximum noise amplitude. Default 0.08.
    
    Returns:
        np.ndarray: Noisy data clipped to [0, max(data)].
    """

    max_thrsh = np.max(data)
    num_timepoints = int(data.shape[0] * portion)
    
    ind_noise = np.random.choice(data.shape[0], num_timepoints, replace=False)
    noise = fast_gaussian_curve(s, noise_max)
    
    data_w_noise = data.copy()
    l = len(noise)//2
    for ind in ind_noise:
        if ind - l < 0:
            left = ind
            right = l
        elif ind + l > data.shape[0]:
            left = l
            right = data.shape[0] - ind
        else:
            left = l
            right = l
        data_w_noise[ind-left:ind+right] += noise[l-left:l+right]
    
    data_w_noise[data_w_noise > max_thrsh] = max_thrsh
    data_w_noise[data_w_noise < 0] = 0
    
    return data_w_noise

def add_noise_2d(data: np.ndarray, portion: float, s: float = 50, noise_max: float = 0.08) -> np.ndarray:
    """
    Add Gaussian noise to a 2D data array.
    
    Args:
        data (np.ndarray): Input 2D array (timepoints, features)
        portion (float): Fraction of timepoints to add noise to (0.0-1.0)
        s (float): Standard deviation of Gaussian noise. Default 50.
        noise_max (float): Maximum noise amplitude. Default 0.08.
        
    Returns:
        np.ndarray: Noisy data clipped to original range [0, max_per_column]
    """
    max_thrsh = np.max(data, axis=0)
    num_timepoints = int(data.shape[0] * portion)
    
    ind_noise = np.zeros((num_timepoints, data.shape[1]), dtype=int)
    for i in range(data.shape[1]):
        ind_noise[:, i] = np.random.choice(data.shape[0], num_timepoints, replace=False)
        
    noise_matrix = np.zeros_like(data)
    for i in range(noise_matrix.shape[1]):
        for ind in ind_noise[:, i]:
            curve = gaussian(np.arange(data.shape[0]), ind, s)
            curve *= noise_max / np.max(curve)
            noise_matrix[:, i] += curve
    
    data_w_noise = data + noise_matrix
    
    data_w_noise = np.clip(data_w_noise, 0, max_thrsh)
    
    return data_w_noise

class NoiseAdder:
    def __init__(self, portion: float, s: float = 50, noise_max: float = 0.08) -> None:
        self.portion = portion
        self.s = s
        self.noise_max = noise_max
    
    def add_noise_1d(self, data: np.ndarray) -> np.ndarray:
        return add_noise_1d(data, self.portion, self.s, self.noise_max).T
    
    def add_noise_1d_fast(self, data: np.ndarray) -> np.ndarray:
        return add_noise_1d_fast(data, self.portion, self.s, self.noise_max).T
    

def add_noise_2d_parallel(data: np.ndarray, portion: float, s: float = 50, noise_max: float = 0.08) -> np.ndarray:
    """
    Add Gaussian noise to 2D data in parallel.

    Args:
        data: Input 2D data array (timepoints x features).
        portion: Fraction of timepoints to add noise.
        s: Standard deviation of the Gaussian noise. Default 50.
        noise_max: Maximum amplitude of the noise. Default 0.08.

    Returns:
        np.ndarray: Modified data with added noise.
    """
    # Use ThreadPoolExecutor for parallel processing
    with cf.ProcessPoolExecutor(max_workers=25) as executor:
        results = list(executor.map(
            NoiseAdder(portion, s, noise_max).add_noise_1d_fast, data.T
        ))

    # Combine results back into 2D data
    noisy_data = np.array(results).T
    return noisy_data

if __name__ == "__main__":
    grid_rates = pickle.load(open(os.path.join(GRID_FIELDS_PATH, "world_0holes", "simulation_result.pkl"), "rb")).T
    print(grid_rates.shape)
    
    # grid_rates_noisy = add_noise_1d(grid_rates[:, 0].copy(), 0.01)
    grid_rates_noisy = add_noise_2d_parallel(grid_rates[:, :40].copy(), 0.01)
    # grid_rates_noisy = np.zeros_like(grid_rates)
    # for i in range(1):
    #     start = time.time()
    #     grid_rates_noisy[:, i] = add_noise_1d(grid_rates[:, i].copy(), 0.01)
    #     print("Time taken:", time.time() - start)
    plt.subplot(2, 1, 1)
    plt.plot(grid_rates[:10000, 0])
    plt.title("Original")
    plt.subplot(2, 1, 2)
    plt.plot(grid_rates_noisy[:10000, 0])
    plt.title("Noisy")
    plt.show()