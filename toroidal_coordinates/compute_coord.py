import os, sys
import numpy as np
import matplotlib.pyplot as plt
import ripser # type: ignore
import pickle
import math
import matplotlib.ticker as mticker
import matplotlib as mpl

sys.path.append(".")
from trajectory import World
from toroidal_coordinates.Gardner_toroidal import Gardner_coord
from constants import GRID_FIELDS_PATH, TRAJ_PATH


def interp_arr(arr: np.ndarray, new_len: int) -> np.ndarray:
    """
    Interpolates an array to a new length.

    Args:
        arr (np.ndarray): The input array to be interpolated.
        new_len (int): The desired length of the interpolated array.

    Returns:
        np.ndarray: The interpolated array with the new length.
    """
    new_indices = np.linspace(0, len(arr) - 1, new_len)
    return np.interp(new_indices, np.arange(len(arr)), arr)


def compute_grid_coord_simulated(num_holes: int) -> None:
    """
    Compute the toroidal coordinates for simulated grid data.

    Args:
        num_holes (int): The number of holes in the grid.
    """
    print(f"Computing toroidal coordinates for {num_holes}-hole world grid fields with Gardner...")
    grid_path = os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes")
    traj_path = os.path.join(TRAJ_PATH, f"random_walk_{num_holes}holes.pkl")

    grid_rates = pickle.load(
        open(os.path.join(grid_path, "simulation_result.pkl"), "rb")
    ).T

    traj = pickle.load(open(traj_path, "rb"))[0]
    xx, yy = traj[:, 0], traj[:, 1]

    xx = interp_arr(xx, grid_rates.shape[0])
    yy = interp_arr(yy, grid_rates.shape[0])
    coords, times, centcosall, centsinall = Gardner_coord(grid_rates)
    visualize_tor_coords_x_traj(xx, yy, coords, times)
    np.savez_compressed(
        os.path.join(grid_path, "toroidal_coords.npz"),
        coords=coords,
        times=times,
        traj=np.concatenate([xx[:, None], yy[:, None]], axis=1),
        centcosall=centcosall,
        centsinall=centsinall,
    )
    plt.savefig(os.path.join(grid_path, "toroidal_coords.png"), dpi=150)

def visualize_tor_coords_x_traj(xx: np.ndarray, yy: np.ndarray, coords:np.ndarray, times: np.ndarray, cos_2pi: bool = True) -> None:
    """
    Visualizes toroidal coordinates.

    Args:
        xx (np.ndarray): X-coordinates of simulated trajectory.
        yy (np.ndarray): Y-coordinates of simulated trajectory.
        coords (np.ndarray): Array of toroidal coordinates.
        times (np.ndarray): Array of time values.
    """
    print(coords.shape)
    if len(times) > 1e4:
        plot_times = np.arange(0, len(times), 10)
    else:
        plot_times = times
    xx = interp_arr(xx, coords.shape[0])
    yy = interp_arr(yy, coords.shape[0])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    if cos_2pi:
        coords_color = [np.cos(coords[plot_times, 0]), np.cos(coords[plot_times, 1])]
    else:
        coords_color = [np.cos(coords[plot_times, 0] / 2), np.cos(coords[plot_times, 1] / 2)]

    sc0 = ax[0].scatter(xx[times][plot_times], yy[times][plot_times], c=coords_color[0], cmap='viridis')
    ax[0].axis('off')
    ax[0].set_aspect('equal', 'box')

    sc1 = ax[1].scatter(xx[times][plot_times], yy[times][plot_times], c=coords_color[1], cmap='viridis')
    ax[1].axis('off')
    ax[1].set_aspect('equal', 'box')
    
    pos = ax[1].get_position()
    
    # Compute colorbar dimensions:
    # Width: 1/3 of the scatter plot width.
    width_ax = pos.width
    width_cbar = width_ax * (1/10)
    # Height: 80% of scatter plot height.
    height_ax = pos.height
    height_cbar = height_ax * 0.95
    # Horizontal pad between ax[1] and colorbar.
    pad = 0.02  
    x_cbar = pos.x1 + pad
    y_cbar = pos.y0 + (height_ax - height_cbar) / 2

    # Add a new axis for the colorbar at the computed position.
    cax = fig.add_axes([x_cbar, y_cbar, width_cbar, height_cbar]) # type: ignore
    fig.colorbar(sc0, cax=cax, ticks=[-1, 0, 1])
    
def plot_toroidal_coords(tor_coords: np.ndarray) -> None:
    """
    Plot the trajectory based on the toroidal coordinates.
    
    Args:
        tor_coords (np.ndarray): Array of toroidal coordinates.
    """
    plt.figure()
    plt.plot(tor_coords[:, 0], tor_coords[:, 1])

def plot_toroidal_coordinates(xx: np.ndarray, yy: np.ndarray, coords:np.ndarray, times: np.ndarray, 
                              subsample_times = False,
                              aspect_equal = True,
                              **kwargs) -> None:
    """
    Visualizes toroidal coordinates. 
    Differs from the previous plotting functions by (1) uses "plasma" colormap and (2) uses the original coordinates instead of cosine values.

    Args:
        xx (np.ndarray): X-coordinates of simulated trajectory.
        yy (np.ndarray): Y-coordinates of simulated trajectory.
        coords (np.ndarray): Array of toroidal coordinates.
        times (np.ndarray): Array of time values.
        aspect_equal (bool): Whether to set the aspect ratio to be equal.
    """
    if subsample_times == True:
        plot_times = np.arange(0, len(times), 10)
    else:               
        plot_times = times

    xx = interp_arr(xx, coords.shape[0])
    yy = interp_arr(yy, coords.shape[0])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    coords_color = [coords[times[plot_times], 0], coords[times[plot_times],1]]

    sc0 = ax[0].scatter(xx[times][plot_times], yy[times][plot_times], c=coords_color[0], cmap='plasma', **kwargs)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    if aspect_equal == True:
        ax[0].set_aspect('equal')

    sc1 = ax[1].scatter(xx[times][plot_times], yy[times][plot_times], c=coords_color[1], cmap='plasma', **kwargs)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    if aspect_equal == True:
        ax[1].set_aspect('equal')
    
    pos = ax[1].get_position()
    
    # Compute colorbar dimensions:
    # Width: 1/3 of the scatter plot width.
    width_ax = pos.width
    width_cbar = width_ax * (1/10)
    # Height: 80% of scatter plot height.
    height_ax = pos.height
    height_cbar = height_ax * 0.95
    # Horizontal pad between ax[1] and colorbar.
    pad = 0.02  
    x_cbar = pos.x1 + pad
    y_cbar = pos.y0 + (height_ax - height_cbar) / 2

    # Add a new axis for the colorbar at the computed position.
    cax = fig.add_axes([x_cbar, y_cbar, width_cbar, height_cbar]) # type: ignore
   
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm = mpl.colors.Normalize(vmin = 0, vmax = 2 * math.pi), cmap = "plasma"),  # type: ignore
                        cax=cax, ticks=np.arange(0, 2*math.pi+0.001, math.pi), format=mticker.FixedFormatter(['0', r'$\pi$', r'$2\pi$']))
    
    cbar.set_ticks(np.arange(0, 2*math.pi+0.001, math.pi)) # type: ignore
    cbar.ax.tick_params(labelsize=25)

if __name__ == "__main__":
    for i in range(3):
        compute_grid_coord_simulated(i)
    
        with np.load(os.path.join(GRID_FIELDS_PATH, f"world_{i}holes/toroidal_coords.npz")) as data:
            tor_coords = data["coords"]
        plot_toroidal_coords(tor_coords)
    plt.show()
