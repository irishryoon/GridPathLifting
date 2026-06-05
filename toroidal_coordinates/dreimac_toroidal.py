import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from dreimac import CircularCoords # type: ignore

sys.path.append(".")
from trajectory import World
from toroidal_coordinates.compute_coord import interp_arr, visualize_tor_coords_x_traj
from noisy_data_test.add_noise import add_noise_1d
from constants import GRID_FIELDS_PATH, TRAJ_PATH, GRID_FIG_PATH

def compute_toroidal_coords_dreimac(grid_rates: np.ndarray, standard_range: bool=True, distance_matrix: bool=False) -> tuple[np.ndarray, np.ndarray]:
    
    cc = CircularCoords(grid_rates, 250, prime=47, distance_matrix=distance_matrix)
    coords = np.concatenate([
        cc.get_coordinates(cocycle_idx=0, standard_range=standard_range)[:, np.newaxis], 
        cc.get_coordinates(cocycle_idx=1, standard_range=standard_range)[:, np.newaxis]
    ], axis=1)
    times = np.arange(grid_rates.shape[0])
    
    return coords, times
    
def compute_toroidal_coords_world(num_holes: int, noisy_level: float | None = None, noisy_variance: float = 50, cos_2pi: bool = True) -> None:
    """
    Compute the toroidal coordinates for simulated grid data using DREiMac package.

    Args:
        num_holes (int): The number of holes in the grid.
    """
    print(f"Computing toroidal coordinates for {num_holes}-hole world grid fields with DREiMac...")
    grid_path = os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes")
    grid_fig_path = os.path.join(GRID_FIG_PATH, f"world_{num_holes}holes")
    traj_path = os.path.join(TRAJ_PATH, f"random_walk_{num_holes}holes.pkl")
    
    grid_rates = pickle.load(open(os.path.join(grid_path, "simulation_result.pkl"), "rb")).T
    
    if noisy_level:
        grid_rates = add_noise_1d(grid_rates, noisy_level, noisy_variance)
    
    # plt.plot(grid_rates[:10000, 0])
    # plt.show()
    
    traj = pickle.load(open(traj_path, "rb"))[0]
    
    coords, times = compute_toroidal_coords_dreimac(grid_rates)
    
    visualize_tor_coords_x_traj(traj[:, 0], traj[:, 1], coords, times, cos_2pi=cos_2pi)

    ### Save
    filepath_npz = os.path.join(grid_path, f"toroidal_coords_dreimac{'_fullrange' if not cos_2pi else ''}{f'_{noisy_level}' if noisy_level else ''}.npz")
    np.savez_compressed(
        filepath_npz,
        coords=coords,
        times=times,
        traj=traj,
    )
    print("Saving toroidal coordinates in ", filepath_npz) 

#     filepath_fig = os.path.join(grid_fig_path, f"toroidal_coords_dreimac{'_fullrange' if not cos_2pi else ''}.png")
#     plt.savefig(filepath_fig, dpi=150)
#     print("Saving toroidal coordinates plots in ", filepath_fig)


if __name__ == "__main__":
    # compute_toroidal_coords_world(0)
    for i in range(3):
        compute_toroidal_coords_world(i, cos_2pi=True)
    
    plt.show()