import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import ripser # type: ignore

sys.path.append(".")
from trajectory import World
from noisy_data_test.add_noise import add_noise_2d_parallel
from persistence_homology import persistence_analysis_spikes
from toroidal_coordinates.dreimac_toroidal import compute_toroidal_coords_dreimac
from toroidal_coordinates.Gardner_toroidal import Gardner_coord, Gardner_persistence
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance_upgrade
from constants import GRID_FIELDS_PATH, TRAJ_PATH, DATA_ROOT, NOISY_DATA_FIG_PATH, NOISY_DATA_PATH

def compute_toroidal_coords_world_with_noise(num_holes: int = 1) -> None:
    grid_path = os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes")
    traj_path = os.path.join(TRAJ_PATH, f"random_walk_{num_holes}holes.pkl")
    
    noise_path = NOISY_DATA_PATH
    noise_fig_path = NOISY_DATA_FIG_PATH
    if not os.path.exists(noise_path):
        os.makedirs(noise_path)
        
    grid_rates = pickle.load(open(os.path.join(grid_path, "simulation_result.pkl"), "rb")).T
    print(grid_rates.shape)
    traj = pickle.load(open(traj_path, "rb"))[0]
    
    portion_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
    variance_list = [1, 10, 50, 100, 500]
    max_noise_list = [0.08]
    # portion_list = [0.001, 0.005, 0.01, 0.05, 0.1]
    # variance_list = [1, 10, 50, 100]
    # max_noise_list = [0.2, 0.3, 0.4]
    
    for max_noise in max_noise_list:
        for portion in portion_list:
            for variance in variance_list:
                name = f'_portion{portion}_variance{variance}{f"_maxnoise{max_noise}" if max_noise != 0.08 else ""}'
                print(f"-- portion={portion}, variance={variance}, max_noise={max_noise} --")
                start = time.time()
                noisy_grid_rates = add_noise_2d_parallel(grid_rates.copy(), portion, variance, max_noise)
                np.savetxt(os.path.join(noise_path, f"noisy_grid_rates{name}.csv"), noisy_grid_rates, delimiter=",")
                try:
                    coords, times = compute_toroidal_coords_dreimac(noisy_grid_rates)
                except:
                    try:
                        print("Dreimac failed")
                        coords, times, _, _ = Gardner_coord(noisy_grid_rates)
                    except:
                        print("Gardner failed")
                        persistence = persistence_analysis_spikes(noisy_grid_rates, maxdim=2)
                        np.savez_compressed(
                            os.path.join(noise_path, f"toroidal_coords{name}.npz"),
                            dgm0=persistence['dgms'][0],
                            dgm1=persistence['dgms'][1],
                            dgm2=persistence['dgms'][2]
                        )
                        continue
                lifted_coords = toroidal_lifting_distance_upgrade(coords.copy())
                print(f"Time: {time.time() - start:.2f}s")
                persistence = Gardner_persistence(noisy_grid_rates, maxdim=2)
                np.savez_compressed(
                    os.path.join(noise_path, f"toroidal_coords{name}.npz"),
                    dgm0=persistence['dgms'][0],
                    dgm1=persistence['dgms'][1],
                    dgm2=persistence['dgms'][2]
                )
                np.savetxt(os.path.join(noise_path, f"toroidal_coords{name}.csv"), lifted_coords, delimiter=",")
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(traj[:, 0], traj[:, 1])
                plt.title("Orginal Trajectory")
                plt.subplot(1, 2, 2)
                plt.plot(lifted_coords[:, 0], lifted_coords[:, 1])
                plt.title(f"Lifted Coords with Noise: Portion={portion}, Var={variance}, Max Noise={max_noise}")
                plt.savefig(os.path.join(noise_fig_path, f"toroidal_coords{name}.png"), dpi=150)
            
if __name__ == "__main__":
    compute_toroidal_coords_world_with_noise()