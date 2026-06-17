"""
Generate 10 random trajectories in a 1-hole world, simulate grid cell activity,
and compute toroidal coordinates for each.

For each trajectory index i, three files are saved to DATA_PATH/revision_cache/:
  - trace_{i}.npy        : processed trajectory, shape (T, 2)
  - grid_activity_{i}.npy: simulated grid cell activity, shape (n_cells, T)
  - toroidal_coords_{i}.npy: toroidal coordinates, shape (T, 2)
"""

import os
import sys
import numpy as np
import concurrent.futures as cf

sys.path.append(".")
from trajectory import random_walk, World
from grid_cell_simulation import preprocess, simulation
from pipelines import recover_traj_pipeline
from ROOT_PATH import DATA_ROOT


DATA_PATH = os.path.join(DATA_ROOT, "revision_cache")
print(f"Data will be saved to: {DATA_PATH}")

WORLD_SIZE = (100, 100)
HOLES = [(30, 30, 70, 70)]  # 1-hole world
STEP_SIZE = 5
TRAJ_LENGTH = 25000
N_TRAJECTORIES = 5


def _run_single(traj_idx: int, save_dir: str) -> int:
    """Generate trajectory, simulate, compute toroidal coords, and save."""
    world = World(WORLD_SIZE, HOLES, step_size=STEP_SIZE)
    trace = random_walk(TRAJ_LENGTH, world, save=False, no_warnings=True)

    processed_trace = preprocess(trace, STEP_SIZE)
    grid_activity = simulation(processed_trace)
    tor_coord, _ = recover_traj_pipeline(grid_activity)

    np.save(os.path.join(save_dir, f"trace_{traj_idx}.npy"), processed_trace)
    np.save(os.path.join(save_dir, f"grid_activity_{traj_idx}.npy"), grid_activity)
    np.save(os.path.join(save_dir, f"toroidal_coords_{traj_idx}.npy"), tor_coord)

    print(f"  traj={traj_idx}: trace={processed_trace.shape}, "
          f"activity={grid_activity.shape}, toroidal={tor_coord.shape}")
    return traj_idx


def main(save_dir: str = DATA_PATH, n_workers: int = 10):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")
    print(f"Launching {N_TRAJECTORIES} jobs (1-hole world) with {n_workers} workers...")

    with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_single, i, save_dir): i
            for i in range(N_TRAJECTORIES)
        }
        for future in cf.as_completed(futures):
            future.result()

    print(f"\nDone. {N_TRAJECTORIES} trajectories saved to {save_dir}")


if __name__ == "__main__":
    main()
