
import os
import sys
import pickle
import numpy as np
from grid_cell_simulation import preprocess, simulation
from pipelines import recover_traj_pipeline, compare_lifted_pipeline
from trajectory import World

from constants import TRAJ_PATH, GRID_FIELDS_PATH

data_dir = TRAJ_PATH
output_dir = GRID_FIELDS_PATH

os.makedirs(output_dir, exist_ok=True)

def process_file(fname):
    pkl_path = os.path.join(data_dir, fname)
    with open(pkl_path, "rb") as f:
        sim_result = pickle.load(f)
        trace = sim_result[0]

    processed_trace = preprocess(trace, goal=600000)
    grid_activity = simulation(processed_trace)
    if processed_trace.shape[0] - grid_activity.shape[1] == 1:
        processed_trace = processed_trace[:grid_activity.shape[1]]
    elif processed_trace.shape[0] != grid_activity.shape[1]:
        min_len = min(processed_trace.shape[0], grid_activity.shape[1])
        processed_trace = processed_trace[:min_len]
        grid_activity = grid_activity[:, :min_len]
    tor_coord_gardner, lifted_coord_gardner = recover_traj_pipeline(grid_activity, method="gardner")
    lifted_coord_transformed_gardner, score_gardner = compare_lifted_pipeline(lifted_coord_gardner, processed_trace)
    outbase = os.path.splitext(fname)[0]
    np.savez(os.path.join(output_dir, f"{outbase}_lifting_results.npz"),
             processed_trace=processed_trace,
             #grid_activity=grid_activity,
             #tor_coord_gardner=tor_coord_gardner,
             lifted_coord_gardner=lifted_coord_gardner,
             score_gardner=score_gardner)
    print(f"Processed and saved results for {fname}")



def simulated_grid_activity(fname):
    pkl_path = os.path.join(data_dir, fname)
    with open(pkl_path, "rb") as f:
        sim_result = pickle.load(f)
        trace = sim_result[0]

    processed_trace = preprocess(trace, goal=600000)
    grid_activity = simulation(processed_trace)
    if processed_trace.shape[0] - grid_activity.shape[1] == 1:
        processed_trace = processed_trace[:grid_activity.shape[1]]
    elif processed_trace.shape[0] != grid_activity.shape[1]:
        min_len = min(processed_trace.shape[0], grid_activity.shape[1])
        processed_trace = processed_trace[:min_len]
        grid_activity = grid_activity[:, :min_len]
    #tor_coord_gardner, lifted_coord_gardner = recover_traj_pipeline(grid_activity, method="gardner")
    #lifted_coord_transformed_gardner, score_gardner = compare_lifted_pipeline(lifted_coord_gardner, processed_trace)
    outbase = os.path.splitext(fname)[0]
    np.savez(os.path.join(output_dir, f"{outbase}_grid_activity.npz"),
             processed_trace=processed_trace,
             grid_activity=grid_activity,
             #tor_coord_gardner=tor_coord_gardner,
             #lifted_coord_gardner=lifted_coord_gardner,
             #score_gardner=score_gardner)
            )
    print(f"Processed and saved results for {fname}")

SIMULATED_GRID_ACTIVITY_FILES = [
    "random_walk_1holes_25000_n0.pkl",
    "random_walk_1holes_25000_n1.pkl",
    "random_walk_1holes_25000_n2.pkl",
    "random_walk_1holes_25000_n3.pkl",
    "random_walk_1holes_25000_n4.pkl",
]

if __name__ == "__main__":
    for fname in SIMULATED_GRID_ACTIVITY_FILES:
        simulated_grid_activity(fname)
