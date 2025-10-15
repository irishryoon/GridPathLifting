"""
This module provides an all-in-one pipeline for recovering the trajectory of a grid cell activity.
"""

import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from trajectory import World
from toroidal_coordinates.Gardner_toroidal import Gardner_coord
from toroidal_coordinates.dreimac_toroidal import compute_toroidal_coords_dreimac
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance_upgrade
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from affine_transform.score_mismatch import score_mismatch

    
def recover_traj_pipeline(grid_activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Recovers trajectory coordinates from grid cell activity using toroidal coordinate transformation and lifting.
    Args:
        grid_activity (np.ndarray): Array representing grid cell activity over time. Activity should be in the shape (n_cells, n_time_steps).
    Returns:
        tor_coord (np.ndarray): The toroidal coordinates recovered from the grid activity.
        lifted_coord (np.ndarray): The lifted coordinates obtained by applying a distance upgrade to the toroidal coordinates.
    """
    try:
        tor_coord, times = compute_toroidal_coords_dreimac(grid_activity.T)
    except:
        print("DREiMac failed")
        tor_coord, times, _, _ = Gardner_coord(grid_activity.T)
    lifted_coord = toroidal_lifting_distance_upgrade(tor_coord)
    
    return tor_coord, lifted_coord

def plot_recovered_trajectory(lifted_coord: np.ndarray, save_path: str | None = None):
    """
    Plots the recovered trajectory coordinates showing the lifted coordinates.
    
    Args:
        lifted_coord (np.ndarray): The lifted coordinates in the shape (n_time_steps, 2).
        save_path (str | None, optional): Path to save the plot. If None, the plot will be displayed.
    """
    plt.figure(figsize=(5, 5))
    
    # Plot lifted coordinates
    plt.scatter(lifted_coord[:, 0], lifted_coord[:, 1], c=np.arange(len(lifted_coord)), 
               cmap='viridis', s=1, alpha=0.7)
    plt.title('Lifted Coordinates')
    plt.xlabel('Lifted X')
    plt.ylabel('Lifted Y')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def compare_lifted_pipeline(
    lifted_coord: np.ndarray,
    original_coord: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compares lifted coordinates to original coordinates by applying a transformation and scoring the mismatch.
    Args:
        lifted_coord (np.ndarray): The coordinates to be transformed and compared. Should be in the shape (n1_time_steps, 2).
        original_coord (np.ndarray): The reference coordinates for comparison. Should be in the shape (n2_time_steps, 2).
    Returns:
        lifted_coord_transformed (np.ndarray): The transformed lifted coordinates.
        score (float): The mismatch score between the transformed lifted coordinates and the original coordinates.
    """
    transform_mat = get_transform_mat(lifted_coord, original_coord)
    lifted_coord_transformed = apply_transform_mat(lifted_coord, transform_mat)

    score = score_mismatch(lifted_coord_transformed, original_coord)
    print(f"Reconstruction Error: {score:.4%}")
    
    return lifted_coord_transformed, score

def plot_trajectory_comparison(
    original_coord: np.ndarray,
    lifted_coord_transformed: np.ndarray,
    score: float,
    save_path: str | None = None
):
    """
    Plots a comparison between original and transformed lifted coordinates.
    
    Args:
        original_coord (np.ndarray): The original reference coordinates in the shape (n_time_steps, 2).
        lifted_coord_transformed (np.ndarray): The transformed lifted coordinates in the shape (n_time_steps, 2).
        score (float): The mismatch score between the trajectories.
        save_path (str | None, optional): Path to save the plot. If None, the plot will be displayed.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot original trajectory
    ax1.scatter(original_coord[:, 0], original_coord[:, 1], 
                c=np.arange(len(original_coord)), cmap='viridis', s=1, alpha=0.7)
    ax1.set_title('Original Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    
    # Plot transformed lifted trajectory
    ax2.scatter(lifted_coord_transformed[:, 0], lifted_coord_transformed[:, 1], 
                c=np.arange(len(lifted_coord_transformed)), cmap='viridis', s=1, alpha=0.7)
    ax2.set_title('Transformed Lifted Trajectory')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    
    # Plot overlay comparison
    ax3.scatter(original_coord[:, 0], original_coord[:, 1], 
                c='#1a80bb', s=1, alpha=0.05, label='Original')
    ax3.scatter(lifted_coord_transformed[:, 0], lifted_coord_transformed[:, 1], 
                c='#ea801c', s=1, alpha=0.05, label='Transformed Lifted')
    ax3.set_title(f'Mismatch Score: {score:.6f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    from constants import GRID_FIELDS_PATH, TRAJ_PATH
    grid_activity = pickle.load(open(os.path.join(GRID_FIELDS_PATH, 'world_1holes', 'simulation_result.pkl'), 'rb')) # Grid cell activity
    print(f"Grid activity shape: {grid_activity.shape}")
    tor_coord, lifted_coord = recover_traj_pipeline(grid_activity)
    print(lifted_coord.shape)

    # Plot recovered trajectory
    plot_recovered_trajectory(lifted_coord)

    original_coord, _ = pickle.load(open(os.path.join(TRAJ_PATH, 'random_walk_1holes.pkl'), 'rb')) # Original trajectory
    lifted_coord_transformed, score = compare_lifted_pipeline(lifted_coord, original_coord)
    print(f'{score:}')
    
    # Plot trajectory comparison
    plot_trajectory_comparison(original_coord, lifted_coord_transformed, score)