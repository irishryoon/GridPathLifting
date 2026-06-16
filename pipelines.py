"""
This module provides an all-in-one pipeline for recovering the tranjectory of a grid cell activity.
"""

import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.append('.')
from trajectory import World
from toroidal_coordinates.Gardner_toroidal import Gardner_coord
from toroidal_coordinates.dreimac_toroidal import compute_toroidal_coords_dreimac
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance_upgrade
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from affine_transform.score_mismatch import score_mismatch, segmental_mismatch

    
def recover_traj_pipeline(grid_activity: np.ndarray, method: str = '') -> tuple[np.ndarray, np.ndarray]:
    """
    Recovers trajectory coordinates from grid cell activity using toroidal coordinate transformation and lifting.
    Args:
        grid_activity (np.ndarray): Array representing grid cell activity over time. Activity should be in the shape (n_cells, n_time_steps).
        method (str): The method to use for computing toroidal coordinates. Options are 'gardner' or 'dreimac'.
    Returns:
        tor_coord (np.ndarray): The toroidal coordinates recovered from the grid activity.
        lifted_coord (np.ndarray): The lifted coordinates obtained by applying a distance upgrade to the toroidal coordinates.
    """
    method = method.lower()
    if method == 'gardner':
        tor_coord, _, _, _ = Gardner_coord(grid_activity.T)
    else:
        try:
            tor_coord, _ = compute_toroidal_coords_dreimac(grid_activity.T)
        except Exception:
            print("dreimac failed, falling back to gardner")
            tor_coord, _, _, _ = Gardner_coord(grid_activity.T)
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
    original_coord: np.ndarray,
    method: str = 'RANSAC',
    world_size: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Compares lifted coordinates to original coordinates by applying a transformation and scoring the mismatch.
    Args:
        lifted_coord (np.ndarray): The coordinates to be transformed and compared. Should be in the shape (n1_time_steps, 2).
        original_coord (np.ndarray): The reference coordinates for comparison. Should be in the shape (n2_time_steps, 2).
        world_size (float | None): The world size used to normalize the mismatch score. If None, it is inferred from the bounding box of original_coord.
    Returns:
        lifted_coord_transformed (np.ndarray): The transformed lifted coordinates.
        score (float): The mismatch score between the transformed lifted coordinates and the original coordinates.
    """
    transform_mat = get_transform_mat(lifted_coord, original_coord, method=method)
    lifted_coord_transformed = apply_transform_mat(lifted_coord, transform_mat)

    score = score_mismatch(lifted_coord_transformed, original_coord, world_size=world_size)
    print(f"Mismatch score: {score}")
    
    return lifted_coord_transformed, score

def _align_1d(x_lift: np.ndarray, x_orig: np.ndarray) -> np.ndarray:
    """Scale + offset x_lift so its start and end match x_orig."""
    a = (x_orig[-1] - x_orig[0]) / (x_lift[-1] - x_lift[0])
    b = x_orig[0] - a * x_lift[0]
    return a * x_lift + b


def compare_lifted_1d_pipeline(
    lifted_coord: np.ndarray,
    original_1d: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compares a lifted coordinate sequence to a 1D reference path by aligning
    start and end points (scale + offset), without using cv2.

    Args:
        lifted_coord (np.ndarray): Lifted coordinates of shape (n_timesteps, 2).
        original_1d (np.ndarray): 1D reference of shape (n_timesteps,) or (n_timesteps, 2).

    Returns:
        lifted_transformed (np.ndarray): Aligned lifted coordinates of shape (n_timesteps, 2).
        score (float): Mismatch score.
    """
    if original_1d.ndim == 1:
        x_orig = original_1d
        original_coord = np.column_stack((x_orig, x_orig))
    else:
        x_orig = original_1d[:, 0]
        original_coord = original_1d

    lifted_x = _align_1d(lifted_coord[:, 0], x_orig)
    lifted_transformed = np.column_stack((lifted_x, lifted_x))

    score = score_mismatch(lifted_transformed, original_coord)
    print(f"Mismatch score: {score}")

    return lifted_transformed, score


def segmental_compare_1d_pipeline(
    lifted_coord: np.ndarray,
    original_1d: np.ndarray,
    len_seg: int,
) -> list[float]:
    """
    Compares lifted and original 1D coordinates segment by segment using
    start/end alignment (scale + offset) per segment, without using cv2.

    Args:
        lifted_coord (np.ndarray): Lifted coordinates of shape (n_timesteps, 2).
        original_1d (np.ndarray): 1D reference of shape (n_timesteps,) or (n_timesteps, 2).
        len_seg (int): Length of each segment.

    Returns:
        list[float]: Mismatch scores for each segment.
    """
    if original_1d.ndim == 1:
        x_orig = original_1d
        original_coord = np.column_stack((x_orig, x_orig))
    else:
        x_orig = original_1d[:, 0]
        original_coord = original_1d

    num_seg = lifted_coord.shape[0] // len_seg
    scores = []
    for i in range(num_seg):
        sl = slice(i * len_seg, (i + 1) * len_seg)
        orig_seg = original_coord[sl]
        lift_seg = lifted_coord[sl]
        if len(lift_seg) < 2:
            continue
        aligned_x = _align_1d(lift_seg[:, 0], orig_seg[:, 0])
        aligned_seg = np.column_stack((aligned_x, aligned_x))
        scores.append(score_mismatch(aligned_seg, orig_seg))

    print(f"Segmental 1D mismatch scores: mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
    return scores


def segmental_compare_pipeline(
    lifted_coord: np.ndarray,
    original_coord: np.ndarray,
    len_seg: int,
) -> list[float]:
    """
    Compares lifted and original coordinates segment by segment, returning a mismatch score per segment.
    Args:
        lifted_coord (np.ndarray): The lifted coordinates in the shape (n_time_steps, 2).
        original_coord (np.ndarray): The reference coordinates in the shape (n_time_steps, 2).
        len_seg (int): Length of each segment.
    Returns:
        list[float]: Mismatch scores for each segment.
    """
    scores = segmental_mismatch(lifted_coord, original_coord, len_seg)
    print(f"Segmental mismatch scores: mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
    return scores


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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Compute shared axis limits from both datasets
    all_coords = np.vstack([original_coord, lifted_coord_transformed])
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    xlim = (x_min - x_margin, x_max + x_margin)
    ylim = (y_min - y_margin, y_max + y_margin)

    # Plot original trajectory
    ax1.scatter(original_coord[:, 0], original_coord[:, 1],
                c=np.arange(len(original_coord)), cmap='viridis', s=1, alpha=0.7)
    ax1.set_title('Original Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Plot transformed lifted trajectory
    ax2.scatter(lifted_coord_transformed[:, 0], lifted_coord_transformed[:, 1],
                c=np.arange(len(lifted_coord_transformed)), cmap='viridis', s=1, alpha=0.7)
    ax2.set_title('Transformed Lifted Trajectory')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Plot overlay comparison
    ax3.scatter(original_coord[:, 0], original_coord[:, 1],
                c='#1a80bb', s=1, alpha=0.05)
    ax3.scatter(lifted_coord_transformed[:, 0], lifted_coord_transformed[:, 1],
                c='#ea801c', s=1, alpha=0.05)
    ax3.set_title(f'Mismatch Score: {score:.6f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.plot([], [], '-', color='#1a80bb', label='Original')
    ax3.plot([], [], '-', color='#ea801c', label='Transformed Lifted')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Apply shared limits and square aspect to all axes
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_box_aspect(1)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    from constants import GRID_FIELDS_PATH, TRAJ_PATH
    grid_activity = pickle.load(open(os.path.join(GRID_FIELDS_PATH, 'world_1holes', 'simulation_result.pkl'), 'rb'))
    print(f"Grid activity shape: {grid_activity.shape}")
    tor_coord, lifted_coord = recover_traj_pipeline(grid_activity)
    print(lifted_coord.shape)

    # Plot recovered trajectory
    plot_recovered_trajectory(lifted_coord)

    original_coord, _ = pickle.load(open(os.path.join(TRAJ_PATH, 'random_walk_1holes.pkl'), 'rb'))
    lifted_coord_transformed, score = compare_lifted_pipeline(lifted_coord, original_coord)
    print(f'{score:}')
    
    # Plot trajectory comparison
    plot_trajectory_comparison(original_coord, lifted_coord_transformed, score)