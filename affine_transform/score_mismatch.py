import os, sys
import numpy as np
import pickle
from scipy.spatial.distance import cdist # type: ignore

sys.path.append('.')
from trajectory import World
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from Gardner_data_test.Gardner_lifting_test import interpolate_2d_along_axis_0
from constants import TRAJ_PATH, GRID_FIELDS_PATH


def score_mismatch(path1: np.ndarray, path2: np.ndarray, world_size: float | None = None) -> float:
    """
    Computes the mismatch score between two paths using Euclidean distances.
    
    Args:
        path1 (np.ndarray): The first path (lifted) as a 2D numpy array.
        path2 (np.ndarray): The second path (original) as a 2D numpy array.
    Returns:
        float: The dismatch score between the two paths.
        
    Note:
        If the two paths do not have the same shape, path2 will be interpolated along axis 0 to match the shape of path1.
        The world_size parameter is used to normalize the score. If it is None, it will be computed based on the maximum distance in path2.
    """
    if path1.shape != path2.shape:
        
        path2 = interpolate_2d_along_axis_0(path2, path1.shape[0])
    
    if world_size is None:
        world_size = np.max([path2[:, 0].max() - path2[:, 0].min(), path2[:, 1].max() - path2[:, 1].min()])
    
    return np.mean(np.linalg.norm(path1 - path2, axis=1)) / world_size


def segmental_mismatch(
    lifted: np.ndarray,
    original: np.ndarray,
    len_seg: int,
) -> list[float]:
    """
    Computes mismatch scores between lifted and original paths segment by segment.

    Args:
        lifted: Lifted path as a 2D numpy array (N, 2).
        original: Original path as a 2D numpy array (N, 2).
        len_seg: Length of each segment.

    Returns:
        List of mismatch scores, one per segment.
    """
    num_seg = lifted.shape[0] // len_seg
    mismatch_scores = []
    for i in range(num_seg):
        original_seg = original[i * len_seg:(i + 1) * len_seg]
        lifted_seg = lifted[i * len_seg:(i + 1) * len_seg]
        if len(lifted_seg) < 4:
            continue
        transform_mat = get_transform_mat(lifted_seg, original_seg)
        recovered_lifted = apply_transform_mat(lifted_seg, transform_mat)
        mismatch_scores.append(score_mismatch(recovered_lifted, original_seg))
    return mismatch_scores


def compute_mean_transform_mat(all_original_segs: np.ndarray, all_lifted_segs: np.ndarray) -> np.ndarray:
    """
    Computes the mean transformation matrix from a set of original and lifted segments.
    """
    length = all_original_segs.shape[0]
    weight_sum = 0.
    all_trans_mats = np.zeros((length, 2, 3))
    for i in range(length):
        original_seg = all_original_segs[i]
        lifted_seg = all_lifted_segs[i]
        transform_mat = get_transform_mat(lifted_seg, original_seg)
        recovered_lifted = apply_transform_mat(lifted_seg, transform_mat)
        mismatch_score = score_mismatch(recovered_lifted, original_seg)
        all_trans_mats[i] = transform_mat[0] * (1/mismatch_score)
        weight_sum += 1 / mismatch_score
    mean_trans_mat = np.sum(all_trans_mats, axis=0) / weight_sum
    return mean_trans_mat
    
    

if __name__ == "__main__":
    num_holes = 2
    traj = pickle.load(open(os.path.join(TRAJ_PATH, f"random_walk_{num_holes}holes.pkl"), "rb"))[0]
    lifted = pickle.load(open(os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes/toroidal_coords_lifted.pkl"), "rb"))
    recovered_path = apply_transform_mat(lifted, get_transform_mat(lifted, traj))
    
    print(score_mismatch(traj, recovered_path))