import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 # type: ignore
import pickle

sys.path.append(".")
from trajectory import World
from Gardner_data_test.Gardner_lifting_test import interpolate_2d_along_axis_0
from constants import GRID_FIELDS_PATH, TRAJ_PATH

def apply_transform_mat(path: np.ndarray, t_mat: np.ndarray) -> np.ndarray:
    """
    Applies the affine transformation matrix to the given path.
    
    Args:
        path (np.ndarray): The path to be transformed as a 2D numpy array of shape (n, 2).
        t_mat (np.ndarray): The affine transformation matrix.
        
    Returns:
        np.ndarray: The transformed path as a 2D numpy array of shape (n, 2).
    """
    new_path = np.zeros_like(path)
    for i in range(path.shape[0]):
        new_path[i] = np.dot(t_mat[0][:2, :2], path[i]) + t_mat[0][:2, 2]
    
    return new_path

def get_transform_mat(path1: np.ndarray, path2: np.ndarray) -> np.ndarray:
    """
    Computes the affine transformation matrix from path1 to path2.
    
    Args:
        path1 (np.ndarray): The reference path as a 2D numpy array of shape (n, 2).
        path2 (np.ndarray): The path to be transformed as a 2D numpy array of shape (m, 2).
        
    Returns:
        np.ndarray: The affine transformation matrix that aligns path2 to path1.
    """
    # make sure the two paths have the same shape
    if path1.shape != path2.shape:
        path2 = interpolate_2d_along_axis_0(path2, path1.shape[0])
    
    t_mat = cv2.estimateAffine2D(path1, path2)
    # print(t_mat)
    
    return t_mat

if __name__ == "__main__":
    num_holes = 2
    traj = pickle.load(open(os.path.join(TRAJ_PATH, f"random_walk_{num_holes}holes.pkl"), "rb"))[0]
    lifted = pickle.load(open(os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes/toroidal_coords_lifted.pkl"), "rb"))
    print(traj.shape, lifted.shape)
    matrix = get_transform_mat(lifted, traj)
    recovered_path = apply_transform_mat(lifted, matrix)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(traj[:, 0], traj[:, 1])
    plt.title("Original")
    plt.subplot(122)
    plt.plot(recovered_path[:, 0], recovered_path[:, 1])
    plt.title("Recovered")
    plt.show()
