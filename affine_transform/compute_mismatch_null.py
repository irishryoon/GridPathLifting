import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

sys.path.append('.')
from trajectory import World, random_walk
from affine_transform.score_mismatch import score_mismatch
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from constants import TRAJ_PATH, AFFINE_TRANSFORM_PATH, AFFINE_TRANSFORM_FIG_PATH

def compute_mismatch_null(num_holes: int) -> list[float]:
    """
    Compute mismatch scores between an original trajectory and randomly generated trajectories.
    This function loads an original trajectory with a specified number of holes, generates or loads
    100 random trajectories in the same world, applies affine transformations to align each random
    trajectory with the original, and computes mismatch scores for each alignment.
    Args:
        num_holes (int): The number of holes in the world/trajectory configuration.
    Returns:
        list[float]: A list of mismatch scores, one for each random trajectory compared to the
                     original trajectory after alignment.
    Note:
        - The original trajectory is loaded from TRAJ_PATH with filename 'random_walk_{num_holes}holes.pkl'
        - Random trajectories are cached in AFFINE_TRANSFORM_PATH with the same filename pattern
        - If cached random trajectories don't exist, 100 new trajectories of 25000 steps are generated
        - Each random trajectory is aligned to the original using affine transformation before scoring
    """
    original_traj, world = pickle.load(open(os.path.join(TRAJ_PATH, f'random_walk_{num_holes}holes.pkl'), "rb"))
    
    if os.path.exists(os.path.join(AFFINE_TRANSFORM_PATH, f'random_walk_{num_holes}holes.pkl')):
        random_traj = pickle.load(open(os.path.join(AFFINE_TRANSFORM_PATH, f'random_walk_{num_holes}holes.pkl'), 'rb'))
    else:
        random_traj = []
        num_traj = 100
        for i in range(num_traj):
            random_traj.append(random_walk(25000, world, save=False, no_warnings=True))
        pickle.dump(random_traj, open(os.path.join(AFFINE_TRANSFORM_PATH, f'random_walk_{num_holes}holes.pkl'), 'wb'))
    
    mismatch_scores = []
    for i in range(len(random_traj)):
        aligned_random = apply_transform_mat(random_traj[i], get_transform_mat(original_traj, random_traj[i]))
        mismatch_scores.append(score_mismatch(original_traj, aligned_random))
        
    return mismatch_scores

if __name__ == '__main__':
   for i in range(3):
        mismatch_scores = compute_mismatch_null(i)
        np.savetxt(os.path.join(AFFINE_TRANSFORM_FIG_PATH, 'null_mismatch', f'mismatch_scores_{i}holes.txt'), mismatch_scores)
        print(f'Mismatch Scores for {i} Holes:')
        print(f'Mean: {np.mean(mismatch_scores)}\tStd: {np.std(mismatch_scores)}')
        plt.figure()
        plt.hist(mismatch_scores, bins=20)
        plt.title(f'Mismatch Scores for {i} Holes')
        plt.xlabel('Mismatch Score')
        plt.ylabel('Frequency')
        plt.show()