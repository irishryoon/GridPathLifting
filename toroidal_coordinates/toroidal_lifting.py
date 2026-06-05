import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy

sys.path.append('.')
from toroidal_coordinates.epsilon import compute_epsilon, compute_delta
from constants import GRID_FIELDS_PATH, TRAJ_PATH, GRID_FIG_PATH

class Covers:
    def __init__(self):
        EPSILON= 0.5
        self.psi_range = [(0, 2*np.pi/3+EPSILON), (2*np.pi/3-EPSILON, 4*np.pi/3+EPSILON), (4*np.pi/3-EPSILON, 2*np.pi)]
        self.theta_range = self.psi_range.copy()
        
        self.cover_range = {}
        for (ii, (psi_min, psi_max)) in enumerate(self.psi_range):
            for (jj, (theta_min, theta_max)) in enumerate(self.theta_range):
                self.cover_range[(ii, jj)] = (theta_min, psi_min, theta_max, psi_max)
    
    def check_in_cover(self, pos: np.ndarray, cover_idx: tuple[int, int]) -> bool:
        """
        Check if the given position is within the specified cover.

        Args:
            pos (np.ndarray): The position to check, represented as a numpy array of shape (2,).
            cover_idx (int): The index of the cover to check against.

        Returns:
            bool: True if the position is within the cover, False otherwise.
        """
        theta, psi = pos
        theta_min, psi_min, theta_max, psi_max = self.cover_range[cover_idx]
        theta_flag = (theta_min <= theta <= theta_max) or (theta_min <= theta + 2*np.pi <= theta_max) or (theta_min <= theta - 2*np.pi <= theta_max)
        psi_flag = (psi_min <= psi <= psi_max) or (psi_min <= psi + 2*np.pi <= psi_max) or (psi_min <= psi - 2*np.pi <= psi_max)
        return theta_flag and psi_flag
    
    def get_cover_idx(self, pos: np.ndarray) -> tuple[int, int]:
        """
        Returns the index of the cover that contains the given position.
        
        Args:
            pos (np.ndarray): The position to check.
            
        Returns:
            tuple[int, int]: The index of the cover that contains the position.
        """
        if not (0 <= pos[0] <= 2*np.pi and 0 <= pos[1] <= 2*np.pi):
            raise ValueError("Invalid position")
        for idx in self.cover_range:
            if self.check_in_cover(pos, idx):
                return idx
            
        raise ValueError("Unknown error")

def toroidal_lifting(cir_coords: np.ndarray) -> np.ndarray:
    """
    Perform toroidal lifting on the toroidal coordinates from the world with the specified number of holes.
    Args:
        cir_coords (np.ndarray): The toroidal coordinates to lift.
    Returns:
        np.ndarray: The toroidal coordinates after lifting.
    """
    covers = Covers()

    current_cover = covers.get_cover_idx(cir_coords[0])
    
    cover_sequence = [current_cover]
    cover_timestamps = []
    current_start = 0
    for i in range(1, cir_coords.shape[0]):
        if covers.check_in_cover(cir_coords[i], current_cover):
            continue
        else:
            cover_timestamps.append((current_start, i))
            
            current_start = i
            current_cover = covers.get_cover_idx(cir_coords[i])
            while covers.check_in_cover(cir_coords[current_start-1], current_cover):
                current_start -= 1
            
            # Remove the last cover if it is fully contained in the current cover
            if current_start <= cover_timestamps[-1][0]:
                cover_timestamps.pop(-1)
                cover_sequence.pop(-1)
                
            cover_sequence.append(current_cover)
    cover_timestamps.append((current_start, cir_coords.shape[0]))
    
    for idx, (ii, jj) in enumerate(cover_sequence):
        if idx == 0:
            prev_cover = cover_sequence[0]
            continue
        if ii - prev_cover[0] == -2:
            cir_coords[cover_timestamps[idx-1][1]:, 1] += 2*np.pi
        elif ii - prev_cover[0] == 2:
            cir_coords[cover_timestamps[idx-1][1]:, 1] -= 2*np.pi
        if jj - prev_cover[1] == -2:
            cir_coords[cover_timestamps[idx-1][1]:, 0] += 2*np.pi
        elif jj - prev_cover[1] == 2:
            cir_coords[cover_timestamps[idx-1][1]:, 0] -= 2*np.pi
        prev_cover = (ii, jj)
    
    
    # if __name__ == '__main__':
    #     plt.figure()
    #     plt.plot(cir_coords[:, 0], cir_coords[:, 1])
    #     pickle.dump(cir_coords, open(os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes", "toroidal_coords_lifted.pkl"), "wb"))
    
    return cir_coords

def toroidal_lifting_distance(cir_coords: np.ndarray, epsilon: float | None = None, plot: bool = False) -> np.ndarray:
    """
    Perform toroidal lifting on the toroidal coordinates from the world with the specified number of holes using
    coordinate-wise distance comparison.
    Args:
        cir_coords (np.ndarray): The toroidal coordinates to lift.
    Returns:
        np.ndarray: The toroidal coordinates after lifting.
    """
    
    if epsilon is None:
        #DELTA = compute_delta(cir_coords) 
        EPSILON = compute_epsilon(cir_coords)
    else:
        EPSILON = epsilon
    # import pdb; pdb.set_trace()
    # EPSILON = 0.1
    
    ### Original version of the algorithm with O(T^2) complexity
    # cir_coords = cir_coords.copy()
    
    # DISTANCE_THRESHOLD = 2 * np.pi - EPSILON
    # for i in range(1, cir_coords.shape[0]):
    #     for j in range(2):
    #         if cir_coords[i, j] - cir_coords[i-1, j] > DISTANCE_THRESHOLD:
    #             cir_coords[i:, j] -= 2*np.pi
    #         elif cir_coords[i, j] - cir_coords[i-1, j] < -DISTANCE_THRESHOLD:
    #             cir_coords[i:, j] += 2*np.pi
        # if cir_coords[i, 0] - cir_coords[i-1, 0] > DISTANCE_THRESHOLD:
        #     cir_coords[i:, 0] -= 2*np.pi
        # elif cir_coords[i, 0] - cir_coords[i-1, 0] < -DISTANCE_THRESHOLD:
        #     cir_coords[i:, 0] += 2*np.pi
        # if cir_coords[i, 1] - cir_coords[i-1, 1] > DISTANCE_THRESHOLD:
        #     cir_coords[i:, 1] -= 2*np.pi
        # elif cir_coords[i, 1] - cir_coords[i-1, 1] < -DISTANCE_THRESHOLD:
        #     cir_coords[i:, 1] += 2*np.pi
    
    ref_coords = cir_coords.copy()
    cir_coords = cir_coords.copy()
    offset = np.zeros(2)
    
    #DISTANCE_THRESHOLD = 2 * np.pi - DELTA
    DISTANCE_THRESHOLD = EPSILON
    for i in range(1, ref_coords.shape[0]):
        for j in range(2):
            diff = ref_coords[i, j] - ref_coords[i-1, j]
            if diff > DISTANCE_THRESHOLD:
                offset[j] -= 2*np.pi
            elif diff < -DISTANCE_THRESHOLD:
                offset[j] += 2*np.pi
            cir_coords[i, j] += offset[j]
    
    
    if plot:
        plt.figure()
        plt.plot(cir_coords[:, 0], cir_coords[:, 1])
        plt.savefig(os.path.join(GRID_FIG_PATH, f"world_{num_holes}holes", "toroidal_coords_lifted.png"), dpi=300)
        
    #     pickle.dump(cir_coords, open(os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes", "toroidal_coords_lifted.pkl"), "wb"))
    
    return cir_coords

def toroidal_lifting_distance_upgrade(cir_coords: np.ndarray, epsilon: float | None = None) -> np.ndarray:
    """
    Adjusts circular coordinates to ensure continuity by lifting them onto a toroidal surface.
    This function modifies the input circular coordinates to account for discontinuities
    caused by wrapping around at the boundaries of a toroidal space. It ensures that the
    distances between consecutive points are minimized, avoiding jumps larger than a
    specified threshold.
    Args:
        cir_coords (np.ndarray): A 2D array of shape (n, 2) representing circular coordinates
            on a toroidal surface. Each row corresponds to a point, and each column corresponds
            to a circular dimension.
        epsilon (float | None, optional): A small positive value used to compute the distance
            threshold. If None, the threshold is computed using the `compute_epsilon` function.
            Defaults to None.
    Returns:
        np.ndarray: A 2D array of the same shape as `cir_coords`, where the coordinates have
        been adjusted to ensure continuity on the toroidal surface.
    """
    if epsilon is None:
        #DELTA = compute_delta(cir_coords) + 1
        EPSILON = compute_epsilon(cir_coords) - 1

    else:
        EPSILON = epsilon
    cir_coords = cir_coords.copy()
    #DISTANCE_THRESHOLD = 2 * np.pi - DELTA 
    DISTANCE_THRESHOLD = EPSILON
    print("DISTANCE_THRESHOLD:", DISTANCE_THRESHOLD)

    for i in range(1, cir_coords.shape[0]):
        for j in range(2):
            diff = abs(cir_coords[i, j] - cir_coords[i-1, j])
            if diff > DISTANCE_THRESHOLD:
                if diff > abs(cir_coords[i-1, j] - (cir_coords[i, j] + 2*np.pi)):
                    cir_coords[i:, j] += 2*np.pi
                elif diff > abs(cir_coords[i-1, j] - (cir_coords[i, j] - 2*np.pi)):
                    cir_coords[i:, j] -= 2*np.pi
            
    return cir_coords


global num_holes
if __name__ == '__main__':
    for i in range(3):
        num_holes = i
        with np.load(os.path.join(GRID_FIELDS_PATH, f"world_{i}holes", "toroidal_coords_dreimac.npz")) as data:
            cir_coords = data["coords"]
        print(cir_coords.shape)
        toroidal_lifting(deepcopy(cir_coords))
        toroidal_lifting_distance(deepcopy(cir_coords))
        toroidal_lifting_distance_upgrade(deepcopy(cir_coords))
    plt.show()