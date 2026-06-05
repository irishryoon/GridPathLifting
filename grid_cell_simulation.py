# simulates grid cell activity using attractor network model from Couey et al. 2013, with connectivity matrix from Gardner et al. 2022.
# https://github.com/erikher/GridCellTorus/blob/main/Simulate%20CANN%20and%20idealized%20Tori.ipynb 
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d # type: ignore
import pickle
from tqdm import tqdm # type: ignore

sys.path.append('.')
from trajectory import random_walk, World, visualize_trajectory
from constants import GRID_FIELDS_PATH, TRAJ_PATH

numbumps = 4  # number of bumps in the connectivity matrix
# parameters of the model
extinp = 1. # external input
alpha = 0.15 # scaling factor
ell = 2. # the distance offset in the connectivity matrix.
inh = -0.02 # global inhibition strength
R  = 15. # radius of interaction
Nx = 28 # number of grid cells along the x-dimension
Ny = 44 # number of grid cells along the y-dimension

if numbumps == 4:
    Nx *= 2
if numbumps == 8:
    Nx *= 2
    Ny *= 2
NG = Nx * Ny


def calc_trace_angle_speed(posx: np.ndarray, posy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the angle and speed of a trajectory given the x and y positions.

    Args:
        posx (np.ndarray): Array of x positions.
        posy (np.ndarray): Array of y positions.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the speed and angle arrays.
    """
    angles = np.arctan2(np.diff(posy), np.diff(posx))
    angles = np.append(angles, angles[-1])
    speeds = 1000. * np.sqrt(np.diff(posx)**2 + np.diff(posy)**2)
    return speeds, angles

def interp_arr(arr: np.ndarray, new_len: int) -> np.ndarray:
    """
    Interpolates the given array to a new length.

    Args:
        arr (np.ndarray): The input array to be interpolated.
        new_len (int): The desired length of the interpolated array.

    Returns:
        np.ndarray: The interpolated array with the new length.
    """
    new_indices = np.linspace(0, len(arr) - 1, new_len)
    return np.interp(new_indices, np.arange(len(arr)), arr)

def preprocess(trace: np.ndarray, goal: int=600000) -> np.ndarray:
    """
    Preprocesses a trace by normalizing the position coordinates.

    Args:
        trace (np.ndarray): The input trace with shape (N, 2), where N is the number of data points.
            Each row represents the x and y coordinates of a data point.
        goal (int, optional): The desired length of the preprocessed trace representing the number of 
            miliseconds of movement. Defaults to 600000, which is equivalent to 10 minutes.

    Returns:
        np.ndarray: The preprocessed trace with length equal to the goal.
    """
    posx, posy = trace[:, 0], trace[:, 1]
    side = max(posx.ptp(), posy.ptp())
    posx, posy = (posx / side) - posx.min(), (posy / side) - posy.min()

    posx = interp_arr(posx, goal)
    posy = interp_arr(posy, goal)

    return np.concatenate((posx[:, None], posy[:, None]), axis=1)


def initialize_connectivity_matrix() -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes the connectivity matrix for the grid cell decoding.

    Returns:
        np.ndarray: The connectivity matrix weights of the attractor network.
        np.ndarray: The angles of the grid cells.

    """
    theta = np.zeros((Nx, Ny))
    theta[0:Nx:2, 0:Ny:2] = 0
    theta[1:Nx:2, 0:Ny:2] = 1
    theta[0:Nx:2, 1:Ny:2] = 2
    theta[1:Nx:2, 1:Ny:2] = 3
    theta = 0.5 * np.pi * theta
    theta = np.ravel(theta)

    xes, yes = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    xes = np.ravel(xes)
    yes = np.ravel(yes)

    Rsqrd = R ** 2
    W = np.zeros((NG, NG))
    for xi in range(Nx):
        xdiffA = np.square(np.abs(xes - xi - ell * np.cos(theta)))
        xdiffB = np.square(Nx - np.abs(xes - xi - ell * np.cos(theta)))
        for y in range(Ny):
            n = xi * Ny + y
            ydiffA = np.square(np.abs(yes - y - ell * np.sin(theta)))
            ydiffB = np.square(Ny - np.abs(yes - y - ell * np.sin(theta)))
            d = xdiffA + ydiffA
            W[d < Rsqrd, n] += inh
            d = xdiffB + ydiffA
            W[d < Rsqrd, n] += inh
            d = xdiffA + ydiffB
            W[d < Rsqrd, n] += inh
            d = xdiffB + ydiffB
            W[d < Rsqrd, n] += inh

    return W, theta

def simulation(trace: np.ndarray):
    """
    Simulates a grid cell decoding process.

    Args:
        W (np.ndarray): The connectivity matrix weights of the attractor network.
        trace (np.ndarray): The trajectory of the rat.
        theta (np.ndarray): The angles of the grid cells.
    """
    W, theta = initialize_connectivity_matrix()

    posx, posy = trace[:, 0], trace[:, 1]
    speeds, angles = calc_trace_angle_speed(posx, posy)
    nums = len(speeds)

    S = (np.random.rand(Nx * Ny) > 0.1).astype(float)
    for t in range(2000):
        S = S + 0.1 * (-S + np.maximum((extinp + np.dot(S, W)), 0.))
        S[S < 0.00001] = 0.

    nodes1 = np.zeros((Nx * Ny, nums))

    print('Running simulation...')
    for t in tqdm(range(0, nums)):
        S = S + 0.1 * (-S + np.maximum((extinp + np.dot(S, W) + alpha * speeds[t] * np.cos(angles[t] - theta)), 0.))
        if t % 10 == 0:
            S[S < 0.0001] = 0.
        nodes1[:, t] = S

    
    return nodes1


def _warmup_state(S: np.ndarray, W: np.ndarray, steps: int = 2000) -> np.ndarray:
    """Warm up network state with recurrent dynamics only."""
    S = S.copy()
    for _ in range(steps):
        S = S + 0.1 * (-S + np.maximum((extinp + np.dot(S, W)), 0.))
        S[S < 0.00001] = 0.
    return S


def _rollout_on_trace(trace: np.ndarray, W: np.ndarray, theta: np.ndarray, S0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run network rollout on a trace and return activity plus final state."""
    posx, posy = trace[:, 0], trace[:, 1]
    speeds, angles = calc_trace_angle_speed(posx, posy)
    nums = len(speeds)

    S = S0.copy()
    nodes = np.zeros((Nx * Ny, nums))

    for t in tqdm(range(0, nums)):
        S = S + 0.1 * (-S + np.maximum((extinp + np.dot(S, W) + alpha * speeds[t] * np.cos(angles[t] - theta)), 0.))
        if t % 10 == 0:
            S[S < 0.0001] = 0.
        nodes[:, t] = S

    return nodes, S


def simulation_transfer(path_a: np.ndarray, path_b: np.ndarray, warmup_steps: int = 2000, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate network dynamics from path A, then continue simulation on path B
    using the final state from path A.

    Args:
        path_a (np.ndarray): First trajectory of shape (T_a, 2).
        path_b (np.ndarray): Second trajectory of shape (T_b, 2).
        warmup_steps (int, optional): Number of recurrent-only warmup steps before path A.
            Defaults to 2000.
        seed (int | None, optional): Seed for state initialization. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - activity_a: Grid activity generated while following path A,
              shape (n_cells, T_a - 1).
            - activity_b: Grid activity generated while following path B,
              shape (n_cells, T_b - 1), conditioned on path A dynamics.
            - final_state: Final network state after path B, shape (n_cells,).
    """
    if path_a.ndim != 2 or path_a.shape[1] != 2:
        raise ValueError(f"path_a must have shape (T, 2), got {path_a.shape}")
    if path_b.ndim != 2 or path_b.shape[1] != 2:
        raise ValueError(f"path_b must have shape (T, 2), got {path_b.shape}")
    if path_a.shape[0] < 2 or path_b.shape[0] < 2:
        raise ValueError("Both path_a and path_b must contain at least 2 timesteps.")

    W, theta = initialize_connectivity_matrix()

    rng = np.random.default_rng(seed)
    S = (rng.random(Nx * Ny) > 0.1).astype(float)
    S = _warmup_state(S, W, steps=warmup_steps)

    print('Running transfer simulation on path A...')
    activity_a, state_after_a = _rollout_on_trace(path_a, W, theta, S)

    print('Running transfer simulation on path B...')
    activity_b, final_state = _rollout_on_trace(path_b, W, theta, state_after_a)

    return activity_a, activity_b, final_state

def grid_fields_decoding_simulation(holes=None):
    """
    Simulates the decoding of grid fields.
    """
    world_size = (100, 100)
    # holes = [(20, 20, 40, 40), (60, 60, 80, 80)]
    # holes = [(30, 30, 70, 70)]
    # holes = None
    step_size = 2
    world = World(world_size, holes, step_size)

    # trace = random_walk(25000, world)
    trace, world = pickle.load(open(os.path.join(TRAJ_PATH, f"random_walk_{len(world.holes)}holes.pkl"), "rb"))
    # visualize_trajectory(world, trace, save=True)
    trace = preprocess(trace, goal=600000)
    plt.plot(trace[:, 0])
    # plt.show()

    nodes1 = simulation(trace) # note that the preprocessed trace is used for the simulation, 
    # but the original trace is used for visualization and saving.
    # This is because the preprocessed trace has been normalized and interpolated, which may not be suitable for visualization or saving in its raw form.

    save_dir = os.path.join(GRID_FIELDS_PATH, f'world_{len(world.holes) if world.holes is not None else 0}holes')
    inds = np.random.permutation(Nx * Ny)
    # for n, i in enumerate(inds[:5]):
    #     mtot, x_edge, y_edge, circ = binned_statistic_2d(trace[:-1, 0], trace[:-1, 1], nodes1[i], 
    #         statistic='mean', bins=50, range=None, expand_binnumbers=True)
    #     plt.figure()
    #     plt.imshow(mtot, origin='lower')
    #     plt.title(f'Grid Cell {i}')

    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     plt.savefig(f'{save_dir}/grid_{n}.png')

    # plt.show()

    pickle.dump(nodes1, open(f'{save_dir}/simulation_result.pkl', 'wb'))

def visualize_main():
    from place_cell_simulation import visualize_place_fields
    for i in range(3):
        save_path = os.path.join(GRID_FIELDS_PATH, f'world_{i}holes')
        trajectory, world = pickle.load(open(os.path.join(TRAJ_PATH, f"random_walk_{i}holes.pkl"), "rb"))
        trajectory = preprocess(trajectory, goal=600000)
        result_path = os.path.join(GRID_FIELDS_PATH, f'world_{i}holes', 'simulation_result.pkl')
        spike_trains = pickle.load(open(result_path, 'rb'))
        visualize_place_fields(trajectory, spike_trains, save_path)

if __name__ == '__main__':
    for holes in [None, [(20, 20, 40, 40), (60, 60, 80, 80)], [(30, 30, 70, 70)]]:
        grid_fields_decoding_simulation(holes=holes)
    visualize_main()