import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as cf
import pickle
from tqdm import tqdm # type: ignore

sys.path.append('.')
from constants import TRAJ_PATH

RESOLUTION = 360  # The number of angles to compute per 360 degrees


class World:
    """
    A class representing the world in which the random walk takes place.

    Attributes:
        size (Tuple[int, int]): The size of the world in each dimension.
        holes (list[tuple[int, int, int, int]]): A list of tuples representing the coordinates of the holes.
            Each tuple should contain four integers representing the x and y coordinates of the top-left and bottom-right corners of the hole.
            Error will be raised for invalid configurations of the two corners.
        step_size (int): The step size of the random walk.
    """
    def __init__(
        self,
        size: tuple[int, int],
        holes: list[tuple[int, int, int, int]] | None = None,
        step_size: int = 2,
    ):
        if holes is not None:
            for hole in holes:
                if (hole[0] > hole[2]) or (hole[1] > hole[3]):
                    raise ValueError(f"Invalid hole configuration with {hole}")
    
        self.size = size
        if holes is None:
            self.holes = []
        else:
            self.holes = holes
        self.step_size = step_size


    def beyond_edge(self, pos: np.ndarray) -> bool:
        """
        Checks if the given position is beyond the world boundaries.

        Args:
            pos (np.ndarray): The position to be checked.

        Returns:
            bool: True if the position is beyond the world boundaries, False otherwise.
        """
        for i, value in enumerate(pos):
            if value < 0 or value > self.size[i]:
                return True
        return False


    def in_holes(self, pos: np.ndarray) -> bool:
        """
        Checks if the given position falls within any of the holes.

        Args:
            pos (np.ndarray): The position to be checked. It should be a numpy array of shape (2,) representing the x and y coordinates.

        Returns:
            bool: True if the position falls within any of the holes, False otherwise.
        """
        if len(self.holes) == 0:
            return False
        for hole in self.holes:
            if (
                (pos[1] >= hole[0])
                and (pos[1] <= hole[2])
                and (pos[0] >= hole[1])
                and (pos[0] <= hole[3])
            ):
                return True
        return False

    
def conv(ang: float) -> np.ndarray:
    """
    Converts an angle in degrees to a 2D vector.

    Args:
        ang (float): The angle in degrees.

    Returns:
        np.ndarray: A 2D vector representing the angle.
    """
    x = np.cos(np.radians(ang))
    y = np.sin(np.radians(ang))
    return np.array([x, y])


def valid_positions(pos: np.ndarray, ranges: list[tuple[float, float]], world: World) -> np.ndarray:
    """
    Calculate the valid ranges of positions based on the given parameters.

    Args:
        pos (np.ndarray): The current position.
        ranges (list[tuple[float, float]]): A list of tuples representing the ranges of angles.
        world (World): The world object containing information about the world size and holes.
        
    Returns:
        np.ndarray: An array of valid positions within the specified ranges.
        
    Raises:
        ValueError: If an invalid range is provided (left bound > right bound).
    """
    valid_directions = np.empty((0, 2))
    step_size = random.uniform(0, world.step_size)
    for l, r in ranges:
        if l > r:
            raise ValueError(f"Invalid range with {l} > {r}")
        new_pos_vec = np.linspace(l, r, int((r - l) / 360 * RESOLUTION))
        new_pos = np.array([pos + step_size * conv(angle) for angle in new_pos_vec])
        new_pos = np.array([pos for pos in new_pos if not world.beyond_edge(pos) and not world.in_holes(pos)])

        if new_pos.size != 0:
            valid_directions = np.concatenate((valid_directions, new_pos))
    return valid_directions


def fix_range(dir_range: tuple[float, float]) -> list[tuple[float, float]]:
    """
    Adjusts the direction range to ensure it falls within the range of 0 to 360 degrees.

    Args:
        dir_range (tuple[int, int]): A tuple representing the direction range.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the adjusted direction range.
    """
    if abs(dir_range[0] - dir_range[1]) > 361:
        raise ValueError('Direction range should not exceed 360 degrees.')
    l, r = dir_range
    output = []
    if l < 0:
        output.append((l + 360, 360.0))
        l = 0.0
    if r > 360:
        output.append((0.0, r - 360))
        r = 360.0
    output.append((l, r))
    return output


def random_walk(time: int, world: World, save: bool = True, no_warnings: bool = False) -> np.ndarray:
    """
    Generate a random walk trajectory.

    Args:
        time (int): The number of time steps for the random walk.
        world (World): The world in which the random walk takes place.
        save (bool): Whether to save the trajectory to a file. Default is True.
        no_warnings (bool): If True, suppresses warnings about invalid directions. Default is False.

    Returns:
        np.ndarray: An array representing the trajectory of the random walk.
    """
    direction = 0.0
    dir_range_one_side = 75
    trace = np.zeros((time, 2))
    # print(world.holes)
    trace[0] = [random.uniform(0, world.size[0]), random.uniform(0, world.size[1])]
    while world.in_holes(trace[0]):
        trace[0] = [random.uniform(0, world.size[0]), random.uniform(0, world.size[1])]
    for idx in tqdm(range(1, time)):
        valid_directions = valid_positions(
            trace[idx - 1],
            fix_range((direction - dir_range_one_side, direction + dir_range_one_side)),
            world
        )
        if valid_directions.size == 0:
            if not no_warnings: print(f"Warning: No valid directions found with direction {direction} at position {trace[idx - 1]}")
            direction = random.uniform(0, 360)
            trace[idx] = trace[idx - 1]
            continue
        trace[idx] = valid_directions[random.randint(0, valid_directions.shape[0] - 1)]
        direction = np.degrees(
                np.arctan2(trace[idx, 1] - trace[idx - 1, 1], trace[idx, 0] - trace[idx - 1, 0])
            )% 360
    if save:
        pickle.dump((trace, world), open(os.path.join(TRAJ_PATH,f"random_walk_{len(world.holes)}holes.pkl"), "wb"))

    return trace


def visualize_trajectory(world: World, trace: np.ndarray, save: bool = True, show: bool = True):
    plt.figure(figsize=(8, 8))
    num_holes = len(world.holes) if world.holes is not None else 0
    plt.xlim(0, world.size[0])
    plt.ylim(0, world.size[1])
    if world.holes: draw_holes(world.holes)
    plt.plot(trace[:, 1], trace[:, 0])
    plt.title(f"Random Walk Trajectory in {num_holes}-Hole World")
    if save: plt.savefig(os.path.join(TRAJ_PATH,f"random_walk_{num_holes}holes.png"), dpi=200)
    if show: plt.show()


def draw_holes(holes):
    for x1, y1, x2, y2 in holes:
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], "black")


if __name__ == "__main__":
    world_size = (100, 100)
    # holes = [(20, 20, 40, 40), (60, 60, 80, 80)]
    # holes = [(30, 30, 70, 70)]
    # holes = None
    import matplotlib.pyplot as plt
    import pickle
    import os
    from constants import TRAJ_PATH
    from multiprocessing import Pool

    def generate_and_save(args):
        n_holes, n, world_size, length, hole_locations, step_size = args
        holes = hole_locations[n_holes]
        world = World(world_size, holes, step_size)
        trace = random_walk(length, world, save=False, no_warnings=True)
        os.makedirs(TRAJ_PATH, exist_ok=True)
        out_path = os.path.join(TRAJ_PATH, f"random_walk_{n_holes}holes_{length}_n{n}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump((trace, world), f)
        # Optionally visualize (disabled by default)
        # visualize_trajectory(world, trace, show=False, save=False)

    length = 25000
    step_size = 2
    hole_locations = {
        0: None,
        1: [(30, 30, 70, 70)],
        2: [(10, 10, 45, 45), (55, 55, 90, 90)]
    }
    jobs = []
    for n_holes in [0, 1, 2]:
        for n in range(10):
            jobs.append((n_holes, n, world_size, length, hole_locations, step_size))
    with Pool() as pool:
        pool.map(generate_and_save, jobs)
    print(f"Saved all trajectories to {TRAJ_PATH}")