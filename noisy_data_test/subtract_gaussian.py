import os, sys

import numpy as np
import matplotlib.pyplot as plt
import pickle
import concurrent.futures as cf

sys.path.append('.')
from add_noise import gaussian, fast_gaussian_curve
from constants import GRID_FIELDS_PATH


def subtract_gaussian_1d(data: np.ndarray, portion: float, s: float = 50, sub_max: float = 0.4) -> np.ndarray:
    """
    Subtract Gaussian curves from a portion of the data.

    Args:
        data: 1D input array.
        portion: Fraction of timepoints to apply subtraction (0 < portion <= 1).
        s: Standard deviation of the Gaussian.
        sub_max: Maximum amplitude of the subtracted Gaussian.

    Returns:
        Modified data array (clipped to [0, max]).
    """
    max_thrsh = np.max(data)
    num_timepoints = int(data.shape[0] * portion)
    ind_sub = np.random.choice(data.shape[0], num_timepoints, replace=False)

    arr = np.arange(data.shape[0])
    data_out = data.copy()
    for i in ind_sub:
        curve = gaussian(arr, i, s)
        curve *= sub_max / np.max(curve)
        data_out -= curve

    data_out = np.clip(data_out, 0, max_thrsh)
    return data_out


def subtract_gaussian_1d_fast(data: np.ndarray, portion: float, s: float = 50, sub_max: float = 0.08) -> np.ndarray:
    max_thrsh = np.max(data)
    num_timepoints = int(data.shape[0] * portion)
    ind_sub = np.random.choice(data.shape[0], num_timepoints, replace=False)
    curve = fast_gaussian_curve(s, sub_max)

    data_out = data.copy()
    l = len(curve) // 2
    for ind in ind_sub:
        if ind - l < 0:
            left = ind
            right = l
        elif ind + l > data.shape[0]:
            left = l
            right = data.shape[0] - ind
        else:
            left = l
            right = l
        data_out[ind - left:ind + right] -= curve[l - left:l + right]

    data_out = np.clip(data_out, 0, max_thrsh)
    return data_out


class GaussianSubtractor:
    def __init__(self, portion: float, s: float = 50, sub_max: float = 0.08) -> None:
        self.portion = portion
        self.s = s
        self.sub_max = sub_max

    def subtract_1d(self, data: np.ndarray) -> np.ndarray:
        return subtract_gaussian_1d_fast(data, self.portion, self.s, self.sub_max).T


def subtract_gaussian_2d_parallel(data: np.ndarray, portion: float, s: float = 50, sub_max: float = 0.08) -> np.ndarray:
    """
    Subtract Gaussian curves from 2D data in parallel.

    Parameters:
    - data (np.ndarray): Input 2D data array (timepoints x features).
    - portion (float): Fraction of timepoints to apply subtraction.
    - s (float): Standard deviation of the Gaussian.
    - sub_max (float): Maximum amplitude of the subtracted Gaussian.

    Returns:
    - np.ndarray: Modified data with Gaussian subtraction applied.
    """
    with cf.ProcessPoolExecutor(max_workers=25) as executor:
        results = list(executor.map(
            GaussianSubtractor(portion, s, sub_max).subtract_1d, data.T
        ))

    return np.array(results).T


def plot_gaussian_param_examples(
    data: np.ndarray,
    std_list: list,
    portion_list: list,
    sub_max: float = 0.4,
    n_timepoints: int = 10000,
    error_list: list | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot panels stacked vertically: original on top, then one panel per (std, portion) pair.

    Args:
        data: 1D input array.
        std_list: List of Gaussian standard deviations (paired with portion_list).
        portion_list: List of portion values paired with std_list.
        sub_max: Maximum amplitude of the subtracted Gaussian.
        n_timepoints: Number of timepoints to display.
        error_list: Optional list of error-level labels (e.g. 'low error', 'high error').
        save_path: If provided, save the figure here instead of showing it.
    """
    segment = data[:n_timepoints]
    n_panels = 1 + len(std_list)

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)

    axes[0].plot(segment, lw=1.5, color='steelblue')
    axes[0].set_title('Original firing rate', fontsize=9)
    axes[0].set_ylabel('Rate')

    for i, (s, portion) in enumerate(zip(std_list, portion_list)):
        result = subtract_gaussian_1d_fast(segment.copy(), portion, s, sub_max)
        axes[i + 1].plot(result, lw=1.5, color='steelblue')
        label = f'  ({error_list[i]})' if error_list else ''
        axes[i + 1].set_title(
            f'$\\sigma$ = {s},  p = {int(portion * 100)}%{label}', fontsize=9
        )
        axes[i + 1].set_ylabel('Rate')

    axes[-1].set_xlabel('Time (steps)')
    fig.suptitle(f'Information deletion examples  (h = {sub_max})', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    grid_rates = pickle.load(
        open(os.path.join(GRID_FIELDS_PATH, "world_0holes", "simulation_result.pkl"), "rb")
    ).T
    print("grid_rates shape:", grid_rates.shape)

    # Pick one cell's firing-rate trace
    cell = grid_rates[:, 0]

    std_list    = [1,    10,   10,   50  ]
    portion_list = [0.10, 0.05, 0.10, 0.01]

    save_path = os.path.join(os.path.dirname(__file__), "subtract_gaussian_examples.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plot_gaussian_param_examples(
        cell,
        std_list=std_list,
        portion_list=portion_list,
        sub_max=0.4,
        n_timepoints=10000,
        error_list=['low error', 'low error', 'high error', 'high error'],
        save_path=save_path,
    )
