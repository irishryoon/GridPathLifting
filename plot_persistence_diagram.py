import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sys.path.append('.')
from constants import *

def plot_persistence_diagram(
    diagrams: list[np.ndarray], 
    life_range: tuple[float, float] | None = None,
    legend_fontsize: int = 20,
    tick_fontsize: int = 16,
    axis_fontsize: int = 25,
    savefig: str | None = None,
    ) -> tuple[float, float]:
    """
    Plots a persistence diagram for a given list of persistence diagrams.
    Args:
        diagrams : list[np.ndarray]
            A list of 2D numpy arrays where each array represents a persistence diagram.
            Each array should have shape (n_points, 2), where the two columns represent
            the birth and death times of topological features.
        life_range : tuple[float, float] | None, optional
            The range of the axes for the plot. If None, the range is determined
            automatically based on the data. Defaults to None.
        legend_fontsize : int, optional
            Font size for the legend. Defaults to 20.
        tick_fontsize : int, optional
            Font size for the tick labels on the axes. Defaults to 16.
        axis_fontsize : int, optional
            Font size for the axis labels. Defaults to 25.
        savefig : str | None, optional
            File path to save the figure. If None, the figure is not saved. Defaults to None.
    Returns:
        tuple[float, float]
            A tuple containing the lower bound of the y-axis (`y_down`) and the y-coordinate
            of the line representing infinity (`b_inf`).
    """
    colormap = sns.color_palette("muted")
    marker_style = dict(markersize=18, alpha=0.6)
    
    if life_range is None:
        concat_dgms = np.concatenate(diagrams).flatten()
        finite_dgms = concat_dgms[np.isfinite(concat_dgms)]
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer
        
        y_down, y_up = x_down, x_up
    else:
        y_down, y_up = life_range
        x_down, x_up = life_range
        
    b_inf = y_down + (y_up - y_down) * 0.95
        
    plt.figure(figsize=(5, 5))
    for i, dgm in enumerate(diagrams):
        plt.plot(dgm[:, 0], dgm[:, 1], 'o', label=f'$H_{i}$', c=colormap[i], **marker_style)
    plt.plot([x_down, x_up], [y_down, y_up], 'k--')
    plt.plot([x_down, x_up], [b_inf, b_inf], "k--", label=r"$\infty$")
    plt.plot([0], [b_inf], 'o', **marker_style, c=colormap[0])
    
    plt.xlabel('Birth', fontsize=axis_fontsize)
    plt.ylabel('Death', fontsize=axis_fontsize)
    plt.xlim(x_down, x_up)
    plt.ylim(y_down, y_up)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(loc='lower right', fontsize=legend_fontsize)
    
    if savefig:    
        plt.savefig(savefig, transparent=True, bbox_inches='tight')

    return y_down, b_inf

if __name__ == "__main__":
    persistence = pickle.load(open(os.path.join(MANU_FIG_CACHE_PATH, 'sphere.pickle'), 'rb'))
    plot_persistence_diagram(persistence['dgms'], life_range=(-0.5,2), savefig=os.path.join(MANU_FIG_MATERIAL_PATH, 'persistence_diagram.pdf'))