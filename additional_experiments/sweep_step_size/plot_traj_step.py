"""
Run the full pipeline once with a user-provided step_size and plot the first 1000 steps
of the trajectory using a gradient color scheme.

Usage: python plot_traj_step20.py <step_size>
"""

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.append(".")
from trajectory import random_walk, World
from grid_cell_simulation import preprocess, simulation
from pipelines import recover_traj_pipeline, compare_lifted_pipeline, segmental_compare_pipeline


WORLD_SIZE = (100, 100)
HOLES = [(30, 30, 70, 70)]
TRAJ_LENGTH = 10000
SEG_LEN = 500
PLOT_STEPS = 1000
SAVE_DIR = "revision/sweep_step_size/example_trajectories"


def run_and_plot(step_size):
    # --- Generate trajectory ---
    world = World(WORLD_SIZE, HOLES, step_size=step_size)
    trace = random_walk(TRAJ_LENGTH, world, save=False, no_warnings=True)
    processed_trace = preprocess(trace, step_size)
    grid_activity = simulation(processed_trace)

    try:
        _, lifted = recover_traj_pipeline(grid_activity)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return

    lifted_transformed, ms = compare_lifted_pipeline(lifted, processed_trace)
    try:
        seg_scores = segmental_compare_pipeline(lifted_transformed, processed_trace, SEG_LEN)
        seg_mean = float(np.mean(seg_scores))
        seg_std = float(np.std(seg_scores))
    except Exception as e:
        print(f"Segmental pipeline failed: {e}")
        seg_mean = seg_std = float("nan")

    print(f"mismatch={ms:.6f}  seg_mean={seg_mean:.6f}  seg_std={seg_std:.6f}")

    # --- Save CSVs ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    for arr, name in [
        (trace,              f"trace_step{step_size}.csv"),
        (processed_trace,    f"processed_trace_step{step_size}.csv"),
        (lifted,             f"lifted_step{step_size}.csv"),
        (lifted_transformed, f"lifted_transformed_step{step_size}.csv"),
    ]:
        csv_path = os.path.join(SAVE_DIR, name)
        np.savetxt(csv_path, arr, delimiter=",", header="x,y", comments="")
        print(f"Saved {csv_path}")

    # --- Save PNGs ---
    def make_segments(arr):
        pts = arr.reshape(-1, 1, 2)
        return np.concatenate([pts[:-1], pts[1:]], axis=1)

    for arr, name, use_world_lim in [
        (trace,              f"trace_step{step_size}.png",              True),
        (processed_trace,    f"processed_trace_step{step_size}.png",    False),
        (lifted,             f"lifted_step{step_size}.png",             False),
        (lifted_transformed, f"lifted_transformed_step{step_size}.png", False),
    ]:
        t = np.linspace(0, 1, len(arr) - 1)
        fig, ax = plt.subplots(figsize=(7, 7))
        lc = LineCollection(make_segments(arr), cmap="plasma", norm=plt.Normalize(0, 1))
        lc.set_array(t)
        lc.set_linewidth(0.5)
        ax.add_collection(lc)

        if use_world_lim:
            x1, y1, x2, y2 = HOLES[0]
            ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color="black", linewidth=1.5)
            ax.set_xlim(0, WORLD_SIZE[0])
            ax.set_ylim(0, WORLD_SIZE[1])
        else:
            pad = 0.05 * (arr.max() - arr.min())
            ax.set_xlim(arr[:, 0].min() - pad, arr[:, 0].max() + pad)
            ax.set_ylim(arr[:, 1].min() - pad, arr[:, 1].max() + pad)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(name.replace(".png", ""))

        cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Time (early → late)")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["start", "end"])

        fig.tight_layout()
        png_path = os.path.join(SAVE_DIR, name)
        plt.savefig(png_path, dpi=200)
        print(f"Saved {png_path}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step_size", type=int, help="Step size for the random walk")
    args = parser.parse_args()
    run_and_plot(args.step_size)
