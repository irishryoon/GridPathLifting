"""
Sweep the epsilon parameter directly and evaluate the resulting mismatch score.

For each epsilon value, the script:
1. Lifts the coordinates using the specified lifting method (original or upgrade)
2. Aligns via affine transform and computes the mismatch score
3. Plots mismatch score vs epsilon and saves results to CSV
"""

import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(".")
from pipelines import compare_lifted_pipeline
from trajectory import World
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance, toroidal_lifting_distance_upgrade
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from affine_transform.score_mismatch import score_mismatch
from constants import GRID_FIELDS_PATH, TRAJ_PATH
from grid_cell_simulation import preprocess


def _eval_single_epsilon(eps, cir_coords, original_traj, use_upgrade):
    """Evaluate mismatch score for a single epsilon value."""
    lifting_fn = toroidal_lifting_distance_upgrade if use_upgrade else toroidal_lifting_distance
    lifted = lifting_fn(cir_coords.copy(), epsilon=eps)

    processed_trace = preprocess(original_traj, goal = 600000) # the second parameter is world.step_size in the trajectory simulation. In our simlations, 5 is used
    _, ms = compare_lifted_pipeline(lifted, processed_trace)
    #transform_mat = get_transform_mat(lifted, original_traj)
    #lifted_transformed = apply_transform_mat(lifted, transform_mat)
    #ms = score_mismatch(lifted_transformed, original_traj)
    
    return eps, ms


def sweep_epsilon(
    num_holes: int = 1,
    epsilon_values: np.ndarray | None = None,
    save_dir: str | None = None,
    use_upgrade: bool = True,
    n_workers: int | None = None,
):
    """
    Sweep epsilon in toroidal lifting and record the mismatch score.

    Args:
        num_holes: World configuration (0, 1, or 2).
        epsilon_values: Array of epsilon values to sweep. Defaults to linspace(0.01, pi, 50).
        save_dir: Directory to save results. Defaults to revision/epsilon_sweep_results.
        use_upgrade: If True, use toroidal_lifting_distance_upgrade; otherwise use toroidal_lifting_distance.
        n_workers: Number of parallel workers. Defaults to 10.
    """
    if epsilon_values is None:
        epsilon_values = np.linspace(0.01, np.pi, 50)
    if n_workers is None:
        n_workers = 10

    method_tag = "upgrade" if use_upgrade else "original"
    print(f"Starting epsilon sweep ({method_tag}) for world_{num_holes}holes "
          f"with {len(epsilon_values)} epsilon values using {n_workers} workers.")

    if save_dir is None:
        save_dir = "revision/sweep_epsilon/epsilon_sweep_results"
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    grid_path = os.path.join(GRID_FIELDS_PATH, f"world_{num_holes}holes")
    coords_file = os.path.join(grid_path, "toroidal_coords_dreimac.npz")
    traj_file = os.path.join(TRAJ_PATH, f"random_walk_{num_holes}holes.pkl")

    with np.load(coords_file) as data:
        cir_coords = data["coords"]
    original_traj = pickle.load(open(traj_file, "rb"))[0]

    print(f"Circular coords shape: {cir_coords.shape}")
    print(f"Original trajectory shape: {original_traj.shape}")
    print(f"Sweeping {len(epsilon_values)} epsilon values in [{epsilon_values[0]:.4f}, {epsilon_values[-1]:.4f}]")

    # --- Sweep with concurrent.futures ---
    worker_fn = partial(_eval_single_epsilon,
                        cir_coords=cir_coords,
                        original_traj=original_traj,
                        use_upgrade=use_upgrade)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker_fn, eps): i for i, eps in enumerate(epsilon_values)}
        mismatch_scores = np.zeros(len(epsilon_values))
        for future in as_completed(futures):
            idx = futures[future]
            eps, ms = future.result()
            mismatch_scores[idx] = ms

    for eps, ms in zip(epsilon_values, mismatch_scores):
        print(f"  epsilon={eps:.4f}  mismatch={ms:.6f}")

    # --- Save results ---
    results = np.column_stack([epsilon_values, mismatch_scores])
    csv_path = os.path.join(save_dir, f"epsilon_sweep_{method_tag}_{num_holes}holes.csv")
    np.savetxt(csv_path, results, delimiter=",", header="epsilon,mismatch_score", comments="")
    print(f"\nResults saved to {csv_path}")

    # # --- Plot ---
    # fig, ax = plt.subplots(figsize=(8, 5))

    # color_ms = "#1f77b4"
    # ax.set_xlabel("Epsilon")
    # ax.set_ylabel("Mismatch Score", color=color_ms)
    # ax.plot(epsilon_values, mismatch_scores, "o-", color=color_ms, markersize=3, label="Mismatch Score")
    # ax.tick_params(axis="y", labelcolor=color_ms)

    # # best_idx = np.argmin(mismatch_scores)
    # # ax.axvline(epsilon_values[best_idx], color="red", linestyle=":", alpha=0.6)
    # # ax.annotate(
    # #     f"best eps={epsilon_values[best_idx]:.4f}\nmismatch={mismatch_scores[best_idx]:.6f}",
    # #     xy=(epsilon_values[best_idx], mismatch_scores[best_idx]),
    # #     xytext=(15, 15),
    # #     textcoords="offset points",
    # #     arrowprops=dict(arrowstyle="->", color="red"),
    # #     fontsize=9,
    # #     color="red",
    # # )

    # plt.title(f"Epsilon Sweep ({method_tag}) — world_{num_holes}holes")
    # fig.tight_layout()

    # fig_path = os.path.join(save_dir, f"epsilon_sweep_{method_tag}_{num_holes}holes.pdf")
    # plt.savefig(fig_path, dpi=200)
    # print(f"Figure saved to {fig_path}")
    # plt.show()

    return epsilon_values, mismatch_scores


if __name__ == "__main__":
    # epsilon_values = np.linspace(0.01, np.pi, 50)
    epsilon_values = np.linspace(0.01, 2 * np.pi, 20)

    print("=" * 60)
    print("Running with toroidal_lifting_distance_upgrade")
    print("=" * 60)
    sweep_epsilon(epsilon_values=epsilon_values, use_upgrade=True)

    print("\n" + "=" * 60)
    print("Running with toroidal_lifting_distance (original)")
    print("=" * 60)
    sweep_epsilon(epsilon_values=epsilon_values, use_upgrade=False)
