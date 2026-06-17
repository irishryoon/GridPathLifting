"""
Sweep the epsilon parameter on Gardner (real) data and evaluate the resulting mismatch score.

For each epsilon value, the script:
1. Reuses pre-extracted toroidal coordinates (Gardner_coord is expensive)
2. Lifts the coordinates using the specified lifting method (original or upgrade)
3. Applies Gaussian smoothing, aligns via affine transform, and computes the mismatch score
4. Plots mismatch score vs epsilon and saves results to CSV
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(".")
from constants import GARDNER_DATA_PATH
from helper_scripts.utils import get_spikes
from toroidal_coordinates.Gardner_toroidal import Gardner_coord
from toroidal_coordinates.toroidal_lifting import toroidal_lifting_distance, toroidal_lifting_distance_upgrade
from affine_transform.get_transform_mat import get_transform_mat, apply_transform_mat
from affine_transform.score_mismatch import score_mismatch, segmental_mismatch


def _eval_single_epsilon(eps, cir_coords, original_coord, use_upgrade, smooth_sigma, len_seg):
    """Evaluate mismatch score for a single epsilon value."""
    lifting_fn = toroidal_lifting_distance_upgrade if use_upgrade else toroidal_lifting_distance
    lifted = lifting_fn(cir_coords.copy(), epsilon=eps)

    if smooth_sigma > 0:
        lifted[:, 0] = gaussian_filter1d(lifted[:, 0], smooth_sigma)
        lifted[:, 1] = gaussian_filter1d(lifted[:, 1], smooth_sigma)

    transform_mat = get_transform_mat(lifted, original_coord)
    lifted_transformed = apply_transform_mat(lifted, transform_mat)
    ms = score_mismatch(lifted_transformed, original_coord)

    seg_scores = None
    if len_seg is not None:
        seg_scores = segmental_mismatch(lifted, original_coord, len_seg)

    return eps, ms, seg_scores


def sweep_epsilon_gardner(
    rat_name: str = "R",
    mod_name: str = "1",
    day_name: str = "day2",
    sess_name: str = "OF",
    epsilon_values: np.ndarray | None = None,
    save_dir: str | None = None,
    use_upgrade: bool = True,
    smooth_sigma: float = 10.0,
    segmental: bool = False,
    len_seg: int = 500,
    n_workers: int | None = None,
):
    """
    Sweep epsilon in toroidal lifting on Gardner data and record the mismatch score.

    Args:
        rat_name: Rat identifier ('R', 'Q', or 'S').
        mod_name: Grid module name ('1', '2', etc.).
        day_name: Recording day ('day1', 'day2', etc.).
        sess_name: Session name ('OF' for open field, etc.).
        epsilon_values: Array of epsilon values to sweep. Defaults to linspace(0.01, pi, 50).
        save_dir: Directory to save results. Defaults to revision/epsilon_sweep_gardner_results.
        use_upgrade: If True, use toroidal_lifting_distance_upgrade; otherwise use original.
        smooth_sigma: Sigma for Gaussian smoothing of lifted coordinates. Set to 0 to disable.
        segmental: If True, also compute segmental mismatch scores per epsilon.
        len_seg: Segment length for segmental mismatch. Only used when segmental=True.
        n_workers: Number of parallel workers. Defaults to 10.
    """
    if epsilon_values is None:
        epsilon_values = np.linspace(0.01, np.pi, 50)
    if n_workers is None:
        n_workers = 10

    method_tag = "upgrade" if use_upgrade else "original"
    session_tag = f"rat{rat_name}_mod{mod_name}_{day_name}_{sess_name}"
    print(f"Starting epsilon sweep ({method_tag}) for {session_tag} "
          f"with {len(epsilon_values)} epsilon values using {n_workers} workers.")

    if save_dir is None:
        save_dir = "revision/sweep_epsilon/epsilon_sweep_gardner_results"
    os.makedirs(save_dir, exist_ok=True)

    # --- Load Gardner data ---
    print("Loading Gardner data...")
    sspikes, xx, yy, _, _ = get_spikes(
        rat_name, mod_name, day_name, sess_name,
        bType="pure", bSmooth=True, bSpeed=True,
        folder=GARDNER_DATA_PATH,
    )
    original_coord = np.column_stack((xx, yy))
    print(f"Spike matrix shape: {sspikes.shape}")
    print(f"Original trajectory shape: {original_coord.shape}")

    # --- Extract toroidal coordinates (expensive, done once) ---
    print("Extracting toroidal coordinates (Gardner_coord)...")
    tor_coord, times, _, _ = Gardner_coord(sspikes)
    print(f"Toroidal coordinates shape: {tor_coord.shape}")

    print(f"Sweeping {len(epsilon_values)} epsilon values in [{epsilon_values[0]:.4f}, {epsilon_values[-1]:.4f}]")

    # --- Sweep with concurrent.futures ---
    seg_len_arg = len_seg if segmental else None
    worker_fn = partial(
        _eval_single_epsilon,
        cir_coords=tor_coord,
        original_coord=original_coord,
        use_upgrade=use_upgrade,
        smooth_sigma=smooth_sigma,
        len_seg=seg_len_arg,
    )

    mismatch_scores = np.zeros(len(epsilon_values))
    all_seg_scores = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker_fn, eps): i for i, eps in enumerate(epsilon_values)}
        for future in as_completed(futures):
            idx = futures[future]
            eps, ms, seg_scores = future.result()
            mismatch_scores[idx] = ms
            if segmental:
                all_seg_scores[idx] = seg_scores

    for eps, ms in zip(epsilon_values, mismatch_scores):
        print(f"  epsilon={eps:.4f}  mismatch={ms:.6f}")

    # --- Save results ---
    results = np.column_stack([epsilon_values, mismatch_scores])
    csv_path = os.path.join(save_dir, f"epsilon_sweep_{method_tag}_Gardner.csv")
    np.savetxt(csv_path, results, delimiter=",", header="epsilon,mismatch_score", comments="")
    print(f"\nResults saved to {csv_path}")

    if segmental:
        seg_means = np.array([np.mean(all_seg_scores[i]) for i in range(len(epsilon_values))])
        seg_stds = np.array([np.std(all_seg_scores[i]) for i in range(len(epsilon_values))])
        seg_results = np.column_stack([epsilon_values, seg_means, seg_stds])
        seg_csv_path = os.path.join(save_dir, f"epsilon_sweep_segmental_{method_tag}_Gardner.csv")
        np.savetxt(seg_csv_path, seg_results, delimiter=",",
                   header="epsilon,seg_mismatch_mean,seg_mismatch_std", comments="")
        print(f"Segmental results saved to {seg_csv_path}")

    # --- Plot: overall mismatch ---
    # fig, ax = plt.subplots(figsize=(8, 5))

    # color_ms = "#1f77b4"
    # ax.set_xlabel("Epsilon")
    # ax.set_ylabel("Mismatch Score", color=color_ms)
    # ax.plot(epsilon_values, mismatch_scores, "o-", color=color_ms, markersize=3, label="Overall")
    # ax.tick_params(axis="y", labelcolor=color_ms)

    # best_idx = np.argmin(mismatch_scores)
    # ax.axvline(epsilon_values[best_idx], color="red", linestyle=":", alpha=0.6)
    # ax.annotate(
    #     f"best eps={epsilon_values[best_idx]:.4f}\nmismatch={mismatch_scores[best_idx]:.6f}",
    #     xy=(epsilon_values[best_idx], mismatch_scores[best_idx]),
    #     xytext=(15, 15),
    #     textcoords="offset points",
    #     arrowprops=dict(arrowstyle="->", color="red"),
    #     fontsize=9,
    #     color="red",
    # )

    # plt.title(f"Epsilon Sweep ({method_tag}) — Gardner")
    # fig.tight_layout()

    # fig_path = os.path.join(save_dir, f"epsilon_sweep_{method_tag}_Gardner.png")
    # plt.savefig(fig_path, dpi=200)
    # print(f"Figure saved to {fig_path}")
    # plt.show()

    # # --- Plot: segmental mismatch with error bars ---
    # if segmental:
    #     fig_seg, ax_seg = plt.subplots(figsize=(8, 5))

    #     ax_seg.errorbar(epsilon_values, seg_means, yerr=seg_stds,
    #                     fmt="o-", color="#1f77b4", markersize=3, capsize=3,
    #                     label=f"Segmental mean ± std (len={len_seg})")
    #     ax_seg.set_xlabel("Epsilon")
    #     ax_seg.set_ylabel("Segmental Mismatch Score")

    #     best_seg_idx = np.argmin(seg_means)
    #     ax_seg.axvline(epsilon_values[best_seg_idx], color="red", linestyle=":", alpha=0.6)
    #     ax_seg.annotate(
    #         f"best eps={epsilon_values[best_seg_idx]:.4f}\nmismatch={seg_means[best_seg_idx]:.6f}",
    #         xy=(epsilon_values[best_seg_idx], seg_means[best_seg_idx]),
    #         xytext=(15, 15),
    #         textcoords="offset points",
    #         arrowprops=dict(arrowstyle="->", color="red"),
    #         fontsize=9,
    #         color="red",
    #     )

    #     ax_seg.legend()
    #     plt.title(f"Segmental Epsilon Sweep ({method_tag}) — Gardner")
    #     fig_seg.tight_layout()

    #     fig_seg_path = os.path.join(save_dir, f"epsilon_sweep_segmental_{method_tag}_Gardner.png")
    #     plt.savefig(fig_seg_path, dpi=200)
    #     print(f"Segmental figure saved to {fig_seg_path}")
    #     plt.show()

    return epsilon_values, mismatch_scores, all_seg_scores


if __name__ == "__main__":
#    epsilon_values = np.linspace(0.01, np.pi, 50)
    epsilon_values = np.linspace(0.01, 2 * np.pi, 20)

    print("=" * 60)
    print("Running with toroidal_lifting_distance_upgrade (overall)")
    print("=" * 60)
    sweep_epsilon_gardner(epsilon_values=epsilon_values, use_upgrade=True, segmental=False)

    print("\n" + "=" * 60)
    print("Running with toroidal_lifting_distance_upgrade (segmental)")
    print("=" * 60)
    sweep_epsilon_gardner(epsilon_values=epsilon_values, use_upgrade=True, segmental=True)
