# %%
# %% [markdown]
# # Dimension Analysis Experiment
# 
# This notebook investigates the retained dimension and eigenvalue decay for the Markov operators at each time slice.
# It also includes local linear regression residual analysis to identify non-harmonic intrinsic coordinates.

# %%
import sys
import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# --- Caching Infrastructure ---
CACHE_DIR = Path("../data/cache_dimension_analysis")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = CACHE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FORCE_RECOMPUTE = False

def load_or_compute(
    cache_name: str,
    compute_fn: Callable[[], Any],
    force: bool = False
) -> Any:
    """Load result from cache if available, otherwise compute and save."""
    cache_path = CACHE_DIR / f"{cache_name}.pkl"
    if not force and not FORCE_RECOMPUTE and cache_path.exists():
        print(f"Loading {cache_name} from cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cache {cache_name}: {e}. Recomputing...")
    
    print(f"Computing {cache_name}...")
    result = compute_fn()
    
    print(f"Saving {cache_name} to cache: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result

# Make repository root importable when executing from the notebooks directory
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
except NameError:
    import os
    sys.path.append(os.path.abspath(".."))

from diffmap.diffusion_maps import (
    select_optimal_bandwidth,
    select_non_harmonic_coordinates,
    _time_slice_markov,
)
from tran_inclusions.data_prep import load_tran_inclusions_data, compute_bandwidth_statistics
from sklearn.utils.extmath import randomized_svd

# %%
def format_for_paper() -> None:
    """Apply publication-style defaults for matplotlib figures."""
    plt.rcParams.update({"image.cmap": "viridis"})
    plt.rcParams.update(
        {
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "Bitstream Vera Serif",
                "Computer Modern Roman",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "Utopia",
                "ITC Bookman",
                "Bookman",
                "Nimbus Roman No9 L",
                "Palatino",
                "Charter",
                "serif",
            ]
        }
    )
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"mathtext.fontset": "custom"})
    plt.rcParams.update({"mathtext.rm": "serif"})
    plt.rcParams.update({"mathtext.it": "serif:italic"})
    plt.rcParams.update({"mathtext.bf": "serif:bold"})
    plt.close("all")

format_for_paper()

# %%
# Parameters
data_path = Path("../data/tran_inclusions.npz")
epsilon_selection_mode = 'first_local_minimum'
max_eigenvectors = 256  # Number of eigenvectors to compute and analyze
LLR_RESIDUAL_THRESHOLD = 0.1
LLR_SGE_MAX_SEARCHES = 12  # None for exact per-k SGE; lower values interpolate between anchors
LLR_SGE_INTERPOLATION = "log_pchip"

# %%
(
    times_arr,
    held_out_indices,
    held_out_times,
    all_frames,
    components,
    mean_vec,
    explained_variance,
    is_whitened,
    whitening_epsilon,
    resolution,
    raw_marginals,
    held_out_marginals,
) = load_tran_inclusions_data(data_path)

print(f"Loaded data: {all_frames.shape} (Time, Samples, PCA_Comps)")
print(f"Time points: {times_arr}")

# %% [markdown]
# ## Dimension Analysis Loop

# %%
def compute_dimension_analysis():
    results = []
    n_times = all_frames.shape[0]
    
    # Compute base epsilons using the same logic as tran_inclusions_experiment
    bandwidth_stats = compute_bandwidth_statistics(all_frames)
    base_epsilons = bandwidth_stats['median'] # Array of medians per frame
    
    for idx in range(n_times):
        print(f"Processing time slice {idx}/{n_times} (t={times_arr[idx]:.3f})...")
        coords = all_frames[idx]
        
        # 1. Select optimal bandwidth
        # Replicate grid from tran_inclusions/tc_embeddings.py:
        # epsilon_scales = np.geomspace(0.01, 0.2, num=32)
        # candidates = base_epsilon * epsilon_scales
        
        base_eps = base_epsilons[idx]
        scales = np.geomspace(0.01, 0.2, num=32)
        candidates = base_eps * scales
        
        eps_k, score_k, _, _ = select_optimal_bandwidth(
            coords,
            candidate_epsilons=candidates,
            alpha=1.0,
            epsilon_scaling=4.0,
            selection=epsilon_selection_mode,
            return_all=True,
        )
        
        # 2. Compute Diffusion Map (Eigenvalues/Vectors)
        distances2 = squareform(pdist(coords, metric="sqeuclidean"))
        
        # Re-implementing symmetric normalization to get eigenvectors:
        kernel = np.exp(-distances2 / (4.0 * eps_k))
        np.fill_diagonal(kernel, 0.0)
        
        # Alpha normalization (alpha=1.0)
        degrees = kernel.sum(axis=1)
        inv_deg = np.power(degrees, -1.0)
        kernel_alpha = (inv_deg[:, None] * kernel) * inv_deg[None, :]
        
        # Symmetric normalization
        d_alpha = kernel_alpha.sum(axis=1)
        inv_sqrt_d = np.power(d_alpha, -0.5)
        P_sym = (inv_sqrt_d[:, None] * kernel_alpha) * inv_sqrt_d[None, :]
        
        # Eigendecomposition using Randomized SVD
        # P_sym is symmetric, so SVD U S V^T gives U=V and S=eigenvalues (abs).
        # Since P_sym is positive semi-definite (mostly), eigenvalues are S.
        # We need top k eigenvectors.
        
        n_eigs = min(max_eigenvectors, P_sym.shape[0])
        U, S, Vt = randomized_svd(P_sym, n_components=n_eigs, random_state=42)
        
        eigvals = S
        eigvecs = U
        
        # Convert back to diffusion coordinates (right eigenvectors of P)
        # psi_k = D^{-1/2} * v_k
        psi = inv_sqrt_d[:, None] * eigvecs
        
        diff_coords = psi * eigvals[None, :]
        
        # 3. Local Linear Regression Residuals
        # Subsample for speed
        n_sub = min(512, diff_coords.shape[0])
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(diff_coords.shape[0], size=n_sub, replace=False)
        diff_coords_sub = diff_coords[sub_idx]
        
        # Note on two kernel scales (per Dsilva et al. 2018):
        # - eps_k: diffusion map kernel scale (used to compute eigenvectors)
        # - LLR kernel scale: used for local linear regression in eigenspace
        # 
        # Option 1: Use semigroup selection for LLR (default, selects scale appropriate for eigenspace)
        # Option 2: Use same scale as diffusion map: llr_kernel_scale=4.0*eps_k
        #
        # Here we use the default semigroup selection for the LLR kernel,
        # which is appropriate when eigenspace geometry differs from ambient space.
        intrinsic, mask, residuals = select_non_harmonic_coordinates(
            eigvals,
            diff_coords_sub,
            residual_threshold=LLR_RESIDUAL_THRESHOLD,
            min_coordinates=2,
            llr_bandwidth_strategy='semigroup',
            llr_semigroup_selection='first_local_minimum',
            llr_semigroup_max_searches=LLR_SGE_MAX_SEARCHES,
            llr_semigroup_interpolation=LLR_SGE_INTERPOLATION,
            # To use the same scale as diffusion map, uncomment:
            # llr_kernel_scale=4.0 * eps_k,
        )
        
        results.append({
            "time": times_arr[idx],
            "epsilon": eps_k,
            "eigenvalues": eigvals,
            "residuals": residuals,
            "mask": mask,
        })
        
    return results

analysis_results = load_or_compute(
    "dimension_analysis",
    compute_dimension_analysis,
    force=True
)

# %%
def save_dimension_analysis_results(
    results,
    output_dir: Path,
    *,
    residual_threshold: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_rows = []
    for res in results:
        time_val = float(res["time"])
        eps_val = float(res["epsilon"])
        evals = np.asarray(res["eigenvalues"])
        residuals = np.asarray(res["residuals"])
        mask = res.get("mask")
        mask_arr = np.asarray(mask) if mask is not None else None
        num_unique = int(mask_arr.sum()) if mask_arr is not None else int((residuals >= residual_threshold).sum())

        summary_rows.append(
            {
                "time": time_val,
                "epsilon": eps_val,
                "num_eigenvectors": int(evals.shape[0]),
                "num_unique": num_unique,
            }
        )

        for idx, (eig, resid) in enumerate(zip(evals, residuals)):
            rows.append(
                {
                    "time": time_val,
                    "epsilon": eps_val,
                    "index": int(idx),
                    "eigenvalue": float(eig),
                    "abs_eigenvalue": float(abs(eig)),
                    "residual": float(resid),
                    "is_unique": bool(mask_arr[idx]) if mask_arr is not None else None,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "dimension_analysis_long.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "dimension_analysis_summary.csv", index=False)

    if metadata is None:
        metadata = {}
    meta_payload = dict(metadata)
    meta_payload["residual_threshold"] = float(residual_threshold)
    with open(output_dir / "dimension_analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, sort_keys=True)

# %% [markdown]
# ## Visualization

# %%
def plot_dimension_analysis(
    results,
    *,
    residual_threshold: float,
    output_dir: Optional[Path] = None,
    show: bool = True,
):
    n_times = len(results)
    n_cols = 2
    n_rows = int(np.ceil(n_times / n_cols))
    
    # Plot Eigenvalue Decay
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
    fig.suptitle("Eigenvalue Decay per Time Slice", fontsize=16)
    
    for idx, res in enumerate(results):
        ax = axes.flat[idx] if n_times > 1 else axes
        evals = res["eigenvalues"]
        # Plot log eigenvalues (excluding the first trivial one usually 1.0)
        # But let's plot all to see the gap.
        ax.plot(np.arange(len(evals)), np.abs(evals), 'o-', markersize=3)
        ax.set_yscale('log')
        ax.set_title(f"t={res['time']:.3f}, eps={res['epsilon']:.2e}")
        ax.set_xlabel("Index")
        ax.set_ylabel("|Eigenvalue|")
        ax.grid(True, alpha=0.3)
    if output_dir is not None:
        fig.savefig(output_dir / "eigenvalue_decay.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Plot LLR Residuals
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
    fig.suptitle("Local Linear Regression Residuals per Time Slice", fontsize=16)
    
    for idx, res in enumerate(results):
        ax = axes.flat[idx] if n_times > 1 else axes
        residuals = res["residuals"]
        # Residuals are for indices 1..k (index 0 is trivial constant)
        # residuals array size matches eigenvectors. 
        # Index 0 residual is usually 1.0 or 0.0 depending on implementation, but usually we care about 1+.
        
        ax.plot(np.arange(len(residuals)), residuals, 's-', color='orange', markersize=3)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"t={res['time']:.3f}")
        ax.set_xlabel("Index")
        ax.set_ylabel("LLR Residual")
        ax.grid(True, alpha=0.3)
        
        # Highlight high scores
        high_score_indices = np.where(residuals > residual_threshold)[0]
        ax.plot(
            high_score_indices,
            residuals[high_score_indices],
            'rx',
            markersize=8,
            label=f"> {residual_threshold:.2f}",
        )
        ax.legend()
        
    if output_dir is not None:
        fig.savefig(output_dir / "llr_residuals.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Combined Plot for a few snapshots (e.g. first, middle, last)
    indices_to_plot = [0, n_times // 2, n_times - 1]
    fig, axes = plt.subplots(len(indices_to_plot), 2, figsize=(12, 4 * len(indices_to_plot)), constrained_layout=True)
    fig.suptitle("Combined Analysis: Eigenvalues vs Residuals", fontsize=16)
    
    for i, idx in enumerate(indices_to_plot):
        res = results[idx]
        evals = res["eigenvalues"]
        residuals = res["residuals"]
        
        # Eigenvalues
        ax_eig = axes[i, 0]
        ax_eig.plot(np.arange(len(evals)), np.abs(evals), 'o-', markersize=4)
        ax_eig.set_yscale('log')
        ax_eig.set_title(f"t={res['time']:.3f}: Eigenvalues")
        ax_eig.set_xlabel("Index")
        ax_eig.set_ylabel("|Eigenvalue|")
        ax_eig.grid(True, alpha=0.3)
        
        # Residuals
        ax_res = axes[i, 1]
        ax_res.plot(np.arange(len(residuals)), residuals, 's-', color='orange', markersize=4)
        ax_res.set_ylim(0, 1.1)
        ax_res.set_title(f"t={res['time']:.3f}: LLR Residuals")
        ax_res.set_xlabel("Index")
        ax_res.set_ylabel("Residual")
        ax_res.grid(True, alpha=0.3)
        
        # Annotate peaks
        peaks = np.where(residuals > residual_threshold)[0]
        for p in peaks:
            ax_res.annotate(f"{p}", (p, residuals[p]), xytext=(0, 5), textcoords='offset points', ha='center')

    if output_dir is not None:
        fig.savefig(output_dir / "combined_analysis.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

save_dimension_analysis_results(
    analysis_results,
    OUTPUT_DIR,
    residual_threshold=LLR_RESIDUAL_THRESHOLD,
    metadata={
        "epsilon_selection_mode": epsilon_selection_mode,
        "max_eigenvectors": max_eigenvectors,
        "llr_residual_threshold": LLR_RESIDUAL_THRESHOLD,
    },
)
plot_dimension_analysis(
    analysis_results,
    residual_threshold=LLR_RESIDUAL_THRESHOLD,
    output_dir=OUTPUT_DIR,
    show=True,
)


# %%
