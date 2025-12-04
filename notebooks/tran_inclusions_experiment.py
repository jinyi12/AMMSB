# %%
# %% [markdown]
# Refactored driver for the tran inclusions experiment.
# 
# This script wires together modular components from ``notebooks/tran_inclusions/``.
# The original ``explore_tran_inclusions_pseudo.py`` remains as a backup reference.
# """
# 
# # %% [markdown]
# # # Tran Inclusions Experiment
# # 
# # This notebook drives the experiment for lifting transport inclusions using spatio-temporal geometric harmonics on TCDM embeddings.

# %%
import sys
import os
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Caching Infrastructure ---
CACHE_DIR = Path("../data/cache_tran_inclusions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FORCE_RECOMPUTE = False  # Set to True to invalidate all caches

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


from diffmap.lifting import compute_lift_metrics, print_metric_table
from mmsfm.viz import plot_field_comparisons, plot_error_statistics
from mmsfm.data_utils import pca_decode, to_images


from tran_inclusions.config import LiftingConfig, PseudoDataConfig  # type: ignore  # noqa: E402
from tran_inclusions.data_prep import (  # type: ignore  # noqa: E402
    compute_bandwidth_statistics,
    load_tran_inclusions_data,
)
from tran_inclusions.tc_embeddings import build_time_coupled_embeddings  # type: ignore  # noqa: E402
from tran_inclusions.pseudo_data import choose_pseudo_times_per_interval  # type: ignore  # noqa: E402
from tran_inclusions.interpolation import build_dense_latent_trajectories, sample_latent_at_times  # type: ignore  # noqa: E402
from tran_inclusions.lifting import fit_lifting_models, lift_pseudo_latents  # type: ignore  # noqa: E402
from tran_inclusions.metrics import evaluate_interpolation_at_observed_times  # type: ignore  # noqa: E402
from tran_inclusions.viz import visualize_fractional_pca_intervals, plot_latent_trajectories_comparison  # type: ignore  # noqa: E402

# %%
# Parameters
data_path = Path("../data/tran_inclusions.npz")
epsilon_selection_mode = 'first_local_minimum'
gh_max_modes = 300

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
# ## Pseudo-Time Selection

# %%
tc_k = 12
alpha = 1.0
beta = -0.2
use_variable_bandwidth = False

pseudo_config = PseudoDataConfig(
    n_dense=20,
    n_pseudo_per_interval=3,
    eta_fuse=0.5,
    alpha_grid=None,
)

pseudo_times, alphas_per_interval = choose_pseudo_times_per_interval(
    times_arr,
    n_per_interval=pseudo_config.n_pseudo_per_interval,
    include_endpoints=False,
    alpha_grid=pseudo_config.alpha_grid,
)
print(f"Planned pseudo times (raw union): {pseudo_times}")

# %% [markdown]
# ## Time-Coupled Embeddings

# %%
# %%
def compute_tc_embeddings():
    bandwidth_stats = compute_bandwidth_statistics(all_frames)
    base_epsilons = bandwidth_stats['median']
    epsilon_scales = np.geomspace(0.01, 0.2, num=32)
    semigroup_sample_size = min(1024, all_frames.shape[1])
    semigroup_rng_seed = 0

    return build_time_coupled_embeddings(
        all_frames=all_frames,
        times_arr=times_arr,
        tc_k=tc_k,
        alpha=alpha,
        beta=beta,
        use_variable_bandwidth=use_variable_bandwidth,
        base_epsilons=base_epsilons,
        scales=epsilon_scales,
        sample_size=semigroup_sample_size,
        rng_seed=semigroup_rng_seed,
        epsilon_selection=epsilon_selection_mode,
    )

tc_result, tc_embeddings_time, selected_epsilons, kde_bandwidths, semigroup_df = load_or_compute(
    "tc_embeddings",
    compute_tc_embeddings
)

# %% [markdown]
# ## Pseudo-Data Generation (Augmentation)

# %%
# --- Semigroup bandwidth diagnostics ---
from IPython.display import display

if semigroup_df is None or len(getattr(semigroup_df, "index", [])) == 0:
    print("Semigroup diagnostics unavailable.")
else:
    semigroup_best = (
        semigroup_df.sort_values("semigroup_error")
        .groupby("time_index")
        .first()
        .reset_index(drop=True)
    )
    semigroup_best = semigroup_best[
        ["time", "epsilon", "semigroup_error", "kde_bandwidth"]
    ]

    # Elbow summary (if available)
    if "epsilon_elbow" in semigroup_df.columns:
        elbow_rows = []
        for idx in sorted(semigroup_df["time_index"].unique()):
            subset = semigroup_df[semigroup_df["time_index"] == idx]
            eps_elb = float(subset["epsilon_elbow"].iloc[0])
            row = subset.loc[(subset["epsilon"].sub(eps_elb)).abs().idxmin()]
            elbow_rows.append(
                {
                    "time": float(row["time"]),
                    "epsilon_elbow": eps_elb,
                    "semigroup_error_elbow": float(row["semigroup_error"]),
                }
            )
        elbow_df = pd.DataFrame(elbow_rows)
        summary = elbow_df.merge(
            semigroup_best.rename(
                columns={
                    "epsilon": "epsilon_min",
                    "semigroup_error": "semigroup_error_min",
                }
            ),
            on="time",
            how="left",
        )
        print("Elbow-selected epsilons (with min-SGE reference):")
        display(summary)
    else:
        print("Semigroup-selected epsilons (min SGE):")
        display(semigroup_best)

    unique_times = semigroup_df["time_index"].unique()
    n_rows = int(np.ceil(len(unique_times) / 3)) if len(unique_times) else 1
    n_cols = min(3, max(1, len(unique_times)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, idx in zip(axes.flat, unique_times):
        subset = semigroup_df[semigroup_df["time_index"] == idx].sort_values(
            "epsilon"
        )
        ax.plot(subset["epsilon"], subset["semigroup_error"], "o-", label="SGE(eps)")
        
        if 'selected_epsilons' in locals() and selected_epsilons is not None:
            selected_eps = selected_epsilons[int(idx)]
        elif "epsilon_elbow" in subset.columns and epsilon_selection_mode == 'elbow':
            selected_eps = subset["epsilon_elbow"].iloc[0]
        else:
            best_row = subset.loc[subset["semigroup_error"].idxmin()]
            selected_eps = best_row["epsilon"]

        ax.axvline(
            selected_eps,
            color="r",
            linestyle="--",
            label="selected eps",
        )
        ax.set_xscale("log")
        ax.set_title(f"t={subset['time'].iloc[0]:.2f}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("semigroup error")
        ax.grid(alpha=0.3)
    if axes.size:
        axes.flat[0].legend()
    plt.show()

# %%
# %%
def compute_pseudo_data():
    entries = []
    for idx, t_val in enumerate(times_arr):
        entries.append(
            {
                "time": float(t_val),
                "kind": "observed",
                "interval_index": idx,
                "alpha": 0.0,
                "source_time": float(t_val),
                "target_time": float(t_val),
                "eta": pseudo_config.eta_fuse,
                "aug_index": idx,
            }
        )
    meta = {
        "entries": entries,
        "alphas_per_interval": alphas_per_interval,
        "eta": pseudo_config.eta_fuse,
    }
    return times_arr, all_frames, meta

times_aug, frames_aug, pseudo_meta = load_or_compute(
    "pseudo_data",
    compute_pseudo_data
)
pseudo_entries = [entry for entry in pseudo_meta['entries'] if entry['kind'] == 'pseudo']
print(
    f"Augmented dataset: {len(times_aug)} scales "
    f"({len(times_arr)} observed, {len(pseudo_entries)} pseudo)."
)

# %% [markdown]
# ## Latent Interpolation

# %%
# %%
def compute_interpolation():
    return build_dense_latent_trajectories(
        tc_result,
        times_train=times_arr,
        tc_embeddings_time=tc_embeddings_time,
        n_dense=pseudo_config.n_dense,
        frechet_mode=pseudo_config.frechet_mode,
    )

interpolation = load_or_compute(
    "interpolation",
    compute_interpolation
)

# 2. CRITICAL STEP: PERMANENTLY update the reference embeddings to the aligned versions
if interpolation.tc_embeddings_aligned is not None:
    print("Updating tc_embeddings_time to aligned embeddings for visualization and downstream tasks.")
    tc_embeddings_time = interpolation.tc_embeddings_aligned
else:
    print("WARNING: Aligned embeddings not found in interpolation result. Using original embeddings.")

# %% [markdown]
# ## Lifting Model Training

# %%
# Compute global Frechet mean interpolation for comparison
interpolation_frechet_global = build_dense_latent_trajectories(
    tc_result,
    times_train=times_arr,
    tc_embeddings_time=tc_embeddings_time,
    n_dense=pseudo_config.n_dense,
    frechet_mode='global',
)

# Visualize the latent trajectories for a few samples
sample_indices_to_plot = list(range(pseudo_config.n_samples_vis))
plot_latent_trajectories_comparison(
    times_train=times_arr,
    tc_embeddings_time=tc_embeddings_time,
    interpolation_triplet=interpolation,
    interpolation_global=interpolation_frechet_global,
    sample_indices=sample_indices_to_plot,
)

# %%
# %%
lifting_config = LiftingConfig(
    holdout_time=times_arr[1] if len(times_arr) > 1 else times_arr[0],
    gh_max_modes=gh_max_modes,
    gh_energy_threshold=0.99,
    gh_epsilon_grid_size=32,
    gh_ridge=1e-4,
    use_continuous_gh=True,
)

def compute_lifting_models():
    return fit_lifting_models(
        tc_embeddings_time,
        all_frames,
        times_arr,
        lifting_config,
        trajectory=tc_result,
    )

models, lifting_metadata = load_or_compute(
    "lifting_models",
    compute_lifting_models,
    force=True,
)

# %% [markdown]
# ## Semigroup Error Diagnostics for Continuous GH

# %%
def plot_cgh_semigroup_errors():
    cgh_model = getattr(models, "continuous_gh", None)
    if cgh_model is None or getattr(cgh_model, "_result", None) is None:
        print("Continuous GH model unavailable; skipping semigroup error plot.")
        return
    res = cgh_model._result
    if not hasattr(res, "semigroup_errors"):
        print("Semigroup errors not recorded; skipping plot.")
        return

    eps_grid = getattr(res, "epsilon_grid", None)
    sge_grid = getattr(res, "semigroup_error_grid", None)
    if eps_grid is not None and sge_grid is not None:
        eps_grid = np.asarray(eps_grid)
        sge_grid = np.asarray(sge_grid)
        n_times = min(eps_grid.shape[0], sge_grid.shape[0])
        if n_times:
            n_rows = int(np.ceil(n_times / 3))
            n_cols = min(3, n_times)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4 * n_cols, 3.2 * n_rows),
                squeeze=False,
                constrained_layout=True,
            )
            for ax, idx in zip(axes.flat, range(n_times)):
                ax.plot(eps_grid[idx], sge_grid[idx], "o-", label="SGE(eps)")
                ax.axvline(
                    res.epsilons[idx],
                    color="r",
                    linestyle="--",
                    label="selected eps",
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"t={res.times[idx]:.2f}")
                ax.set_xlabel("epsilon")
                ax.set_ylabel("semigroup error")
                ax.grid(alpha=0.3)
            if axes.size:
                axes.flat[0].legend()
            plt.show()
        else:
            print("Semigroup error grid appears empty; skipping epsilon sweep plots.")
    else:
        print("Semigroup error grid unavailable; skipping epsilon sweep plots.")

    plt.figure(figsize=(6, 4))
    plt.plot(res.times, res.semigroup_errors, "o-", label="SGE (selected eps)")
    plt.axhline(getattr(cgh_model, "semigroup_tolerance", 0.1), color="r", linestyle="--", label="tolerance")
    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel("semigroup error")
    plt.title("Continuous GH semigroup error per snapshot")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

plot_cgh_semigroup_errors()

# %% [markdown]
# ## Evaluation

# %%
metrics = evaluate_interpolation_at_observed_times(
    tc_embeddings_time=tc_embeddings_time,
    all_frames=all_frames,
    times_arr=times_arr,
    interpolation=interpolation,
    models=models,
    lifting_metadata=lifting_metadata,
    config=lifting_config,
    components=components,
    mean_vec=mean_vec,
    explained_variance=explained_variance,
    is_whitened=is_whitened,
    whitening_epsilon=whitening_epsilon,
    resolution=resolution,
)
print("Evaluation metrics:", metrics)



# %%
# reload mmsfm.viz (py file) modules to ensure latest versions are used
import importlib
import mmsfm.viz
importlib.reload
import importlib
import mmsfm.viz
importlib.reload(mmsfm.viz)
from mmsfm.viz import plot_field_comparisons, plot_error_statistics
print("Reloaded mmsfm.viz")

# --- Step 6: Lift held-out times and compare against ground truth ---
if held_out_times:
    eval_pairs: list[tuple[float, np.ndarray]] = []
    for t in held_out_times:
        matched: Optional[tuple[float, np.ndarray]] = None
        for t_key, frame in held_out_marginals.items():
            if abs(float(t_key) - float(t)) <= lifting_config.time_match_tol:
                matched = (float(t_key), frame)
                break
        if matched is None:
            print(f"No PCA marginal found for held-out time {t:.3f}; skipping.")
            continue
        eval_pairs.append(matched)

    if eval_pairs:
        eval_times = np.array([t for t, _ in eval_pairs], dtype=np.float64)
        eval_times, phi_eval = sample_latent_at_times(
            interpolation,
            eval_times,
            method=pseudo_config.interp_method,
        )
        pseudo_micro_eval = lift_pseudo_latents(
            phi_eval,
            eval_times,
            models,
            tc_embeddings_time,
            all_frames,
            times_arr,
            config=lifting_config,
            lifting_metadata=lifting_metadata,
            training_interpolation=interpolation,
        )
        sample_indices = tuple(idx for idx in lifting_config.plot_samples if idx < all_frames.shape[1])
        if not sample_indices:
            sample_indices = (0,)

        stacked_truth = []
        stacked_g = []
        stacked_preds: Dict[str, List[np.ndarray]] = {name: [] for name in pseudo_micro_eval}

        for i, (t_eval, X_true_pca) in enumerate(eval_pairs):
            print(f"--- Hold-out t={t_eval:.3f} ---")
            X_true_flat = pca_decode(
                X_true_pca,
                components,
                mean_vec,
                explained_variance,
                is_whitened,
                whitening_epsilon,
            )
            imgs_true = to_images(X_true_flat, resolution)

            preds_flat = {
                name: pca_decode(
                    pseudo_micro_eval[name][i],
                    components,
                    mean_vec,
                    explained_variance,
                    is_whitened,
                    whitening_epsilon,
                )
                for name in pseudo_micro_eval
            }
            imgs_preds = {name: to_images(arr, resolution) for name, arr in preds_flat.items()}

            lift_metrics = compute_lift_metrics(
                X_true_flat,
                None,
                None,
                preds_flat["convex"],
                None,
                X_cgh=preds_flat.get("cgh"),
            )
            print_metric_table(lift_metrics)
            plot_field_comparisons(
                imgs_true,
                imgs_preds.get("cgh", imgs_preds["convex"]),
                None,
                imgs_preds["convex"],
                sample_indices=sample_indices,
                imgs_krr=None,
                vmax_mode=lifting_config.vmax_mode,
            )
            plot_error_statistics(lift_metrics, phi_eval[i])

            stacked_truth.append(X_true_flat)
            stacked_g.append(phi_eval[i])
            for name, arr in preds_flat.items():
                stacked_preds[name].append(arr)

        if stacked_truth:
            X_true_all = np.vstack(stacked_truth)
            agg_preds = {name: np.vstack(arrs) for name, arrs in stacked_preds.items() if arrs}
            agg_metrics = compute_lift_metrics(
                X_true_all,
                None,
                None,
                agg_preds["convex"],
                None,
                X_cgh=agg_preds.get("cgh"),
            )
            print("Aggregate hold-out lift metrics across times:")
            print_metric_table(agg_metrics)
            plot_error_statistics(agg_metrics, np.vstack(stacked_g))
    else:
        print("Held-out times provided but no matching PCA marginals; skipping lift comparison.")
else:
    print("No held-out times provided; skipping lift comparison.")

# %% [markdown]
# ## Visualization of Reconstructed Fields
#  
# ## Visualize the fractional PCA intervals and reconstructed fields.

# %%
visualize_fractional_pca_intervals(
    all_frames=all_frames,
    times_arr=times_arr,
    frames_aug=frames_aug,
    pseudo_meta=pseudo_meta,
    components=components,
    mean_vec=mean_vec,
    explained_variance=explained_variance,
    is_whitened=is_whitened,
    whitening_epsilon=whitening_epsilon,
    resolution=resolution,
    sample_indices=tuple(range(min(pseudo_config.n_samples_fields, all_frames.shape[1]))),
)



