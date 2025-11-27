# %%
# %% [markdown]
# # Tran et al. inclusions: micro → meso → macro

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Dict, Tuple, List, Any
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path.cwd().parent))

from diffmap.lifting import (
    build_training_pairs,
    build_time_augmented_training_pairs,
    compute_lift_metrics,
    lift_with_convex_hull,
    lift_with_geometric_harmonics,
    lift_with_time_local_kernel_ridge,
    print_metric_table,
)
from mmsfm.data_utils import (
    find_holdout_index,
    get_marginals,
    invert_pca,
    pca_decode,
    split_train_holdout_marginals,
    to_images,
)
from mmsfm.viz import format_for_paper, plot_error_statistics, plot_field_comparisons

import scipy.linalg
from scipy.spatial.distance import pdist, squareform

data_path = Path("../data/tran_inclusions.npz")
npz = np.load(data_path)
resolution = int(np.sqrt(int(npz["data_dim"])))

# Debug: print all available keys
print("Available keys in npz:", list(npz.files))

# Extract available times from marginals (not raw_marginal_)
time_keys = [k for k in npz.files if k.startswith("marginal_")]
times = sorted(float(k.replace("marginal_", "")) for k in time_keys)
print("Times:", times)

format_for_paper()

# %%
# Debug: show available keys
print("Available keys in npz:", list(npz.files))

# Acquire marginals (raw preferred; PCA otherwise)
times, marginals, mode = get_marginals(npz, choice="pca")
_, raw_marginals, _ = get_marginals(npz, choice="raw")
held_out_indices = npz.get("held_out_indices")
if held_out_indices is not None:
    held_out_indices = np.asarray(held_out_indices, dtype=int)
else:
    held_out_indices = np.array([], dtype=int)

train_times, train_marginals, held_out_times, held_out_marginals, used_held_out_indices = split_train_holdout_marginals(
    times,
    marginals,
    held_out_indices,
)
times, marginals = train_times, train_marginals
print("Using mode:", mode, "Found times:", times)
if held_out_times:
    print(
        f"Held-out times at indices {used_held_out_indices}: {held_out_times} (excluded from training)"
    )

# Load PCA metadata (used only when mode == 'pca')
components = npz.get("pca_components")
mean_vec = npz.get("pca_mean")
explained_variance = npz.get("pca_explained_variance")
is_whitened = bool(npz.get("is_whitened", False))
whitening_epsilon = float(npz.get("whitening_epsilon", 0.0))

# Visualise one sample across times
sample_idx = 0
ncols = len(times)
if ncols == 0:
    raise RuntimeError("No times found to plot. Check the npz keys printed above.")

fig, axes = plt.subplots(1, max(1, ncols), figsize=(4 * max(1, ncols), 4))
if ncols == 1:
    axes = [axes]

for ax, (i,t) in zip(axes, enumerate(times)):
    data = marginals[t]
    if sample_idx >= data.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range for time {t} (len={data.shape[0]})")

    if mode == "pca":
        if i == 0:
            field = raw_marginals[i][sample_idx].reshape(resolution, resolution)
        else:
            recon = invert_pca(data, components, mean_vec, explained_variance, is_whitened, whitening_epsilon)
            field = recon[sample_idx].reshape(resolution, resolution)
    else:
        field = data[sample_idx].reshape(resolution, resolution)

    im = ax.imshow(field, cmap="viridis")
    ax.set_title(f"t={t}")
    ax.axis("off")

fig.colorbar(im, ax=axes[:ncols], shrink=0.8)
plt.show()

# %% [markdown]
# ## Data Loading & Preparation
# Convert the PCA marginal dictionary into a time-major tensor for downstream analysis.

# %%
# Assemble PCA snapshots into a tensor (time, samples, components)
times_arr = np.array(times, dtype=np.float64)
all_frames = np.stack([marginals[t] for t in times_arr])
# exclude first time since it is for visualization only
all_frames = all_frames[1:]
times_arr = times_arr[1:]
n_times, n_samples, n_components = all_frames.shape

print(f"All frames tensor shape: {all_frames.shape} (time x samples x components)")
print(f"Unique times: {times_arr}")
print(f"Number of PCA components: {n_components}")

# %% [markdown]
# ## PCA Visualization
# Inspect how the first two PCA coefficients evolve over time before constructing diffusion maps.

# %%
rng = np.random.default_rng(0)
subset = min(2000, n_samples)
if subset < n_samples:
    sample_idx_subset = rng.choice(n_samples, size=subset, replace=False)
else:
    sample_idx_subset = np.arange(n_samples)

fig, axes = plt.subplots(
    1, n_times, figsize=(4 * n_times, 4), sharex=True, sharey=True, constrained_layout=True
)
if n_times == 1:
    axes = [axes]

# color by sample id (consistent across panels)
sample_ids = np.arange(len(sample_idx_subset))
for i, ax in enumerate(axes):
    coords = all_frames[i, sample_idx_subset, :]
    sc = ax.scatter(coords[:,0], coords[:,1], c=sample_ids,
                    cmap='viridis', s=10, alpha=0.8)
fig.colorbar(sc, ax=axes, label='sample index', fraction=0.046, pad=0.04)

plt.show()


# %% [markdown]
# ## Diffusion Hyperparameters
# Set the parameters for the variable-bandwidth time-coupled diffusion map experiment.

# %%
tc_k = 8
tc_alpha = 0.5
tc_beta = -0.2
use_variable_bandwidth = True

print(f"Diffusion config → k={tc_k}, alpha={tc_alpha}, beta={tc_beta}, variable_bandwidth={use_variable_bandwidth}")


# %% [markdown]
# ## Bandwidth Analysis
# Tune per-time diffusion bandwidths by targeting a fixed effective neighbour count.

# %%

from diffmap.diffusion_maps import (
    select_epsilons_by_connectivity,
    time_coupled_diffusion_map,
    build_time_coupled_trajectory,
    fit_coordinate_splines,
    evaluate_coordinate_splines,
    fit_geometric_harmonics,
    geometric_harmonics_lift,
    geometric_harmonics_lift_local,
    ConvexHullInterpolator,
    GeometricHarmonicsModel,
)


def compute_bandwidth_statistics(frames: np.ndarray) -> dict[str, np.ndarray]:
    medians, q1, q3, maxima = [], [], [], []
    for snapshot in frames:
        d2 = squareform(pdist(snapshot, metric='sqeuclidean'))
        mask = d2 > 0
        if np.any(mask):
            vals = d2[mask]
        else:
            vals = np.array([1.0])
        medians.append(float(np.median(vals)))
        q1.append(float(np.percentile(vals, 25)))
        q3.append(float(np.percentile(vals, 75)))
        maxima.append(float(np.max(vals)))
    return {
        'median': np.array(medians),
        'q1': np.array(q1),
        'q3': np.array(q3),
        'max': np.array(maxima),
    }


bandwidth_stats = compute_bandwidth_statistics(all_frames)
base_epsilons = bandwidth_stats['median']
print('Base epsilon estimates (median squared distances):')
for idx, eps in enumerate(base_epsilons):
    print(f"  t={times_arr[idx]:.2f}: median={eps:.3e}, IQR=({bandwidth_stats['q1'][idx]:.3e}, {bandwidth_stats['q3'][idx]:.3e})")

epsilon_scales = np.geomspace(0.1, 4.0, num=32)
sample_size = min(1024, n_samples)
selected_epsilons, kde_bandwidths, connectivity_df = select_epsilons_by_connectivity(
    all_frames,
    times_arr,
    base_epsilons=base_epsilons,
    scales=epsilon_scales,
    alpha=tc_alpha,
    target_neighbors=64.0,
    sample_size=sample_size,
    rng_seed=0,
    variable_bandwidth=use_variable_bandwidth,
    beta=tc_beta,
)

print('Chosen epsilons by connectivity:')
print(selected_epsilons)
print('KDE bandwidths used:', kde_bandwidths)
connectivity_df.head()


# %%
import pandas as pd

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
ax = axes[0]
ax.plot(times_arr, base_epsilons, 'k--', label='median $d^2$')
ax.plot(times_arr, selected_epsilons, 'o-', label='connectivity-tuned $\epsilon_t$')
ax.set(xlabel='time', ylabel='$\epsilon$', title='Per-time bandwidths')
ax.grid(alpha=0.3)
ax.legend()

ax = axes[1]
ax.axhline(64, color='k', linestyle='--', label='target neighbours')
best = connectivity_df.sort_values('score').groupby('time_idx').first().reset_index()
ax.plot(best['time'], best['mean_neighbors'], 'o-')
ax.set(xlabel='time', ylabel='mean neighbours (> e^{-1})', title='Connectivity diagnostics')
ax.grid(alpha=0.3)
ax.legend()
plt.show()


# %% [markdown]
# ## Time-Coupled Diffusion Maps
# Apply the Marshall–Hirn construction with the tuned bandwidths and variable kernels.

# %%
tc_result_finaltime = time_coupled_diffusion_map(
    list(all_frames),
    k=tc_k,
    alpha=tc_alpha,
    epsilons=selected_epsilons,
    variable_bandwidth=use_variable_bandwidth,
    beta=tc_beta,
    density_bandwidths=kde_bandwidths.tolist(),
    t=len(times_arr),
)

print(f"Constructed {len(tc_result_finaltime.transition_operators)} operators with embedding shape {tc_result_finaltime.embedding.shape}.")
print('Stationary distribution min/max:', tc_result_finaltime.stationary_distribution.min(), tc_result_finaltime.stationary_distribution.max())

# Visualize spectrum for each time point
# We need to compute the spectrum for each time step t=1...T
# The function build_time_coupled_trajectory computes this internally.
# Let's use the result from build_time_coupled_trajectory which is computed in the next cell,
# or we can compute it here if we want to see it immediately.

# For immediate visualization, let's compute the trajectory result here temporarily or just wait for the next cell.
# However, the user asked to visualize it *instead* of only the last time.
# The `tc_result_finaltime` only contains the result for the final time `t`.
# To get all spectra, we should use `build_time_coupled_trajectory`.

temp_trajectory = build_time_coupled_trajectory(
    tc_result_finaltime.transition_operators,
    embed_dim=tc_k,
)

plt.figure(figsize=(8, 5))
# Plot spectrum for each time step
for t_idx, sigma_t in enumerate(temp_trajectory.singular_values):
    plt.plot(sigma_t, 'o-', label=f't={times_arr[t_idx]:.2f}', alpha=0.7, markersize=4)

plt.yscale('log')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Time-Coupled Operator Spectrum Evolution')
plt.grid(alpha=0.4, which='both')
# Put legend outside if too many lines
if len(times_arr) > 10:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
else:
    plt.legend()
plt.tight_layout()
plt.show()

# %%
tc_result = build_time_coupled_trajectory(
    tc_result_finaltime.transition_operators,
    embed_dim=tc_k,
)
tc_embeddings_time = tc_result.embeddings
print('Time-coupled embeddings tensor:', tc_embeddings_time.shape)

from matplotlib import colors as mcolors
# use index as coloring
coloring = np.arange(tc_embeddings_time.shape[1])
norm = mcolors.Normalize(vmin=np.min(coloring), vmax=np.max(coloring))
fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4), sharex=True, sharey=True, constrained_layout=True)
if n_times == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    coords = tc_embeddings_time[i]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=coloring, cmap='viridis', norm=norm, s=10, alpha=0.7)
    ax.set_title(f"t = {times_arr[i]:.2f}")
    ax.set_xlabel('DC 1')
    if i == 0:
        ax.set_ylabel('DC 2')
    ax.grid(alpha=0.3)

fig.colorbar(sc, ax=axes, label='index', fraction=0.046, pad=0.01)
plt.suptitle('Time-coupled diffusion coordinates', y=1.02)
plt.show()

traj_times = times_arr[: tc_embeddings_time.shape[0]]
fig, ax = plt.subplots(figsize=(6, 4))
traj_count = min(8, n_samples)
subset_idx = rng.choice(n_samples, size=traj_count, replace=False)
for idx in subset_idx:
    ax.plot(traj_times, tc_embeddings_time[:, idx, 0], alpha=0.7)
ax.set(xlabel='time', ylabel='DC 1')
ax.set_title('Sample diffusion coordinate trajectories')
ax.grid(alpha=0.3)
plt.show()


# %%
train_times = times_arr
holdout_time = 0.75

spline_type = 'pchip'
spline_window_mode = 'triplet'

sample_splines = []
for sample_idx in range(n_samples):
    coords_sample = tc_embeddings_time[:, sample_idx, :]
    splines = fit_coordinate_splines(
        coords_sample,
        train_times,
        spline_type=spline_type,
        window_mode=spline_window_mode,
    )
    sample_splines.append(splines)

g_star = np.vstack([
    evaluate_coordinate_splines(splines, holdout_time).ravel()
    for splines in sample_splines
])
print(f"Spline interpolation complete: {len(sample_splines)} samples; latent dim = {g_star.shape[1]}")


# %%

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print('matplotlib is not installed; skipping latent spline visualization.')
else:
    sample_to_plot = 0
    t_dense = np.linspace(train_times.min(), train_times.max(), 200)
    splines = sample_splines[sample_to_plot]
    phi_dense = np.vstack([
        evaluate_coordinate_splines(splines, t).ravel()
        for t in t_dense
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(train_times, tc_embeddings_time[:, sample_to_plot, 0], 'o', label='training $\\psi_1$')
    axes[0].plot(t_dense, phi_dense[:, 0], '-', label='spline $\\psi_1$')
    axes[0].axvline(holdout_time, color='red', linestyle='--', label='held-out t')
    axes[0].scatter([holdout_time], [g_star[sample_to_plot, 0]], color='red')
    axes[0].set(
        xlabel='time',
        ylabel='$\\psi_1$',
        title=f'Latent spline (sample {sample_to_plot})',
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if phi_dense.shape[1] >= 2:
        axes[1].plot(phi_dense[:, 0], phi_dense[:, 1], '-', color='tab:blue', label='spline path')
        axes[1].scatter(
            tc_embeddings_time[:, sample_to_plot, 0],
            tc_embeddings_time[:, sample_to_plot, 1],
            c=train_times,
            cmap='viridis',
            marker='o',
            label='training samples',
        )
        axes[1].scatter(
            g_star[sample_to_plot, 0],
            g_star[sample_to_plot, 1],
            color='red',
            marker='*',
            s=120,
            label='held-out',
        )
        axes[1].set(
            xlabel='$\\psi_1$',
            ylabel='$\\psi_2$',
            title='Latent trajectory',
        )
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'Need ≥2 latent dims', ha='center', va='center')
        
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[1], label='time')
    plt.show()

# %%
print("Shape of tc_result.singular_values[0].shape:", tc_result.singular_values[0].shape)
print("Shape of tc_result.left_singular_vectors[0].shape:", tc_result.left_singular_vectors[0].shape)
print("Shape of tc_result.embeddings[0].shape:", tc_result.embeddings[0].shape)

# %%
from stiefel.stiefel import batch_stiefel_log, Stiefel_Log, Stiefel_Exp
from stiefel.barycenter import R_barycenter
from stiefel.projection_retraction import (
    st_retr_orthographic,
    st_inv_retr_orthographic,
)
from stiefel.error_measures import err_st
from diffmap.diffusion_maps import _orient_svd
from sklearn.decomposition import TruncatedSVD
# use the symmetric markov operators (used in SVD) and concatenate them as global snapshots
A_concatenated = np.concatenate([tc_result.A_operators[i] for i in range(len(tc_result.A_operators))], axis=1)
svd = TruncatedSVD(n_components=len(tc_result.singular_values[0]) + 1, algorithm='randomized', random_state=42)
svd.fit(A_concatenated)
U_global = svd.transform(A_concatenated)
Vt_global = svd.components_
Sigma_global = svd.singular_values_
U_global, Vt_global = _orient_svd(A_concatenated, U_global, Vt_global)

# remove first trivial component
U_global = U_global[:, 1:]
Vt_global = Vt_global[1:, :]

# Stack trajectory on Stiefel
U_train_list = np.stack(tc_result.left_singular_vectors, axis=0)
n_times, n_samples, stiefel_dim = U_train_list.shape
times_train = train_times  # from notebook
print(f"Stacked left singular vectors: {U_train_list.shape}")

# Compute Frechet mean with orthographic retraction
barycenter_kwargs = dict(stepsize=1.0, max_it=100, tol=1e-8, verbosity=False)
U_frechet, U_bary_iters = R_barycenter(
    points=U_train_list,
    retr=st_retr_orthographic,
    inv_retr=st_inv_retr_orthographic,
    init=U_global,
    **barycenter_kwargs,
)
deltas_global = batch_stiefel_log(U_global, U_train_list, metric_alpha=1e-8, tau=1e-2)
deltas_fm = batch_stiefel_log(U_frechet, U_train_list, metric_alpha=1e-8, tau=1e-2)

# Basic diagnostics for basepoints
err_to_global = np.array([err_st(U_global, U_train_list[t]) for t in range(n_times)])
err_to_frechet = np.array([err_st(U_frechet, U_train_list[t]) for t in range(n_times)])
print(f"Mean err_st to U_global: {err_to_global.mean():.4e}")
print(f"Mean err_st to U_frechet: {err_to_frechet.mean():.4e}")

M = U_global.T @ U_frechet
s = np.linalg.svd(M, compute_uv=False)
principal_angles = np.arccos(np.clip(s, -1.0, 1.0))
print("Principal angles U_global vs U_frechet (rad):", principal_angles)


def _log_spread(deltas_arr: np.ndarray) -> float:
    flat = deltas_arr.reshape(deltas_arr.shape[0], -1)
    return float(np.mean(np.sum(flat * flat, axis=1)))


def stiefel_log_smoothness(deltas_arr: np.ndarray) -> float:
    if deltas_arr.shape[0] < 3:
        return float("nan")
    d2 = deltas_arr[2:] - 2 * deltas_arr[1:-1] + deltas_arr[:-2]
    flat = d2.reshape(d2.shape[0], -1)
    return float(np.mean(np.sum(flat * flat, axis=1)))


spread_global = _log_spread(deltas_global)
spread_fm = _log_spread(deltas_fm)
smooth_global = stiefel_log_smoothness(deltas_global)
smooth_fm = stiefel_log_smoothness(deltas_fm)
print(f"Log spread → global: {spread_global:.4e}, Frechet: {spread_fm:.4e}")
print(f"Log smoothness → global: {smooth_global:.4e}, Frechet: {smooth_fm:.4e}")

if __name__ == "__main__":
    vmax = max(
        np.max(np.abs(U_global)),
        np.max(np.abs(U_frechet)),
        np.max(np.abs(U_frechet - U_global)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ims = [
        axes[0].imshow(U_global, vmin=-vmax, vmax=vmax),
        axes[1].imshow(U_frechet, vmin=-vmax, vmax=vmax),
        axes[2].imshow(U_frechet - U_global, vmin=-vmax, vmax=vmax),
    ]
    axes[0].set_title("$U_{\\mathrm{global}}$")
    axes[1].set_title("$U_{\\mathrm{FM}}$")
    axes[2].set_title("Difference")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(ims[-1], ax=axes, shrink=0.7)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(times_train, err_to_global, "o-", label="U_global")
    axes[0].plot(times_train, err_to_frechet, "s-", label="U_frechet")
    axes[0].set(xlabel="time", ylabel="err_st", title="Basepoint centrality")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(principal_angles, bins=min(20, len(principal_angles)), color="tab:orange")
    axes[1].set(xlabel="principal angle (rad)", ylabel="count", title="Column-space angles")
    axes[1].grid(alpha=0.3)
    plt.show()

# %%
from scipy.interpolate import PchipInterpolator

# Define dense time grid for interpolation
t_dense = np.linspace(times_train.min(), times_train.max(), 200)

# --- 1. Stiefel Interpolation (Deltas) for both basepoints ---
def interpolate_deltas(deltas_arr: np.ndarray, base_point: np.ndarray):
    deltas_flat = deltas_arr.reshape(deltas_arr.shape[0], -1)
    pchip = PchipInterpolator(times_train, deltas_flat, axis=0)
    deltas_dense_flat = pchip(t_dense)
    deltas_dense = deltas_dense_flat.reshape(len(t_dense), *base_point.shape)
    U_dense = np.array([Stiefel_Exp(U0=base_point, Delta=d) for d in deltas_dense])
    return deltas_dense, U_dense, pchip


deltas_global_dense, U_dense_global, pchip_delta_global = interpolate_deltas(deltas_global, U_global)
deltas_fm_dense, U_dense_fm, pchip_delta_fm = interpolate_deltas(deltas_fm, U_frechet)
print(f"Stiefel interpolation complete. Shapes → global: {U_dense_global.shape}, Frechet: {U_dense_fm.shape}")

# --- 2. Interpolate Singular Values (Sigma) ---
# Stack singular values from trajectory result
sigmas = np.stack(tc_result.singular_values) # Shape: (n_times, k)

# Interpolate in log-domain
log_sigmas = np.log(sigmas + 1e-16)
pchip_sigma = PchipInterpolator(times_train, log_sigmas, axis=0)
log_sigmas_dense = pchip_sigma(t_dense)
sigmas_dense = np.exp(log_sigmas_dense) # Shape: (n_dense, k)

# --- 3. Interpolate Stationary Distributions (Pi) ---
# Stack stationary distributions
pis = np.stack(tc_result.stationary_distributions) # Shape: (n_times, n_samples)

# Interpolate in log-domain
log_pis = np.log(pis + 1e-16)
pchip_pi = PchipInterpolator(times_train, log_pis, axis=0)
log_pis_dense = pchip_pi(t_dense)
pis_dense_unnorm = np.exp(log_pis_dense)

# Normalize to sum to 1 at each time step
pis_dense = pis_dense_unnorm / pis_dense_unnorm.sum(axis=1, keepdims=True) # Shape: (n_dense, n_samples)

print(f"Sigma dense shape: {sigmas_dense.shape}")
print(f"Pi dense shape: {pis_dense.shape}")

# --- 4. Reconstruct Diffusion Embeddings for both basepoints ---
def reconstruct_embeddings(U_dense: np.ndarray) -> np.ndarray:
    phi_list = []
    for i in range(len(t_dense)):
        U = U_dense[i]
        S = sigmas_dense[i]
        Pi = pis_dense[i]
        phi_list.append((U * S[None, :]) / np.sqrt(Pi)[:, None])
    return np.array(phi_list)


phi_global_dense = reconstruct_embeddings(U_dense_global)
phi_frechet_dense = reconstruct_embeddings(U_dense_fm)
phi_stiefel_dense = phi_global_dense  # keep legacy name for downstream plots
print(f"Reconstructed embeddings → global: {phi_global_dense.shape}, Frechet: {phi_frechet_dense.shape}")

# %%
# Hold-out evaluation on odd time indices
all_indices = np.arange(n_times)
train_idx = all_indices[::2]
val_idx = all_indices[1::2]

train_times_sub = times_train[train_idx]
val_times_sub = times_train[val_idx]


def interpolate_split(deltas_arr: np.ndarray, base_point: np.ndarray):
    deltas_train = deltas_arr[train_idx].reshape(len(train_idx), -1)
    pchip = PchipInterpolator(train_times_sub, deltas_train, axis=0)
    deltas_val_flat = pchip(val_times_sub)
    deltas_val = deltas_val_flat.reshape(len(val_idx), *base_point.shape)
    U_pred = np.array([Stiefel_Exp(U0=base_point, Delta=d) for d in deltas_val])
    return U_pred, pchip


U_pred_global_val, pchip_delta_global_cv = interpolate_split(deltas_global, U_global)
U_pred_fm_val, pchip_delta_fm_cv = interpolate_split(deltas_fm, U_frechet)

U_true_val = U_train_list[val_idx]
err_st_global_val = np.array([err_st(U_pred_global_val[i], U_true_val[i]) for i in range(len(val_idx))])
err_st_fm_val = np.array([err_st(U_pred_fm_val[i], U_true_val[i]) for i in range(len(val_idx))])
print(f"Mean validation err_st → global: {err_st_global_val.mean():.4e}, Frechet: {err_st_fm_val.mean():.4e}")

# Embedding hold-out
log_sigmas_train = log_sigmas[train_idx]
pchip_sigma_cv = PchipInterpolator(train_times_sub, log_sigmas_train, axis=0)
sigmas_val = np.exp(pchip_sigma_cv(val_times_sub))

log_pis_train = log_pis[train_idx]
pchip_pi_cv = PchipInterpolator(train_times_sub, log_pis_train, axis=0)
pis_val_unnorm = np.exp(pchip_pi_cv(val_times_sub))
pis_val = pis_val_unnorm / pis_val_unnorm.sum(axis=1, keepdims=True)


def reconstruct_val_embeddings(U_pred: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(val_idx)):
        out.append((U_pred[i] * sigmas_val[i][None, :]) / np.sqrt(pis_val[i])[:, None])
    return np.array(out)


phi_global_val = reconstruct_val_embeddings(U_pred_global_val)
phi_frechet_val = reconstruct_val_embeddings(U_pred_fm_val)
phi_true_val = tc_embeddings_time[val_idx]

embedding_mse_global = float(np.mean((phi_global_val - phi_true_val) ** 2))
embedding_mse_fm = float(np.mean((phi_frechet_val - phi_true_val) ** 2))
print(f"Validation embedding MSE → global: {embedding_mse_global:.4e}, Frechet: {embedding_mse_fm:.4e}")


# %%
# --- 5. Comparison: Naive Spline vs Global Stiefel vs Frechet Stiefel ---

sample_idx = 0  # Choose sample to visualize

# 1. Get Naive Spline Trajectory for this sample
splines_sample = sample_splines[sample_idx]
phi_naive_sample = np.vstack(
    [evaluate_coordinate_splines(splines_sample, t).ravel() for t in t_dense]
)

# 2. Stiefel Interpolations
phi_global_sample = phi_global_dense[:, sample_idx, :]
phi_frechet_sample = phi_frechet_dense[:, sample_idx, :]

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Component 1 vs Time
axes[0].plot(times_train, tc_embeddings_time[:, sample_idx, 0], 'ko', label='Training Points')
axes[0].plot(t_dense, phi_naive_sample[:, 0], 'b--', label='Naive Spline')
axes[0].plot(t_dense, phi_global_sample[:, 0], 'r-', label='Global basepoint')
axes[0].plot(t_dense, phi_frechet_sample[:, 0], 'g-', label='Frechet basepoint')
axes[0].set_title(f'Sample {sample_idx}: Component 1 Evolution')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('$\\psi_1$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Component 2 vs Time
axes[1].plot(times_train, tc_embeddings_time[:, sample_idx, 1], 'ko', label='Training Points')
axes[1].plot(t_dense, phi_naive_sample[:, 1], 'b--', label='Naive Spline')
axes[1].plot(t_dense, phi_global_sample[:, 1], 'r-', label='Global basepoint')
axes[1].plot(t_dense, phi_frechet_sample[:, 1], 'g-', label='Frechet basepoint')
axes[1].set_title(f'Sample {sample_idx}: Component 2 Evolution')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('$\\psi_2$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.show()

# 2D Trajectory Plot (Phase Plane)
plt.figure(figsize=(8, 7))
plt.plot(phi_naive_sample[:, 0], phi_naive_sample[:, 1], 'b--', label='Naive Spline')
plt.plot(phi_global_sample[:, 0], phi_global_sample[:, 1], 'r-', label='Global basepoint')
plt.plot(phi_frechet_sample[:, 0], phi_frechet_sample[:, 1], 'g-', label='Frechet basepoint')
plt.scatter(
    tc_embeddings_time[:, sample_idx, 0],
    tc_embeddings_time[:, sample_idx, 1],
    c='k',
    zorder=5,
    label='Training Points',
)

# Mark start and end (global view)
plt.scatter(phi_global_sample[0, 0], phi_global_sample[0, 1], c='r', marker='^', s=100, label='Start (global)')
plt.scatter(phi_global_sample[-1, 0], phi_global_sample[-1, 1], c='r', marker='v', s=100, label='End (global)')

plt.xlabel('$\\psi_1$')
plt.ylabel('$\\psi_2$')
plt.title(f'Latent Trajectory Comparison (Sample {sample_idx})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()


# %%
# --- 6. Visualize Interpolation for All Samples (Fréchet Basepoint) ---

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Plot parameters
alpha_val = 0.05
color_interp = 'g'
color_train = 'k'

n_samples_total = phi_frechet_dense.shape[1]

# Component 1 vs Time
# Plot all trajectories at once: (n_dense, n_samples)
axes[0].plot(t_dense, phi_frechet_dense[:, :, 0], color=color_interp, alpha=alpha_val, linewidth=0.5)
# Scatter training points
# Flatten time to match flattened embeddings: (n_times * n_samples)
t_flat = np.repeat(times_train, n_samples_total)
axes[0].scatter(t_flat, tc_embeddings_time[:, :, 0].flatten(), 
                color=color_train, s=1, alpha=0.1, zorder=2)
axes[0].set_title(f'Component 1 Evolution (All {n_samples_total} Samples)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('$\psi_1$')
axes[0].grid(True, alpha=0.3)

# Component 2 vs Time
axes[1].plot(t_dense, phi_frechet_dense[:, :, 1], color=color_interp, alpha=alpha_val, linewidth=0.5)
axes[1].scatter(t_flat, tc_embeddings_time[:, :, 1].flatten(), 
                color=color_train, s=1, alpha=0.1, zorder=2)
axes[1].set_title(f'Component 2 Evolution (All {n_samples_total} Samples)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('$\psi_2$')
axes[1].grid(True, alpha=0.3)

# Phase Plane
axes[2].plot(phi_frechet_dense[:, :, 0], phi_frechet_dense[:, :, 1], color=color_interp, alpha=alpha_val, linewidth=0.5)
axes[2].scatter(tc_embeddings_time[:, :, 0].flatten(), tc_embeddings_time[:, :, 1].flatten(), 
                color=color_train, s=1, alpha=0.1, zorder=2)
axes[2].set_title('Latent Trajectories (Phase Plane)')
axes[2].set_xlabel('$\psi_1$')
axes[2].set_ylabel('$\psi_2$')
axes[2].grid(True, alpha=0.3)
axes[2].axis('equal')

# Add a custom legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=color_interp, lw=2),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_train, markersize=5)]
axes[2].legend(custom_lines, ['Fréchet Interpolation', 'Training Points'], loc='best')

plt.show()


@dataclass
class LiftingConfig:
    holdout_time: float = 0.75
    gh_delta: float = 1e-3
    gh_ridge: float = 1e-6
    gh_local_delta: float = 5e-3
    gh_local_ridge: float = 1e-3
    gh_local_neighbors: int = 128
    gh_local_max_modes: int = 8
    convex_k: int = 64
    convex_max_iter: int = 200
    krr_spatial_length_scale: Optional[float] = None
    krr_temporal_length_scale: Optional[float] = None
    krr_time_scaling: Optional[float] = None
    krr_ridge_penalty: float = 1e-3
    plot_samples: tuple[int, ...] = (0, 1, 2)
    vmax_mode: Literal["global", "per_sample"] = "global"
    time_match_tol: float = 1e-8


@dataclass
class LatentInterpolationResult:
    t_dense: np.ndarray                 # (n_dense,)
    phi_global_dense: np.ndarray        # (n_dense, n_samples, latent_dim)
    phi_frechet_dense: np.ndarray       # (n_dense, n_samples, latent_dim)
    phi_naive_dense: Optional[np.ndarray] = None # (n_dense, n_samples, latent_dim)
    
    # Store interpolators for exact evaluation if needed
    pchip_delta_global: Optional[PchipInterpolator] = None
    pchip_delta_fm: Optional[PchipInterpolator] = None
    pchip_sigma: Optional[PchipInterpolator] = None
    pchip_pi: Optional[PchipInterpolator] = None
    
    # Base points
    U_global: Optional[np.ndarray] = None
    U_frechet: Optional[np.ndarray] = None


@dataclass
class LiftingModels:
    gh_model: GeometricHarmonicsModel
    convex: ConvexHullInterpolator
    # KRR is stateless in current implementation, but we store config in LiftingConfig


@dataclass
class PseudoDataConfig:
    n_dense: int = 200
    n_pseudo_per_interval: int = 3
    interp_method: Literal['frechet', 'global', 'naive'] = 'frechet'
    n_samples_vis: int = 8        # for trajectory plots
    n_samples_fields: int = 3     # for field strips


def load_tran_inclusions_data(data_path: Path = Path("../data/tran_inclusions.npz")):
    npz = np.load(data_path)
    resolution = int(np.sqrt(int(npz["data_dim"])))

    # Debug: print all available keys
    print("Available keys in npz:", list(npz.files))

    # Extract available times from marginals (not raw_marginal_)
    time_keys = [k for k in npz.files if k.startswith("marginal_")]
    times = sorted(float(k.replace("marginal_", "")) for k in time_keys)
    print("Times:", times)

    # Acquire marginals (raw preferred; PCA otherwise)
    times, marginals, mode = get_marginals(npz, choice="pca")
    _, raw_marginals, _ = get_marginals(npz, choice="raw")
    held_out_indices = npz.get("held_out_indices")
    if held_out_indices is not None:
        held_out_indices = np.asarray(held_out_indices, dtype=int)
    else:
        held_out_indices = np.array([], dtype=int)

    train_times, train_marginals, held_out_times, held_out_marginals, used_held_out_indices = split_train_holdout_marginals(
        times,
        marginals,
        held_out_indices,
    )
    times, marginals = train_times, train_marginals
    print("Using mode:", mode, "Found times:", times)
    if held_out_times:
        print(
            f"Held-out times at indices {used_held_out_indices}: {held_out_times} (excluded from training)"
        )

    # Load PCA metadata (used only when mode == 'pca')
    components = npz.get("pca_components")
    mean_vec = npz.get("pca_mean")
    explained_variance = npz.get("pca_explained_variance")
    is_whitened = bool(npz.get("is_whitened", False))
    whitening_epsilon = float(npz.get("whitening_epsilon", 0.0))
    
    # Assemble PCA snapshots into a tensor (time, samples, components)
    times_arr = np.array(times, dtype=np.float64)
    all_frames = np.stack([marginals[t] for t in times_arr])
    # exclude first time since it is for visualization only (as per original script)
    all_frames = all_frames[1:]
    times_arr = times_arr[1:]
    
    return times_arr, all_frames, components, mean_vec, explained_variance, is_whitened, whitening_epsilon, resolution, raw_marginals

def compute_bandwidth_statistics(frames: np.ndarray) -> dict[str, np.ndarray]:
    medians, q1, q3, maxima = [], [], [], []
    for snapshot in frames:
        d2 = squareform(pdist(snapshot, metric='sqeuclidean'))
        mask = d2 > 0
        if np.any(mask):
            vals = d2[mask]
        else:
            vals = np.array([1.0])
        medians.append(float(np.median(vals)))
        q1.append(float(np.percentile(vals, 25)))
        q3.append(float(np.percentile(vals, 75)))
        maxima.append(float(np.max(vals)))
    return {
        'median': np.array(medians),
        'q1': np.array(q1),
        'q3': np.array(q3),
        'max': np.array(maxima),
    }

def build_time_coupled_embeddings(
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    tc_k: int = 8,
    alpha: float = 0.5,
    beta: float = -0.2,
    use_variable_bandwidth: bool = True,
):
    bandwidth_stats = compute_bandwidth_statistics(all_frames)
    base_epsilons = bandwidth_stats['median']
    print('Base epsilon estimates (median squared distances):')
    for idx, eps in enumerate(base_epsilons):
        print(f"  t={times_arr[idx]:.2f}: median={eps:.3e}, IQR=({bandwidth_stats['q1'][idx]:.3e}, {bandwidth_stats['q3'][idx]:.3e})")

    epsilon_scales = np.geomspace(0.1, 4.0, num=32)
    sample_size = min(1024, all_frames.shape[1])
    selected_epsilons, kde_bandwidths, connectivity_df = select_epsilons_by_connectivity(
        all_frames,
        times_arr,
        base_epsilons=base_epsilons,
        scales=epsilon_scales,
        alpha=alpha,
        target_neighbors=64.0,
        sample_size=sample_size,
        rng_seed=0,
        variable_bandwidth=use_variable_bandwidth,
        beta=beta,
    )

    print('Chosen epsilons by connectivity:')
    print(selected_epsilons)
    print('KDE bandwidths used:', kde_bandwidths)

    tc_result_finaltime = time_coupled_diffusion_map(
        list(all_frames),
        k=tc_k,
        alpha=alpha,
        epsilons=selected_epsilons,
        variable_bandwidth=use_variable_bandwidth,
        beta=beta,
        density_bandwidths=kde_bandwidths.tolist(),
        t=len(times_arr),
    )
    
    tc_result = build_time_coupled_trajectory(
        tc_result_finaltime.transition_operators,
        embed_dim=tc_k,
    )
    tc_embeddings_time = tc_result.embeddings
    print('Time-coupled embeddings tensor:', tc_embeddings_time.shape)
    
    return tc_result, tc_embeddings_time, selected_epsilons, kde_bandwidths, connectivity_df


def build_dense_latent_trajectories(
    tc_result,
    times_train: np.ndarray,
    tc_embeddings_time: np.ndarray,
    n_dense: int = 200,
) -> LatentInterpolationResult:
    # 1. Build U_train_list, U_global, U_frechet
    # use the symmetric markov operators (used in SVD) and concatenate them as global snapshots
    A_concatenated = np.concatenate([tc_result.A_operators[i] for i in range(len(tc_result.A_operators))], axis=1)
    svd = TruncatedSVD(n_components=len(tc_result.singular_values[0]) + 1, algorithm='randomized', random_state=42)
    svd.fit(A_concatenated)
    U_global = svd.transform(A_concatenated)
    Vt_global = svd.components_
    U_global, Vt_global = _orient_svd(A_concatenated, U_global, Vt_global)

    # remove first trivial component
    U_global = U_global[:, 1:]
    
    # Stack trajectory on Stiefel
    U_train_list = np.stack(tc_result.left_singular_vectors, axis=0)
    n_times, n_samples, stiefel_dim = U_train_list.shape
    print(f"Stacked left singular vectors: {U_train_list.shape}")

    # Compute Frechet mean with orthographic retraction
    barycenter_kwargs = dict(stepsize=1.0, max_it=100, tol=1e-8, verbosity=False)
    U_frechet, U_bary_iters = R_barycenter(
        points=U_train_list,
        retr=st_retr_orthographic,
        inv_retr=st_inv_retr_orthographic,
        init=U_global,
        **barycenter_kwargs,
    )
    deltas_global = batch_stiefel_log(U_global, U_train_list, metric_alpha=1e-8, tau=1e-2)
    deltas_fm = batch_stiefel_log(U_frechet, U_train_list, metric_alpha=1e-8, tau=1e-2)

    # 2. Interpolate deltas on Stiefel (global + Frechet) to get U_dense_global / U_dense_fm.
    t_dense = np.linspace(times_train.min(), times_train.max(), n_dense)
    
    def interpolate_deltas(deltas_arr: np.ndarray, base_point: np.ndarray):
        deltas_flat = deltas_arr.reshape(deltas_arr.shape[0], -1)
        pchip = PchipInterpolator(times_train, deltas_flat, axis=0)
        deltas_dense_flat = pchip(t_dense)
        deltas_dense = deltas_dense_flat.reshape(len(t_dense), *base_point.shape)
        U_dense = np.array([Stiefel_Exp(U0=base_point, Delta=d) for d in deltas_dense])
        return deltas_dense, U_dense, pchip

    deltas_global_dense, U_dense_global, pchip_delta_global = interpolate_deltas(deltas_global, U_global)
    deltas_fm_dense, U_dense_fm, pchip_delta_fm = interpolate_deltas(deltas_fm, U_frechet)
    
    # 3. Interpolate singular values + stationary distributions (log-domain)
    sigmas = np.stack(tc_result.singular_values) # Shape: (n_times, k)
    log_sigmas = np.log(sigmas + 1e-16)
    pchip_sigma = PchipInterpolator(times_train, log_sigmas, axis=0)
    log_sigmas_dense = pchip_sigma(t_dense)
    sigmas_dense = np.exp(log_sigmas_dense) # Shape: (n_dense, k)

    pis = np.stack(tc_result.stationary_distributions) # Shape: (n_times, n_samples)
    log_pis = np.log(pis + 1e-16)
    pchip_pi = PchipInterpolator(times_train, log_pis, axis=0)
    log_pis_dense = pchip_pi(t_dense)
    pis_dense_unnorm = np.exp(log_pis_dense)
    pis_dense = pis_dense_unnorm / pis_dense_unnorm.sum(axis=1, keepdims=True) # Shape: (n_dense, n_samples)

    # Reconstruct embeddings
    def reconstruct_embeddings(U_dense: np.ndarray) -> np.ndarray:
        phi_list = []
        for i in range(len(t_dense)):
            U = U_dense[i]
            S = sigmas_dense[i]
            Pi = pis_dense[i]
            phi_list.append((U * S[None, :]) / np.sqrt(Pi)[:, None])
        return np.array(phi_list)

    phi_global_dense = reconstruct_embeddings(U_dense_global)
    phi_frechet_dense = reconstruct_embeddings(U_dense_fm)
    
    # 4. Naive splines
    sample_splines = []
    for sample_idx in range(n_samples):
        coords_sample = tc_embeddings_time[:, sample_idx, :]
        splines = fit_coordinate_splines(
            coords_sample,
            times_train,
            spline_type='pchip',
            window_mode='triplet',
        )
        sample_splines.append(splines)

    phi_naive_dense = np.stack([
        np.vstack([
            evaluate_coordinate_splines(splines, t).ravel()
            for t in t_dense
        ])
        for splines in sample_splines
    ], axis=1) # (n_dense, n_samples, latent_dim)

    return LatentInterpolationResult(
        t_dense=t_dense,
        phi_global_dense=phi_global_dense,
        phi_frechet_dense=phi_frechet_dense,
        phi_naive_dense=phi_naive_dense,
        pchip_delta_global=pchip_delta_global,
        pchip_delta_fm=pchip_delta_fm,
        pchip_sigma=pchip_sigma,
        pchip_pi=pchip_pi,
        U_global=U_global,
        U_frechet=U_frechet,
    )

def choose_pseudo_times_per_interval(
    times_train: np.ndarray,
    n_per_interval: int = 3,
    include_endpoints: bool = False,
) -> np.ndarray:
    """
    For each interval [t_i, t_{i+1}], sample n_per_interval times strictly inside
    or including endpoints depending on flag.
    Return a sorted 1D array of unique pseudo times.
    """
    pseudo_times = []
    for i in range(len(times_train) - 1):
        t_start = times_train[i]
        t_end = times_train[i+1]
        if include_endpoints:
            pts = np.linspace(t_start, t_end, n_per_interval + 2)
        else:
            pts = np.linspace(t_start, t_end, n_per_interval + 2)[1:-1]
        pseudo_times.extend(pts)
    
    if include_endpoints:
        # Add the very last point if not covered (though linspace covers it)
        # But we iterate intervals, so endpoints might be duplicated.
        pass
        
    return np.unique(np.array(pseudo_times))

def sample_latent_at_times(
    interpolation: LatentInterpolationResult,
    pseudo_times: np.ndarray,
    method: Literal['naive', 'global', 'frechet'] = 'frechet',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      - t_pseudo: (n_pseudo,)
      - phi_pseudo: (n_pseudo, n_samples, latent_dim)
    """
    # We can use the stored pchips for exact evaluation if available
    # Or interpolate from dense grid.
    # Using pchips is more accurate.
    
    if method == 'naive':
        # Naive splines are not stored as pchips in the result object in a way that's easy to evaluate here
        # without refitting or storing the splines.
        # But we have phi_naive_dense. Let's just interpolate from dense grid for simplicity
        # or we could store the splines in the result object.
        
        # Actually, let's just use interpolation from dense grid for all for consistency and simplicity
        # unless we want high precision.
        # The prompt suggests: "Simplest: use np.interp / PchipInterpolator on index vs time mapping"
        
        phi_dense = interpolation.phi_naive_dense
    elif method == 'global':
        phi_dense = interpolation.phi_global_dense
    elif method == 'frechet':
        phi_dense = interpolation.phi_frechet_dense
    else:
        raise ValueError(f"Unknown method {method}")
        
    # Interpolate from dense grid
    phi_pseudo = np.zeros((len(pseudo_times), phi_dense.shape[1], phi_dense.shape[2]))
    for i in range(phi_dense.shape[1]): # per sample
        for j in range(phi_dense.shape[2]): # per dim
            # PchipInterpolator over the dense grid
            pchip = PchipInterpolator(interpolation.t_dense, phi_dense[:, i, j])
            phi_pseudo[:, i, j] = pchip(pseudo_times)
            
    return pseudo_times, phi_pseudo

def fit_lifting_models(
    tc_embeddings_time: np.ndarray,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    config: LiftingConfig,
) -> Tuple[LiftingModels, Dict[str, Any]]:
    """
    - Build macro_train, micro_train (using existing build_training_pairs).
    - Fit GH (global), construct ConvexHullInterpolator, and compute KRR hyperparams.
    - Return lifting models + any metadata you want.
    """
    macro_train, micro_train = build_training_pairs(
        tc_embeddings_time,
        all_frames,
        times_arr,
        config.holdout_time,
        tol=config.time_match_tol,
    )
    macro_train_coords, train_time_values, micro_train_states = build_time_augmented_training_pairs(
        macro_time_series=tc_embeddings_time,
        micro_time_series=all_frames,
        snapshot_times=times_arr,
        holdout_snapshot_time=config.holdout_time,
        tol=config.time_match_tol,
    )
    print(f"Training pairs collected: macro {macro_train.shape}, micro {micro_train.shape}")
    
    # KRR Hyperparameters
    time_span = float(times_arr.max() - times_arr.min()) if times_arr.size > 0 else 1.0
    if config.krr_spatial_length_scale is None:
        centered_macro = macro_train_coords - macro_train_coords.mean(axis=0)
        macro_norms = np.linalg.norm(centered_macro, axis=1)
        median_norm = float(np.median(macro_norms)) if macro_norms.size > 0 else 1.0
        config.krr_spatial_length_scale = median_norm if median_norm > 0 else 1.0
    if config.krr_temporal_length_scale is None:
        temporal_guess = 0.5 * time_span / max(len(times_arr), 1)
        config.krr_temporal_length_scale = temporal_guess if temporal_guess > 0 else 1.0
    if config.krr_time_scaling is None:
        config.krr_time_scaling = 1.0 / time_span if time_span > 0 else 1.0
    
    print(
        "KRR hyperparameters \u2192 "
        f"sigma_g={config.krr_spatial_length_scale:.3e}, "
        f"sigma_t={config.krr_temporal_length_scale:.3e}, "
        f"gamma={config.krr_time_scaling:.3e}, "
        f"lambda={config.krr_ridge_penalty:.3e}"
    )

    # Fit GH
    gh_model = fit_geometric_harmonics(
        intrinsic_coords=macro_train,
        samples=micro_train,
        epsilon_star=None,
        delta=config.gh_delta,
        ridge=config.gh_ridge,
        grid_shape=None,
        center=True,
    )
    
    # Fit Convex Hull
    chi = ConvexHullInterpolator(macro_states=macro_train, micro_states=micro_train)
    
    models = LiftingModels(gh_model=gh_model, convex=chi)
    metadata = {
        'macro_train': macro_train,
        'micro_train': micro_train,
        'macro_train_coords': macro_train_coords,
        'train_time_values': train_time_values,
        'micro_train_states': micro_train_states
    }
    return models, metadata

def lift_pseudo_latents(
    phi_pseudo: np.ndarray,          # (n_pseudo, n_samples, latent_dim)
    t_pseudo: np.ndarray,            # (n_pseudo,)
    models: LiftingModels,
    tc_embeddings_time: np.ndarray,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    config: LiftingConfig,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict of pseudo microstates in PCA space:
        {
            'gh_global': (n_pseudo, n_samples, n_components),
            'gh_local':  ...,
            'convex':    ...,
            'krr':       ...,
        }
    """
    n_pseudo, n_samples, latent_dim = phi_pseudo.shape
    n_components = all_frames.shape[2]
    
    # Flatten for batch processing
    query_coords = phi_pseudo.reshape(-1, latent_dim) # (n_pseudo * n_samples, latent_dim)
    
    # 1. GH Global
    print("Lifting with GH Global...")
    gh_lifts = lift_with_geometric_harmonics(
        models.gh_model,
        query_coords,
        local_delta=config.gh_local_delta,
        local_ridge=config.gh_local_ridge,
        local_neighbors=config.gh_local_neighbors,
        max_local_modes=config.gh_local_max_modes,
    )
    X_gh_global = gh_lifts["global"].reshape(n_pseudo, n_samples, n_components)
    X_gh_local = gh_lifts["local"].reshape(n_pseudo, n_samples, n_components)
    
    # 2. Convex Hull
    print("Lifting with Convex Hull...")
    X_convex_flat = lift_with_convex_hull(
        models.convex,
        query_coords,
        k=config.convex_k,
        max_iter=config.convex_max_iter,
    )
    X_convex = X_convex_flat.reshape(n_pseudo, n_samples, n_components)
    
    # 3. KRR
    print("Lifting with KRR...")
    # KRR needs to be done per time step because it depends on the target time t_star
    X_krr = np.zeros((n_pseudo, n_samples, n_components))
    
    for i, t_star in enumerate(t_pseudo):
        g_star_t = phi_pseudo[i] # (n_samples, latent_dim)
        
        X_krr_t = lift_with_time_local_kernel_ridge(
            macro_time_series=tc_embeddings_time,
            micro_time_series=all_frames,
            snapshot_times=times_arr,
            holdout_snapshot_time=t_star,
            g_star=g_star_t,
            spatial_length_scale=config.krr_spatial_length_scale,
            temporal_length_scale=config.krr_temporal_length_scale,
            time_scaling=config.krr_time_scaling,
            ridge_penalty=config.krr_ridge_penalty,
        )
        X_krr[i] = X_krr_t
        
    return {
        'gh_global': X_gh_global,
        'gh_local': X_gh_local,
        'convex': X_convex,
        'krr': X_krr,
    }

def decode_pseudo_microstates(
    pseudo_micro: Dict[str, np.ndarray],
    components, mean_vec, explained_variance, is_whitened, whitening_epsilon,
    resolution: int,
) -> Dict[str, np.ndarray]:
    """
    For each key in pseudo_micro (gh_global, gh_local, convex, krr),
    decode the PCA coefficients to flat arrays and then to images.

    Return:
        {
            'gh_global': imgs_gh_global,  # (n_pseudo, n_samples, H, W)
            ...
        }
    """
    decoded_imgs = {}
    for key, X_pca in pseudo_micro.items():
        # X_pca shape: (n_pseudo, n_samples, n_components)
        n_pseudo, n_samples, _ = X_pca.shape
        
        # Flatten for decoding
        X_pca_flat = X_pca.reshape(-1, X_pca.shape[-1])
        
        X_flat = pca_decode(
            X_pca_flat, components, mean_vec, explained_variance, is_whitened, whitening_epsilon
        )
        
        imgs = to_images(X_flat, resolution) # (n_pseudo * n_samples, H, W)
        imgs = imgs.reshape(n_pseudo, n_samples, resolution, resolution)
        decoded_imgs[key] = imgs
        
    return decoded_imgs

def evaluate_interpolation_at_observed_times(
    tc_embeddings_time: np.ndarray,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    interpolation: LatentInterpolationResult,
    models: LiftingModels,
    config: LiftingConfig,
    components, mean_vec, explained_variance, is_whitened, whitening_epsilon, resolution,
) -> Dict[str, Any]:
    """
    For each observed time:
      - compute latent interpolation at that time
      - compute embedding MSE
      - lift to microspace and compute field-space errors
    Return summary statistics (per-time and aggregated).
    """
    print("Evaluating interpolation at observed times...")
    
    # 1. Latent interpolation at observed times
    # We use the 'frechet' method as the primary one for evaluation as per prompt implication
    # or we could evaluate all. Let's stick to frechet for now or what's in config.
    
    # Actually, we need to pass the config or decide. Let's assume Frechet for now as it's the advanced one.
    
    # Get interpolated latents at observed times
    _, phi_interp_true = sample_latent_at_times(interpolation, times_arr, method='frechet')
    
    # Compute embedding MSE
    # tc_embeddings_time: (n_times, n_samples, latent_dim)
    # phi_interp_true: (n_times, n_samples, latent_dim)
    
    embedding_mse = np.mean((tc_embeddings_time - phi_interp_true) ** 2, axis=(1, 2)) # per time
    print(f"Embedding MSE per time: {embedding_mse}")
    print(f"Mean Embedding MSE: {np.mean(embedding_mse)}")
    
    # 2. Lift to microspace
    # We lift the interpolated latents at the observed times
    pseudo_micro = lift_pseudo_latents(
        phi_interp_true,
        times_arr,
        models,
        tc_embeddings_time,
        all_frames,
        times_arr,
        config
    )
    
    # Decode to fields
    decoded_imgs = decode_pseudo_microstates(
        pseudo_micro, components, mean_vec, explained_variance, is_whitened, whitening_epsilon, resolution
    )
    
    # True images
    # all_frames is PCA. Need to decode to images for comparison.
    X_true_flat = pca_decode(
        all_frames.reshape(-1, all_frames.shape[-1]), 
        components, mean_vec, explained_variance, is_whitened, whitening_epsilon
    )
    imgs_true = to_images(X_true_flat, resolution).reshape(len(times_arr), all_frames.shape[1], resolution, resolution)
    
    # Compute metrics
    metrics = {}
    for method, imgs_pred in decoded_imgs.items():
        # imgs_pred: (n_times, n_samples, H, W)
        mse = np.mean((imgs_true - imgs_pred) ** 2, axis=(1, 2, 3)) # per time
        metrics[f"{method}_mse"] = mse
        print(f"Method {method} Mean MSE: {np.mean(mse)}")
        
    return {
        'embedding_mse': embedding_mse,
        'field_metrics': metrics,
        'times': times_arr
    }

def plot_field_time_strips(
    imgs_true_times: Dict[float, np.ndarray],   # {t_true: (n_samples, H, W)}
    imgs_pseudo_times: Dict[float, Dict[str, np.ndarray]],  # {t_star: {method: (n_samples, H, W)}}
    sample_indices: Tuple[int, ...],
    times_arr: np.ndarray,
):
    """
    Produce multi-row, multi-column matplotlib figures.
    Columns: Time (sorted union of true and pseudo times)
    Rows: True (if available), then Methods
    """
    # Collect all unique times
    true_times = sorted(imgs_true_times.keys())
    pseudo_times = sorted(imgs_pseudo_times.keys())
    all_times = sorted(list(set(true_times + pseudo_times)))
    
    methods = list(list(imgs_pseudo_times.values())[0].keys())
    
    n_samples = len(sample_indices)
    n_times = len(all_times)
    n_rows = 1 + len(methods) # True + methods
    
    for sample_idx in sample_indices:
        fig, axes = plt.subplots(n_rows, n_times, figsize=(2 * n_times, 2 * n_rows), constrained_layout=True)
        if n_times == 1:
            axes = axes[:, None] # Ensure 2D array
            
        # Plot True
        for j, t in enumerate(all_times):
            ax = axes[0, j]
            if t in imgs_true_times:
                img = imgs_true_times[t][sample_idx]
                ax.imshow(img, cmap='viridis')
                ax.set_title(f"True t={t:.2f}")
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                ax.set_title(f"t={t:.2f}")
            ax.axis('off')
            
        # Plot Methods
        for i, method in enumerate(methods):
            for j, t in enumerate(all_times):
                ax = axes[i + 1, j]
                if t in imgs_pseudo_times:
                    img = imgs_pseudo_times[t][method][sample_idx]
                    ax.imshow(img, cmap='viridis')
                    if j == 0:
                        ax.set_ylabel(method)
                elif t in imgs_true_times:
                    # If we evaluated at observed times too, we might have it.
                    # But usually imgs_pseudo_times contains the lifted results.
                    # If we lifted at observed times (which we did in evaluation), we should pass them here too.
                    # For now, assume N/A if not in pseudo dict
                    ax.text(0.5, 0.5, "-", ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                ax.axis('off')
                
        plt.suptitle(f"Sample {sample_idx} Time Strip")
        plt.show()

def plot_evaluation_metrics(eval_results: Dict[str, Any]):
    times = eval_results['times']
    embedding_mse = eval_results['embedding_mse']
    field_metrics = eval_results['field_metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    # Embedding MSE
    axes[0].plot(times, embedding_mse, 'o-', label='Embedding MSE')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Latent Interpolation Error at Observed Times')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Field MSE
    for method_key, mse in field_metrics.items():
        axes[1].plot(times, mse, 's-', label=method_key)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Field Reconstruction Error at Observed Times')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.show()

def plot_latent_trajectories_comparison(
    times_train, tc_embeddings_time, interpolation, sample_indices
):
    t_dense = interpolation.t_dense
    phi_frechet = interpolation.phi_frechet_dense
    phi_naive = interpolation.phi_naive_dense
    
    for sample_idx in sample_indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        
        # Component 1
        axes[0].plot(times_train, tc_embeddings_time[:, sample_idx, 0], 'ko', label='True')
        axes[0].plot(t_dense, phi_frechet[:, sample_idx, 0], 'g-', label='Frechet')
        if phi_naive is not None:
            axes[0].plot(t_dense, phi_naive[:, sample_idx, 0], 'b--', label='Naive')
        axes[0].set_title(f'Sample {sample_idx}: Comp 1')
        axes[0].legend()
        
        # Component 2
        axes[1].plot(times_train, tc_embeddings_time[:, sample_idx, 1], 'ko', label='True')
        axes[1].plot(t_dense, phi_frechet[:, sample_idx, 1], 'g-', label='Frechet')
        if phi_naive is not None:
            axes[1].plot(t_dense, phi_naive[:, sample_idx, 1], 'b--', label='Naive')
        axes[1].set_title(f'Sample {sample_idx}: Comp 2')
        
        # Phase Plane
        axes[2].plot(phi_frechet[:, sample_idx, 0], phi_frechet[:, sample_idx, 1], 'g-', label='Frechet')
        if phi_naive is not None:
            axes[2].plot(phi_naive[:, sample_idx, 0], phi_naive[:, sample_idx, 1], 'b--', label='Naive')
        axes[2].scatter(tc_embeddings_time[:, sample_idx, 0], tc_embeddings_time[:, sample_idx, 1], c='k', label='True')
        axes[2].set_title(f'Sample {sample_idx}: Phase Plane')
        
        plt.show()

def run_pseudo_preimage_experiment():
    # 1. Load data
    times_arr, all_frames, components, mean_vec, explained_variance, is_whitened, whitening_epsilon, resolution, raw_marginals = load_tran_inclusions_data()
    
    # 2. Build time-coupled embeddings
    tc_result, tc_embeddings_time, selected_epsilons, kde_bandwidths, connectivity_df = build_time_coupled_embeddings(
        all_frames, times_arr, tc_k=8, alpha=0.5, beta=-0.2, use_variable_bandwidth=True
    )
    
    # 3. Build dense latent interpolation
    pseudo_config = PseudoDataConfig()
    interpolation = build_dense_latent_trajectories(
        tc_result, times_arr, tc_embeddings_time, n_dense=pseudo_config.n_dense
    )
    
    # 4. Choose pseudo_times
    pseudo_times = choose_pseudo_times_per_interval(
        times_arr, n_per_interval=pseudo_config.n_pseudo_per_interval
    )
    print(f"Chosen pseudo times: {pseudo_times}")

    # 4b. Generate pseudo PCA states (Layer 1) & Augment Data
    print("Generating pseudo PCA states (Layer 1) and augmenting training data...")
    
    aug_times_list = list(times_arr)
    aug_embeddings_list = [tc_embeddings_time[i] for i in range(len(times_arr))]
    aug_frames_list = [all_frames[i] for i in range(len(times_arr))]
    
    sanity_check_data = [] 
    
    for i in range(len(times_arr) - 1):
        t_start = times_arr[i]
        t_end = times_arr[i+1]
        X_i = all_frames[i]
        
        interval_pseudo_times = [t for t in pseudo_times if t_start < t < t_end]
        if not interval_pseudo_times:
            continue
            
        thetas = [(t - t_start) / (t_end - t_start) for t in interval_pseudo_times]
        
        d2 = pdist(X_i, metric='sqeuclidean')
        eps = np.median(d2) 
        
        P = build_pca_markov_operator_for_interval(X_i, eps)
        pseudo_states_dict = generate_pseudo_pca_states_for_interval(X_i, P, thetas)
        
        if i == 0:
            sanity_check_data.append((X_i, all_frames[i+1], pseudo_states_dict, t_start, t_end))
            
        _, phi_pseudo_interval = sample_latent_at_times(
            interpolation, np.array(interval_pseudo_times), method=pseudo_config.interp_method
        )
        
        for j, t in enumerate(interval_pseudo_times):
            theta = thetas[j]
            X_pseudo = pseudo_states_dict[theta]
            z_pseudo = phi_pseudo_interval[j]
            
            aug_times_list.append(t)
            aug_embeddings_list.append(z_pseudo)
            aug_frames_list.append(X_pseudo)
            
    aug_data = sorted(zip(aug_times_list, aug_embeddings_list, aug_frames_list), key=lambda x: x[0])
    times_aug = np.array([x[0] for x in aug_data])
    tc_embeddings_aug = np.stack([x[1] for x in aug_data])
    all_frames_aug = np.stack([x[2] for x in aug_data])
    
    print(f"Augmented data shapes: times {times_aug.shape}, embeddings {tc_embeddings_aug.shape}, frames {all_frames_aug.shape}")
    
    if sanity_check_data:
        X_start, X_end, p_dict, t_s, t_e = sanity_check_data[0]
        plt.figure(figsize=(8, 6))
        n_vis = 5
        for s_idx in range(n_vis):
            plt.scatter(X_start[s_idx, 0], X_start[s_idx, 1], c='blue', marker='o', label='Start' if s_idx==0 else "")
            plt.scatter(X_end[s_idx, 0], X_end[s_idx, 1], c='red', marker='x', label='End' if s_idx==0 else "")
            for theta, X_p in p_dict.items():
                plt.scatter(X_p[s_idx, 0], X_p[s_idx, 1], c='green', marker='.', s=20, alpha=0.5, label='Pseudo' if s_idx==0 and theta==list(p_dict.keys())[0] else "")
        plt.title(f"PCA Layer Interpolation Sanity Check (Interval {t_s:.2f}-{t_e:.2f})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.savefig("sanity_check_pca.png")
        plt.close()
    
    # 5. Fit lifting models
    lifting_config = LiftingConfig()
    models, metadata = fit_lifting_models(
        tc_embeddings_aug, all_frames_aug, times_aug, lifting_config
    )
    
    # 6. Evaluate at observed times (error curves)
    eval_results = evaluate_interpolation_at_observed_times(
        tc_embeddings_time, all_frames, times_arr, interpolation, models, lifting_config,
        components, mean_vec, explained_variance, is_whitened, whitening_epsilon, resolution
    )
    plot_evaluation_metrics(eval_results)
    
    # 7. Lift pseudo latents and decode to fields
    # Sample latent at pseudo times
    _, phi_pseudo = sample_latent_at_times(interpolation, pseudo_times, method=pseudo_config.interp_method)
    
    # Lift
    pseudo_micro = lift_pseudo_latents(
        phi_pseudo, pseudo_times, models, tc_embeddings_time, all_frames, times_arr, lifting_config
    )
    
    # Decode
    decoded_imgs_pseudo = decode_pseudo_microstates(
        pseudo_micro, components, mean_vec, explained_variance, is_whitened, whitening_epsilon, resolution
    )
    
    # Prepare true images for plotting
    # We need true images at observed times
    X_true_flat = pca_decode(
        all_frames.reshape(-1, all_frames.shape[-1]), 
        components, mean_vec, explained_variance, is_whitened, whitening_epsilon
    )
    imgs_true_all = to_images(X_true_flat, resolution).reshape(len(times_arr), all_frames.shape[1], resolution, resolution)
    imgs_true_times = {t: imgs_true_all[i] for i, t in enumerate(times_arr)}
    
    # Prepare pseudo images dict
    # {t_star: {method: (n_samples, H, W)}}
    imgs_pseudo_times = {}
    for i, t in enumerate(pseudo_times):
        imgs_pseudo_times[t] = {}
        for method, imgs in decoded_imgs_pseudo.items():
            imgs_pseudo_times[t][method] = imgs[i]
            
    # 8. Generate plots
    # Latent trajectories
    plot_latent_trajectories_comparison(
        times_arr, tc_embeddings_time, interpolation, 
        sample_indices=tuple(range(min(pseudo_config.n_samples_vis, all_frames.shape[1])))
    )
    
    # Field time strips
    plot_field_time_strips(
        imgs_true_times, imgs_pseudo_times, 
        sample_indices=tuple(range(min(pseudo_config.n_samples_fields, all_frames.shape[1]))),
        times_arr=times_arr
    )


def build_pca_markov_operator_for_interval(
    X_i: np.ndarray,  # (n_samples, n_pca_dims)
    eps: float,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Build a row-stochastic Markov operator P_{i+1} on samples at time s_i,
    using the same kernel construction as in diffusion_maps.py but applied
    to PCA coordinates.
    """
    # 1. Compute kernel K_ij = exp(-||x_i - x_j||^2 / eps)
    d2 = squareform(pdist(X_i, metric='sqeuclidean'))
    K = np.exp(-d2 / eps)
    
    # 2. Alpha-normalization (optional but recommended if density varies)
    if alpha > 0:
        q = np.sum(K, axis=1, keepdims=True)
        q_alpha = q ** (-alpha)
        K = q_alpha * K * q_alpha.T
        
    # 3. Compute degree vector d_i = sum_j K_ij
    d = np.sum(K, axis=1, keepdims=True)
    
    # 4. Compute normalized Markov operator P = D^{-1} K
    P = K / d
    return P

def fractional_markov_power(P: np.ndarray, theta: float) -> np.ndarray:
    """
    Given a Markov matrix P (n_samples x n_samples),
    compute P^{theta} using spectral decomposition.
    """
    # 1. eigvals, eigvecs
    eigvals, eigvecs = np.linalg.eig(P)
    
    # 2. Construct Λ^{theta}
    # Handle complex values carefully, though P is usually real.
    # eigvals might be complex due to numerical noise or non-symmetry (though P is similar to symmetric)
    # We take complex power then real part at the end.
    Lambda_theta = np.diag(eigvals ** theta)
    
    # 3. Rebuild P_theta = eigvecs @ Λ_theta @ inv(eigvecs)
    P_theta = eigvecs @ Lambda_theta @ np.linalg.inv(eigvecs)
    
    return P_theta.real

def generate_pseudo_pca_states_for_interval(
    X_i: np.ndarray,    # (n_samples, n_pca_dims)
    P_i_plus_1: np.ndarray,   # (n_samples, n_samples)
    thetas: List[float],
) -> Dict[float, np.ndarray]:
    """
    For each theta in [0,1], compute X_{i,theta} = P_{i+1}^{(theta)} X_i.
    Returns a dict mapping theta -> (n_samples, n_pca_dims).
    """
    pseudo_states = {}
    for theta in thetas:
        P_theta = fractional_markov_power(P_i_plus_1, theta)
        X_i_theta = P_theta @ X_i
        pseudo_states[theta] = X_i_theta
    return pseudo_states

