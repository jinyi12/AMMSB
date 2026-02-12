# %% [markdown]
# # Tran et al. inclusions: micro → meso → macro

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy.io
import subprocess
import os

sys.path.append(str(Path.cwd().parent))

data_path = Path("../data/tran_inclusions.npz")
npz = np.load(data_path)
resolution = int(np.sqrt(int(npz["data_dim"])))

# Debug: print all available keys
print("Available keys in npz:", list(npz.files))

# Extract available times from marginals (not raw_marginal_)
time_keys = [k for k in npz.files if k.startswith("marginal_")]
times = sorted(float(k.replace("marginal_", "")) for k in time_keys)
print("Times:", times)

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
# Debug: show available keys
print("Available keys in npz:", list(npz.files))

def get_marginals(npz_obj, choice="pca"):
    raw_keys = [k for k in npz_obj.files if k.startswith("raw_marginal_")]
    pca_keys = [k for k in npz_obj.files if k.startswith("marginal_")]
    
    if choice == "raw" and raw_keys:
        pairs = []
        for k in raw_keys:
            suffix = k[len("raw_marginal_"):]
            try:
                t = float(suffix)
            except ValueError:
                continue
            pairs.append((t, k))
        pairs.sort(key=lambda x: x[0])
        times = [t for t, _ in pairs]
        return times, {t: npz_obj[k] for t, k in pairs}, "raw"

    if choice == "pca" and pca_keys:
        pairs = []
        for k in pca_keys:
            suffix = k[len("marginal_"):]
            try:
                t = float(suffix)
            except ValueError:
                continue
            pairs.append((t, k))
        pairs.sort(key=lambda x: x[0])
        times = [t for t, _ in pairs]
        return times, {t: npz_obj[k] for t, k in pairs}, "pca"

    raise ValueError(f"No marginal keys found in npz. Available keys: {list(npz_obj.files)}")

def invert_pca(coeffs, components, mean, explained_variance, whitened, whitening_epsilon):
    coeffs = np.asarray(coeffs)
    components = np.asarray(components)
    mean = np.asarray(mean)

    if whitened:
        eig_floor = np.maximum(explained_variance, whitening_epsilon)
        coeffs = coeffs * np.sqrt(eig_floor)[None, :]

    recon = coeffs @ components  # (N, n_features)
    recon = recon + mean[None, :]
    return recon

# Acquire marginals (raw preferred; PCA otherwise)
times, marginals, mode = get_marginals(npz, choice="pca")
print("Using mode:", mode, "Found times:", times)

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

for ax, t in zip(axes, times):
    data = marginals[t]
    if sample_idx >= data.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range for time {t} (len={data.shape[0]})")

    if mode == "pca":
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

from scipy.spatial.distance import pdist, squareform
from diffmap.diffusion_maps import (
    select_epsilons_by_connectivity,
    time_coupled_diffusion_map,
    build_time_coupled_trajectory,
    fit_coordinate_splines,
    evaluate_coordinate_splines,
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

plt.figure(figsize=(6, 4))
plt.plot(tc_result_finaltime.singular_values, 'o-')
plt.yscale('log')
plt.xlabel('index')
plt.ylabel('singular value')
plt.title('Time-coupled operator spectrum')
plt.grid(alpha=0.4, which='both')
plt.show()


# %% [markdown]
# ## Trajectory Visualization
# Propagate the diffusion coordinates through time and inspect the latent trajectories.

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
from stiefel.stiefel import batch_stiefel_log, Stiefel_Log, Stiefel_Exp

# %%
# Stiefel_Log(tc_result.left_singular_vectors[1], tc_result.left_singular_vectors[2], metric_alpha=1e-8, tau=1e-3)[0]


class GrassmannInterpolator:
    def __init__(self, matlab_pkg_path="GrassmannInterpolation-main"):
        """
        Initialize the interface to the MATLAB GrassmannInterpolation package.
        
        Args:
            matlab_pkg_path: Relative path to the folder containing the .m files
                             (Interpolate_Gr.m, maxvol.m, etc.)
        """
        self.pkg_path = Path(matlab_pkg_path).resolve()
        if not self.pkg_path.exists():
            raise FileNotFoundError(f"Could not find MATLAB package at {self.pkg_path}")

    def interpolate(self, 
                    U_list: list[np.ndarray], 
                    t_train: np.ndarray, 
                    t_query: np.ndarray, 
                    method: str = 'local_lag', 
                    use_maxvol: bool = True) -> np.ndarray:
        """
        Interpolate Grassmann points (Stiefel matrices) using MATLAB.
        
        Args:
            U_list: List of (N, k) numpy arrays representing the left singular vectors 
                    at training times.
            t_train: Array of training time points.
            t_query: Array of target time points to interpolate.
            method: Interpolation method ('local_lag', 'normal_lag'). 
                    'local_lag' is recommended for stability.
            use_maxvol: If True, applies MaxVol preconditioning (recommended).
            
        Returns:
            U_interpolated: (N, k, n_query) array of interpolated frames.
        """
        
        # 1. Prepare Data for MATLAB
        # Stack list into (N, k, n_train) for easier MATLAB ingestion
        U_tensor = np.stack(U_list, axis=2)
        
        temp_input = "bridge_input.mat"
        temp_output = "bridge_output.mat"
        script_name = "run_grassmann_interp.m"
        
        scipy.io.savemat(temp_input, {
            "U_tensor": U_tensor,
            "t_train": t_train,
            "t_query": t_query,
            "method": method,
            "use_maxvol": float(use_maxvol) # Convert bool to number for MATLAB
        })

        # 2. Generate MATLAB Driver Script
        # This script sets paths, transforms data, runs interpolation, and handles I/O
        # We implement a piecewise interpolation scheme similar to FN_interpolate.m
        # to handle long trajectories by using local charts.
        matlab_script = f"""
        try
            % Add package to path
            addpath('{self.pkg_path.as_posix()}');
            
            % Load data
            load('{temp_input}');
            
            % Helper to initialize tools
            M = matrix_tools();
            
            % Convert 3D tensor to Cell Array
            [N, P, n_train] = size(U_tensor);
            Data = cell(1, n_train);
            for i = 1:n_train
                Data{{i}} = U_tensor(:,:,i);
            end
            
            n_query = length(t_query);
            Y_tensor = zeros(N, P, n_query);
            
            % Piecewise interpolation loop
            for i = 1:n_query
                t = t_query(i);
                
                % Find interval [t_a, t_b]
                idx = find(t_train <= t, 1, 'last');
                if isempty(idx)
                    idx = 1;
                end
                if idx == n_train
                    idx = n_train - 1;
                end
                
                t_a = t_train(idx);
                t_b = t_train(idx+1);
                
                % Extract data for interval
                U_a = Data{{idx}};
                U_b = Data{{idx+1}};
                
                % Select Chart (MaxVol)
                if use_maxvol
                    % Try maxvol on both and pick best condition number
                    [~, P_a] = maxvol(U_a, 100);
                    [~, P_b] = maxvol(U_b, 100);
                    
                    U_a_loc_a = P_a * U_a; U_b_loc_a = P_a * U_b;
                    U_a_loc_b = P_b * U_a; U_b_loc_b = P_b * U_b;
                    
                    cond_a = max(cond(U_a_loc_a(1:P, 1:P)), cond(U_b_loc_a(1:P, 1:P)));
                    cond_b = max(cond(U_a_loc_b(1:P, 1:P)), cond(U_b_loc_b(1:P, 1:P)));
                    
                    if cond_a <= cond_b
                        P_mat = P_a;
                    else
                        P_mat = P_b;
                    end
                else
                    P_mat = speye(N);
                end
                
                % Apply permutation
                Data_local = {{P_mat * U_a, P_mat * U_b}};
                Time_local = [t_a, t_b];
                
                % Interpolate
                Y_local = Interpolate_Gr(Time_local, Data_local, t, method);
                
                % Map back
                Y_tensor(:,:,i) = P_mat' * Y_local;
            end
            
            % Save Result
            save('{temp_output}', 'Y_tensor');
            exit(0);
            
        catch ME
            disp(getReport(ME));
            exit(1);
        end
        """
        
        with open(script_name, "w") as f:
            f.write(matlab_script)

        # 3. Execute MATLAB
        # -batch implies non-interactive mode, specifically for scripts
        print("Running MATLAB interpolation...")
        result = subprocess.run(
            ["matlab", "-batch", "run_grassmann_interp"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("MATLAB Error Output:\n", result.stdout)
            raise RuntimeError("MATLAB execution failed.")

        # 4. Retrieve Results
        if not os.path.exists(temp_output):
            raise RuntimeError("MATLAB did not produce an output file.")
            
        mat_result = scipy.io.loadmat(temp_output)
        U_interpolated = mat_result['Y_tensor']
        
        # Cleanup temporary files
        for f in [temp_input, temp_output, script_name]:
            try:
                os.remove(f)
            except OSError:
                pass
                
        return U_interpolated

# --- Usage Example with your tc_result ---

# 1. Extract data from your existing tc_result
# tc_result.left_singular_vectors is a list of numpy arrays
U_train_list = tc_result.left_singular_vectors
times_train = train_times # From your existing variables

# 2. Define query times (e.g., a dense grid)
times_dense = np.linspace(times_train.min(), times_train.max(), 200)

# 3. Run Interpolation
interpolator = GrassmannInterpolator(matlab_pkg_path="../GrassmannInterpolation-main")

try:
    # This returns a (N, k, 200) array
    U_dense = interpolator.interpolate(
        U_list=U_train_list,
        t_train=times_train,
        t_query=times_dense,
        method='local_lag' # Uses local coordinates (log map) + Lagrange
    )
    
    print(f"Interpolation successful. Output shape: {U_dense.shape}")
    
    # save
    np.save("../data/U_dense_interp.npy", U_dense)

    # 4. Visualize or use U_dense
    # Example: Plot the evolution of the first coordinate of the first vector
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(times_dense, U_dense[0, 0, :], label='Interpolated')
    # Plot training points for comparison
    train_vals = [u[0,0] for u in U_train_list]
    plt.plot(times_train, train_vals, 'o', label='Training')
    plt.title("Evolution of U(1,1) over time")
    plt.legend()
    plt.show()

except Exception as e:
    print(f"Interpolation failed: {e}")