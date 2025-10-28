"""Visualization tools for reconstructed random fields from PCA coefficients.

This module provides functions to visualize the reconstructed 2D random fields
from PCA coefficient trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
import wandb


def reconstruct_fields_from_coefficients(coeffs, pca_info, resolution):
    """Reconstruct 2D random fields from PCA coefficients.
    
    Args:
        coeffs: numpy array of shape (T, N, d) or (N, d) where:
            T = number of time points (optional)
            N = number of samples
            d = number of PCA components
        pca_info: dict with 'components', 'mean', 'explained_variance'
        resolution: int, spatial resolution of the original fields (assumes square)
    
    Returns:
        fields: numpy array of shape (T, N, resolution, resolution) or (N, resolution, resolution)
    """
    mean = pca_info['mean']
    is_whitened = bool(pca_info.get('is_whitened', True))
    eigenvectors = pca_info['components'].T
    eigenvalues = pca_info['explained_variance']
    if is_whitened:
        sqrt_eig = np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
    else:
        components = pca_info['components']
    
    # Handle both 2D and 3D inputs
    original_shape = coeffs.shape
    if coeffs.ndim == 2:
        coeffs = coeffs[np.newaxis, ...]  # Add time dimension
        squeeze_time = True
    else:
        squeeze_time = False
    
    T, N, d = coeffs.shape
    data_dim = resolution * resolution
    
    fields = []
    for t in range(T):
        if is_whitened:
            scaled_coeffs = coeffs[t] @ sqrt_eig
            reconstructed = scaled_coeffs @ eigenvectors.T + mean
        else:
            reconstructed = coeffs[t] @ components + mean
        # Reshape to 2D fields
        fields_2d = reconstructed.reshape(N, resolution, resolution)
        fields.append(fields_2d)
    
    fields = np.array(fields)
    
    if squeeze_time:
        fields = fields.squeeze(0)
    
    return fields


def plot_field_snapshots(fields, zt, outdir, run, n_samples=5, score=False, cmap='viridis'):
    """Plot snapshots of reconstructed fields at different time points.
    
    Args:
        fields: numpy array of shape (T, N, H, W)
        zt: array of time values corresponding to marginals (length M)
        outdir: output directory for saving plots
        run: wandb run object
        n_samples: number of sample fields to display
        score: bool, whether this is from SDE (True) or ODE (False)
        cmap: matplotlib colormap
    """
    T, N, H, W = fields.shape
    M = len(zt)  # Number of marginals in original data
    n_samples = min(n_samples, N)
    
    # Select evenly spaced time indices from the trajectory that correspond to marginals
    # Map marginal times to trajectory indices
    if T >= M:
        time_indices = np.linspace(0, T-1, M).astype(int)
    else:
        # If fewer trajectory points than marginals, just use all trajectory points
        time_indices = np.arange(T)
        M = T
    
    # Create grid: rows = samples, cols = time points
    fig = plt.figure(figsize=(3 * M, 3 * n_samples))
    gs = gridspec.GridSpec(n_samples, M, figure=fig, hspace=0.3, wspace=0.3)
    
    # Find global vmin/vmax for consistent coloring
    vmin = np.percentile(fields, 1)
    vmax = np.percentile(fields, 99)
    
    for i in range(n_samples):
        for j, t_idx in enumerate(time_indices):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(fields[t_idx, i], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                # Use the actual zt time if we have it, otherwise interpolate
                t_val = zt[j] if j < len(zt) else zt[0] + (zt[-1] - zt[0]) * t_idx / (T - 1)
                ax.set_title(f't = {t_val:.2f}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}', fontsize=10)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    diffeq = 'sde' if score else 'ode'
    filename = f'{outdir}/field_snapshots_{diffeq}'
    fig.savefig(f'{filename}.png', bbox_inches='tight', dpi=150)
    fig.savefig(f'{filename}.pdf', bbox_inches='tight')
    
    img = wandb.Image(
        data_or_path=f'{filename}.png',
        mode='RGB'
    )
    run.log({f'visualizations/field_snapshots_{diffeq}': img})
    
    plt.close(fig)


def plot_field_evolution_gif(fields, zt, outdir, run, sample_idx=0, score=False, cmap='viridis', fps=5):
    """Create animated GIF showing evolution of a single field through time.
    
    Args:
        fields: numpy array of shape (T, N, H, W)
        zt: array of time values for marginals (length M)
        outdir: output directory
        run: wandb run object
        sample_idx: which sample to animate
        score: bool, whether from SDE or ODE
        cmap: matplotlib colormap
        fps: frames per second
    """
    T, N, H, W = fields.shape
    
    # Create interpolated time values for all trajectory points
    t_interp = np.linspace(zt[0], zt[-1], T)
    
    # Find global vmin/vmax
    vmin = np.percentile(fields[:, sample_idx], 1)
    vmax = np.percentile(fields[:, sample_idx], 99)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def animate(i):
        ax.clear()
        im = ax.imshow(fields[i, sample_idx], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'Sample {sample_idx + 1}, t = {t_interp[i]:.2f}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return [im]
    
    ani = animation.FuncAnimation(
        fig, animate, frames=T, interval=1000//fps, blit=True
    )
    
    diffeq = 'sde' if score else 'ode'
    filename = f'{outdir}/field_evolution_{diffeq}_sample{sample_idx}.gif'
    ani.save(filename, writer='imagemagick', fps=fps)
    
    gif = wandb.Video(data_or_path=filename, format='gif')
    run.log({f'visualizations/field_evolution_{diffeq}_sample{sample_idx}': gif})
    
    plt.close(fig)


def plot_field_statistics(fields, zt, testdata_fields, outdir, run, score=False):
    """Plot statistical comparisons between generated and test fields.
    
    Args:
        fields: generated fields of shape (T, N, H, W)
        zt: time values for marginals (length M)
        testdata_fields: test fields of shape (M, N_test, H, W) where M is number of marginals
        outdir: output directory
        run: wandb run object
        score: bool, whether from SDE or ODE
    """
    T, N, H, W = fields.shape
    M = len(testdata_fields)
    
    # Create interpolated time values for all trajectory points
    t_interp = np.linspace(zt[0], zt[-1], T)
    
    # Compute mean and std for generated fields
    gen_means = np.mean(fields, axis=(1, 2, 3))  # Mean over samples and spatial dims
    gen_stds = np.std(fields, axis=(1, 2, 3))
    
    # Compute mean and std for test fields
    test_means = np.array([np.mean(testdata_fields[m]) for m in range(M)])
    test_stds = np.array([np.std(testdata_fields[m]) for m in range(M)])
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean comparison
    axs[0].plot(t_interp, gen_means, 'o-', label='Generated', markersize=6, alpha=0.7)
    axs[0].plot(zt[:M], test_means, 's-', label='Test Data', markersize=8)
    axs[0].set_xlabel('Time', fontsize=12)
    axs[0].set_ylabel('Mean Field Value', fontsize=12)
    axs[0].set_title('Mean Field Value Over Time', fontsize=14)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Std comparison
    axs[1].plot(t_interp, gen_stds, 'o-', label='Generated', markersize=6, alpha=0.7)
    axs[1].plot(zt[:M], test_stds, 's-', label='Test Data', markersize=8)
    axs[1].set_xlabel('Time', fontsize=12)
    axs[1].set_ylabel('Std Field Value', fontsize=12)
    axs[1].set_title('Std Field Value Over Time', fontsize=14)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    diffeq = 'sde' if score else 'ode'
    filename = f'{outdir}/field_statistics_{diffeq}'
    fig.savefig(f'{filename}.png', bbox_inches='tight', dpi=150)
    fig.savefig(f'{filename}.pdf', bbox_inches='tight')
    
    img = wandb.Image(data_or_path=f'{filename}.png', mode='RGB')
    run.log({f'evalplots/field_statistics_{diffeq}': img})
    
    plt.close(fig)


def plot_spatial_correlation(fields, zt, outdir, run, score=False):
    """Plot spatial autocorrelation structure of reconstructed fields.
    
    Args:
        fields: numpy array of shape (T, N, H, W)
        zt: time values for marginals (length M)
        outdir: output directory
        run: wandb run object
        score: bool, whether from SDE or ODE
    """
    from scipy.signal import correlate2d
    
    T, N, H, W = fields.shape
    n_times = min(5, T)  # Plot correlation for up to 5 time points
    time_indices = np.linspace(0, T-1, n_times).astype(int)
    
    # Create interpolated time values
    t_interp = np.linspace(zt[0], zt[-1], T)
    
    fig, axs = plt.subplots(1, n_times, figsize=(4 * n_times, 4))
    if n_times == 1:
        axs = [axs]
    
    for idx, t_idx in enumerate(time_indices):
        # Compute average autocorrelation across samples
        autocorrs = []
        for i in range(min(20, N)):  # Use first 20 samples
            field = fields[t_idx, i]
            # Normalize field
            field_norm = (field - field.mean()) / (field.std() + 1e-9)
            # Compute 2D autocorrelation
            autocorr = correlate2d(field_norm, field_norm, mode='same')
            autocorr = autocorr / (autocorr.max() + 1e-9)  # Normalize
            autocorrs.append(autocorr)
        
        avg_autocorr = np.mean(autocorrs, axis=0)
        
        im = axs[idx].imshow(avg_autocorr, cmap='RdBu_r', vmin=-0.3, vmax=1.0, origin='lower')
        axs[idx].set_title(f't = {t_interp[t_idx]:.2f}', fontsize=12)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        fig.colorbar(im, ax=axs[idx], fraction=0.046, pad=0.04)
    
    fig.suptitle('Spatial Autocorrelation', fontsize=16)
    fig.tight_layout()
    
    diffeq = 'sde' if score else 'ode'
    filename = f'{outdir}/spatial_correlation_{diffeq}'
    fig.savefig(f'{filename}.png', bbox_inches='tight', dpi=150)
    fig.savefig(f'{filename}.pdf', bbox_inches='tight')
    
    img = wandb.Image(data_or_path=f'{filename}.png', mode='RGB')
    run.log({f'evalplots/spatial_correlation_{diffeq}': img})
    
    plt.close(fig)


def visualize_all_field_reconstructions(
    traj_coeffs, testdata, pca_info, zt, outdir, run, score=False
):
    """Main function to create all field reconstruction visualizations.
    
    Args:
        traj_coeffs: trajectory in PCA coefficient space, shape (T, N, d)
        testdata: list of test data arrays in PCA coefficient space
        pca_info: dict with PCA components and stats
        zt: array of time values
        outdir: output directory
        run: wandb run object
        score: bool, whether from SDE or ODE
    """
    # Infer resolution from data dimension
    data_dim = int(pca_info['data_dim'])
    resolution = int(np.sqrt(data_dim))
    
    print(f'Reconstructing fields with resolution {resolution}x{resolution}...')
    
    # Reconstruct fields from trajectory coefficients
    fields = reconstruct_fields_from_coefficients(traj_coeffs, pca_info, resolution)
    print(f'Generated fields shape: {fields.shape}')
    
    # Reconstruct test data fields for comparison
    testdata_fields = []
    for test_marginal in testdata:
        test_fields = reconstruct_fields_from_coefficients(
            test_marginal, pca_info, resolution
        )
        testdata_fields.append(test_fields)
    
    print('Creating field visualizations...')
    
    # Create all visualizations
    plot_field_snapshots(fields, zt, outdir, run, n_samples=5, score=score)
    plot_field_evolution_gif(fields, zt, outdir, run, sample_idx=0, score=score, fps=5)
    plot_field_statistics(fields, zt, testdata_fields, outdir, run, score=score)
    plot_spatial_correlation(fields, zt, outdir, run, score=score)
    
    # Create additional GIFs for a few more samples
    for sample_idx in [1, 2]:
        if sample_idx < fields.shape[1]:
            plot_field_evolution_gif(
                fields, zt, outdir, run, sample_idx=sample_idx, score=score, fps=5
            )
    
    print('Field visualizations complete!')
