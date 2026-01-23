from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
from mmsfm.data_utils import pca_decode, to_images
from tran_inclusions.config import LatentInterpolationResult

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

def visualize_fractional_pca_intervals(
    *,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    frames_aug: np.ndarray,
    pseudo_meta: Dict[str, Any],
    components,
    mean_vec,
    explained_variance,
    is_whitened: bool,
    whitening_epsilon: float,
    resolution: int,
    sample_indices: Tuple[int, ...],
) -> None:
    """Decode and plot pseudo PCA states between successive times."""
    pseudo_entries = [e for e in pseudo_meta['entries'] if e['kind'] == 'pseudo']
    if not pseudo_entries:
        print("No pseudo entries to visualise.")
        return

    pseudo_by_interval: Dict[int, list[dict[str, Any]]] = {}
    for entry in pseudo_entries:
        pseudo_by_interval.setdefault(entry['interval_index'], []).append(entry)
    for entries in pseudo_by_interval.values():
        entries.sort(key=lambda e: e['time'])

    for interval_idx, entries in pseudo_by_interval.items():
        t_start = times_arr[interval_idx]
        t_end = times_arr[interval_idx + 1]
        frame_start = all_frames[interval_idx]
        frame_end = all_frames[interval_idx + 1]

        # Decode true endpoints
        imgs_true_times: Dict[float, np.ndarray] = {}
        X_start_flat = pca_decode(frame_start, components, mean_vec, explained_variance, is_whitened, whitening_epsilon)
        X_end_flat = pca_decode(frame_end, components, mean_vec, explained_variance, is_whitened, whitening_epsilon)
        imgs_true_times[float(t_start)] = to_images(X_start_flat, resolution)
        imgs_true_times[float(t_end)] = to_images(X_end_flat, resolution)

        # Decode pseudo frames
        imgs_pseudo_times: Dict[float, Dict[str, np.ndarray]] = {}
        for entry in entries:
            t_star = float(entry['time'])
            idx = entry['aug_index']
            X_pseudo_flat = pca_decode(
                frames_aug[idx],
                components,
                mean_vec,
                explained_variance,
                is_whitened,
                whitening_epsilon,
            )
            imgs_pseudo_times[t_star] = {'fused_step': to_images(X_pseudo_flat, resolution)}

        print(f"Interval [{t_start:.2f}, {t_end:.2f}] → {len(entries)} pseudo states at {sorted(imgs_pseudo_times.keys())}")
        plot_field_time_strips(
            imgs_true_times,
            imgs_pseudo_times,
            sample_indices=sample_indices,
            times_arr=times_arr,
        )

def plot_latent_trajectories_comparison(
    times_train,
    tc_embeddings_time,
    interpolation_triplet,
    sample_indices,
    interpolation_global: Optional[LatentInterpolationResult] = None,
):
    t_dense = interpolation_triplet.t_dense
    
    # Try to get specific triplet interpolation, fallback to generic if not available
    phi_frechet_triplet = getattr(interpolation_triplet, 'phi_frechet_triplet_dense', None)
    if phi_frechet_triplet is None:
        phi_frechet_triplet = interpolation_triplet.phi_frechet_dense

    phi_linear_triplet = getattr(interpolation_triplet, 'phi_linear_dense', None)
    phi_naive = interpolation_triplet.phi_naive_dense
    
    phi_frechet_global = None
    phi_linear_global = None
    if interpolation_global is not None:
        # Try to get specific global interpolation, fallback to generic
        phi_frechet_global = getattr(interpolation_global, 'phi_frechet_global_dense', None)
        if phi_frechet_global is None:
            phi_frechet_global = interpolation_global.phi_frechet_dense
        phi_linear_global = getattr(interpolation_global, 'phi_linear_dense', None)
    
    for sample_idx in sample_indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        
        # Component 1
        axes[0].plot(times_train, tc_embeddings_time[:, sample_idx, 0], 'ko', label='True')
        axes[0].plot(t_dense, phi_frechet_triplet[:, sample_idx, 0], 'g-', label='Frechet triplet (pchip)')
        if phi_linear_triplet is not None:
            axes[0].plot(t_dense, phi_linear_triplet[:, sample_idx, 0], 'm-.', label='Frechet triplet (linear)')
        if phi_frechet_global is not None:
            axes[0].plot(t_dense, phi_frechet_global[:, sample_idx, 0], 'c--', label='Frechet global (pchip)')
        if phi_linear_global is not None:
            axes[0].plot(t_dense, phi_linear_global[:, sample_idx, 0], color='tab:cyan', linestyle=':', label='Frechet global (linear)')
        if phi_naive is not None:
            axes[0].plot(t_dense, phi_naive[:, sample_idx, 0], 'b--', label='Naive')
        axes[0].set_title(f'Sample {sample_idx}: Comp 1')
        axes[0].legend()
        
        # Component 2
        axes[1].plot(times_train, tc_embeddings_time[:, sample_idx, 1], 'ko', label='True')
        axes[1].plot(t_dense, phi_frechet_triplet[:, sample_idx, 1], 'g-', label='Frechet triplet (pchip)')
        if phi_linear_triplet is not None:
            axes[1].plot(t_dense, phi_linear_triplet[:, sample_idx, 1], 'm-.', label='Frechet triplet (linear)')
        if phi_frechet_global is not None:
            axes[1].plot(t_dense, phi_frechet_global[:, sample_idx, 1], 'c--', label='Frechet global (pchip)')
        if phi_linear_global is not None:
            axes[1].plot(t_dense, phi_linear_global[:, sample_idx, 1], color='tab:cyan', linestyle=':', label='Frechet global (linear)')
        if phi_naive is not None:
            axes[1].plot(t_dense, phi_naive[:, sample_idx, 1], 'b--', label='Naive')
        axes[1].set_title(f'Sample {sample_idx}: Comp 2')
        
        # Phase Plane
        axes[2].plot(phi_frechet_triplet[:, sample_idx, 0], phi_frechet_triplet[:, sample_idx, 1], 'g-', label='Frechet triplet (pchip)')
        if phi_linear_triplet is not None:
            axes[2].plot(phi_linear_triplet[:, sample_idx, 0], phi_linear_triplet[:, sample_idx, 1], 'm-.', label='Frechet triplet (linear)')
        if phi_frechet_global is not None:
            axes[2].plot(phi_frechet_global[:, sample_idx, 0], phi_frechet_global[:, sample_idx, 1], 'c--', label='Frechet global (pchip)')
        if phi_linear_global is not None:
            axes[2].plot(phi_linear_global[:, sample_idx, 0], phi_linear_global[:, sample_idx, 1], color='tab:cyan', linestyle=':', label='Frechet global (linear)')
        if phi_naive is not None:
            axes[2].plot(phi_naive[:, sample_idx, 0], phi_naive[:, sample_idx, 1], 'b--', label='Naive')
        axes[2].scatter(tc_embeddings_time[:, sample_idx, 0], tc_embeddings_time[:, sample_idx, 1], c='k', label='True')
        axes[2].set_title(f'Sample {sample_idx}: Phase Plane')
        
        plt.show()
