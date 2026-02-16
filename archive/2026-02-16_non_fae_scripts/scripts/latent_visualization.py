"""Latent space visualization utilities for geodesic autoencoders.

This module provides unified visualization functions for comparing learned vs reference
latent spaces across different autoencoder types (base GeodesicAutoencoder and
CascadedResidualAutoencoder).

Functions:
    visualize_latent_comparison: Create multi-panel scatter plots comparing learned
        and reference latent embeddings across time marginals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor


def visualize_latent_comparison(
    encoder: Union[torch.nn.Module, object],
    x_train: Tensor,
    latent_ref: Tensor,
    zt_train_times: np.ndarray,
    save_path: Path,
    device: torch.device,
    *,
    n_samples: int = 500,
    run: Optional[object] = None,
    step: int = 0,
    use_cascaded: bool = False,
    wandb_key: str = "latent_comparison",
    title: str = "Latent Comparison: Ref(blue) vs Learned(red)",
) -> None:
    """Visualize learned vs reference latent space (first 3 dims) per time marginal.

    Creates a grid of 2D or 3D scatter plots comparing reference (blue) and learned (red)
    latent embeddings at each time point. Uses consistent axis limits across all subplots
    to make contraction/expansion patterns visible.

    Args:
        encoder: Encoder model. Can be GeodesicAutoencoder, CascadedResidualAutoencoder,
            or any model with either .encode(x, t) or .encoder(x, t) methods.
        x_train: Input data tensor of shape (T, N, D) where T is number of time points,
            N is number of samples, D is input dimension.
        latent_ref: Reference latent embeddings of shape (T, N, d) where d is latent dim.
        zt_train_times: Array of normalized time values, shape (T,).
        save_path: Path where the visualization PNG will be saved.
        device: torch.device for computation.
        n_samples: Maximum number of samples to plot per time point (default: 500).
            If N > n_samples, a random subset is selected.
        run: Optional WandB run object for logging. If None, no WandB logging.
        step: Training step number for WandB logging (default: 0).
        use_cascaded: If True, forces use of .encode() method (for CascadedResidualAutoencoder).
            If False, prefers .encoder() method. If None (default), auto-detects based on
            whether encoder has an 'encode' attribute.
        wandb_key: Key name for WandB logging (default: "latent_comparison").
        title: Super title for the plot (default: "Latent Comparison: Ref(blue) vs Learned(red)").

    Returns:
        None. Saves visualization to save_path and optionally logs to WandB.

    Notes:
        - If latent dimension < 2, visualization is skipped
        - Uses 3D scatter plots if latent dimension >= 3, otherwise 2D
        - Computes and displays per-time standard deviations for diagnostic purposes
        - Axis limits are global (computed from all time points) for easy comparison
        - Falls back gracefully if matplotlib or WandB imports fail
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except Exception as e:
        print(f"  Warning: Failed to import matplotlib: {e}")
        return

    encoder.eval()

    # Check dimensions
    if latent_ref.shape[-1] < 2:
        print(f"  Skipping visualization: latent dimension {latent_ref.shape[-1]} < 2")
        return

    try:
        T = len(zt_train_times)
        n_cols = min(T, 4)
        n_rows = (T + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(4 * n_cols, 4.5 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)

        # First pass: collect all data to determine global axis limits
        all_learned = []
        all_ref = []

        with torch.no_grad():
            for t_idx in range(T):
                # Sample subset
                n_avail = x_train.shape[1]
                n_samp = min(n_samples, n_avail)

                # Generate indices on CPU, then move data to device
                j = torch.randperm(n_avail)[:n_samp]

                # Move batch to device for inference
                x_samp = x_train[t_idx, j].to(device, non_blocking=True)
                t_samp = torch.full((n_samp,), float(zt_train_times[t_idx]), device=device)

                # Auto-detect encoder type and call appropriate method
                if use_cascaded or (hasattr(encoder, "encode") and not hasattr(encoder, "encoder")):
                    # CascadedResidualAutoencoder uses .encode(x, t)
                    y_learned = encoder.encode(x_samp, t_samp).cpu().numpy()
                elif hasattr(encoder, "encoder"):
                    # GeodesicAutoencoder: use .encoder() to access base encoder
                    y_learned = encoder.encoder(x_samp, t_samp).cpu().numpy()
                else:
                    # Fallback: treat encoder as callable
                    y_learned = encoder(x_samp, t_samp).cpu().numpy()

                y_ref = latent_ref[t_idx, j].cpu().numpy()

                all_learned.append(y_learned)
                all_ref.append(y_ref)

            # Compute global axis limits from first 3 dims
            all_points = np.concatenate([arr[:, :3] for arr in all_learned + all_ref], axis=0)
            global_min = all_points.min()
            global_max = all_points.max()
            margin = (global_max - global_min) * 0.1
            axis_lim = (global_min - margin, global_max + margin)

            # Second pass: plot
            for t_idx in range(T):
                row, col = t_idx // n_cols, t_idx % n_cols

                # 3D if dim >= 3, else 2D
                is_3d = (latent_ref.shape[2] >= 3)
                ax = fig.add_subplot(gs[row, col], projection='3d' if is_3d else None)

                y_learned = all_learned[t_idx]
                y_ref = all_ref[t_idx]

                # Compute statistics (standard deviations)
                ref_std = float(y_ref.std())
                learned_std = float(y_learned.std())

                if is_3d:
                    ax.scatter(y_ref[:, 0], y_ref[:, 1], y_ref[:, 2],
                              c='blue', alpha=0.3, s=5, label='Reference')
                    ax.scatter(y_learned[:, 0], y_learned[:, 1], y_learned[:, 2],
                              c='red', alpha=0.3, s=5, label='Learned')
                    ax.set_zlim(axis_lim)
                else:
                    ax.scatter(y_ref[:, 0], y_ref[:, 1],
                              c='blue', alpha=0.3, s=5, label='Reference')
                    ax.scatter(y_learned[:, 0], y_learned[:, 1],
                              c='red', alpha=0.3, s=5, label='Learned')

                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                ax.set_title(
                    f't={zt_train_times[t_idx]:.2f}\nstd_ref={ref_std:.2f} std_lrn={learned_std:.2f}',
                    fontsize=9
                )

                if t_idx == 0:
                    ax.legend(fontsize=6)

        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved latent comparison visualization to {save_path}")

        # WandB logging (optional)
        if run is not None:
            try:
                # Verify file exists before logging
                if not save_path.exists():
                    print(f"  Warning: Visualization file not found at {save_path}")
                else:
                    # Use PIL to load image for more reliable wandb logging
                    from PIL import Image as PILImage

                    # Import wandb dynamically (in case it's not available)
                    try:
                        from scripts.wandb_compat import wandb
                    except ImportError:
                        import wandb as wandb_module
                        wandb = wandb_module

                    img = PILImage.open(save_path)
                    wandb_img = wandb.Image(img, caption=f"{title} at step {step}")
                    if wandb_img is not None:
                        run.log({wandb_key: wandb_img}, step=step)
                        print(f"  Logged visualization to wandb at step {step}")
                    else:
                        print(f"  Warning: wandb.Image returned None (wandb may be in offline/noop mode)")
            except Exception as e:
                print(f"  Warning: Failed to log visualization to wandb: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"  Error during visualization: {e}")
        import traceback
        traceback.print_exc()
