"""Archive of deprecated PCA/Lifter logic.

Contains:
- lift_latent_trajectory (from pca_precomputed_main.py)
- Visualization calls requiring coefficient-space data (from pca_precomputed_main.py)
"""

import numpy as np
import torch
from pathlib import Path

# Placeholder for imports that might be needed if code is resurrected
# from diffmap.diffusion_maps import ConvexHullInterpolator
# from scripts.time_local_lifting import map_times_to_training_indices


def lift_latent_trajectory(
    traj_latent: np.ndarray,
    lifter,  # ConvexHullInterpolator
    *,
    neighbor_k: int,
    batch_size: int,
    target_times: np.ndarray | None = None,
    training_times: np.ndarray | None = None,
):
    """
    [DEPRECATED] Lift latent MMSFM trajectories back into PCA coefficient space.

    For time-coupled lifters, each trajectory slice is lifted using the nearest
    training marginal in time to respect time locality.
    """
    from scripts.time_local_lifting import map_times_to_training_indices

    if lifter is None:
        raise ValueError('ConvexHullInterpolator lifter is required for lifting trajectories.')
    traj_latent = np.asarray(traj_latent, dtype=np.float64)
    if traj_latent.ndim != 3:
        raise ValueError('Expected trajectory array with shape (T, N, latent_dim).')

    # If lifter is time-coupled, enforce time-aware lifting via nearest training slice
    if getattr(lifter, 'is_time_coupled', False):
        if training_times is None:
            raise ValueError('Time-coupled lifting requires training_times for time alignment.')

        T_traj = traj_latent.shape[0]
        T_lifter = lifter.time_len

        training_times = np.asarray(training_times, dtype=float).reshape(-1)
        if training_times.shape[0] != T_lifter:
            raise ValueError(
                f'training_times length {training_times.shape[0]} does not match lifter.time_len {T_lifter}.'
            )

        if target_times is None:
            if T_traj != T_lifter:
                raise ValueError(
                    'target_times must be provided when trajectory length does not match lifter time dimension.'
                )
            return lifter.batch_lift(
                traj_latent,
                k=neighbor_k,
                batch_size=batch_size,
            )

        target_times = np.asarray(target_times, dtype=float).reshape(-1)
        if target_times.shape[0] != T_traj:
            raise ValueError(
                f'target_times length {target_times.shape[0]} does not match trajectory length {T_traj}.'
            )

        time_indices = map_times_to_training_indices(target_times, training_times)
        flat = np.ascontiguousarray(traj_latent.reshape(T_traj * traj_latent.shape[1], -1))
        time_indices_flat = np.repeat(time_indices, traj_latent.shape[1])
        lifted = lifter.batch_lift(
            flat,
            k=neighbor_k,
            batch_size=batch_size,
            time_indices=time_indices_flat
        )
        return lifted.reshape(traj_latent.shape[0], traj_latent.shape[1], -1)

    # Legacy loop for 2D lifter
    T, N, _ = traj_latent.shape
    flat = np.ascontiguousarray(traj_latent.reshape(T * N, -1))
    lifted = lifter.batch_lift(
        flat,
        k=neighbor_k,
        batch_size=batch_size,
    )
    return lifted.reshape(T, N, -1)


def _archived_visualization_block(
    # Arguments that were present in the deprecated block
    ode_traj_coeff_at_zt,
    coeff_testdata,
    pca_info,
    zt,
    latent_outdir,
    run,
    sde_back_traj_coeff_at_zt=None,
    visualize_all_field_reconstructions=None,
):
    """Archived block for field reconstruction visualization."""
    if visualize_all_field_reconstructions is None:
        return

    print('Creating field reconstruction visualizations for latent plots...')
    # NOTE: we pass the LIFTED trajectory (coeff space) here, not the latent one!
    visualize_all_field_reconstructions(
        ode_traj_coeff_at_zt, coeff_testdata, pca_info, zt,
        latent_outdir, run, score=False,
        prefix='latent_'
    )

    if sde_back_traj_coeff_at_zt is not None:
        print('Creating field reconstruction visualizations for latent backward SDE plots...')
        # Visualizing the backward SDE (score=True usually, but here likely effectively ODE+noise)
        # We treat it as "sde" for plotting label purposes
        # Reverse arrays to visualize backward process (T -> 0)
        visualize_all_field_reconstructions(
            sde_back_traj_coeff_at_zt[::-1],
            coeff_testdata[::-1],
            pca_info,
            zt[::-1],
            latent_outdir, run, score=True,  # usage of 'score=True' labels it as SDE in plots
            prefix='latent_backward_'
        )
