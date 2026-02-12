"""Cycle consistency training utilities for geodesic autoencoders.

This module provides sampling and cycle consistency loss functions for training
autoencoders with intermediate-time embeddings from time-coupled diffusion maps (TCDM).

Functions:
    sample_stratified_times: Sample time points using stratified sampling
    sample_psi_batch: Sample interpolated embeddings at specified times
    cycle_pairwise_losses: Compute cycle consistency + distance/GM losses
    build_psi_provider: Build PsiProvider from interpolation results
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmsfm.psi_provider import PsiProvider


def sample_stratified_times(
    batch_size: int,
    t_min: float,
    t_max: float,
    n_strata: int,
    device: torch.device,
) -> Tensor:
    """Sample time points using stratified sampling for better temporal coverage.

    Divides [t_min, t_max] into n_strata bins and samples uniformly from each bin.
    This ensures better coverage across the time dimension compared to uniform random
    sampling, which is particularly important for:
    - Cycle consistency training with intermediate times
    - Learning smooth time-dependent transformations
    - Preventing temporal clustering in mini-batches

    Args:
        batch_size: Total number of time points to sample.
        t_min: Minimum time value (normalized, typically 0.0).
        t_max: Maximum time value (normalized, typically 1.0).
        n_strata: Number of time bins/strata. Higher values give more uniform
            coverage but smaller sample counts per bin.
        device: torch.device for tensor allocation.

    Returns:
        Tensor of shape (batch_size,) with stratified time samples, shuffled to
        avoid temporal ordering bias within the batch.

    Raises:
        ValueError: If n_strata <= 0.

    Examples:
        >>> # Sample 100 times with 10 strata (10 samples per bin)
        >>> t_batch = sample_stratified_times(100, 0.0, 1.0, 10, torch.device('cuda'))
        >>> # Distribution will have ~10 samples in [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]

    Notes:
        - If batch_size is not divisible by n_strata, remainder samples are distributed
          across the first few strata
        - If t_max <= t_min, returns constant tensor filled with t_min
        - Samples are shuffled after generation to break temporal ordering
    """
    if n_strata <= 0:
        raise ValueError(f"n_strata must be > 0, got {n_strata}")

    if t_max <= t_min:
        return torch.full((batch_size,), t_min, device=device, dtype=torch.float32)

    # Number of samples per stratum
    samples_per_stratum = batch_size // n_strata
    remainder = batch_size % n_strata

    # Compute stratum boundaries
    stratum_width = (t_max - t_min) / n_strata

    time_samples = []
    for i in range(n_strata):
        # Number of samples for this stratum (distribute remainder across first few strata)
        n_samples = samples_per_stratum + (1 if i < remainder else 0)

        if n_samples > 0:
            # Stratum boundaries
            stratum_min = t_min + i * stratum_width
            stratum_max = t_min + (i + 1) * stratum_width

            # Sample uniformly within this stratum
            t_stratum = torch.rand((n_samples,), device=device, dtype=torch.float32)
            t_stratum = stratum_min + t_stratum * (stratum_max - stratum_min)
            time_samples.append(t_stratum)

    # Concatenate and shuffle to avoid temporal ordering bias
    t_batch = torch.cat(time_samples, dim=0)
    perm = torch.randperm(batch_size, device=device)
    return t_batch[perm]


def sample_psi_batch(
    psi_provider: PsiProvider,
    t_batch: Tensor,
    sample_idx: Tensor,
    *,
    mode: str,
) -> Tensor:
    """Sample a batch of interpolated embeddings at specified times and sample indices.

    Efficiently retrieves interpolated TCDM embeddings ψ(t) at arbitrary time points
    without materializing the full (B, N, K) tensor. Uses either nearest-neighbor
    or linear interpolation based on the specified mode.

    Args:
        psi_provider: PsiProvider instance containing dense trajectories and time grid.
        t_batch: Time values at which to sample, shape (B,).
        sample_idx: Sample indices to retrieve, shape (B,). These index into the
            sample dimension of psi_provider.psi_dense.
        mode: Sampling mode. One of:
            - "nearest": Use nearest time point (faster, less smooth)
            - "linear" or "interpolation": Linear interpolation between adjacent time points
            If None or empty, uses psi_provider.mode as default.

    Returns:
        Tensor of shape (B, K) containing interpolated embeddings, where K is the
        embedding dimension from psi_provider.psi_dense.

    Raises:
        ValueError: If mode is not 'nearest', 'linear', or 'interpolation'.

    Examples:
        >>> psi_prov = PsiProvider(t_dense, psi_dense, mode='linear')
        >>> t_batch = torch.tensor([0.25, 0.5, 0.75], device='cuda')
        >>> sample_idx = torch.tensor([0, 5, 10], device='cuda')
        >>> psi_batch = sample_psi_batch(psi_prov, t_batch, sample_idx, mode='linear')
        >>> # psi_batch.shape = (3, K) where K is embedding dimension

    Notes:
        - For "linear" mode, uses psi_provider._bracket() to find bracketing time indices
        - Memory efficient: only retrieves (B, K) embeddings instead of full (B, N, K)
        - Assumes psi_provider.psi_dense has shape (T, N, K) where T is dense time grid
    """
    mode_eff = (mode or psi_provider.mode).lower().strip()

    if mode_eff == "nearest":
        # Quantize times to nearest time index
        tidx = psi_provider.quantize(t_batch)
        return psi_provider.psi_dense[tidx, sample_idx]

    elif mode_eff in {"linear", "interpolation"}:
        # Linear interpolation between adjacent time points
        idx0, idx1, w = psi_provider._bracket(t_batch)
        psi0 = psi_provider.psi_dense[idx0, sample_idx]
        psi1 = psi_provider.psi_dense[idx1, sample_idx]
        return (1.0 - w).view(-1, 1) * psi0 + w.view(-1, 1) * psi1

    else:
        raise ValueError(f"Unknown psi sampling mode '{mode_eff}'. Use 'nearest', 'linear', or 'interpolation'.")


def cycle_pairwise_losses(
    autoencoder,
    psi_batch: Tensor,
    t_batch: Tensor,
    *,
    dist_weight: float,
    dist_mode: str,
    min_ref_dist: float,
    gm_weight: float,
    gm_normalization: str = "paper",
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute cycle consistency + distance/GM losses for intermediate-time embeddings.

    For intermediate times t' between observed marginals, we don't have ground truth
    ambient data x. Instead, we have precomputed interpolated embeddings ψ(t') from TCDM.

    The cycle consistency ensures the autoencoder is coherent across the latent-ambient cycle:
        L_cycle = ||E(D(ψ, t'), t') - ψ||²

    where:
        - ψ: Interpolated TCDM embedding at time t'
        - D(ψ, t'): Decoder output (reconstructed ambient data)
        - E(D(ψ, t'), t'): Encoder output after decoding (cycled embedding)

    We can also enforce distance preservation on the cycle-reconstructed embeddings:
        y_cycle = E(D(ψ, t'), t')
        d_pred = ||y_cycle[i] - y_cycle[j]||
        d_ref = ||ψ[i] - ψ[j]||  (the TCDM diffusion distance)

    This leverages the key property of TCDM embeddings (Lemma 2.1):
        D^{(t)}(x_j, x_k) = ||δ_j^T Ψ^{(t)} - δ_k^T Ψ^{(t)}||_{L²}

    Args:
        autoencoder: GeodesicAutoencoder with .encoder() and .decoder() methods.
        psi_batch: Interpolated TCDM embeddings at time t', shape (B, K).
        t_batch: Time values, shape (B,).
        dist_weight: Weight for distance loss. If 0.0, distance loss is not computed.
        dist_mode: Distance loss mode ('mse', 'relative', or 'normalized_mse').
        min_ref_dist: Minimum reference distance threshold for distance loss.
        gm_weight: Weight for graph matching loss. If 0.0, GM loss is not computed.
        eps: Numerical stability constant for distance loss.

    Returns:
        Tuple of (loss_cycle, loss_dist, loss_gm):
            - loss_cycle: MSE between cycled embedding and original ψ
            - loss_dist: Distance preservation loss (0 if dist_weight=0)
            - loss_gm: Graph matching loss (0 if gm_weight=0)

    Notes:
        - Cycle consistency is always computed (core constraint)
        - Distance and GM losses are optional (controlled by weights)
        - Uses functions from training_losses module for distance computations
        - This function assumes autoencoder has both encoder and decoder

    References:
        - TCDM: Time-Coupled Diffusion Maps
        - Geodesic autoencoders for learning low-dimensional representations
    """
    from scripts.training_losses import (
        pairwise_distance_matrix,
        distance_loss_from_distance_matrices,
        graph_matching_loss_from_distance_matrices,
    )

    device = psi_batch.device

    # Cycle: ψ -> D(ψ, t) -> x_cycle -> E(x_cycle, t) -> y_cycle
    x_cycle = autoencoder.decoder(psi_batch, t_batch)
    y_cycle = autoencoder.encoder(x_cycle, t_batch)

    # Cycle consistency loss (always computed)
    loss_cycle = F.mse_loss(y_cycle, psi_batch)

    # Distance and GM losses (optional, controlled by weights)
    if float(dist_weight) == 0.0 and float(gm_weight) == 0.0:
        loss_dist = torch.tensor(0.0, device=device, dtype=loss_cycle.dtype)
        loss_gm = torch.tensor(0.0, device=device, dtype=loss_cycle.dtype)
    else:
        # Compute distance matrices once (shared computation)
        d_pred = pairwise_distance_matrix(y_cycle)
        d_ref = pairwise_distance_matrix(psi_batch)

        if float(dist_weight) == 0.0:
            loss_dist = torch.tensor(0.0, device=device, dtype=loss_cycle.dtype)
        else:
            loss_dist = distance_loss_from_distance_matrices(
                d_pred, d_ref, mode=dist_mode, min_ref_dist=min_ref_dist, eps=eps
            )

        if float(gm_weight) == 0.0:
            loss_gm = torch.tensor(0.0, device=device, dtype=loss_cycle.dtype)
        else:
            loss_gm = graph_matching_loss_from_distance_matrices(d_pred, d_ref)
            gm_normalization_eff = (gm_normalization or "paper").lower().strip()
            if gm_normalization_eff not in {"paper", "n_pairs"}:
                raise ValueError(
                    f"Unknown gm_normalization '{gm_normalization}'. Use 'paper' or 'n_pairs'."
                )
            if gm_normalization_eff == "n_pairs":
                n = int(d_ref.shape[0])
                if n >= 2:
                    loss_gm = loss_gm / float(n * (n - 1))

    return loss_cycle, loss_dist, loss_gm


def build_psi_provider(
    interp,
    *,
    scaler,
    frechet_mode: str,
    psi_mode: str,
    sample_idx: Optional[np.ndarray] = None,
) -> PsiProvider:
    """Build a PsiProvider from interpolation results for cycle consistency training.

    Extracts dense TCDM trajectories from interpolation cache, applies normalization,
    and wraps in a PsiProvider for efficient sampling during training.

    Args:
        interp: Interpolation result object with dense trajectory attributes:
            - phi_frechet_triplet_dense: Dense trajectories from triplet Fréchet mean
            - phi_frechet_global_dense: Dense trajectories from global Fréchet mean
            - phi_frechet_dense: Fallback dense trajectories
            - t_dense: Dense time grid
        scaler: DistanceCurveScaler for normalizing embeddings across time.
        frechet_mode: Which dense trajectories to use:
            - 'triplet': Use phi_frechet_triplet_dense (per-sample triplet interpolation)
            - 'global': Use phi_frechet_global_dense (global Fréchet mean)
            Falls back to phi_frechet_dense if specified mode is not available.
        psi_mode: Sampling mode for PsiProvider ('nearest' or 'interpolation').
        sample_idx: Optional array of sample indices to extract a subset. If None,
            uses all samples. Useful for train/test splitting.

    Returns:
        PsiProvider instance ready for sampling during training.

    Raises:
        ValueError: If dense trajectories are missing from interpolation cache,
                    or if sample_idx is provided but empty.

    Examples:
        >>> from scripts.pca_precomputed_utils import load_interpolation_cache
        >>> interp = load_interpolation_cache('interp_cache.pkl')
        >>> scaler = DistanceCurveScaler(...)
        >>> psi_prov = build_psi_provider(
        ...     interp, scaler=scaler, frechet_mode='global',
        ...     psi_mode='linear', sample_idx=train_indices
        ... )
        >>> # Use psi_prov with sample_psi_batch() during training

    Notes:
        - Dense trajectories should have shape (T_dense, N, K)
        - Scaler should be fit on training data before calling this function
        - PsiProvider wraps normalized trajectories for efficient time-based sampling
        - If sample_idx is provided, extracts only those samples (reduces memory)
    """
    # Select dense trajectories based on frechet_mode
    if frechet_mode == "triplet":
        dense_trajs = interp.phi_frechet_triplet_dense
    else:
        dense_trajs = interp.phi_frechet_global_dense

    # Fallback if specified mode is not available
    if dense_trajs is None:
        dense_trajs = interp.phi_frechet_dense

    if dense_trajs is None:
        raise ValueError("Dense trajectories are missing from interpolation cache.")

    # Subset samples if specified
    if sample_idx is not None:
        idx = np.asarray(sample_idx, dtype=int).reshape(-1)
        if idx.size == 0:
            raise ValueError("sample_idx must be non-empty.")
        dense_trajs = dense_trajs[:, idx, :]

    # Normalize trajectories across time
    t_dense = interp.t_dense
    norm_dense_trajs = scaler.transform_at_times(dense_trajs, t_dense).astype(np.float32)

    # Build and return PsiProvider
    return PsiProvider(t_dense.astype(np.float32), norm_dense_trajs, mode=psi_mode)
