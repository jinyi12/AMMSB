"""Distance-based loss functions for training geodesic autoencoders.

This module provides distance preservation and graph matching loss functions used
in training GeodesicAutoencoder and CascadedResidualAutoencoder models.

Functions:
    pairwise_distance_matrix: Compute pairwise Euclidean distance matrix
    distance_loss_from_distance_matrices: Compute distance preservation loss
    graph_matching_loss_from_distance_matrices: Compute graph matching loss
    normalized_graph_matching_loss: Normalized variant for TCDM embeddings
    submanifold_loss: Penalize off-submanifold coordinates
    pairwise_losses_all_pairs: Compute both distance and GM losses together
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def pairwise_distance_matrix(y: Tensor, *, stable: bool = True) -> Tensor:
    """Compute pairwise Euclidean distance matrix.

    Args:
        y: Input tensor of shape (n, d) where n is number of points, d is dimension.
        stable: If True, uses numerically stable computation mode and clamps
            negative values from sqrt to 0. If False, uses basic cdist.

    Returns:
        Distance matrix of shape (n, n) where element (i, j) is ||y_i - y_j||_2.

    Notes:
        - Always converts to float32 for stability
        - When stable=True, uses compute_mode='use_mm_for_euclid_dist' which is
          more stable for large distances by avoiding some intermediate squares
        - Clamping prevents tiny negative values from sqrt due to numerical errors
    """
    y32 = y.to(dtype=torch.float32)
    if stable:
        # Use stable computation mode and clamp negative values from sqrt
        d = torch.cdist(y32, y32, p=2, compute_mode='use_mm_for_euclid_dist')
        return d.clamp(min=0.0)
    else:
        return torch.cdist(y32, y32, p=2)


def distance_loss_from_distance_matrices(
    d_pred: Tensor,
    d_ref: Tensor,
    *,
    mode: str = "mse",
    min_ref_dist: float = 0.0,
    eps: float = 1e-8,
) -> Tensor:
    """Compute distance preservation loss from pre-computed distance matrices.

    Measures how well predicted distances match reference distances using various
    loss modes. Only considers upper triangular elements (unique pairs).

    Args:
        d_pred: Predicted distance matrix, shape (n, n).
        d_ref: Reference distance matrix, shape (n, n).
        mode: Loss mode. One of:
            - "mse": Mean squared error of distances (default)
            - "relative": Mean squared relative error (d_pred/d_ref - 1)^2
            - "normalized_mse" or "norm_mse": MSE normalized by mean reference distance
        min_ref_dist: Minimum reference distance threshold. Pairs with d_ref < threshold
            are excluded from loss computation (default: 0.0, no filtering).
        eps: Small constant for numerical stability in division (default: 1e-8).

    Returns:
        Scalar loss tensor. Returns 0 if n < 2 or no valid pairs after filtering.

    Examples:
        >>> d_pred = torch.tensor([[0., 1., 2.], [1., 0., 1.5], [2., 1.5, 0.]])
        >>> d_ref = torch.tensor([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
        >>> loss = distance_loss_from_distance_matrices(d_pred, d_ref, mode="mse")
        >>> # loss ≈ mean((1.5-1)^2, (2-2)^2, ...) for upper triangular
    """
    n = int(d_ref.shape[0])
    if n < 2:
        return torch.tensor(0.0, device=d_ref.device, dtype=d_ref.dtype)

    # Extract upper triangular elements (unique pairs)
    idx = torch.triu_indices(n, n, offset=1, device=d_ref.device)
    d_pred_u = d_pred[idx[0], idx[1]]
    d_ref_u = d_ref[idx[0], idx[1]]

    # Filter by minimum reference distance if specified
    if float(min_ref_dist) > 0.0:
        mask = d_ref_u > float(min_ref_dist)
        if mask.sum() < 1:
            return torch.tensor(0.0, device=d_ref.device, dtype=d_ref.dtype)
        d_pred_u = d_pred_u[mask]
        d_ref_u = d_ref_u[mask]

    # Compute loss based on mode
    mode = (mode or "mse").lower().strip()
    if mode == "relative":
        # Relative error: (d_pred / d_ref - 1)^2
        # Use clamp for safer division by very small distances
        rel = (d_pred_u / d_ref_u.clamp(min=float(eps))) - 1.0
        return (rel ** 2).mean()
    elif mode == "mse":
        return F.mse_loss(d_pred_u, d_ref_u)
    elif mode in {"normalized_mse", "norm_mse"}:
        # MSE normalized by mean reference distance
        scale = d_ref_u.mean().clamp(min=float(eps))
        diff = (d_pred_u - d_ref_u) / scale
        return (diff ** 2).mean()
    else:
        raise ValueError(f"Unknown dist_mode '{mode}'. Use 'mse', 'relative', or 'normalized_mse'.")


def graph_matching_loss_from_distance_matrices(
    d_pred: Tensor,
    d_ref: Tensor,
    *,
    clamp: bool = True,
) -> Tensor:
    """Compute graph matching loss from distance matrices (efficient O(n^2) form).

    Measures consistency of distance differences between all pairs of points.
    Ensures that the learned embedding preserves not just individual distances,
    but also the relative structure (differences between distances).

    The full paper form is:
        L_gm = (1/n) Σ_i Σ_{j≠i} ||(d_pred(:,i) - d_pred(:,j)) - (d_ref(:,i) - d_ref(:,j))||^2

    This is O(n^4) naively but simplifies to O(n^2):
        Let E = d_pred - d_ref and v_i = E(:,i). Then:
        L_gm = 2 * ||E||_F^2 - (2/n) * ||E·1||_2^2

    Args:
        d_pred: Predicted distance matrix, shape (n, n).
        d_ref: Reference distance matrix, shape (n, n).
        clamp: If True, clamps error values to [-100, 100] for numerical stability.
            Prevents extreme gradients from outlier distance predictions.

    Returns:
        Scalar graph matching loss. Returns 0 if n < 2.

    Notes:
        - This loss is sensitive to batch size (scales as O(n^2))
        - For normalized variant (scale-invariant), use normalized_graph_matching_loss()
        - The clamping (when enabled) prevents gradient explosions from outliers

    References:
        Based on graph matching formulation in time-coupled diffusion maps literature.
    """
    n = int(d_ref.shape[0])
    if n < 2:
        return torch.tensor(0.0, device=d_ref.device, dtype=d_ref.dtype)

    err = d_pred - d_ref

    # Clamp error to prevent extreme values (optional but recommended)
    if clamp:
        err = err.clamp(-100.0, 100.0)

    # Efficient O(n^2) computation
    fro = (err * err).sum()  # ||E||_F^2
    colsum = err.sum(dim=1)  # E·1 (column sums)
    return 2.0 * fro - (2.0 / float(n)) * (colsum * colsum).sum()


def normalized_graph_matching_loss(
    d_pred: Tensor,
    d_ref: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Normalized graph matching loss for TCDM embeddings.

    Variant of graph_matching_loss_from_distance_matrices that normalizes by batch size
    to make the loss scale-invariant. This is particularly useful for TCDM embeddings
    where the embedding preserves time-coupled diffusion distances.

    The TCDM embedding preserves:
        D^{(t)}(x_j, x_k) = ||δ_j^T Ψ^{(t)} - δ_k^T Ψ^{(t)}||_{L^2}

    The graph matching loss measures consistency of distance differences:
        L_gm = (1/n) Σ_i Σ_{j≠i} ||(d_pred(:,i) - d_pred(:,j)) - (d_ref(:,i) - d_ref(:,j))||^2

    This is O(n^4) in naive form but simplifies to O(n^2):
        L_gm = 2*||E||_F^2 - (2/n)*||E·1||_2^2  where E = d_pred - d_ref

    Issue with raw GM loss: It scales as O(n^2 * mean_dist^2) while distance MSE scales
    as O(mean_dist^2). For typical TCDM embeddings with mean distances ~1, GM loss is
    ~100x larger with n≈10 samples/time.

    This normalized version divides by n(n-1), making it scale as O(1) with respect to
    batch size, and comparable in magnitude to distance MSE loss.

    Args:
        d_pred: Predicted distance matrix, shape (n, n).
        d_ref: Reference distance matrix, shape (n, n).
        eps: Numerical stability constant (unused currently, kept for API consistency).

    Returns:
        Normalized graph matching loss (scalar). Returns 0 if n < 2.

    Notes:
        - This normalization makes the loss invariant to batch size
        - Retains the scale of distances (like MSE)
        - More suitable when balancing with other losses in multi-objective training
        - Does NOT divide by mean distance squared (preserves distance scale)
    """
    n = int(d_ref.shape[0])
    if n < 2:
        return torch.tensor(0.0, device=d_ref.device, dtype=d_ref.dtype)

    err = d_pred - d_ref
    fro = (err * err).sum()
    colsum = err.sum(dim=1)
    raw_gm = 2.0 * fro - (2.0 / float(n)) * (colsum * colsum).sum()

    # Normalize by n(n-1) to make loss invariant to batch size
    normalization = float(n * (n - 1))
    return raw_gm / normalization


def submanifold_loss(
    phi: Tensor,
    latent_dim: int,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Submanifold loss: penalize non-zero tail coordinates beyond latent_dim.

    Matches the formulation:
        L_sub = (1/N) Σ_i || [0_{d'}; I_{d-d'}] φ_t(x_i) ||_1

    i.e., for a representation φ(x) in R^d, encourage φ(x) to lie near the
    d'-dimensional coordinate subspace by penalizing the last (d-d') entries.

    Args:
        phi: Tensor of shape (..., d). Typically φ_t(x) (not projected).
        latent_dim: d' (number of leading coordinates to keep unpenalized).
        reduction: One of {"mean", "sum", "none"}.

    Returns:
        Scalar loss if reduction != "none", else per-sample losses of shape (...,).
    """
    latent_dim = int(latent_dim)
    if latent_dim < 0:
        raise ValueError(f"latent_dim must be >= 0, got {latent_dim}")
    if phi.shape[-1] <= latent_dim:
        # Nothing to penalize.
        out = torch.zeros(phi.shape[:-1], device=phi.device, dtype=phi.dtype)
    else:
        tail = phi[..., latent_dim:]
        out = tail.abs().sum(dim=-1)

    reduction = (reduction or "mean").lower().strip()
    if reduction == "none":
        return out
    if reduction == "mean":
        return out.mean()
    if reduction == "sum":
        return out.sum()
    raise ValueError(f"Unknown reduction '{reduction}'. Use 'mean', 'sum', or 'none'.")


def stability_regularization_loss(
    vector_field_fn,
    z: Tensor,
    t_emb: Tensor,
    *,
    n_random_vectors: int = 1,
    reduction: str = "mean",
) -> Tensor:
    """Stability regularization: penalize Jacobian norm of vector field.

    Computes: (1/n) Σ_i ||ε^T ∇f_θ(z_i)||^2

    where ε is a random direction (Rademacher ±1). Uses VJP (vector-Jacobian product)
    for efficient computation without materializing the full d×d Jacobian.

    For n_random_vectors > 1, uses Hutchinson-style stochastic trace estimation:
        E[||ε^T ∇f||^2] = E[ε^T (∇f ∇f^T) ε] = tr(∇f ∇f^T) = ||∇f||_F^2

    Args:
        vector_field_fn: Callable that takes (z, t_emb) and returns f(z; t).
            This should be the vector field f of the ODE dz/ds = f(z; t).
        z: State tensor of shape (B, d) where B is batch size, d is dimension.
        t_emb: Time embedding tensor of shape (B, t_dim).
        n_random_vectors: Number of random vectors for stochastic estimation.
            More vectors = lower variance but higher cost. Default: 1.
        reduction: One of {"mean", "sum", "none"}.

    Returns:
        Scalar loss if reduction != "none", else per-sample losses of shape (B,).

    Notes:
        - Uses Rademacher random vectors (±1) which are unbiased estimators
        - VJP is computed via torch.autograd.grad with create_graph=True
        - Cost: O(d * n_random_vectors) per sample (vs O(d^2) for full Jacobian)
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z with shape (B, d), got {tuple(z.shape)}")
    batch_size, dim = z.shape
    device, dtype = z.device, z.dtype

    # Ensure z requires grad for Jacobian computation
    z_leaf = z if z.requires_grad else z.detach().requires_grad_(True)

    # Compute f(z; t)
    f_z = vector_field_fn(z_leaf, t_emb)  # (B, d)

    # Accumulate squared VJP norms over random vectors
    loss_accum = torch.zeros(batch_size, device=device, dtype=dtype)

    for _ in range(int(n_random_vectors)):
        # Rademacher random vector: ε ∈ {-1, +1}^d
        eps = torch.randint(0, 2, (batch_size, dim), device=device, dtype=dtype) * 2 - 1

        # VJP: v = ε^T ∇f = ∂(ε·f)/∂z (row vector per sample)
        # grad_outputs = eps makes this compute Σ_j ε_j * ∂f_j/∂z
        vjp = torch.autograd.grad(
            outputs=f_z,
            inputs=z_leaf,
            grad_outputs=eps,
            create_graph=True,  # Keep graph for backprop through this loss
            retain_graph=True,
        )[0]  # (B, d)

        # ||ε^T ∇f||^2 per sample
        loss_accum = loss_accum + (vjp ** 2).sum(dim=-1)

    # Average over random vectors
    loss_per_sample = loss_accum / float(n_random_vectors)

    reduction = (reduction or "mean").lower().strip()
    if reduction == "none":
        return loss_per_sample
    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    raise ValueError(f"Unknown reduction '{reduction}'. Use 'mean', 'sum', or 'none'.")


def pairwise_losses_all_pairs(
    y_pred: Tensor,
    y_ref: Tensor,
    *,
    dist_weight: float,
    dist_mode: str = "mse",
    min_ref_dist: float = 0.0,
    gm_weight: float,
    eps: float = 1e-8,
    stable: bool = True,
    clamp_gm: bool = True,
    gm_normalization: str = "paper",
) -> tuple[Tensor, Tensor]:
    """Compute both distance loss and graph matching loss from embedding pairs.

    Convenience function that computes distance matrices once and returns both
    distance preservation loss and graph matching loss. Skips computation if
    corresponding weight is 0.

    Args:
        y_pred: Predicted embeddings, shape (n, d).
        y_ref: Reference embeddings, shape (n, d).
        dist_weight: Weight for distance loss. If 0, distance loss is not computed.
        dist_mode: Distance loss mode ("mse", "relative", or "normalized_mse").
        min_ref_dist: Minimum reference distance threshold for distance loss.
        gm_weight: Weight for graph matching loss. If 0, GM loss is not computed.
        eps: Numerical stability constant for distance loss.
        stable: If True, uses stable distance matrix computation (see pairwise_distance_matrix).
        clamp_gm: If True, clamps errors in GM loss for numerical stability.
        gm_normalization: How to normalize the graph matching loss. One of:
            - "paper": Use the paper form (1/n sum over i and j≠i), which scales ~O(n^2).
            - "n_pairs": Further divide by n(n-1) to make it roughly invariant to batch size.

    Returns:
        Tuple of (distance_loss, graph_matching_loss). Each is 0 if corresponding weight is 0.

    Examples:
        >>> y_pred = torch.randn(10, 5)  # 10 samples, 5-dim embedding
        >>> y_ref = torch.randn(10, 5)
        >>> loss_dist, loss_gm = pairwise_losses_all_pairs(
        ...     y_pred, y_ref,
        ...     dist_weight=1.0, dist_mode="mse", min_ref_dist=0.0,
        ...     gm_weight=0.5, eps=1e-8
        ... )
        >>> total_loss = 1.0 * loss_dist + 0.5 * loss_gm
    """
    # Compute distance matrices once (shared computation)
    d_pred = pairwise_distance_matrix(y_pred, stable=stable)
    d_ref = pairwise_distance_matrix(y_ref, stable=stable)

    # Compute distance loss if weight > 0
    if float(dist_weight) == 0.0:
        loss_dist = torch.tensor(0.0, device=y_pred.device, dtype=d_ref.dtype)
    else:
        loss_dist = distance_loss_from_distance_matrices(
            d_pred, d_ref, mode=dist_mode, min_ref_dist=min_ref_dist, eps=eps
        )

    # Compute graph matching loss if weight > 0
    if float(gm_weight) == 0.0:
        loss_gm = torch.tensor(0.0, device=y_pred.device, dtype=loss_dist.dtype)
    else:
        gm_normalization_eff = (gm_normalization or "paper").lower().strip()
        if gm_normalization_eff not in {"paper", "n_pairs"}:
            raise ValueError(
                f"Unknown gm_normalization '{gm_normalization}'. Use 'paper' or 'n_pairs'."
            )
        loss_gm = graph_matching_loss_from_distance_matrices(d_pred, d_ref, clamp=clamp_gm)
        if gm_normalization_eff == "n_pairs":
            n = int(d_ref.shape[0])
            if n >= 2:
                loss_gm = loss_gm / float(n * (n - 1))

    return loss_dist, loss_gm
