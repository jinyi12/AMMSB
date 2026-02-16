"""Time-stratified scaling for TCDM embeddings with contractive geometry.

The TCDM embeddings naturally contract over time as diffusion progresses
(eigenvalues decay exponentially). Standard global scaling (MinMaxScaler)
preserves this contraction but makes later time slices nearly point-masses,
making them hard to learn.

This module provides principled scaling approaches that:
1. Normalize each time slice for learnable targets
2. Preserve relative distance structure within each time
3. Store scale factors to recover the true contractive geometry

Key Insight:
-----------
If we denote the raw TCDM embedding at time t as Φ(t), the natural distance
ratio r(t) = E[||Φ_i(t) - Φ_j(t)||] / E[||Φ_i(0) - Φ_j(0)||] captures the
contraction. We want to:
- Train the encoder to output normalized embeddings Φ̃(t) = Φ(t) / σ(t)
- Store σ(t) to recover Φ(t) = σ(t) * Φ̃(t)
- The decoder must learn the inverse: X̃ = D(Φ̃(t), t)

Scaling Strategies:
------------------
1. "per_time_std": Normalize each time to unit std (σ(t) = std(Φ(t)))
2. "per_time_iqr": Normalize using interquartile range (robust to outliers)
3. "per_time_median_dist": Normalize by median pairwise distance
4. "global_aware": Normalize to unit std but store the contraction ratios

Reference: Diffusion map eigenvalue decay theory (Coifman & Lafon, 2006)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class TimeStratifiedScaler:
    """Scaler that normalizes each time slice while preserving relative structure.
    
    Attributes:
        strategy: Scaling strategy to use.
        target_std: Target standard deviation after scaling.
        per_time_scales: Computed scale factors per time slice.
        per_time_centers: Computed center per time slice.
        contraction_ratios: Ratio of scale at time t to scale at time 0.
    """
    strategy: Literal["per_time_std", "per_time_iqr", "per_time_median_dist", "global_aware", "contraction_preserving"] = "per_time_std"
    target_std: float = 0.5
    center_data: bool = True
    contraction_power: float = 0.3  # For contraction_preserving: compress contraction to ratio^power
    
    # Fitted parameters
    per_time_scales: Optional[np.ndarray] = field(default=None, repr=False)
    per_time_centers: Optional[np.ndarray] = field(default=None, repr=False)
    contraction_ratios: Optional[np.ndarray] = field(default=None, repr=False)
    damped_contraction_ratios: Optional[np.ndarray] = field(default=None, repr=False)
    fitted: bool = False
    
    def fit(self, X: list[np.ndarray] | np.ndarray) -> "TimeStratifiedScaler":
        """Fit the scaler to training data.
        
        Args:
            X: List of (N, D) arrays or (T, N, D) array of embeddings per time.
            
        Returns:
            self
        """
        if isinstance(X, list):
            X = np.stack(X, axis=0)
        
        T, N, D = X.shape
        
        # Compute per-time statistics
        self.per_time_centers = np.zeros((T, D), dtype=np.float64)
        self.per_time_scales = np.zeros(T, dtype=np.float64)
        
        for t in range(T):
            data_t = X[t]
            
            # Center: mean of each dimension
            if self.center_data:
                self.per_time_centers[t] = data_t.mean(axis=0)
            else:
                self.per_time_centers[t] = 0.0
            
            # Scale: depends on strategy
            centered = data_t - self.per_time_centers[t]
            
            if self.strategy == "per_time_std":
                # Standard deviation of all values
                self.per_time_scales[t] = centered.std() + 1e-10
                
            elif self.strategy == "per_time_iqr":
                # Interquartile range (more robust)
                flat = centered.flatten()
                q75, q25 = np.percentile(flat, [75, 25])
                iqr = q75 - q25
                # IQR ~ 1.35 * std for Gaussian, so normalize
                self.per_time_scales[t] = (iqr / 1.35) + 1e-10
                
            elif self.strategy == "per_time_median_dist":
                # Median pairwise distance (captures geometry directly)
                n_sample = min(500, N)
                idx = np.random.permutation(N)[:n_sample]
                dists = np.linalg.norm(
                    centered[idx, None, :] - centered[None, idx, :], axis=-1
                )
                median_dist = np.median(dists[np.triu_indices(n_sample, k=1)])
                self.per_time_scales[t] = median_dist + 1e-10
                
            elif self.strategy == "global_aware":
                # Same as per_time_std but we'll use contraction ratios later
                self.per_time_scales[t] = centered.std() + 1e-10
            
            elif self.strategy == "contraction_preserving":
                # Record the raw std for computing contraction ratios
                # But we'll use a SINGLE global scale factor (from t=0) for all times
                self.per_time_scales[t] = centered.std() + 1e-10
                
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Compute contraction ratios (relative to t=0) BEFORE modifying per_time_scales
        raw_scales = self.per_time_scales.copy()  # Save raw scales
        self.contraction_ratios = raw_scales / (raw_scales[0] + 1e-10)
        
        # For global_aware, we want to preserve some sense of the contraction
        # by using a damped scale factor
        if self.strategy == "global_aware":
            # Use geometric mean between per-time and global scale
            global_scale = raw_scales[0]  # scale of t=0
            damping = 0.5  # how much to preserve contraction
            self.per_time_scales = (
                (raw_scales ** (1 - damping)) * 
                (global_scale ** damping)
            )
        
        # For contraction_preserving: use damped contraction that preserves the structure
        # but compresses the dynamic range to be learnable
        # If contraction_ratio[t] = 0.00002 (4500x contraction), ratio^0.3 ≈ 0.03
        # This preserves: monotonicity, relative ordering, direction of contraction
        # But compresses: 4500x range → ~30x range (learnable!)
        if self.strategy == "contraction_preserving":
            # Damped contraction: damped_ratio[t] = contraction_ratio[t] ^ power
            damped_ratios = np.power(self.contraction_ratios, self.contraction_power)
            self.damped_contraction_ratios = damped_ratios
            
            # To achieve scaled_std[t] = target_std * damped_ratio[t], we need:
            # scale[t] = raw_std[t] / damped_ratio[t]
            # Because: scaled_std = raw_std / scale * target_std = damped_ratio * target_std
            self.per_time_scales = raw_scales / (damped_ratios + 1e-10)
            
            print(f"[TimeStratifiedScaler] contraction_preserving with power={self.contraction_power}")
            print(f"  Original contraction ratios: {self.contraction_ratios}")
            print(f"  Damped contraction ratios: {damped_ratios}")
            print(f"  Expected scaled stds: {self.target_std * damped_ratios}")
        
        self.fitted = True
        return self
    
    def transform(
        self, X: list[np.ndarray] | np.ndarray
    ) -> list[np.ndarray] | np.ndarray:
        """Transform data using fitted parameters.
        
        Args:
            X: List of (N, D) arrays or (T, N, D) array.
            
        Returns:
            Scaled data in same format as input.
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted!")
        
        is_list = isinstance(X, list)
        if is_list:
            X_arr = np.stack(X, axis=0)
        else:
            X_arr = X
        
        T = X_arr.shape[0]
        assert T == len(self.per_time_scales), f"Time dimension mismatch: {T} vs {len(self.per_time_scales)}"
        
        result = np.zeros_like(X_arr, dtype=np.float32)
        for t in range(T):
            centered = X_arr[t] - self.per_time_centers[t]
            # Scale to target_std
            scaled = (centered / self.per_time_scales[t]) * self.target_std
            result[t] = scaled.astype(np.float32)
        
        if is_list:
            return [result[t] for t in range(T)]
        return result
    
    def inverse_transform(
        self, X: list[np.ndarray] | np.ndarray
    ) -> list[np.ndarray] | np.ndarray:
        """Inverse transform to recover original scale.
        
        Args:
            X: Scaled data.
            
        Returns:
            Data in original scale.
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted!")
        
        is_list = isinstance(X, list)
        if is_list:
            X_arr = np.stack(X, axis=0)
        else:
            X_arr = X
        
        T = X_arr.shape[0]
        result = np.zeros_like(X_arr, dtype=np.float32)
        for t in range(T):
            # Undo scale
            unscaled = (X_arr[t] / self.target_std) * self.per_time_scales[t]
            result[t] = (unscaled + self.per_time_centers[t]).astype(np.float32)
        
        if is_list:
            return [result[t] for t in range(T)]
        return result
    
    def fit_transform(
        self, X: list[np.ndarray] | np.ndarray
    ) -> list[np.ndarray] | np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def get_distance_normalizers(self) -> np.ndarray:
        """Get per-time distance normalization factors.
        
        Returns:
            Array of shape (T,) where result[t] = 1/scale[t].
            Multiplying raw distances by this gives normalized distances.
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted!")
        return 1.0 / self.per_time_scales
    
    def get_state_dict(self) -> dict:
        """Get serializable state for checkpointing."""
        return {
            "strategy": self.strategy,
            "target_std": self.target_std,
            "center_data": self.center_data,
            "per_time_scales": self.per_time_scales,
            "per_time_centers": self.per_time_centers,
            "contraction_ratios": self.contraction_ratios,
            "fitted": self.fitted,
        }
    
    @classmethod
    def from_state_dict(cls, state: dict) -> "TimeStratifiedScaler":
        """Reconstruct scaler from state dict."""
        scaler = cls(
            strategy=state["strategy"],
            target_std=state["target_std"],
            center_data=state.get("center_data", True),
        )
        scaler.per_time_scales = state["per_time_scales"]
        scaler.per_time_centers = state["per_time_centers"]
        scaler.contraction_ratios = state["contraction_ratios"]
        scaler.fitted = state["fitted"]
        return scaler


@dataclass
class DistanceCurveScaler:
    """Continuous scaler derived from a dense latent trajectory via pairwise distances.

    This scaler computes a time-dependent contraction ratio r(t) from a dense latent
    trajectory Psi(t) by estimating a typical pairwise distance d(t) (median of random
    pairs). It then defines a smooth time-conditioned isotropic scale so that the
    scaled contraction follows r(t)^p (p = contraction_power).

    Transform at time t:
        X_scaled(t) = ((X(t) - c(t)) / s(t)) * target_std

    where:
        r(t) = d(t) / d(t0)
        s(t) = std0 * r(t)^(1 - contraction_power)
        c(t) is a (linearly interpolated) time-dependent mean when center_data=True.

    Notes:
        - Distances are translation-invariant, so c(t) only affects absolute location
          in scaled coordinates (useful for learnability; does not change geometry).
        - contraction_power = 1 preserves contraction (global scale s(t)=std0).
        - contraction_power = 0 removes contraction (s(t) tracks r(t) so distances
          become roughly time-normalized).
    """

    target_std: float = 0.25
    contraction_power: float = 1.0
    center_data: bool = True
    n_pairs: int = 4096
    seed: int = 0

    t_dense: Optional[np.ndarray] = field(default=None, repr=False)
    centers_dense: Optional[np.ndarray] = field(default=None, repr=False)  # (T_dense, K)
    log_r_dense: Optional[np.ndarray] = field(default=None, repr=False)  # (T_dense,)
    std0: Optional[float] = field(default=None, repr=False)
    d0: Optional[float] = field(default=None, repr=False)
    fitted: bool = False

    def fit(self, dense_trajs: np.ndarray, t_dense: np.ndarray) -> "DistanceCurveScaler":
        dense_trajs = np.asarray(dense_trajs, dtype=np.float64)
        if dense_trajs.ndim != 3:
            raise ValueError("dense_trajs must have shape (T_dense, N, K).")
        t_dense = np.asarray(t_dense, dtype=float).reshape(-1)
        if t_dense.ndim != 1 or t_dense.shape[0] != dense_trajs.shape[0]:
            raise ValueError("t_dense must be 1D and match dense_trajs time dimension.")
        if t_dense.shape[0] < 2:
            raise ValueError("t_dense must contain at least two time points.")
        if np.any(t_dense[1:] <= t_dense[:-1]):
            raise ValueError("t_dense must be strictly increasing.")

        T_dense, n, k = dense_trajs.shape
        if n < 2:
            raise ValueError("Need at least 2 samples to compute distance curve.")

        if self.center_data:
            centers = dense_trajs.mean(axis=1)  # (T_dense, K)
        else:
            centers = np.zeros((T_dense, k), dtype=np.float64)

        std0 = float(np.std(dense_trajs[0] - centers[0]) + 1e-12)

        n_pairs = int(self.n_pairs)
        if n_pairs <= 0:
            raise ValueError("n_pairs must be > 0.")
        rng = np.random.default_rng(int(self.seed))
        i = rng.integers(0, n, size=n_pairs, dtype=np.int64)
        j = rng.integers(0, n, size=n_pairs, dtype=np.int64)
        j = np.where(j == i, (j + 1) % n, j)

        d_curve = np.empty(T_dense, dtype=np.float64)
        for ti in range(T_dense):
            diff = dense_trajs[ti, i] - dense_trajs[ti, j]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            d_curve[ti] = np.median(dists)

        d0 = float(d_curve[0] + 1e-12)
        r = np.clip(d_curve / d0, 1e-12, None)
        log_r = np.log(r)

        self.t_dense = t_dense.astype(np.float64)
        self.centers_dense = centers.astype(np.float64)
        self.log_r_dense = log_r.astype(np.float64)
        self.std0 = std0
        self.d0 = d0
        self.fitted = True
        return self

    def _bracket(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.t_dense is None:
            raise RuntimeError("DistanceCurveScaler is not fitted.")
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        idx1 = np.searchsorted(self.t_dense, t_arr, side="left")
        idx1 = np.clip(idx1, 1, self.t_dense.shape[0] - 1)
        idx0 = idx1 - 1
        t0 = self.t_dense[idx0]
        t1 = self.t_dense[idx1]
        w = (t_arr - t0) / (t1 - t0 + 1e-12)
        w = np.clip(w, 0.0, 1.0)
        return idx0, idx1, w

    def contraction_ratio(self, t: np.ndarray) -> np.ndarray:
        if not self.fitted or self.log_r_dense is None:
            raise RuntimeError("DistanceCurveScaler is not fitted.")
        idx0, idx1, w = self._bracket(t)
        log_r = (1.0 - w) * self.log_r_dense[idx0] + w * self.log_r_dense[idx1]
        return np.exp(log_r).astype(np.float64)

    def center(self, t: np.ndarray) -> np.ndarray:
        if not self.fitted or self.centers_dense is None:
            raise RuntimeError("DistanceCurveScaler is not fitted.")
        if not self.center_data:
            return np.zeros((np.asarray(t).reshape(-1).shape[0], self.centers_dense.shape[1]), dtype=np.float64)
        idx0, idx1, w = self._bracket(t)
        w2 = w[:, None]
        return ((1.0 - w2) * self.centers_dense[idx0] + w2 * self.centers_dense[idx1]).astype(np.float64)

    def scale(self, t: np.ndarray) -> np.ndarray:
        if not self.fitted or self.std0 is None:
            raise RuntimeError("DistanceCurveScaler is not fitted.")
        r = self.contraction_ratio(t)
        expo = float(1.0 - float(self.contraction_power))
        return (float(self.std0) * np.power(r, expo)).astype(np.float64)

    def transform_at_times(self, X: list[np.ndarray] | np.ndarray, times: np.ndarray) -> list[np.ndarray] | np.ndarray:
        if not self.fitted:
            raise RuntimeError("DistanceCurveScaler is not fitted.")
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        is_list = isinstance(X, list)
        X_arr = np.stack(X, axis=0) if is_list else np.asarray(X)
        if X_arr.ndim != 3:
            raise ValueError("X must have shape (T, N, K).")
        if X_arr.shape[0] != times_arr.shape[0]:
            raise ValueError("times must match X time dimension.")

        c = self.center(times_arr)  # (T, K)
        s = self.scale(times_arr)  # (T,)
        out = (X_arr.astype(np.float64) - c[:, None, :]) / s[:, None, None]
        out *= float(self.target_std)
        out = out.astype(np.float32)
        if is_list:
            return [out[t] for t in range(out.shape[0])]
        return out

    def inverse_transform_at_times(self, X: list[np.ndarray] | np.ndarray, times: np.ndarray) -> list[np.ndarray] | np.ndarray:
        if not self.fitted:
            raise RuntimeError("DistanceCurveScaler is not fitted.")
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        is_list = isinstance(X, list)
        X_arr = np.stack(X, axis=0) if is_list else np.asarray(X)
        if X_arr.ndim != 3:
            raise ValueError("X must have shape (T, N, K).")
        if X_arr.shape[0] != times_arr.shape[0]:
            raise ValueError("times must match X time dimension.")

        c = self.center(times_arr)  # (T, K)
        s = self.scale(times_arr)  # (T,)
        out = (X_arr.astype(np.float64) / float(self.target_std)) * s[:, None, None] + c[:, None, :]
        out = out.astype(np.float32)
        if is_list:
            return [out[t] for t in range(out.shape[0])]
        return out

    def get_state_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("DistanceCurveScaler is not fitted.")
        return {
            "type": "distance_curve",
            "target_std": float(self.target_std),
            "contraction_power": float(self.contraction_power),
            "center_data": bool(self.center_data),
            "n_pairs": int(self.n_pairs),
            "seed": int(self.seed),
            "t_dense": self.t_dense,
            "centers_dense": self.centers_dense,
            "log_r_dense": self.log_r_dense,
            "std0": float(self.std0) if self.std0 is not None else None,
            "d0": float(self.d0) if self.d0 is not None else None,
            "fitted": bool(self.fitted),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DistanceCurveScaler":
        scaler = cls(
            target_std=float(state["target_std"]),
            contraction_power=float(state["contraction_power"]),
            center_data=bool(state.get("center_data", True)),
            n_pairs=int(state.get("n_pairs", 4096)),
            seed=int(state.get("seed", 0)),
        )
        scaler.t_dense = np.asarray(state["t_dense"], dtype=np.float64)
        scaler.centers_dense = np.asarray(state["centers_dense"], dtype=np.float64)
        scaler.log_r_dense = np.asarray(state["log_r_dense"], dtype=np.float64)
        scaler.std0 = float(state["std0"])
        scaler.d0 = float(state.get("d0", 0.0))
        scaler.fitted = bool(state.get("fitted", True))
        return scaler


def compute_local_geometry_loss(
    y_pred: "torch.Tensor",
    y_ref: "torch.Tensor",
    knn_idx: "torch.Tensor",
    *,
    n_neighbors: int = 8,
    temperature: float = 0.1,
) -> "torch.Tensor":
    """Compute local geometry preservation loss using softmax-normalized distances.
    
    Instead of matching absolute distances (which fail for contractive geometry),
    this loss matches the *relative distribution* of distances to neighbors.
    
    For each point i, we compute:
        p_i(j) = softmax(-d(y_pred_i, y_pred_j) / τ)  for j ∈ neighbors(i)
        q_i(j) = softmax(-d(y_ref_i, y_ref_j) / τ)   for j ∈ neighbors(i)
    
    Loss = KL(p || q) averaged over all points.
    
    This preserves:
    - Nearest neighbor ordering (who is closest)
    - Relative distance ratios (how much closer is j than k)
    
    Args:
        y_pred: Predicted latent positions, shape (B, D).
        y_ref: Reference latent positions, shape (B, D).
        knn_idx: Precomputed k-nearest neighbor indices, shape (B, k).
        n_neighbors: Number of neighbors to consider from knn_idx.
        temperature: Softmax temperature (smaller = sharper distribution).
        
    Returns:
        Scalar loss value.
    """
    import torch
    import torch.nn.functional as F
    
    B, D = y_pred.shape
    k = min(n_neighbors, knn_idx.shape[1])
    
    # Get neighbor embeddings
    neighbors_idx = knn_idx[:, :k]  # (B, k)
    
    # Compute distances to neighbors for predicted
    y_neighbors_pred = y_pred[neighbors_idx]  # (B, k, D)
    d_pred = torch.linalg.norm(
        y_pred.unsqueeze(1) - y_neighbors_pred, dim=-1
    )  # (B, k)
    
    # Compute distances to neighbors for reference
    y_neighbors_ref = y_ref[neighbors_idx]  # (B, k, D)
    d_ref = torch.linalg.norm(
        y_ref.unsqueeze(1) - y_neighbors_ref, dim=-1
    )  # (B, k)
    
    # Compute softmax distributions (negative distance for "similarity")
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    
    # Normalize distances within each sample for numerical stability
    d_pred_norm = d_pred / (d_pred.mean(dim=1, keepdim=True) + eps)
    d_ref_norm = d_ref / (d_ref.mean(dim=1, keepdim=True) + eps)
    
    p_pred = F.softmax(-d_pred_norm / temperature, dim=-1)  # (B, k)
    p_ref = F.softmax(-d_ref_norm / temperature, dim=-1)   # (B, k)
    
    # KL divergence: KL(p_ref || p_pred) = sum(p_ref * log(p_ref / p_pred))
    kl = (p_ref * (torch.log(p_ref + eps) - torch.log(p_pred + eps))).sum(dim=-1)
    
    return kl.mean()


def compute_scale_matching_loss(
    d_pred: "torch.Tensor",
    d_ref: "torch.Tensor",
    eps: float = 1e-8,
) -> "torch.Tensor":
    """Compute scale matching loss to ensure learned distances have correct scale.
    
    Loss = (mean(d_pred) / mean(d_ref) - 1)^2
    
    This penalizes when the average learned distance differs from the average
    reference distance, encouraging the encoder to preserve the overall scale.
    
    Args:
        d_pred: Predicted distances, shape (B,).
        d_ref: Reference distances, shape (B,).
        eps: Small constant for numerical stability.
        
    Returns:
        Scalar loss value.
    """
    import torch
    
    mean_pred = d_pred.mean() + eps
    mean_ref = d_ref.mean() + eps
    
    # Ratio should be 1.0
    ratio = mean_pred / mean_ref
    return (ratio - 1.0) ** 2


def compute_normalized_distance_loss(
    d_pred: "torch.Tensor",
    d_ref: "torch.Tensor",
    eps: float = 1e-8,
) -> "torch.Tensor":
    """Compute normalized distance loss that matches distance distributions.
    
    Normalizes both predicted and reference distances by their respective means,
    then computes MSE. This focuses on matching the *shape* of the distance
    distribution rather than absolute scale.
    
    Loss = MSE(d_pred / mean(d_pred), d_ref / mean(d_ref))
    
    Args:
        d_pred: Predicted distances, shape (B,).
        d_ref: Reference distances, shape (B,).
        eps: Small constant for numerical stability.
        
    Returns:
        Scalar loss value.
    """
    import torch
    import torch.nn.functional as F
    
    # Normalize by mean
    d_pred_norm = d_pred / (d_pred.mean() + eps)
    d_ref_norm = d_ref / (d_ref.mean() + eps)
    
    return F.mse_loss(d_pred_norm, d_ref_norm)


def compute_rank_correlation_loss(
    y_pred: "torch.Tensor",
    y_ref: "torch.Tensor",
    n_triplets: int = 256,
) -> "torch.Tensor":
    """Compute rank correlation loss using triplet comparisons.
    
    For random triplets (i, j, k), we check if the ordering of distances
    is preserved:
        d(y_pred_i, y_pred_j) < d(y_pred_i, y_pred_k)  iff
        d(y_ref_i, y_ref_j) < d(y_ref_i, y_ref_k)
    
    This is a differentiable relaxation using soft ranking.
    
    Args:
        y_pred: Predicted latent positions, shape (B, D).
        y_ref: Reference latent positions, shape (B, D).
        n_triplets: Number of triplets to sample.
        
    Returns:
        Scalar loss in [0, 1] where 0 = perfect rank agreement.
    """
    import torch
    
    B = y_pred.shape[0]
    device = y_pred.device
    
    # Sample triplets
    n_triplets = min(n_triplets, B * (B - 1) * (B - 2) // 6)
    idx_i = torch.randint(0, B, (n_triplets,), device=device)
    idx_j = torch.randint(0, B, (n_triplets,), device=device)
    idx_k = torch.randint(0, B, (n_triplets,), device=device)
    
    # Ensure distinct indices
    valid = (idx_i != idx_j) & (idx_j != idx_k) & (idx_i != idx_k)
    idx_i, idx_j, idx_k = idx_i[valid], idx_j[valid], idx_k[valid]
    
    if len(idx_i) < 10:
        return torch.tensor(0.0, device=device, dtype=y_pred.dtype)
    
    # Compute distances
    d_ij_pred = torch.linalg.norm(y_pred[idx_i] - y_pred[idx_j], dim=-1)
    d_ik_pred = torch.linalg.norm(y_pred[idx_i] - y_pred[idx_k], dim=-1)
    d_ij_ref = torch.linalg.norm(y_ref[idx_i] - y_ref[idx_j], dim=-1)
    d_ik_ref = torch.linalg.norm(y_ref[idx_i] - y_ref[idx_k], dim=-1)
    
    # Soft ranking: sigmoid((d_ij - d_ik) / τ) ∈ [0, 1]
    # = 1 if d_ij > d_ik (j is farther), = 0 if d_ij < d_ik (j is closer)
    tau = 0.1  # temperature
    
    # Normalize by scale for numerical stability
    scale_pred = (d_ij_pred + d_ik_pred).mean() / 2 + 1e-8
    scale_ref = (d_ij_ref + d_ik_ref).mean() / 2 + 1e-8
    
    rank_pred = torch.sigmoid((d_ij_pred - d_ik_pred) / (tau * scale_pred))
    rank_ref = torch.sigmoid((d_ij_ref - d_ik_ref) / (tau * scale_ref))
    
    # MSE between soft ranks
    return torch.nn.functional.mse_loss(rank_pred, rank_ref)


def compute_variance_matching_loss(
    y_pred: "torch.Tensor",
    y_ref: "torch.Tensor",
) -> "torch.Tensor":
    """Compute loss that penalizes mismatch between predicted and reference variance.
    
    This directly addresses the issue where learned_std != ref_std.
    Uses relative error: (std_pred / std_ref - 1)^2
    
    Args:
        y_pred: Predicted latent positions, shape (B, D).
        y_ref: Reference latent positions, shape (B, D).
        
    Returns:
        Scalar loss >= 0 where 0 = perfect variance match.
    """
    import torch
    
    eps = 1e-8
    
    # Compute std across batch for each dimension, then mean across dims
    std_pred = y_pred.std(dim=0).mean() + eps
    std_ref = y_ref.std(dim=0).mean() + eps
    
    # Relative error: penalize deviation from ratio = 1
    ratio = std_pred / std_ref
    loss = (ratio - 1.0) ** 2
    
    return loss
