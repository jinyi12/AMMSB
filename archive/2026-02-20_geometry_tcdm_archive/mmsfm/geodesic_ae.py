"""Geometry-aware autoencoder for TCDM distance-preserving embeddings.

The autoencoder is time-conditioned and trained to preserve TCDM distances,
with cycle consistency at intermediate times.

Reference: preimage_experiment_main_ae.py, GAGA/train_autoencoder.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def _build_fc_stack(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    *,
    activation_cls: type[nn.Module],
    dropout: float,
    use_spectral_norm: bool,
) -> nn.Sequential:
    """Build an MLP where each intermediate Linear is followed by:
    spectral norm (on the Linear), BatchNorm1d, activation, and Dropout.
    """
    dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [int(output_dim)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        is_intermediate = i < len(dims) - 2
        linear: nn.Module = nn.Linear(dims[i], dims[i + 1])
        if use_spectral_norm and is_intermediate:
            linear = nn.utils.spectral_norm(linear)
        layers.append(linear)
        if is_intermediate:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(activation_cls())
            layers.append(nn.Dropout(p=float(dropout)))
    return nn.Sequential(*layers)


class AugmentedMLP(nn.Module):
    """Modified MLP with dual input encoders and gated mixing (Wang et al., 2021).

    Given an input x, compute two parallel encodings (U, V), and then mix them at
    every hidden layer via an element-wise gate:

        U = σ(W_u x + b_u),  V = σ(W_v x + b_v)
        f^(l) = W^(l) g^(l-1) + b^(l)
        g^(l) = σ(f^(l)) ⊙ U + (1 - σ(f^(l))) ⊙ V
        out = W_out g^(L) + b_out

    To support varying hidden widths, we learn (U_l, V_l) per hidden layer.
    Spectral normalization (optional), BatchNorm, and Dropout are kept as
    regularization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        *,
        activation_cls: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        hidden_dims = [int(d) for d in hidden_dims]
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("AugmentedMLP requires positive input/output dimensions.")

        self.activation = activation_cls()
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else None

        def _linear(in_dim: int, out_dim: int, *, is_intermediate: bool) -> nn.Module:
            layer = nn.Linear(in_dim, out_dim)
            if use_spectral_norm and is_intermediate:
                layer = nn.utils.spectral_norm(layer)
            return layer

        self.gates = nn.ModuleList()
        self.hat_layers = nn.ModuleList()
        self.tilde_layers = nn.ModuleList()
        self.step_norms = nn.ModuleList()

        if len(hidden_dims) == 0:
            self.first = _linear(input_dim, output_dim, is_intermediate=False)
            self.first_norm = nn.Identity()
            self.out = nn.Identity()
            return

        # Hidden layer 1 (uses `first` for backwards compatibility in state dict keys)
        self.first = _linear(input_dim, hidden_dims[0], is_intermediate=True)
        self.first_norm = nn.BatchNorm1d(hidden_dims[0])
        self.hat_layers.append(_linear(input_dim, hidden_dims[0], is_intermediate=True))
        self.tilde_layers.append(_linear(input_dim, hidden_dims[0], is_intermediate=True))

        # Hidden layers 2..L
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.gates.append(_linear(in_dim, out_dim, is_intermediate=True))
            self.step_norms.append(nn.BatchNorm1d(out_dim))
            self.hat_layers.append(_linear(input_dim, out_dim, is_intermediate=True))
            self.tilde_layers.append(_linear(input_dim, out_dim, is_intermediate=True))

        # Final linear readout
        self.out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        if len(self.hat_layers) == 0:
            return self.first(x)

        # U, V encodings per hidden layer
        hat_vals = [self.activation(layer(x)) for layer in self.hat_layers]
        tilde_vals = [self.activation(layer(x)) for layer in self.tilde_layers]

        # Hidden layer 1
        # f = self.first_norm(self.first(x))
        f = self.first(x)
        gate = self.activation(f)
        g = gate * hat_vals[0] + (1.0 - gate) * tilde_vals[0]
        if self.dropout is not None:
            g = self.dropout(g)

        # Hidden layers 2..L
        for idx, gate_layer in enumerate(self.gates, start=1):
            # f = self.step_norms[idx - 1](gate_layer(g))
            f = gate_layer(g)
            gate = self.activation(f)
            g = gate * hat_vals[idx] + (1.0 - gate) * tilde_vals[idx]
            if self.dropout is not None:
                g = self.dropout(g)

        return self.out(g)


class TimeConditionedEncoder(nn.Module):
    """Time-conditioned encoder using the augmented MLP architecture.
    
    Maps ambient space X to latent space Y conditioned on time t.
    E: (x, t) -> y where x ∈ R^D, t ∈ [0,1], y ∈ R^K.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        time_dim: int = 32,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.time_dim = int(time_dim)
        
        # Time embedding
        self.time_emb = _build_fc_stack(
            input_dim=1,
            output_dim=int(time_dim),
            hidden_dims=[int(time_dim)],
            activation_cls=activation_cls,
            dropout=float(dropout),
            use_spectral_norm=use_spectral_norm,
        )
        
        self.net = AugmentedMLP(
            input_dim=int(in_dim) + int(time_dim),
            output_dim=int(out_dim),
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
            dropout=float(dropout),
            use_spectral_norm=use_spectral_norm,
        )
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, D).
            t: Time tensor of shape (B,) or (B, 1).
            
        Returns:
            Latent tensor of shape (B, K).
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        t_emb = self.time_emb(t)  # (B, time_dim)
        xt = torch.cat([x, t_emb], dim=-1)  # (B, D + time_dim)
        return self.net(xt)


class TimeConditionedDecoder(nn.Module):
    """Time-conditioned decoder using the augmented MLP architecture.
    
    Maps latent space Y to ambient space X conditioned on time t.
    D: (y, t) -> x where y ∈ R^K, t ∈ [0,1], x ∈ R^D.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        time_dim: int = 32,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.time_dim = int(time_dim)
        
        # Time embedding
        self.time_emb = _build_fc_stack(
            input_dim=1,
            output_dim=int(time_dim),
            hidden_dims=[int(time_dim)],
            activation_cls=activation_cls,
            dropout=float(dropout),
            use_spectral_norm=use_spectral_norm,
        )
        
        self.net = AugmentedMLP(
            input_dim=int(in_dim) + int(time_dim),
            output_dim=int(out_dim),
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
            dropout=float(dropout),
            use_spectral_norm=use_spectral_norm,
        )
    
    def forward(self, y: Tensor, t: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            y: Latent tensor of shape (B, K).
            t: Time tensor of shape (B,) or (B, 1).
            
        Returns:
            Reconstructed tensor of shape (B, D).
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        t_emb = self.time_emb(t)  # (B, time_dim)
        yt = torch.cat([y, t_emb], dim=-1)  # (B, K + time_dim)
        return self.net(yt)


class UnconditionedEncoder(nn.Module):
    """Encoder without explicit time conditioning.

    Maps ambient space X to latent space Y:
        E: x -> y where x ∈ R^D, y ∈ R^K.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.net = AugmentedMLP(
            input_dim=self.in_dim,
            output_dim=self.out_dim,
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
            dropout=float(dropout),
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UnconditionedDecoder(nn.Module):
    """Decoder without explicit time conditioning.

    Maps latent space Y to ambient space X:
        D: y -> x where y ∈ R^K, x ∈ R^D.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.net = AugmentedMLP(
            input_dim=self.in_dim,
            output_dim=self.out_dim,
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
            dropout=float(dropout),
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, y: Tensor) -> Tensor:
        return self.net(y)


def robust_distance_loss(
    d_pred: Tensor,
    d_ref: Tensor,
    *,
    mode: str = "normalized_mse",
    min_dist: float = 0.01,
    eps: float = 1e-8,
) -> Tensor:
    """Robust distance preservation loss with multiple modes.
    
    Preserves pairwise distances between predicted and reference latents.
    Filters out very small distances to avoid numerical instability.
    
    Args:
        d_pred: Predicted distances of shape (B,).
        d_ref: Reference distances of shape (B,).
        mode: Loss mode:
            - 'normalized_mse': MSE normalized by mean ref distance (stable, recommended)
            - 'log_mse': MSE in log space (scale-invariant)
            - 'huber': Huber loss on differences (robust to outliers)
            - 'correlation': 1 - Pearson correlation (preserves relative ordering)
        min_dist: Minimum reference distance to include (filters very close pairs).
        eps: Small constant for numerical stability.
        
    Returns:
        Scalar distance loss.
    """
    # Filter out very small reference distances (cause instability)
    mask = d_ref > min_dist
    if mask.sum() < 2:
        # Not enough valid pairs, return zero loss
        return torch.tensor(0.0, device=d_pred.device, dtype=d_pred.dtype)
    
    d_pred_f = d_pred[mask]
    d_ref_f = d_ref[mask]
    
    if mode == "normalized_mse":
        # MSE normalized by mean reference distance
        # Stable and interpretable: error relative to typical distance
        mean_ref = d_ref_f.mean() + eps
        normalized_diff = (d_pred_f - d_ref_f) / mean_ref
        return (normalized_diff ** 2).mean()
    
    elif mode == "log_mse":
        # MSE in log space: (log d_pred - log d_ref)²
        # Scale-invariant, more robust but can amplify small distance errors
        log_pred = torch.log(d_pred_f + eps)
        log_ref = torch.log(d_ref_f + eps)
        return F.mse_loss(log_pred, log_ref)
    
    elif mode == "huber":
        # Huber loss on differences: robust to outliers
        # Delta=1 means linear penalty beyond 1 unit difference
        return F.huber_loss(d_pred_f, d_ref_f, delta=1.0)
    
    elif mode == "correlation":
        # Negative correlation: preserves relative ordering
        # Loss = 1 - corr(d_pred, d_ref), range [0, 2]
        d_pred_c = d_pred_f - d_pred_f.mean()
        d_ref_c = d_ref_f - d_ref_f.mean()
        corr = (d_pred_c * d_ref_c).sum() / (
            torch.sqrt((d_pred_c ** 2).sum() * (d_ref_c ** 2).sum()) + eps
        )
        return 1.0 - corr
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Keep old name for backwards compatibility
def scale_invariant_distance_loss(
    d_pred: Tensor,
    d_ref: Tensor,
    *,
    mode: str = "normalized_mse",
    eps: float = 1e-8,
) -> Tensor:
    """Alias for robust_distance_loss (backwards compatibility)."""
    return robust_distance_loss(d_pred, d_ref, mode=mode, eps=eps)


def compute_latent_metrics(
    y: Tensor,
    y_ref: Tensor,
) -> dict[str, float]:
    """Compute comparison metrics between learned and reference latents.
    
    Args:
        y: Learned latent embeddings of shape (B, K).
        y_ref: Reference latent embeddings of shape (B, K).
        
    Returns:
        Dictionary of metrics.
    """
    # MSE
    mse = F.mse_loss(y, y_ref).item()
    
    # Relative MSE (normalized by reference norm)
    ref_norm_sq = (y_ref ** 2).mean()
    rel_mse = (mse / (ref_norm_sq + 1e-8))
    
    # Cosine similarity (averaged over samples)
    cos_sim = F.cosine_similarity(y, y_ref, dim=-1).mean().item()
    
    # Correlation coefficient (Pearson's r)
    y_flat = y.flatten()
    y_ref_flat = y_ref.flatten()
    y_centered = y_flat - y_flat.mean()
    y_ref_centered = y_ref_flat - y_ref_flat.mean()
    corr = (y_centered * y_ref_centered).sum() / (
        torch.sqrt((y_centered ** 2).sum() * (y_ref_centered ** 2).sum()) + 1e-8
    )
    
    # Explained variance (R²)
    ss_res = ((y - y_ref) ** 2).sum()
    ss_tot = ((y_ref - y_ref.mean(dim=0, keepdim=True)) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        "latent/mse": mse,
        "latent/rel_mse": float(rel_mse),
        "latent/cosine_sim": cos_sim,
        "latent/correlation": float(corr.item()),
        "latent/r2": float(r2.item()),
    }


def compute_distance_preservation_metrics(
    y: Tensor,
    y_ref: Tensor,
    *,
    n_pairs: int = 1000,
) -> dict[str, float]:
    """Compute distance preservation metrics between learned and reference latents.
    
    Args:
        y: Learned latent embeddings of shape (B, K).
        y_ref: Reference latent embeddings of shape (B, K).
        n_pairs: Number of random pairs to sample for distance comparison.
        
    Returns:
        Dictionary of distance preservation metrics.
    """
    B = y.shape[0]
    if n_pairs > B * (B - 1) // 2:
        n_pairs = B * (B - 1) // 2
    
    # Sample random pairs
    idx = torch.randperm(B, device=y.device)[:min(n_pairs * 2, B)]
    i = idx[:len(idx)//2]
    j = idx[len(idx)//2:2*(len(idx)//2)]
    
    # Compute pairwise distances
    d_pred = torch.linalg.norm(y[i] - y[j], dim=-1)
    d_ref = torch.linalg.norm(y_ref[i] - y_ref[j], dim=-1)
    
    # Relative error
    rel_error = torch.abs(d_pred - d_ref) / (d_ref + 1e-8)
    
    # Correlation of distances
    d_pred_centered = d_pred - d_pred.mean()
    d_ref_centered = d_ref - d_ref.mean()
    dist_corr = (d_pred_centered * d_ref_centered).sum() / (
        torch.sqrt((d_pred_centered ** 2).sum() * (d_ref_centered ** 2).sum()) + 1e-8
    )
    
    return {
        "dist/mean_rel_error": float(rel_error.mean().item()),
        "dist/median_rel_error": float(rel_error.median().item()),
        "dist/correlation": float(dist_corr.item()),
    }


class GeodesicAutoencoder(nn.Module):
    """Time-conditioned autoencoder for TCDM distance-preserving embeddings.
    
    Training objectives:
    1. Reconstruction: ||D(E(x,t), t) - x||²
    2. Distance preservation: scale-invariant distance matching
    3. Cycle consistency: ||E(D(ψ(t'), t'), t') - ψ(t')||² at intermediate times
    """
    
    def __init__(
        self,
        ambient_dim: int,
        latent_dim: int,
        encoder_hidden: list[int] = [512, 256],
        decoder_hidden: list[int] = [256, 512],
        time_dim: int = 32,
        dropout: float = 0.2,
        use_spectral_norm: bool = True,
        distance_mode: str = "normalized_mse",
        activation_cls: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.ambient_dim = int(ambient_dim)
        self.latent_dim = int(latent_dim)
        self.distance_mode = distance_mode
        
        self.encoder = TimeConditionedEncoder(
            in_dim=ambient_dim,
            out_dim=latent_dim,
            hidden_dims=encoder_hidden,
            time_dim=time_dim,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            activation_cls=activation_cls,
        )
        self.decoder = TimeConditionedDecoder(
            in_dim=latent_dim,
            out_dim=ambient_dim,
            hidden_dims=decoder_hidden,
            time_dim=time_dim,
            dropout=dropout,
            use_spectral_norm=use_spectral_norm,
            activation_cls=activation_cls,
        )
    
    def encode(self, x: Tensor, t: Tensor) -> Tensor:
        """Encode x at time t to latent y."""
        return self.encoder(x, t)
    
    def decode(self, y: Tensor, t: Tensor) -> Tensor:
        """Decode latent y at time t to x."""
        return self.decoder(y, t)
    
    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass: encode then decode.
        
        Returns:
            (y, x_hat) where y is latent and x_hat is reconstruction.
        """
        y = self.encode(x, t)
        x_hat = self.decode(y, t)
        return y, x_hat
    
    def reconstruction_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """MSE reconstruction loss."""
        return F.mse_loss(x_hat, x)
    
    def distance_loss(
        self,
        y: Tensor,
        y_ref: Tensor,
        y_neighbor: Tensor,
        y_ref_neighbor: Tensor,
        *,
        eps: float = 1e-8,
    ) -> Tensor:
        """Scale-invariant distance preservation loss.
        
        Args:
            y: Encoded points of shape (B, K).
            y_ref: Reference latent positions of shape (B, K).
            y_neighbor: Encoded neighbor points of shape (B, K).
            y_ref_neighbor: Reference neighbor latent positions of shape (B, K).
            eps: Numerical stability constant.
            
        Returns:
            Scalar distance loss.
        """
        d_pred = torch.linalg.norm(y - y_neighbor, dim=-1)
        d_ref = torch.linalg.norm(y_ref - y_ref_neighbor, dim=-1)
        return scale_invariant_distance_loss(d_pred, d_ref, mode=self.distance_mode, eps=eps)
    
    def cycle_consistency_loss(
        self,
        psi: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Cycle consistency loss at intermediate times.
        
        For intermediate embeddings ψ(t):
            L_cycle = ||E(D(ψ, t), t) - ψ||²
            
        Args:
            psi: Intermediate latent embeddings of shape (B, K).
            t: Time values of shape (B,).
            
        Returns:
            Scalar cycle consistency loss.
        """
        x_cycle = self.decode(psi, t)
        y_cycle = self.encode(x_cycle, t)
        return F.mse_loss(y_cycle, psi)
    
    def compute_metrics(
        self,
        y: Tensor,
        y_ref: Tensor,
    ) -> dict[str, float]:
        """Compute all latent space comparison metrics.
        
        Args:
            y: Learned latent embeddings of shape (B, K).
            y_ref: Reference latent embeddings of shape (B, K).
            
        Returns:
            Dictionary of metrics for WandB logging.
        """
        metrics = {}
        metrics.update(compute_latent_metrics(y, y_ref))
        metrics.update(compute_distance_preservation_metrics(y, y_ref))
        return metrics
