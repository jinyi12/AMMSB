"""Off-grid spectral loss functions for FAE training.

This module avoids any full-grid FFT or grid finite-difference assumptions.
All losses operate directly on sampled points ``(x_dec, u_dec)``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from functional_autoencoders.losses import _call_autoencoder_fn


def _parse_multiscale_sigmas(raw: Optional[str | Sequence[float]]) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]
    return [float(v) for v in raw]


def sample_frequency_vectors(
    key: jax.Array,
    n_freqs: int,
    coord_dim: int = 2,
    sigma: float = 1.0,
    multiscale_sigmas: Optional[str | Sequence[float]] = None,
) -> jnp.ndarray:
    """Sample frequency vectors for nonuniform Fourier projections."""
    sigmas = _parse_multiscale_sigmas(multiscale_sigmas)
    if sigmas:
        n_scales = len(sigmas)
        base_block = n_freqs // n_scales
        remainder = n_freqs % n_scales
        blocks = []
        for i, scale in enumerate(sigmas):
            block_size = base_block + (1 if i < remainder else 0)
            key, subkey = jax.random.split(key)
            blocks.append(jax.random.normal(subkey, (block_size, coord_dim)) * scale)
        return jnp.concatenate(blocks, axis=0)
    return jax.random.normal(key, (n_freqs, coord_dim)) * sigma


def fourier_coefficients(
    u: jnp.ndarray,
    x: jnp.ndarray,
    freq_vectors: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate Fourier coefficients on sampled points.

    Uses Monte-Carlo quadrature over sampled coordinates:
        \hat u(k) \approx mean_j u(x_j) exp(i 2π k·x_j)
    """
    values = u[..., 0][..., None]  # [B, N, 1]
    phase = 2.0 * jnp.pi * jnp.einsum("bnc,fc->bnf", x, freq_vectors)  # [B, N, F]
    coeff_real = jnp.mean(values * jnp.cos(phase), axis=1)  # [B, F]
    coeff_imag = jnp.mean(values * jnp.sin(phase), axis=1)  # [B, F]
    return coeff_real, coeff_imag


def fourier_mse_by_frequency(
    u_pred: jnp.ndarray,
    u_true: jnp.ndarray,
    x: jnp.ndarray,
    freq_vectors: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-frequency coefficient MSE and true signal energy."""
    pred_r, pred_i = fourier_coefficients(u_pred, x, freq_vectors)
    true_r, true_i = fourier_coefficients(u_true, x, freq_vectors)
    error_sq = (pred_r - true_r) ** 2 + (pred_i - true_i) ** 2
    true_energy = true_r**2 + true_i**2
    return error_sq, true_energy


def pairwise_gradient_mismatch(
    u_pred: jnp.ndarray,
    u_true: jnp.ndarray,
    x: jnp.ndarray,
    key: jax.Array,
    n_pairs: int = 1024,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Off-grid Sobolev-like mismatch via random point pairs.

    Approximates gradient mismatch using directional finite differences on random
    pairs of sampled points, without requiring a grid.
    """
    n_points = x.shape[1]
    n_pairs_eff = min(int(n_pairs), max(1, n_points * 4))

    key_i, key_j = jax.random.split(key)
    idx_i = jax.random.randint(key_i, (n_pairs_eff,), 0, n_points)
    idx_j = jax.random.randint(key_j, (n_pairs_eff,), 0, n_points)

    x_i = jnp.take(x, idx_i, axis=1)
    x_j = jnp.take(x, idx_j, axis=1)
    dist_sq = jnp.sum((x_i - x_j) ** 2, axis=-1)  # [B, P]

    u_pred_i = jnp.take(u_pred[..., 0], idx_i, axis=1)
    u_pred_j = jnp.take(u_pred[..., 0], idx_j, axis=1)
    u_true_i = jnp.take(u_true[..., 0], idx_i, axis=1)
    u_true_j = jnp.take(u_true[..., 0], idx_j, axis=1)

    delta_err_sq = ((u_pred_i - u_pred_j) - (u_true_i - u_true_j)) ** 2
    scaled = delta_err_sq / (dist_sq + eps)

    valid = (dist_sq > eps).astype(scaled.dtype)
    denom = jnp.maximum(jnp.sum(valid), 1.0)
    return jnp.sum(scaled * valid) / denom


def weighted_fourier_mismatch(
    u_pred: jnp.ndarray,
    u_true: jnp.ndarray,
    x: jnp.ndarray,
    key: jax.Array,
    freq_weight_power: float = 1.0,
    n_freqs: int = 128,
    spectral_sigma: float = 1.0,
    spectral_multiscale_sigmas: Optional[str | Sequence[float]] = None,
) -> jnp.ndarray:
    """Frequency-weighted mismatch using off-grid Fourier projections."""
    freq_vectors = sample_frequency_vectors(
        key=key,
        n_freqs=n_freqs,
        coord_dim=x.shape[-1],
        sigma=spectral_sigma,
        multiscale_sigmas=spectral_multiscale_sigmas,
    )
    error_sq, _ = fourier_mse_by_frequency(u_pred, u_true, x, freq_vectors)
    freq_mag = jnp.linalg.norm(freq_vectors, axis=-1)
    weights = (1.0 + freq_mag) ** freq_weight_power
    return jnp.mean(error_sq * weights[None, :])


def high_pass_mismatch(
    u_pred: jnp.ndarray,
    u_true: jnp.ndarray,
    x: jnp.ndarray,
    key: jax.Array,
    sigma: float = 0.05,
    n_freqs: int = 128,
    spectral_sigma: float = 1.0,
    spectral_multiscale_sigmas: Optional[str | Sequence[float]] = None,
) -> jnp.ndarray:
    """High-pass mismatch via Gaussian high-pass weighting in frequency space."""
    freq_vectors = sample_frequency_vectors(
        key=key,
        n_freqs=n_freqs,
        coord_dim=x.shape[-1],
        sigma=spectral_sigma,
        multiscale_sigmas=spectral_multiscale_sigmas,
    )
    error_sq, _ = fourier_mse_by_frequency(u_pred, u_true, x, freq_vectors)
    freq_mag_sq = jnp.sum(freq_vectors**2, axis=-1)
    high_pass_weight = 1.0 - jnp.exp(-0.5 * sigma**2 * freq_mag_sq)
    return jnp.mean(error_sq * high_pass_weight[None, :])


def h1_loss(
    u_pred: jnp.ndarray,
    u_true: jnp.ndarray,
    x: jnp.ndarray,
    key: jax.Array,
    lambda_grad: float = 1.0,
    n_pairs: int = 1024,
) -> jnp.ndarray:
    """Off-grid H1-style loss: L2 + pairwise gradient mismatch."""
    l2_term = jnp.mean((u_pred - u_true) ** 2)
    if lambda_grad <= 0.0:
        return l2_term
    grad_term = pairwise_gradient_mismatch(
        u_pred=u_pred,
        u_true=u_true,
        x=x,
        key=key,
        n_pairs=n_pairs,
    )
    return l2_term + lambda_grad * grad_term


def fourier_weighted_loss(
    u_pred: jnp.ndarray,
    u_true: jnp.ndarray,
    x: jnp.ndarray,
    key: jax.Array,
    freq_weight_power: float = 1.0,
    n_freqs: int = 128,
    spectral_sigma: float = 1.0,
    spectral_multiscale_sigmas: Optional[str | Sequence[float]] = None,
) -> jnp.ndarray:
    """Off-grid Fourier-weighted loss."""
    return weighted_fourier_mismatch(
        u_pred=u_pred,
        u_true=u_true,
        x=x,
        key=key,
        freq_weight_power=freq_weight_power,
        n_freqs=n_freqs,
        spectral_sigma=spectral_sigma,
        spectral_multiscale_sigmas=spectral_multiscale_sigmas,
    )


def high_pass_residual_loss(
    u_pred: jnp.ndarray,
    u_true: jnp.ndarray,
    x: jnp.ndarray,
    key: jax.Array,
    sigma: float = 0.05,
    lambda_residual: float = 1.0,
    n_freqs: int = 128,
    spectral_sigma: float = 1.0,
    spectral_multiscale_sigmas: Optional[str | Sequence[float]] = None,
) -> jnp.ndarray:
    """Off-grid high-pass residual loss."""
    l2_term = jnp.mean((u_pred - u_true) ** 2)
    if lambda_residual <= 0.0:
        return l2_term
    hp_term = high_pass_mismatch(
        u_pred=u_pred,
        u_true=u_true,
        x=x,
        key=key,
        sigma=sigma,
        n_freqs=n_freqs,
        spectral_sigma=spectral_sigma,
        spectral_multiscale_sigmas=spectral_multiscale_sigmas,
    )
    return l2_term + lambda_residual * hp_term


def get_spectral_loss_fn(
    autoencoder,
    domain,
    beta: float = 1e-4,
    loss_type: str = "l2",
    # H1 loss params
    lambda_grad: float = 0.0,
    h1_n_pairs: int = 1024,
    # Fourier-weighted loss params
    freq_weight_power: float = 0.0,
    # High-pass residual params
    lambda_residual: float = 0.0,
    residual_sigma: float = 0.05,
    # Off-grid spectral projection params
    n_spectral_freqs: int = 128,
    spectral_sigma: float = 1.0,
    spectral_multiscale_sigmas: Optional[str | Sequence[float]] = None,
) -> Callable:
    """Get off-grid spectral loss function."""
    del domain

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        key, k_enc = jax.random.split(key)
        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=k_enc,
        )

        key, k_dec = jax.random.split(key)
        u_pred, decoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.decoder.apply,
            u=latents,
            x=x_dec,
            name="decoder",
            dropout_key=k_dec,
        )

        key, k_h1 = jax.random.split(key)
        key, k_fourier = jax.random.split(key)
        key, k_residual = jax.random.split(key)

        if loss_type == "l2":
            recon_loss = 0.5 * jnp.mean((u_dec - u_pred) ** 2)
        elif loss_type == "h1":
            recon_loss = h1_loss(
                u_pred=u_pred,
                u_true=u_dec,
                x=x_dec,
                key=k_h1,
                lambda_grad=lambda_grad,
                n_pairs=h1_n_pairs,
            )
        elif loss_type == "fourier_weighted":
            recon_loss = fourier_weighted_loss(
                u_pred=u_pred,
                u_true=u_dec,
                x=x_dec,
                key=k_fourier,
                freq_weight_power=freq_weight_power,
                n_freqs=n_spectral_freqs,
                spectral_sigma=spectral_sigma,
                spectral_multiscale_sigmas=spectral_multiscale_sigmas,
            )
        elif loss_type == "high_pass_residual":
            recon_loss = high_pass_residual_loss(
                u_pred=u_pred,
                u_true=u_dec,
                x=x_dec,
                key=k_residual,
                sigma=residual_sigma,
                lambda_residual=lambda_residual,
                n_freqs=n_spectral_freqs,
                spectral_sigma=spectral_sigma,
                spectral_multiscale_sigmas=spectral_multiscale_sigmas,
            )
        elif loss_type == "combined":
            recon_loss = 0.5 * jnp.mean((u_dec - u_pred) ** 2)
            if lambda_grad > 0:
                recon_loss = recon_loss + lambda_grad * pairwise_gradient_mismatch(
                    u_pred=u_pred,
                    u_true=u_dec,
                    x=x_dec,
                    key=k_h1,
                    n_pairs=h1_n_pairs,
                )
            if freq_weight_power > 0:
                recon_loss = recon_loss + 0.1 * weighted_fourier_mismatch(
                    u_pred=u_pred,
                    u_true=u_dec,
                    x=x_dec,
                    key=k_fourier,
                    freq_weight_power=freq_weight_power,
                    n_freqs=n_spectral_freqs,
                    spectral_sigma=spectral_sigma,
                    spectral_multiscale_sigmas=spectral_multiscale_sigmas,
                )
            if lambda_residual > 0:
                recon_loss = recon_loss + lambda_residual * high_pass_mismatch(
                    u_pred=u_pred,
                    u_true=u_dec,
                    x=x_dec,
                    key=k_residual,
                    sigma=residual_sigma,
                    n_freqs=n_spectral_freqs,
                    spectral_sigma=spectral_sigma,
                    spectral_multiscale_sigmas=spectral_multiscale_sigmas,
                )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        latent_reg = jnp.mean(beta * jnp.sum(latents**2, axis=-1))
        total_loss = recon_loss + latent_reg

        updated_batch_stats = {
            "encoder": encoder_updates["batch_stats"],
            "decoder": decoder_updates["batch_stats"],
        }
        return total_loss, updated_batch_stats

    return loss_fn
