from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from mmsfm.fae.sigreg import (  # noqa: E402
    compute_projected_variance_floor_loss,
    compute_sigreg_loss_from_latents,
    flatten_token_latents,
    sample_normalized_gaussian_slices,
)


def test_sample_normalized_gaussian_slices_is_deterministic_and_unit_norm() -> None:
    key = jax.random.PRNGKey(0)
    slices_a = sample_normalized_gaussian_slices(key, latent_dim=8, num_slices=16)
    slices_b = sample_normalized_gaussian_slices(key, latent_dim=8, num_slices=16)

    np.testing.assert_allclose(slices_a, slices_b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jnp.linalg.norm(slices_a, axis=0),
        jnp.ones((16,), dtype=slices_a.dtype),
        rtol=1e-6,
        atol=1e-6,
    )


def test_flatten_token_latents_flattens_last_two_dimensions() -> None:
    latents = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    flattened = flatten_token_latents(latents)

    assert flattened.shape == (2, 12)
    np.testing.assert_allclose(flattened[0], np.arange(12, dtype=np.float32))
    np.testing.assert_allclose(flattened[1], np.arange(12, 24, dtype=np.float32))


def test_sigreg_statistic_is_lower_for_standard_normal_than_shifted_or_non_gaussian() -> None:
    key = jax.random.PRNGKey(1)
    key, k_std, k_shift, k_uniform, k_aniso, k_loss = jax.random.split(key, 6)

    standard = jax.random.normal(k_std, (1024, 8))
    shifted = jax.random.normal(k_shift, (1024, 8)) + 1.0
    uniform = jax.random.uniform(k_uniform, (1024, 8), minval=-jnp.sqrt(3.0), maxval=jnp.sqrt(3.0))
    anisotropic = jax.random.normal(k_aniso, (1024, 8)) * jnp.linspace(0.2, 2.0, 8, dtype=jnp.float32)

    standard_loss, standard_residual = compute_sigreg_loss_from_latents(
        standard,
        key=k_loss,
        num_slices=128,
        num_points=9,
        t_max=3.0,
    )
    shifted_loss, _ = compute_sigreg_loss_from_latents(
        shifted,
        key=k_loss,
        num_slices=128,
        num_points=9,
        t_max=3.0,
    )
    uniform_loss, _ = compute_sigreg_loss_from_latents(
        uniform,
        key=k_loss,
        num_slices=128,
        num_points=9,
        t_max=3.0,
    )
    anisotropic_loss, _ = compute_sigreg_loss_from_latents(
        anisotropic,
        key=k_loss,
        num_slices=128,
        num_points=9,
        t_max=3.0,
    )

    assert standard_residual.shape == (128, 18)
    assert float(standard_loss) < float(shifted_loss)
    assert float(standard_loss) < float(uniform_loss)
    assert float(standard_loss) < float(anisotropic_loss)


def test_projected_variance_floor_penalizes_collapsed_latents_more_than_spread_latents() -> None:
    key = jax.random.PRNGKey(7)
    collapsed = jnp.zeros((64, 8), dtype=jnp.float32)
    spread = jax.random.normal(key, (64, 8))

    collapsed_loss, collapsed_diag = compute_projected_variance_floor_loss(
        collapsed,
        key=jax.random.PRNGKey(8),
        num_directions=16,
        variance_floor=1e-2,
    )
    spread_loss, spread_diag = compute_projected_variance_floor_loss(
        spread,
        key=jax.random.PRNGKey(9),
        num_directions=16,
        variance_floor=1e-2,
    )

    assert float(collapsed_loss) > float(spread_loss)
    assert float(collapsed_diag["projected_var_min"]) == pytest.approx(0.0, abs=1e-8)
    assert float(spread_diag["projected_var_mean"]) > 0.0
