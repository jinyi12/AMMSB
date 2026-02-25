import sys
from pathlib import Path

import pytest


# Ensure repo root is on sys.path so `scripts.*` imports work under pytest,
# even when the project is not installed as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scripts.fae.fae_naive.diffusion_denoiser_decoder import ScaledDenoiserDecoder
from scripts.fae.fae_naive.spacetime_geometry_fae_jax import (
    compute_spacetime_energy_on_decoder_trajectory,
)


def test_fae_decoder_spacetime_energy_smoke():
    key = jax.random.PRNGKey(0)
    b = 1
    latent_dim = 8
    n_points = 16
    x_dim = 1
    out_dim = 1

    decoder = ScaledDenoiserDecoder(
        out_dim=out_dim,
        time_emb_dim=16,
        scaling=1.0,
        diffusion_steps=8,
        beta_schedule="linear",
        sampler="ode",
        time_eps=1e-2,
        norm_type="layernorm",
    )

    z = jax.random.normal(key, (b, latent_dim), dtype=jnp.float32)
    x = jnp.linspace(-1.0, 1.0, n_points, dtype=jnp.float32)[:, None]  # [P,1]
    x = jnp.broadcast_to(x[None, :, :], (b, n_points, x_dim))
    # Dummy noisy field + times for init.
    noisy = jax.random.normal(key, (b, n_points, out_dim), dtype=jnp.float32)
    t = jnp.full((b,), 0.5, dtype=jnp.float32)

    key, init_key, run_key = jax.random.split(key, 3)
    variables = decoder.init(init_key, z, x, noisy, t, train=False, method=decoder.predict_x)
    decoder_vars = {"params": variables["params"]}

    res = compute_spacetime_energy_on_decoder_trajectory(
        decoder=decoder,
        decoder_vars=decoder_vars,
        z_cond=z,
        x_coords=x,
        key=run_key,
        num_steps=6,
        sampler="ode",
        sde_sigma=0.0,
        num_probes=1,
        probe="rademacher",
        stabilize_nonneg=True,
    )

    assert res.energy.shape == ()
    assert jnp.isfinite(res.energy)
    assert res.energy >= 0.0
    assert res.t_grid.ndim == 1
    assert res.z_curve.ndim == 3
    assert res.x0_hat_curve.shape == res.z_curve.shape

