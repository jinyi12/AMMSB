import jax
import jax.numpy as jnp
import numpy as np


def _mean_sum_sq(x: jax.Array) -> jax.Array:
    # Match the convention used in scripts/fae/fae_naive/sobolev_losses.py:
    # sum over channels, mean over batch+points.
    return jnp.mean(jnp.sum(x**2, axis=-1))


def _mean_sum_prod(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.mean(jnp.sum(x * y, axis=-1))


def test_sobolev_energy_identity_value_term():
    key = jax.random.PRNGKey(0)
    key_u, key_v = jax.random.split(key)
    u = jax.random.normal(key_u, (3, 17, 1))
    v = jax.random.normal(key_v, (3, 17, 1))

    # Residual form: 0.5 ||v - u||^2
    residual = 0.5 * _mean_sum_sq(v - u)
    # Data norm constant: 0.5 ||u||^2
    data_norm = 0.5 * _mean_sum_sq(u)
    # Energy form: 0.5 ||v||^2 - <v, u>
    energy = 0.5 * _mean_sum_sq(v) - _mean_sum_prod(v, u)

    np.testing.assert_allclose(np.array(energy), np.array(residual - data_norm), rtol=1e-6, atol=1e-6)


def test_sobolev_energy_identity_gradient_term():
    key = jax.random.PRNGKey(1)
    key_u, key_v = jax.random.split(key)
    du = jax.random.normal(key_u, (2, 19, 2))
    dv = jax.random.normal(key_v, (2, 19, 2))

    residual = 0.5 * _mean_sum_sq(dv - du)
    data_norm = 0.5 * _mean_sum_sq(du)
    energy = 0.5 * _mean_sum_sq(dv) - _mean_sum_prod(dv, du)

    np.testing.assert_allclose(np.array(energy), np.array(residual - data_norm), rtol=1e-6, atol=1e-6)


def test_sobolev_energy_and_residual_have_same_grad_wrt_prediction():
    # The optimizer only "sees" gradients w.r.t. model outputs (and then params).
    # Energy form differs from residual form by a constant independent of prediction.
    key = jax.random.PRNGKey(2)
    key_u, key_v = jax.random.split(key)
    u = jax.random.normal(key_u, (2, 11, 1))
    v0 = jax.random.normal(key_v, (2, 11, 1))

    def residual_loss(v):
        return 0.5 * _mean_sum_sq(v - u)

    def energy_loss(v):
        return 0.5 * _mean_sum_sq(v) - _mean_sum_prod(v, u)

    grad_res = jax.grad(residual_loss)(v0)
    grad_eng = jax.grad(energy_loss)(v0)

    np.testing.assert_allclose(np.array(grad_res), np.array(grad_eng), rtol=1e-6, atol=1e-6)

