from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmsfm.fae.fae_training_components import build_autoencoder
from mmsfm.fae.fae_latent_utils import make_fae_apply_fns
from mmsfm.fae.transformer_dit_prior import TransformerDiTPrior
from mmsfm.fae.transformer_downstream import make_transformer_fae_apply_fns
from mmsfm.fae.transformer_prior_support import setup_transformer_prior_training


def test_build_autoencoder_supports_transformer_encoder_and_decoder() -> None:
    key = jax.random.PRNGKey(0)
    autoencoder, architecture_info = build_autoencoder(
        key=key,
        latent_dim=12,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="points",
        n_heads=4,
    )

    batch_size = 2
    n_points = 9
    u = jnp.ones((batch_size, n_points, 1), dtype=jnp.float32)
    coords = jnp.linspace(0.0, 1.0, n_points, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords[:3], coords[:3], indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))

    variables = autoencoder.init(jax.random.PRNGKey(1), u, x, x, train=False)
    encoder_vars = {
        "params": variables["params"]["encoder"],
        "batch_stats": variables.get("batch_stats", {}).get("encoder", {}),
    }
    latents = autoencoder.encoder.apply(
        encoder_vars,
        u,
        x,
        train=False,
    )
    decoded = autoencoder.apply(variables, u, x, x, train=False)

    assert latents.shape == (batch_size, 4, 16)
    assert decoded.shape == (batch_size, n_points, 1)
    assert architecture_info["encoder_type"] == "transformer"
    assert architecture_info["decoder_type"] == "transformer"
    assert architecture_info["latent_representation"] == "token_sequence"
    assert architecture_info["transformer_latent_shape"] == [4, 16]
    assert architecture_info["transformer_num_latents"] == 4


def test_build_autoencoder_supports_patchified_transformer_encoder_and_decoder() -> None:
    key = jax.random.PRNGKey(2)
    autoencoder, architecture_info = build_autoencoder(
        key=key,
        latent_dim=8,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )

    batch_size = 2
    side = 4
    n_points = side * side
    u = jnp.ones((batch_size, n_points, 1), dtype=jnp.float32)
    coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))

    variables = autoencoder.init(jax.random.PRNGKey(3), u, x, x, train=False)
    encoder_vars = {
        "params": variables["params"]["encoder"],
        "batch_stats": variables.get("batch_stats", {}).get("encoder", {}),
    }
    decoder_vars = {
        "params": variables["params"]["decoder"],
        "batch_stats": variables.get("batch_stats", {}).get("decoder", {}),
    }
    latents = autoencoder.encoder.apply(
        encoder_vars,
        u,
        x,
        train=False,
    )
    decoded = autoencoder.apply(variables, u, x, x, train=False)
    subset_x = x[:, ::2, :]
    decoded_subset = autoencoder.decoder.apply(
        decoder_vars,
        latents,
        subset_x,
        train=False,
    )
    shifted_x = (x + jnp.array([0.125, 0.0], dtype=jnp.float32)) % 1.0
    decoded_shifted = autoencoder.decoder.apply(
        decoder_vars,
        latents,
        shifted_x,
        train=False,
    )
    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder,
        variables["params"],
        variables.get("batch_stats", {}),
    )
    flat_latents = encode_fn(np.array(u), np.array(x))
    roundtrip = decode_fn(flat_latents, np.array(x))

    assert latents.shape == (batch_size, 4, 16)
    assert decoded.shape == (batch_size, n_points, 1)
    assert decoded_subset.shape == (batch_size, n_points // 2, 1)
    assert not jnp.allclose(decoded, decoded_shifted)
    assert flat_latents.shape == (batch_size, 64)
    assert roundtrip.shape == (batch_size, n_points, 1)
    assert architecture_info["transformer_tokenization"] == "patches"
    assert architecture_info["transformer_patch_size"] == 2
    assert architecture_info["transformer_max_grid_size"] == [4, 4]
    assert architecture_info["transformer_grid_size"] == [4, 4]
    assert architecture_info["transformer_decoder_query_mode"] == "coordinates"


def test_transformer_decoder_subset_queries_match_dense_decode_selection() -> None:
    key = jax.random.PRNGKey(21)
    autoencoder, _ = build_autoencoder(
        key=key,
        latent_dim=8,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )

    batch_size = 2
    side = 4
    n_points = side * side
    u = jnp.ones((batch_size, n_points, 1), dtype=jnp.float32)
    coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))
    subset_idx = jnp.array([0, 3, 5, 9, 12, 15], dtype=jnp.int32)
    subset_x = x[:, subset_idx, :]
    perm = jnp.array([5, 0, 3, 2, 1, 4], dtype=jnp.int32)

    variables = autoencoder.init(jax.random.PRNGKey(22), u, x, x, train=False)
    encoder_vars = {
        "params": variables["params"]["encoder"],
        "batch_stats": variables.get("batch_stats", {}).get("encoder", {}),
    }
    decoder_vars = {
        "params": variables["params"]["decoder"],
        "batch_stats": variables.get("batch_stats", {}).get("decoder", {}),
    }

    latents = autoencoder.encoder.apply(encoder_vars, u, x, train=False)
    dense_decode = autoencoder.decoder.apply(decoder_vars, latents, x, train=False)
    subset_decode = autoencoder.decoder.apply(decoder_vars, latents, subset_x, train=False)
    subset_decode_unbatched = autoencoder.decoder.apply(
        decoder_vars,
        latents,
        subset_x[0],
        train=False,
    )
    permuted_subset_decode = autoencoder.decoder.apply(
        decoder_vars,
        latents,
        subset_x[:, perm, :],
        train=False,
    )

    np.testing.assert_allclose(
        np.asarray(subset_decode),
        np.asarray(dense_decode[:, subset_idx, :]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(permuted_subset_decode),
        np.asarray(subset_decode[:, perm, :]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(subset_decode_unbatched),
        np.asarray(subset_decode),
        rtol=1e-5,
        atol=1e-5,
    )


def test_transformer_downstream_apply_fns_are_owned_separately() -> None:
    key = jax.random.PRNGKey(23)
    autoencoder, _ = build_autoencoder(
        key=key,
        latent_dim=8,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )

    batch_size = 2
    side = 4
    n_points = side * side
    u = jnp.ones((batch_size, n_points, 1), dtype=jnp.float32)
    coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))

    variables = autoencoder.init(jax.random.PRNGKey(24), u, x, x, train=False)
    generic_encode, generic_decode = make_fae_apply_fns(
        autoencoder,
        variables["params"],
        variables.get("batch_stats", {}),
    )
    transport_encode, transport_decode = make_transformer_fae_apply_fns(
        autoencoder,
        variables["params"],
        variables.get("batch_stats", {}),
        latent_format="flattened",
    )
    token_encode, token_decode = make_transformer_fae_apply_fns(
        autoencoder,
        variables["params"],
        variables.get("batch_stats", {}),
        latent_format="tokens",
    )

    z_generic = generic_encode(np.asarray(u), np.asarray(x))
    z_transport = transport_encode(np.asarray(u), np.asarray(x))
    z_tokens = token_encode(np.asarray(u), np.asarray(x))
    u_generic = generic_decode(z_generic, np.asarray(x))
    u_transport = transport_decode(z_transport, np.asarray(x))
    u_tokens = token_decode(z_tokens, np.asarray(x))

    assert z_generic.shape == (batch_size, 64)
    assert z_transport.shape == (batch_size, 64)
    assert z_tokens.shape == (batch_size, 4, 16)
    np.testing.assert_allclose(z_generic, z_transport, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        z_generic,
        np.reshape(z_tokens, (batch_size, -1)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(u_generic, u_transport, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(u_generic, u_tokens, rtol=1e-5, atol=1e-5)


def test_patch_transformer_encoder_is_permutation_robust_under_jit() -> None:
    key = jax.random.PRNGKey(4)
    autoencoder, _ = build_autoencoder(
        key=key,
        latent_dim=8,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )

    batch_size = 2
    side = 4
    n_points = side * side
    u = (
        jnp.arange(batch_size * n_points, dtype=jnp.float32).reshape(batch_size, n_points, 1)
        / float(n_points)
    )
    coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))
    perm = jnp.array([5, 0, 15, 2, 9, 4, 8, 14, 1, 10, 6, 13, 3, 12, 7, 11])
    u_perm = u[:, perm, :]
    x_perm = x[:, perm, :]

    variables = autoencoder.init(jax.random.PRNGKey(5), u, x, x, train=False)
    encode_fn, _ = make_fae_apply_fns(
        autoencoder,
        variables["params"],
        variables.get("batch_stats", {}),
    )

    latents = encode_fn(np.asarray(u), np.asarray(x))
    latents_perm = encode_fn(np.asarray(u_perm), np.asarray(x_perm))

    np.testing.assert_allclose(latents_perm, latents, rtol=1e-5, atol=1e-5)


def test_patch_transformer_encoder_supports_lower_resolution_runtime_grid() -> None:
    key = jax.random.PRNGKey(6)
    autoencoder, architecture_info = build_autoencoder(
        key=key,
        latent_dim=8,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )

    batch_size = 2
    runtime_side = 2
    n_points = runtime_side * runtime_side
    u = jnp.ones((batch_size, n_points, 1), dtype=jnp.float32)
    coords = jnp.linspace(0.0, 1.0, runtime_side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))

    variables = autoencoder.init(jax.random.PRNGKey(7), u, x, x, train=False)
    encoder_vars = {
        "params": variables["params"]["encoder"],
        "batch_stats": variables.get("batch_stats", {}).get("encoder", {}),
    }
    latents = autoencoder.encoder.apply(encoder_vars, u, x, train=False)
    decoded = autoencoder.apply(variables, u, x, x, train=False)

    assert latents.shape == (batch_size, 4, 16)
    assert decoded.shape == (batch_size, n_points, 1)
    assert architecture_info["transformer_max_grid_size"] == [4, 4]


def test_build_autoencoder_rejects_mixed_transformer_and_vector_components() -> None:
    key = jax.random.PRNGKey(8)

    with pytest.raises(ValueError, match="requires encoder_type='transformer' and decoder_type='transformer' together"):
        build_autoencoder(
            key=key,
            latent_dim=8,
            n_freqs=4,
            fourier_sigma=1.0,
            decoder_features=(16, 16),
            encoder_type="pooling",
            decoder_type="transformer",
            transformer_emb_dim=16,
            transformer_num_latents=4,
            transformer_encoder_depth=2,
            transformer_cross_attn_depth=1,
            transformer_decoder_depth=2,
            transformer_mlp_ratio=2,
            transformer_tokenization="patches",
            transformer_patch_size=2,
            transformer_grid_size=(4, 4),
            n_heads=4,
        )


def test_transformer_dit_prior_matches_encoder_token_shapes_for_point_and_patch_modes() -> None:
    configs = [
        {
            "key": jax.random.PRNGKey(30),
            "tokenization": "points",
            "patch_size": 2,
            "grid_size": None,
            "side": 3,
        },
        {
            "key": jax.random.PRNGKey(31),
            "tokenization": "patches",
            "patch_size": 2,
            "grid_size": (4, 4),
            "side": 4,
        },
    ]

    for cfg in configs:
        autoencoder, _ = build_autoencoder(
            key=cfg["key"],
            latent_dim=8,
            n_freqs=4,
            fourier_sigma=1.0,
            decoder_features=(16, 16),
            encoder_type="transformer",
            decoder_type="transformer",
            transformer_emb_dim=16,
            transformer_num_latents=4,
            transformer_encoder_depth=2,
            transformer_cross_attn_depth=1,
            transformer_decoder_depth=2,
            transformer_mlp_ratio=2,
            transformer_tokenization=cfg["tokenization"],
            transformer_patch_size=cfg["patch_size"],
            transformer_grid_size=cfg["grid_size"],
            n_heads=4,
        )

        side = int(cfg["side"])
        n_points = side * side
        batch_size = 2
        u = jnp.ones((batch_size, n_points, 1), dtype=jnp.float32)
        coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
        x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
        x = jnp.reshape(x, (1, n_points, 2))
        x = jnp.broadcast_to(x, (batch_size, n_points, 2))

        variables = autoencoder.init(jax.random.PRNGKey(side + 40), u, x, x, train=False)
        encoder_vars = {
            "params": variables["params"]["encoder"],
            "batch_stats": variables.get("batch_stats", {}).get("encoder", {}),
        }
        latents = autoencoder.encoder.apply(encoder_vars, u, x, train=False)

        prior = TransformerDiTPrior(
            hidden_dim=32,
            n_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            time_emb_dim=8,
            prior_logsnr_max=5.0,
        )
        prior_variables = prior.init(jax.random.PRNGKey(side + 50), latents, jnp.zeros((batch_size,)))
        pred = prior.apply(prior_variables, latents, jnp.zeros((batch_size,)))

        assert pred.shape == latents.shape


def test_transformer_dit_prior_velocity_from_x0_prediction_matches_linear_flow_formula() -> None:
    prior = TransformerDiTPrior(
        hidden_dim=32,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    z_clean = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10.0
    noise = jnp.flip(z_clean, axis=-1) + 0.25
    t = jnp.array([0.2, 0.7], dtype=jnp.float32)

    z_t = prior.mix_latent(z_clean, t, noise)
    pred_velocity = prior.velocity_from_x0_prediction(z_t, z_clean, t)
    target_velocity = prior.closed_form_velocity_target(z_clean, noise)

    t_b = t.reshape(2, 1, 1)
    x1_pred = (z_t - (1.0 - t_b) * z_clean) / t_b
    endpoint_velocity = (x1_pred - z_t) / (1.0 - t_b)

    np.testing.assert_allclose(pred_velocity, target_velocity, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pred_velocity, endpoint_velocity, rtol=1e-6, atol=1e-6)


def test_transformer_prior_setup_initializes_prior_params_and_reconstruction_hook() -> None:
    autoencoder, _ = build_autoencoder(
        key=jax.random.PRNGKey(60),
        latent_dim=8,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )

    args = SimpleNamespace(
        n_heads=4,
        beta=0.0,
        loss_type="l2",
        prior_hidden_dim=32,
        prior_n_layers=2,
        prior_time_emb_dim=8,
        prior_logsnr_max=5.0,
        prior_loss_weight=1.0,
        prior_num_heads=4,
        prior_mlp_ratio=2.0,
        ntk_scale_norm=10.0,
        ntk_epsilon=1e-8,
        ntk_estimate_total_trace=False,
        ntk_total_trace_ema_decay=0.99,
        ntk_n_loss_terms=1,
        ntk_trace_update_interval=100,
        ntk_hutchinson_probes=1,
        ntk_output_chunk_size=0,
        ntk_trace_estimator="rhutch",
    )

    loss_fn, metrics, reconstruct_fn, extra_init_params_fn = setup_transformer_prior_training(
        autoencoder,
        args,
    )
    extra = extra_init_params_fn(jax.random.PRNGKey(61))

    assert callable(loss_fn)
    assert metrics == []
    assert callable(reconstruct_fn)
    assert "prior" in extra


def test_transformer_apply_fns_ignore_extra_prior_params() -> None:
    autoencoder, _ = build_autoencoder(
        key=jax.random.PRNGKey(70),
        latent_dim=8,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )

    batch_size = 2
    side = 4
    n_points = side * side
    u = jnp.ones((batch_size, n_points, 1), dtype=jnp.float32)
    coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))

    variables = autoencoder.init(jax.random.PRNGKey(71), u, x, x, train=False)
    params_with_prior = dict(variables["params"])
    params_with_prior["prior"] = {"dummy": jnp.ones((1,), dtype=jnp.float32)}

    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder,
        params_with_prior,
        variables.get("batch_stats", {}),
    )
    latents = encode_fn(np.asarray(u), np.asarray(x))
    decoded = decode_fn(latents, np.asarray(x))

    assert latents.shape == (batch_size, 64)
    assert decoded.shape == (batch_size, n_points, 1)
