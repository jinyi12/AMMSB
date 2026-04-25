from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch

from data.transform_utils import apply_inverse_transform, load_transform_info
from mmsfm.fae.fae_latent_utils import (
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.fae.tran_evaluation.coarse_consistency_cache import (
    build_or_load_global_decoded_cache,
    build_or_load_global_latent_cache,
    build_or_load_interval_decoded_cache,
    build_or_load_interval_latent_cache,
)
from scripts.csp.conditional_eval.condition_set import (
    build_interval_condition_batch,
    build_root_condition_batch,
)
from scripts.csp.conditional_eval.seed_policy import build_seed_policy
from scripts.fae.tran_evaluation.coarse_consistency import (
    aggregate_conditionwise_dirac_statistics,
    select_conditioned_qualitative_examples,
    summarize_conditioned_residuals,
)
from scripts.fae.tran_evaluation.conditional_support import make_pair_label
from scripts.fae.tran_evaluation.core import FilterLadder
from scripts.fae.tran_evaluation.latent_msbm_runtime import (
    build_latent_msbm_agent,
    load_policy_checkpoints,
    sample_backward_full_trajectory_knots,
    sample_backward_one_interval,
)
from scripts.csp.token_decode_runtime import (
    AUTO_COARSE_DECODE_BATCH_CAP,
    decode_token_latent_batch,
    resolve_decode_point_batch_size,
    resolve_decode_sample_batch_size,
    resolve_requested_jax_device,
)
from scripts.csp.run_context import (
    build_sigma_fn_from_config,
    load_decode_dataset_metadata,
    normalize_condition_mode,
)
from csp._trajectory_layout import generation_zt_from_data_zt
from csp.sde import interval_save_times
from scripts.fae.tran_evaluation.run_support import (
    parse_key_value_args_file as parse_args_file,
    resolve_existing_path,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TOKEN_CONDITIONED_SAMPLING_MAX_BATCH_SIZE = 4


@dataclass(frozen=True)
class CoarseConsistencyRuntime:
    provider: str
    run_dir: Path
    split: dict[str, Any]
    latent_train: np.ndarray
    latent_test: np.ndarray
    zt: np.ndarray
    time_indices: np.ndarray
    decode_latents_to_fields: Callable[[np.ndarray], np.ndarray]
    sample_interval_latents: Callable[[np.ndarray, int, int, int, float | None], np.ndarray]
    sample_full_rollout_knots: Callable[[np.ndarray, int, int, float | None], np.ndarray]
    sample_full_rollout_dense: Callable[
        [np.ndarray, int, int, float | None],
        tuple[np.ndarray, np.ndarray],
    ]
    supports_conditioned_metrics: bool
    metadata: dict[str, Any]
    sample_full_rollout_knots_for_split: (
        Callable[[str, np.ndarray, int, int, float | None], np.ndarray] | None
    ) = None

    def __post_init__(self) -> None:
        if self.sample_full_rollout_knots_for_split is None:
            def _sample_full_rollout_knots_for_split(
                split: str,
                sample_indices: np.ndarray,
                n_realizations: int,
                seed: int,
                drift_clip_norm: float | None,
            ) -> np.ndarray:
                if str(split) != "test":
                    raise ValueError(
                        "This runtime was constructed with the legacy test-only full-rollout sampler; "
                        "split-aware sampling is required for non-test population correlation."
                    )
                return self.sample_full_rollout_knots(
                    sample_indices,
                    n_realizations,
                    seed,
                    drift_clip_norm,
                )

            object.__setattr__(
                self,
                "sample_full_rollout_knots_for_split",
                _sample_full_rollout_knots_for_split,
            )


def _unsupported_runtime_op(name: str) -> Callable[..., np.ndarray]:
    def _raise(*_args, **_kwargs) -> np.ndarray:
        raise RuntimeError(
            f"{name} is unavailable on the lightweight coarse-consistency runtime contract loader."
        )

    return _raise


def _resolve_token_conditioned_sampling_max_batch_size(
    sampling_max_batch_size: int | None,
) -> int:
    if sampling_max_batch_size is None:
        return int(DEFAULT_TOKEN_CONDITIONED_SAMPLING_MAX_BATCH_SIZE)
    resolved = int(sampling_max_batch_size)
    if resolved <= 0:
        raise ValueError(
            "sampling_max_batch_size must be a positive integer when provided, "
            f"got {sampling_max_batch_size!r}."
        )
    return resolved


def _dense_data_time_coordinates_from_zt(
    zt: np.ndarray,
    *,
    dt0: float,
) -> np.ndarray:
    zt_arr = np.asarray(zt, dtype=np.float32).reshape(-1)
    generation_zt = np.asarray(generation_zt_from_data_zt(zt_arr), dtype=np.float32).reshape(-1)
    dense_generation_parts: list[np.ndarray] = []
    for interval_idx in range(int(generation_zt.shape[0]) - 1):
        interval_times = np.asarray(
            interval_save_times(
                float(generation_zt[interval_idx]),
                float(generation_zt[interval_idx + 1]),
                float(dt0),
                dtype=jnp.float32,
            ),
            dtype=np.float32,
        ).reshape(-1)
        dense_generation_parts.append(interval_times if interval_idx == 0 else interval_times[1:])
    if not dense_generation_parts:
        return zt_arr.copy()
    dense_generation = np.concatenate(dense_generation_parts, axis=0)
    return np.asarray(float(zt_arr[-1]) - dense_generation[::-1], dtype=np.float32)


def _latent_msbm_dense_time_coordinates(agent: LatentMSBMAgent) -> np.ndarray:
    num_intervals = int(agent.t_dists.numel() - 1)
    dense_parts: list[np.ndarray] = []
    ts_np = np.asarray(agent.ts.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    t_dists_np = np.asarray(agent.t_dists.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    for idx in range(num_intervals - 1, -1, -1):
        rev_idx = (num_intervals - 1) - idx
        interval_times = np.asarray(float(t_dists_np[rev_idx]) + ts_np, dtype=np.float32)
        dense_parts.append(interval_times if idx == num_intervals - 1 else interval_times[1:])
    if not dense_parts:
        return np.asarray(t_dists_np, dtype=np.float32)
    dense_forward = np.concatenate(dense_parts, axis=0)
    return np.asarray(dense_forward[::-1], dtype=np.float32)


def _empty_token_latent_train(root_latents: np.ndarray) -> np.ndarray:
    token_shape = tuple(int(dim) for dim in np.asarray(root_latents, dtype=np.float32).shape[1:])
    return np.zeros((0, 0, *token_shape), dtype=np.float32)


def _root_only_token_latent_test(root_latents: np.ndarray) -> np.ndarray:
    return np.asarray(root_latents, dtype=np.float32)[None, ...]


def _token_rollout_metadata(
    *,
    cfg: dict[str, Any],
    source,
    clip_bounds: tuple[float, float] | None,
    sampling_max_batch_size: int,
) -> dict[str, Any]:
    return {
        "provider": "csp_token_dit",
        "model_type": str(cfg.get("model_type", "conditional_bridge_token_dit")),
        "condition_mode": str(
            normalize_condition_mode(cfg.get("condition_mode", "previous_state"), default="previous_state")
        ),
        "decode_mode": "token_native",
        "dataset_path": str(source.dataset_path),
        "latents_path": str(source.latents_path),
        "fae_checkpoint_path": (
            str(source.fae_checkpoint_path)
            if source.fae_checkpoint_path is not None
            else None
        ),
        "source_run_dir": (
            str(source.source_run_dir)
            if source.source_run_dir is not None
            else None
        ),
        "clip_to_dataset_range": bool(clip_bounds is not None),
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "use_ema": None,
        "sampling_max_batch_size": int(sampling_max_batch_size),
    }


def _make_decode_latents_to_fields(
    *,
    decode_fn: Any,
    grid_coords: np.ndarray,
    transform_info: dict[str, Any],
    decode_batch_size: int,
    clip_bounds: tuple[float, float] | None = None,
    adaptive_decode_batch_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
    runtime_metadata: dict[str, Any] | None = None,
    stage_label: str = "decode",
) -> Callable[[np.ndarray], np.ndarray]:
    def decode_latents_to_fields(latents: np.ndarray) -> np.ndarray:
        z = np.asarray(latents, dtype=np.float32)
        if z.ndim < 2:
            raise ValueError(
                "Expected latents with shape (N, ...) and an explicit batch axis, "
                f"got {z.shape}."
            )
        if adaptive_decode_batch_fn is not None:
            decode_result = adaptive_decode_batch_fn(z)
            decoded_model = np.asarray(
                decode_result["fields_log"],
                dtype=np.float32,
            )
            if runtime_metadata is not None:
                runtime_metadata["decode_device_resolved"] = str(decode_result["resolved_device"])
                runtime_metadata["decode_jit_enabled"] = bool(decode_result["jit_decode"])
                runtime_metadata["decode_batch_size"] = int(decode_result["sample_batch_size"])
                runtime_metadata["decode_point_batch_size"] = int(decode_result["point_batch_size"])
            return apply_inverse_transform(decoded_model, transform_info)
        parts = []
        for start in range(0, int(z.shape[0]), int(decode_batch_size)):
            z_batch = z[start : start + int(decode_batch_size)]
            x_batch = np.broadcast_to(grid_coords[None, ...], (z_batch.shape[0], *grid_coords.shape))
            decoded = np.asarray(decode_fn(z_batch, x_batch), dtype=np.float32)
            if decoded.ndim == 3:
                decoded = decoded.squeeze(-1)
            if clip_bounds is not None:
                clip_min, clip_max = clip_bounds
                decoded = np.clip(decoded, clip_min, clip_max)
            parts.append(decoded)
        decoded_model = np.concatenate(parts, axis=0)
        return apply_inverse_transform(decoded_model, transform_info)

    return decode_latents_to_fields


def _load_latent_msbm_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    device: str | torch.device,
    decode_mode: str,
    decode_batch_size: int,
    use_ema: bool,
) -> CoarseConsistencyRuntime:
    train_cfg = parse_args_file(run_dir / "args.txt")

    lat_path = run_dir / "fae_latents.npz"
    if not lat_path.exists():
        raise FileNotFoundError(f"Missing {lat_path}")
    with np.load(lat_path, allow_pickle=True) as lat_npz:
        latent_train = np.asarray(lat_npz["latent_train"], dtype=np.float32)
        latent_test = np.asarray(lat_npz["latent_test"], dtype=np.float32)
        zt = np.asarray(lat_npz["zt"], dtype=np.float32)
        time_indices = np.asarray(lat_npz["time_indices"], dtype=np.int64)
        grid_coords = np.asarray(lat_npz["grid_coords"], dtype=np.float32)
        split = dict(np.asarray(lat_npz["split"], dtype=object).reshape(-1)[0])

    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        transform_info = load_transform_info(dataset_npz)

    fae_checkpoint_path = resolve_existing_path(
        train_cfg.get("fae_checkpoint"),
        repo_root=REPO_ROOT,
        roots=[run_dir, Path.cwd()],
    )
    if fae_checkpoint_path is None:
        raise FileNotFoundError("Could not resolve FAE checkpoint from args.txt")
    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_fae_from_checkpoint(ckpt)
    _encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=str(decode_mode),
    )

    agent = build_latent_msbm_agent(
        train_cfg,
        zt,
        int(latent_test.shape[-1]),
        device,
        latent_train=latent_train,
        latent_test=latent_test,
    )
    load_policy_checkpoints(
        agent,
        run_dir,
        device,
        use_ema=bool(use_ema),
        load_forward=False,
        load_backward=True,
        weights_only=True,
    )

    def sample_interval_latents(
        test_sample_indices: np.ndarray,
        interval_idx: int,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        generated_batches = []
        for local_idx, test_idx in enumerate(np.asarray(test_sample_indices, dtype=np.int64)):
            z_start = torch.from_numpy(latent_test[int(interval_idx) + 1, int(test_idx)][None, :]).float().to(device)
            z_gen = sample_backward_one_interval(
                agent=agent,
                policy=agent.z_b,
                z_start=z_start,
                interval_idx=int(interval_idx),
                n_realizations=int(n_realizations),
                seed=int(seed) + 100_000 * int(interval_idx) + 1_000 * int(local_idx),
                drift_clip_norm=drift_clip_norm,
            )
            generated_batches.append(z_gen.detach().cpu().numpy().astype(np.float32))
        return np.stack(generated_batches, axis=0)

    def sample_full_rollout_knots(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        return sample_full_rollout_knots_for_split(
            "test",
            test_sample_indices,
            int(n_realizations),
            int(seed),
            drift_clip_norm,
        )

    def sample_full_rollout_knots_for_split(
        split_name: str,
        sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        split_latents = (
            latent_train if str(split_name) == "train" else latent_test
        )
        rollout_batches = []
        for local_idx, sample_idx in enumerate(np.asarray(sample_indices, dtype=np.int64)):
            z_start = torch.from_numpy(split_latents[-1, int(sample_idx)][None, :]).float().to(device)
            knots = sample_backward_full_trajectory_knots(
                agent=agent,
                policy=agent.z_b,
                z_start=z_start,
                n_realizations=int(n_realizations),
                seed=int(seed) + 10_000 * int(local_idx),
                drift_clip_norm=drift_clip_norm,
            )
            rollout_batches.append(np.transpose(knots.detach().cpu().numpy().astype(np.float32), (1, 0, 2)))
        return np.stack(rollout_batches, axis=0)

    def sample_full_rollout_dense(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rollout_knots: list[np.ndarray] = []
        rollout_dense: list[np.ndarray] = []
        for local_idx, test_idx in enumerate(np.asarray(test_sample_indices, dtype=np.int64)):
            z_start = torch.from_numpy(latent_test[-1, int(test_idx)][None, :]).float().to(device)
            knots_list: list[np.ndarray] = []
            dense_list: list[np.ndarray] = []
            for realization_idx in range(int(n_realizations)):
                torch.manual_seed(int(seed) + 10_000 * int(local_idx) + int(realization_idx))
                knots_i, dense_i = sample_full_trajectory(
                    agent=agent,
                    policy=agent.z_b,
                    y_init=z_start,
                    direction="backward",
                    drift_clip_norm=drift_clip_norm,
                )
                knots_list.append(np.asarray(knots_i[:, 0, :].detach().cpu().numpy(), dtype=np.float32))
                dense_list.append(np.asarray(dense_i[:, 0, :].detach().cpu().numpy(), dtype=np.float32))
            rollout_knots.append(np.stack(knots_list, axis=0))
            rollout_dense.append(np.stack(dense_list, axis=0))
        return np.stack(rollout_knots, axis=0), np.stack(rollout_dense, axis=0)

    return CoarseConsistencyRuntime(
        provider="latent_msbm",
        run_dir=run_dir,
        split=split,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=zt,
        time_indices=time_indices,
        decode_latents_to_fields=_make_decode_latents_to_fields(
            decode_fn=decode_fn,
            grid_coords=grid_coords,
            transform_info=transform_info,
            decode_batch_size=decode_batch_size,
        ),
        sample_interval_latents=sample_interval_latents,
        sample_full_rollout_knots=sample_full_rollout_knots,
        sample_full_rollout_knots_for_split=sample_full_rollout_knots_for_split,
        sample_full_rollout_dense=sample_full_rollout_dense,
        supports_conditioned_metrics=True,
        metadata={
            "provider": "latent_msbm",
            "model_type": "latent_msbm",
            "condition_mode": None,
            "decode_mode": str(decode_mode),
            "dataset_path": str(dataset_path),
            "latents_path": str(lat_path),
            "fae_checkpoint_path": str(fae_checkpoint_path),
            "source_run_dir": None,
            "clip_to_dataset_range": False,
            "clip_bounds": None,
            "rollout_dense_time_coordinates": _latent_msbm_dense_time_coordinates(agent).tolist(),
            "rollout_dense_time_semantics": "latent_msbm_internal_time",
            "use_ema": bool(use_ema),
        },
    )


def _load_csp_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    decode_mode: str,
    decode_batch_size: int,
) -> CoarseConsistencyRuntime:
    from scripts.csp.run_context import load_csp_sampling_runtime, load_fae_decode_context

    csp_runtime = load_csp_sampling_runtime(
        run_dir,
        dataset_override=str(dataset_path),
    )
    if csp_runtime.source.fae_checkpoint_path is None:
        raise ValueError(
            "The CSP run does not record a resolvable FAE checkpoint for conditioned coarse evaluation."
        )
    decode_context = load_fae_decode_context(
        dataset_path=Path(dataset_path),
        fae_checkpoint_path=Path(csp_runtime.source.fae_checkpoint_path),
        decode_mode=decode_mode,
    )
    split = csp_runtime.archive.split
    if not isinstance(split, dict):
        raise ValueError("The CSP latent archive does not contain split metadata required for held-out evaluation.")

    supports_conditioned = (
        csp_runtime.model_type in {"conditional_bridge", "paired_prior_bridge"}
        and csp_runtime.condition_mode is not None
    )

    def sample_interval_latents(
        test_sample_indices: np.ndarray,
        interval_idx: int,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This CSP runtime does not support conditioned interval sampling.")

        import jax
        import jax.numpy as jnp
        from csp import sample_conditional_batch
        from csp.paired_prior_bridge import sample_paired_prior_conditional_batch

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(csp_runtime.archive.latent_test[int(interval_idx) + 1, test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        interval_zt = np.asarray(csp_runtime.archive.zt[int(interval_idx) : int(interval_idx) + 2], dtype=np.float32)
        interval_offset = int(csp_runtime.archive.num_intervals - 1 - int(interval_idx))
        if csp_runtime.model_type == "paired_prior_bridge":
            traj = sample_paired_prior_conditional_batch(
                csp_runtime.model,
                jnp.asarray(repeated_conditions, dtype=jnp.float32),
                jnp.asarray(interval_zt, dtype=jnp.float32),
                float(csp_runtime.delta_v),
                float(csp_runtime.dt0),
                jax.random.PRNGKey(int(seed)),
                condition_num_intervals=int(csp_runtime.archive.num_intervals),
                interval_offset=interval_offset,
                theta_feature_clip=float(csp_runtime.theta_feature_clip or 0.0),
            )
        else:
            global_conditions = np.asarray(csp_runtime.archive.latent_test[-1, test_indices], dtype=np.float32)
            repeated_global = np.repeat(global_conditions, int(n_realizations), axis=0)
            traj = sample_conditional_batch(
                csp_runtime.model,
                jnp.asarray(repeated_conditions, dtype=jnp.float32),
                jnp.asarray(interval_zt, dtype=jnp.float32),
                csp_runtime.sigma_fn,
                float(csp_runtime.dt0),
                jax.random.PRNGKey(int(seed)),
                condition_mode=str(csp_runtime.condition_mode),
                global_condition_batch=jnp.asarray(repeated_global, dtype=jnp.float32),
                condition_num_intervals=int(csp_runtime.archive.num_intervals),
                interval_offset=interval_offset,
            )
        generated = np.asarray(traj[:, 0, :], dtype=np.float32)
        return generated.reshape(coarse_conditions.shape[0], int(n_realizations), coarse_conditions.shape[1])

    def sample_full_rollout_knots(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        return sample_full_rollout_knots_for_split(
            "test",
            test_sample_indices,
            int(n_realizations),
            int(seed),
            drift_clip_norm,
        )

    def sample_full_rollout_knots_for_split(
        split_name: str,
        sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This CSP runtime does not support conditioned full-rollout sampling.")

        import jax
        import jax.numpy as jnp
        from csp import sample_conditional_batch
        from csp.paired_prior_bridge import sample_paired_prior_conditional_batch

        if str(split_name) == "train":
            source_latents = np.asarray(csp_runtime.archive.latent_train, dtype=np.float32)
        elif str(split_name) == "test":
            source_latents = np.asarray(csp_runtime.archive.latent_test, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported rollout split {split_name!r}; expected 'train' or 'test'.")
        split_indices = np.asarray(sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(source_latents[-1, split_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        if csp_runtime.model_type == "paired_prior_bridge":
            traj = sample_paired_prior_conditional_batch(
                csp_runtime.model,
                jnp.asarray(repeated_conditions, dtype=jnp.float32),
                jnp.asarray(csp_runtime.archive.zt, dtype=jnp.float32),
                float(csp_runtime.delta_v),
                float(csp_runtime.dt0),
                jax.random.PRNGKey(int(seed)),
                condition_num_intervals=int(csp_runtime.archive.num_intervals),
                interval_offset=0,
                theta_feature_clip=float(csp_runtime.theta_feature_clip or 0.0),
            )
        else:
            traj = sample_conditional_batch(
                csp_runtime.model,
                jnp.asarray(repeated_conditions, dtype=jnp.float32),
                jnp.asarray(csp_runtime.archive.zt, dtype=jnp.float32),
                csp_runtime.sigma_fn,
                float(csp_runtime.dt0),
                jax.random.PRNGKey(int(seed)),
                condition_mode=str(csp_runtime.condition_mode),
                global_condition_batch=jnp.asarray(repeated_conditions, dtype=jnp.float32),
                condition_num_intervals=int(csp_runtime.archive.num_intervals),
                interval_offset=0,
            )
        rollout = np.asarray(traj, dtype=np.float32)
        return rollout.reshape(coarse_conditions.shape[0], int(n_realizations), rollout.shape[1], rollout.shape[2])

    def sample_full_rollout_dense(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This CSP runtime does not support conditioned dense full-rollout sampling.")

        import jax
        import jax.numpy as jnp
        from csp.sample import sample_conditional_dense_batch_from_keys
        from csp.paired_prior_bridge import sample_paired_prior_conditional_dense_batch_from_keys

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(csp_runtime.archive.latent_test[-1, test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        keys = jax.random.split(jax.random.PRNGKey(int(seed)), int(repeated_conditions.shape[0]))
        if csp_runtime.model_type == "paired_prior_bridge":
            if csp_runtime.delta_v is None:
                raise ValueError("Paired-prior CSP runtime is missing delta_v in the run config.")
            knots, dense = sample_paired_prior_conditional_dense_batch_from_keys(
                csp_runtime.model,
                jnp.asarray(repeated_conditions, dtype=jnp.float32),
                jnp.asarray(csp_runtime.archive.zt, dtype=jnp.float32),
                float(csp_runtime.delta_v),
                float(csp_runtime.dt0),
                keys,
                condition_num_intervals=int(csp_runtime.archive.num_intervals),
                interval_offset=0,
                theta_feature_clip=float(csp_runtime.theta_feature_clip or 0.0),
            )
        else:
            knots, dense = sample_conditional_dense_batch_from_keys(
                csp_runtime.model,
                jnp.asarray(repeated_conditions, dtype=jnp.float32),
                jnp.asarray(csp_runtime.archive.zt, dtype=jnp.float32),
                csp_runtime.sigma_fn,
                float(csp_runtime.dt0),
                keys,
                condition_mode=str(csp_runtime.condition_mode),
                global_condition_batch=jnp.asarray(repeated_conditions, dtype=jnp.float32),
                condition_num_intervals=int(csp_runtime.archive.num_intervals),
                interval_offset=0,
            )
        knots_arr = np.asarray(knots, dtype=np.float32)
        dense_arr = np.asarray(dense, dtype=np.float32)
        return (
            knots_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *knots_arr.shape[1:]),
            dense_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *dense_arr.shape[1:]),
        )

    return CoarseConsistencyRuntime(
        provider="csp",
        run_dir=run_dir,
        split=split,
        latent_train=np.asarray(csp_runtime.archive.latent_train, dtype=np.float32),
        latent_test=np.asarray(csp_runtime.archive.latent_test, dtype=np.float32),
        zt=np.asarray(csp_runtime.archive.zt, dtype=np.float32),
        time_indices=np.asarray(csp_runtime.archive.time_indices, dtype=np.int64),
        decode_latents_to_fields=_make_decode_latents_to_fields(
            decode_fn=decode_context.decode_fn,
            grid_coords=np.asarray(decode_context.grid_coords, dtype=np.float32),
            transform_info=decode_context.transform_info,
            decode_batch_size=decode_batch_size,
            clip_bounds=decode_context.clip_bounds,
        ),
        sample_interval_latents=sample_interval_latents,
        sample_full_rollout_knots=sample_full_rollout_knots,
        sample_full_rollout_knots_for_split=sample_full_rollout_knots_for_split,
        sample_full_rollout_dense=sample_full_rollout_dense,
        supports_conditioned_metrics=supports_conditioned,
        metadata={
            "provider": "csp",
            "model_type": str(csp_runtime.model_type),
            "condition_mode": str(csp_runtime.condition_mode),
            "decode_mode": str(decode_mode),
            "dataset_path": str(csp_runtime.source.dataset_path),
            "latents_path": str(csp_runtime.source.latents_path),
            "fae_checkpoint_path": (
                str(csp_runtime.source.fae_checkpoint_path)
                if csp_runtime.source.fae_checkpoint_path is not None
                else None
            ),
            "source_run_dir": (
                str(csp_runtime.source.source_run_dir)
                if csp_runtime.source.source_run_dir is not None
                else None
            ),
            "clip_to_dataset_range": bool(decode_context.clip_bounds is not None),
            "clip_bounds": list(decode_context.clip_bounds) if decode_context.clip_bounds is not None else None,
            "rollout_dense_time_coordinates": _dense_data_time_coordinates_from_zt(
                np.asarray(csp_runtime.archive.zt, dtype=np.float32),
                dt0=float(csp_runtime.dt0),
            ).tolist(),
            "rollout_dense_time_semantics": "stored_data_zt",
            "use_ema": None,
        },
    )


def _load_token_csp_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    decode_mode: str,
    decode_batch_size: int,
    coarse_sampling_device: str = "auto",
    coarse_decode_device: str = "auto",
    coarse_decode_point_batch_size: int | None = None,
    sampling_max_batch_size: int | None = None,
    sampling_adjoint: str = "direct",
) -> CoarseConsistencyRuntime:
    del decode_mode
    from scripts.csp.token_run_context import (
        load_token_csp_sampling_runtime,
        load_token_fae_decode_context,
        sample_token_csp_batch,
    )
    from csp.token_paired_prior_bridge import PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE

    sampling_device_kind, sampling_device = resolve_requested_jax_device(
        coarse_sampling_device,
        auto_preference="gpu",
    )
    csp_runtime = load_token_csp_sampling_runtime(
        run_dir,
        dataset_override=str(dataset_path),
        runtime_device=sampling_device,
        runtime_device_kind=sampling_device_kind,
    )
    if csp_runtime.source.fae_checkpoint_path is None:
        raise ValueError(
            "The token-native CSP run does not record a resolvable FAE checkpoint for conditioned coarse evaluation."
        )
    split = csp_runtime.archive.split
    if not isinstance(split, dict):
        raise ValueError("The token CSP latent archive does not contain split metadata required for held-out evaluation.")

    supports_conditioned = (
        csp_runtime.model_type in {"conditional_bridge_token_dit", PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE}
        and csp_runtime.condition_mode is not None
    )
    runtime_metadata: dict[str, Any] = {
        "sampling_device_requested": str(coarse_sampling_device),
        "sampling_device_resolved": str(sampling_device_kind),
        "decode_device_requested": str(coarse_decode_device),
        "decode_point_batch_size_requested": (
            None if coarse_decode_point_batch_size is None else int(coarse_decode_point_batch_size)
        ),
    }
    resolved_sampling_max_batch_size = _resolve_token_conditioned_sampling_max_batch_size(
        sampling_max_batch_size,
    )
    runtime_metadata["sampling_max_batch_size"] = int(resolved_sampling_max_batch_size)
    sampling_adjoint_norm = str(sampling_adjoint).strip().lower()
    if sampling_adjoint_norm not in {"checkpoint", "direct"}:
        raise ValueError(
            "sampling_adjoint must be 'checkpoint' or 'direct', "
            f"got {sampling_adjoint!r}."
        )
    if sampling_adjoint_norm == "direct":
        import diffrax

        token_sampling_adjoint = diffrax.DirectAdjoint()
    else:
        token_sampling_adjoint = None
    runtime_metadata["sampling_adjoint"] = str(sampling_adjoint_norm)

    def _decode_context_factory(device_kind: str, jit_decode: bool) -> Any:
        resolved_device_kind, decode_device = resolve_requested_jax_device(
            device_kind,
            auto_preference=device_kind,
        )
        return load_token_fae_decode_context(
            dataset_path=Path(dataset_path),
            fae_checkpoint_path=Path(csp_runtime.source.fae_checkpoint_path),
            decode_device=decode_device,
            decode_device_kind=resolved_device_kind,
            jit_decode=jit_decode,
        )

    initial_decode_device_kind, _ = resolve_requested_jax_device(
        coarse_decode_device,
        auto_preference="cpu",
    )
    first_decode_context = _decode_context_factory(
        initial_decode_device_kind,
        bool(initial_decode_device_kind == "gpu"),
    )
    initial_decode_batch_size = resolve_decode_sample_batch_size(
        int(decode_batch_size),
        requested_device=coarse_decode_device,
        auto_cap=AUTO_COARSE_DECODE_BATCH_CAP,
    )
    decode_contexts = {str(initial_decode_device_kind): first_decode_context}
    decode_device_resolved = str(initial_decode_device_kind)
    decode_point_batch_size = resolve_decode_point_batch_size(
        coarse_decode_point_batch_size,
        grid_size=int(first_decode_context.grid_coords.shape[0]),
    )

    def _adaptive_decode_batch(latents: np.ndarray) -> dict[str, Any]:
        nonlocal decode_device_resolved, initial_decode_batch_size, decode_point_batch_size
        result = decode_token_latent_batch(
            latents=np.asarray(latents, dtype=np.float32),
            decode_context_factory=_decode_context_factory,
            grid_coords=np.asarray(first_decode_context.grid_coords, dtype=np.float32),
            requested_device=str(coarse_decode_device),
            auto_preference="cpu",
            sample_batch_size=int(initial_decode_batch_size),
            point_batch_size=int(decode_point_batch_size),
            stage_label="coarse_consistency_token_decode",
            clip_bounds=first_decode_context.clip_bounds,
            resolved_device=decode_device_resolved,
            context_cache=decode_contexts,
            logger=lambda message: print(message, flush=True),
        )
        decode_device_resolved = str(result["resolved_device"])
        initial_decode_batch_size = int(result["sample_batch_size"])
        decode_point_batch_size = int(result["point_batch_size"])
        return result

    runtime_metadata.update(
        {
            "decode_device_resolved": str(initial_decode_device_kind),
            "decode_jit_enabled": bool(first_decode_context.decode_jit_enabled),
            "decode_batch_size": int(initial_decode_batch_size),
            "decode_point_batch_size": int(decode_point_batch_size),
        }
    )

    def sample_interval_latents(
        test_sample_indices: np.ndarray,
        interval_idx: int,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This token-native CSP runtime does not support conditioned interval sampling.")

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(
            csp_runtime.archive.latent_test[int(interval_idx) + 1, test_indices],
            dtype=np.float32,
        )
        global_conditions = np.asarray(
            csp_runtime.archive.latent_test[-1, test_indices],
            dtype=np.float32,
        )
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        repeated_global = np.repeat(global_conditions, int(n_realizations), axis=0)
        interval_zt = np.asarray(
            csp_runtime.archive.zt[int(interval_idx) : int(interval_idx) + 2],
            dtype=np.float32,
        )
        interval_offset = int(csp_runtime.archive.num_intervals - 1 - int(interval_idx))
        traj = sample_token_csp_batch(
            csp_runtime,
            repeated_conditions,
            interval_zt,
            seed=int(seed),
            global_condition_batch=repeated_global,
            interval_offset=interval_offset,
            condition_num_intervals=int(csp_runtime.archive.num_intervals),
            max_batch_size=int(resolved_sampling_max_batch_size),
            adjoint=token_sampling_adjoint,
        )
        generated = np.asarray(traj[:, 0, ...], dtype=np.float32)
        return generated.reshape(
            coarse_conditions.shape[0],
            int(n_realizations),
            *coarse_conditions.shape[1:],
        )

    def sample_full_rollout_knots(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        return sample_full_rollout_knots_for_split(
            "test",
            test_sample_indices,
            int(n_realizations),
            int(seed),
            drift_clip_norm,
        )

    def sample_full_rollout_knots_for_split(
        split_name: str,
        sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This token-native CSP runtime does not support conditioned full-rollout sampling.")

        if str(split_name) == "train":
            source_latents = np.asarray(csp_runtime.archive.latent_train, dtype=np.float32)
        elif str(split_name) == "test":
            source_latents = np.asarray(csp_runtime.archive.latent_test, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported rollout split {split_name!r}; expected 'train' or 'test'.")
        split_indices = np.asarray(sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(source_latents[-1, split_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        traj = sample_token_csp_batch(
            csp_runtime,
            repeated_conditions,
            csp_runtime.archive.zt,
            seed=int(seed),
            global_condition_batch=repeated_conditions,
            interval_offset=0,
            condition_num_intervals=int(csp_runtime.archive.num_intervals),
            max_batch_size=int(resolved_sampling_max_batch_size),
            adjoint=token_sampling_adjoint,
        )
        rollout = np.asarray(traj, dtype=np.float32)
        return rollout.reshape(
            coarse_conditions.shape[0],
            int(n_realizations),
            *rollout.shape[1:],
        )

    def sample_full_rollout_dense(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This token-native CSP runtime does not support conditioned dense full-rollout sampling.")

        from csp.token_sample import sample_token_conditional_dense_batch_from_keys
        from csp.token_paired_prior_bridge import (
            PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE,
            sample_token_paired_prior_conditional_dense_batch_from_keys,
        )

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(root_latents[test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        batch_size = int(resolved_sampling_max_batch_size)
        base_key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0)
        knot_parts: list[np.ndarray] = []
        dense_parts: list[np.ndarray] = []

        for chunk_idx, start in enumerate(range(0, int(repeated_conditions.shape[0]), int(batch_size))):
            stop = min(int(start) + int(batch_size), int(repeated_conditions.shape[0]))
            chunk_key = jax.random.fold_in(base_key, int(chunk_idx))
            chunk_keys = jax.random.split(chunk_key, int(stop - start))
            chunk_conditions = repeated_conditions[int(start) : int(stop)]
            if model_type == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE:
                if sampling_runtime.delta_v is None:
                    raise ValueError("Token paired-prior runtime is missing delta_v in the run config.")
                knot_chunk, dense_chunk = sample_token_paired_prior_conditional_dense_batch_from_keys(
                    sampling_runtime.model,
                    chunk_conditions,
                    np.asarray(archive_contract.zt, dtype=np.float32),
                    float(sampling_runtime.delta_v),
                    float(sampling_runtime.dt0),
                    chunk_keys,
                    condition_num_intervals=int(archive_contract.num_intervals),
                    interval_offset=0,
                    theta_feature_clip=float(
                        sampling_runtime.theta_feature_clip
                        if sampling_runtime.theta_feature_clip is not None
                        else DEFAULT_THETA_FEATURE_CLIP
                    ),
                    adjoint=token_sampling_adjoint,
                )
            else:
                knot_chunk, dense_chunk = sample_token_conditional_dense_batch_from_keys(
                    sampling_runtime.model,
                    chunk_conditions,
                    np.asarray(archive_contract.zt, dtype=np.float32),
                    sampling_runtime.sigma_fn,
                    float(sampling_runtime.dt0),
                    chunk_keys,
                    condition_mode=str(sampling_runtime.condition_mode),
                    global_condition_batch=chunk_conditions,
                    interval_offset=0,
                    adjoint=token_sampling_adjoint,
                )
            knot_parts.append(np.asarray(jax.device_get(knot_chunk), dtype=np.float32))
            dense_parts.append(np.asarray(jax.device_get(dense_chunk), dtype=np.float32))

        knots_arr = np.concatenate(knot_parts, axis=0)
        dense_arr = np.concatenate(dense_parts, axis=0)
        return (
            knots_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *knots_arr.shape[1:]),
            dense_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *dense_arr.shape[1:]),
        )

    def sample_full_rollout_dense(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This token-native CSP runtime does not support conditioned dense full-rollout sampling.")

        from csp.token_sample import sample_token_conditional_dense_batch_from_keys
        from csp.token_paired_prior_bridge import (
            DEFAULT_THETA_FEATURE_CLIP,
            PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE,
            sample_token_paired_prior_conditional_dense_batch_from_keys,
        )

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(csp_runtime.archive.latent_test[-1, test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        batch_size = int(resolved_sampling_max_batch_size)
        base_key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0)
        knot_parts: list[np.ndarray] = []
        dense_parts: list[np.ndarray] = []

        for chunk_idx, start in enumerate(range(0, int(repeated_conditions.shape[0]), int(batch_size))):
            stop = min(int(start) + int(batch_size), int(repeated_conditions.shape[0]))
            chunk_key = jax.random.fold_in(base_key, int(chunk_idx))
            chunk_keys = jax.random.split(chunk_key, int(stop - start))
            chunk_conditions = repeated_conditions[int(start) : int(stop)]
            if csp_runtime.model_type == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE:
                if csp_runtime.delta_v is None:
                    raise ValueError("Token paired-prior runtime is missing delta_v in the run config.")
                knot_chunk, dense_chunk = sample_token_paired_prior_conditional_dense_batch_from_keys(
                    csp_runtime.model,
                    chunk_conditions,
                    np.asarray(csp_runtime.archive.zt, dtype=np.float32),
                    float(csp_runtime.delta_v),
                    float(csp_runtime.dt0),
                    chunk_keys,
                    condition_num_intervals=int(csp_runtime.archive.num_intervals),
                    interval_offset=0,
                    theta_feature_clip=float(
                        csp_runtime.theta_feature_clip
                        if csp_runtime.theta_feature_clip is not None
                        else DEFAULT_THETA_FEATURE_CLIP
                    ),
                    adjoint=token_sampling_adjoint,
                )
            else:
                knot_chunk, dense_chunk = sample_token_conditional_dense_batch_from_keys(
                    csp_runtime.model,
                    chunk_conditions,
                    np.asarray(csp_runtime.archive.zt, dtype=np.float32),
                    csp_runtime.sigma_fn,
                    float(csp_runtime.dt0),
                    chunk_keys,
                    condition_mode=str(csp_runtime.condition_mode),
                    global_condition_batch=chunk_conditions,
                    interval_offset=0,
                    adjoint=token_sampling_adjoint,
                )
            knot_parts.append(np.asarray(jax.device_get(knot_chunk), dtype=np.float32))
            dense_parts.append(np.asarray(jax.device_get(dense_chunk), dtype=np.float32))

        knots_arr = np.concatenate(knot_parts, axis=0)
        dense_arr = np.concatenate(dense_parts, axis=0)
        return (
            knots_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *knots_arr.shape[1:]),
            dense_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *dense_arr.shape[1:]),
        )

    return CoarseConsistencyRuntime(
        provider="csp_token_dit",
        run_dir=run_dir,
        split=split,
        latent_train=np.asarray(csp_runtime.archive.latent_train, dtype=np.float32),
        latent_test=np.asarray(csp_runtime.archive.latent_test, dtype=np.float32),
        zt=np.asarray(csp_runtime.archive.zt, dtype=np.float32),
        time_indices=np.asarray(csp_runtime.archive.time_indices, dtype=np.int64),
        decode_latents_to_fields=_make_decode_latents_to_fields(
            decode_fn=first_decode_context.decode_fn,
            grid_coords=np.asarray(first_decode_context.grid_coords, dtype=np.float32),
            transform_info=first_decode_context.transform_info,
            decode_batch_size=int(initial_decode_batch_size),
            clip_bounds=first_decode_context.clip_bounds,
            adaptive_decode_batch_fn=_adaptive_decode_batch,
            runtime_metadata=runtime_metadata,
            stage_label="coarse_consistency_token_decode",
        ),
        sample_interval_latents=sample_interval_latents,
        sample_full_rollout_knots=sample_full_rollout_knots,
        sample_full_rollout_knots_for_split=sample_full_rollout_knots_for_split,
        sample_full_rollout_dense=sample_full_rollout_dense,
        supports_conditioned_metrics=supports_conditioned,
        metadata={
            "provider": "csp_token_dit",
            "model_type": str(csp_runtime.model_type),
            "condition_mode": str(csp_runtime.condition_mode),
            "decode_mode": "token_native",
            "dataset_path": str(csp_runtime.source.dataset_path),
            "latents_path": str(csp_runtime.source.latents_path),
            "fae_checkpoint_path": (
                str(csp_runtime.source.fae_checkpoint_path)
                if csp_runtime.source.fae_checkpoint_path is not None
                else None
            ),
            "source_run_dir": (
                str(csp_runtime.source.source_run_dir)
                if csp_runtime.source.source_run_dir is not None
                else None
            ),
            "clip_to_dataset_range": bool(first_decode_context.clip_bounds is not None),
            "clip_bounds": list(first_decode_context.clip_bounds) if first_decode_context.clip_bounds is not None else None,
            "rollout_dense_time_coordinates": _dense_data_time_coordinates_from_zt(
                np.asarray(csp_runtime.archive.zt, dtype=np.float32),
                dt0=float(csp_runtime.dt0),
            ).tolist(),
            "rollout_dense_time_semantics": "stored_data_zt",
            "use_ema": None,
            **runtime_metadata,
        },
    )


def _load_token_csp_sampling_only_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    coarse_sampling_device: str = "auto",
    sampling_max_batch_size: int | None = None,
    sampling_adjoint: str = "direct",
) -> CoarseConsistencyRuntime:
    import equinox as eqx

    from csp.token_dit import build_token_conditional_dit
    from scripts.csp.token_run_context import resolve_token_csp_source_contract, sample_token_csp_batch
    from csp.token_paired_prior_bridge import (
        DEFAULT_THETA_FEATURE_CLIP,
        PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE,
    )

    cfg, source, archive_contract = resolve_token_csp_source_contract(
        run_dir,
        dataset_override=str(dataset_path),
    )
    _resolution, _grid_coords, _transform_info, clip_bounds = load_decode_dataset_metadata(dataset_path)
    sampling_device_kind, sampling_device = resolve_requested_jax_device(
        coarse_sampling_device,
        auto_preference="gpu",
    )
    resolved_sampling_max_batch_size = _resolve_token_conditioned_sampling_max_batch_size(
        sampling_max_batch_size,
    )
    with jax.default_device(sampling_device):
        model = build_token_conditional_dit(
            token_shape=archive_contract.token_shape,
            hidden_dim=int(cfg["dit_hidden_dim"]),
            n_layers=int(cfg["dit_n_layers"]),
            num_heads=int(cfg["dit_num_heads"]),
            mlp_ratio=float(cfg["dit_mlp_ratio"]),
            time_emb_dim=int(cfg["dit_time_emb_dim"]),
            num_intervals=int(archive_contract.num_intervals),
            key=jax.random.PRNGKey(0),
            conditioning_style=str(cfg.get("token_conditioning", "mixed_sequence")),
        )
        checkpoint_path = source.run_dir / "checkpoints" / "conditional_bridge_token_dit.eqx"
        model = eqx.tree_deserialise_leaves(checkpoint_path, model)

    model_type = str(cfg.get("model_type", "conditional_bridge_token_dit"))
    condition_mode = str(
        normalize_condition_mode(cfg.get("condition_mode", "previous_state"), default="previous_state")
    )
    dt0 = float(cfg["dt0"])
    sigma_fn = (
        None
        if model_type == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE
        else build_sigma_fn_from_config(cfg, np.asarray(1.0 - archive_contract.zt, dtype=np.float32))
    )
    supports_conditioned = (
        model_type in {"conditional_bridge_token_dit", PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE}
        and condition_mode is not None
    )
    sampling_adjoint_norm = str(sampling_adjoint).strip().lower()
    if sampling_adjoint_norm not in {"checkpoint", "direct"}:
        raise ValueError(
            "sampling_adjoint must be 'checkpoint' or 'direct', "
            f"got {sampling_adjoint!r}."
        )
    if sampling_adjoint_norm == "direct":
        import diffrax

        token_sampling_adjoint = diffrax.DirectAdjoint()
    else:
        token_sampling_adjoint = None
    runtime_metadata = _token_rollout_metadata(
        cfg=cfg,
        source=source,
        clip_bounds=clip_bounds,
        sampling_max_batch_size=int(resolved_sampling_max_batch_size),
    )
    runtime_metadata.update(
        {
            "sampling_device_requested": str(coarse_sampling_device),
            "sampling_device_resolved": str(sampling_device_kind),
            "sampling_adjoint": str(sampling_adjoint_norm),
            "rollout_dense_time_coordinates": _dense_data_time_coordinates_from_zt(
                np.asarray(archive_contract.zt, dtype=np.float32),
                dt0=float(dt0),
            ).tolist(),
            "rollout_dense_time_semantics": "stored_data_zt",
        }
    )
    sampling_runtime = SimpleNamespace(
        model=model,
        model_type=model_type,
        sigma_fn=sigma_fn,
        dt0=dt0,
        condition_mode=condition_mode,
        delta_v=(float(cfg["delta_v"]) if cfg.get("delta_v") is not None else None),
        theta_feature_clip=float(cfg.get("theta_feature_clip", DEFAULT_THETA_FEATURE_CLIP)),
        sampling_device=sampling_device,
        sampling_device_kind=sampling_device_kind,
    )
    root_latents = np.asarray(archive_contract.latent_test_root, dtype=np.float32)

    def sample_full_rollout_knots(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This token-native CSP runtime does not support conditioned full-rollout sampling.")

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(root_latents[test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        traj = sample_token_csp_batch(
            sampling_runtime,
            repeated_conditions,
            np.asarray(archive_contract.zt, dtype=np.float32),
            seed=int(seed),
            global_condition_batch=repeated_conditions,
            interval_offset=0,
            condition_num_intervals=int(archive_contract.num_intervals),
            max_batch_size=int(resolved_sampling_max_batch_size),
            adjoint=token_sampling_adjoint,
        )
        rollout = np.asarray(traj, dtype=np.float32)
        return rollout.reshape(
            coarse_conditions.shape[0],
            int(n_realizations),
            *rollout.shape[1:],
        )

    def sample_full_rollout_dense(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This token-native CSP runtime does not support conditioned dense full-rollout sampling.")

        from csp.token_sample import sample_token_conditional_dense_batch_from_keys
        from csp.token_paired_prior_bridge import (
            sample_token_paired_prior_conditional_dense_batch_from_keys,
        )

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(root_latents[test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        batch_size = int(resolved_sampling_max_batch_size)
        base_key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0)
        knot_parts: list[np.ndarray] = []
        dense_parts: list[np.ndarray] = []

        for chunk_idx, start in enumerate(range(0, int(repeated_conditions.shape[0]), int(batch_size))):
            stop = min(int(start) + int(batch_size), int(repeated_conditions.shape[0]))
            chunk_key = jax.random.fold_in(base_key, int(chunk_idx))
            chunk_keys = jax.random.split(chunk_key, int(stop - start))
            chunk_conditions = repeated_conditions[int(start) : int(stop)]
            if model_type == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE:
                if sampling_runtime.delta_v is None:
                    raise ValueError("Token paired-prior runtime is missing delta_v in the run config.")
                knot_chunk, dense_chunk = sample_token_paired_prior_conditional_dense_batch_from_keys(
                    sampling_runtime.model,
                    chunk_conditions,
                    np.asarray(archive_contract.zt, dtype=np.float32),
                    float(sampling_runtime.delta_v),
                    float(sampling_runtime.dt0),
                    chunk_keys,
                    condition_num_intervals=int(archive_contract.num_intervals),
                    interval_offset=0,
                    theta_feature_clip=float(
                        sampling_runtime.theta_feature_clip
                        if sampling_runtime.theta_feature_clip is not None
                        else DEFAULT_THETA_FEATURE_CLIP
                    ),
                    adjoint=token_sampling_adjoint,
                )
            else:
                knot_chunk, dense_chunk = sample_token_conditional_dense_batch_from_keys(
                    sampling_runtime.model,
                    chunk_conditions,
                    np.asarray(archive_contract.zt, dtype=np.float32),
                    sampling_runtime.sigma_fn,
                    float(sampling_runtime.dt0),
                    chunk_keys,
                    condition_mode=str(sampling_runtime.condition_mode),
                    global_condition_batch=chunk_conditions,
                    interval_offset=0,
                    adjoint=token_sampling_adjoint,
                )
            knot_parts.append(np.asarray(jax.device_get(knot_chunk), dtype=np.float32))
            dense_parts.append(np.asarray(jax.device_get(dense_chunk), dtype=np.float32))

        knots_arr = np.concatenate(knot_parts, axis=0)
        dense_arr = np.concatenate(dense_parts, axis=0)
        return (
            knots_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *knots_arr.shape[1:]),
            dense_arr.reshape(coarse_conditions.shape[0], int(n_realizations), *dense_arr.shape[1:]),
        )

    return CoarseConsistencyRuntime(
        provider="csp_token_dit",
        run_dir=run_dir,
        split=archive_contract.split,
        latent_train=_empty_token_latent_train(root_latents),
        latent_test=_root_only_token_latent_test(root_latents),
        zt=np.asarray(archive_contract.zt, dtype=np.float32),
        time_indices=np.asarray(archive_contract.time_indices, dtype=np.int64),
        decode_latents_to_fields=_unsupported_runtime_op("decode_latents_to_fields"),
        sample_interval_latents=_unsupported_runtime_op("sample_interval_latents"),
        sample_full_rollout_knots=sample_full_rollout_knots,
        sample_full_rollout_knots_for_split=_unsupported_runtime_op("sample_full_rollout_knots_for_split"),
        sample_full_rollout_dense=sample_full_rollout_dense,
        supports_conditioned_metrics=supports_conditioned,
        metadata=runtime_metadata,
    )


def _load_token_csp_decode_only_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    decode_batch_size: int,
    coarse_decode_device: str = "auto",
    coarse_decode_point_batch_size: int | None = None,
    sampling_max_batch_size: int | None = None,
) -> CoarseConsistencyRuntime:
    from scripts.csp.token_run_context import resolve_token_csp_source_contract, load_token_fae_decode_context

    cfg, source, archive_contract = resolve_token_csp_source_contract(
        run_dir,
        dataset_override=str(dataset_path),
    )
    resolved_sampling_max_batch_size = _resolve_token_conditioned_sampling_max_batch_size(
        sampling_max_batch_size,
    )
    runtime_metadata = _token_rollout_metadata(
        cfg=cfg,
        source=source,
        clip_bounds=None,
        sampling_max_batch_size=int(resolved_sampling_max_batch_size),
    )

    def _decode_context_factory(device_kind: str, jit_decode: bool) -> Any:
        resolved_device_kind, decode_device = resolve_requested_jax_device(
            device_kind,
            auto_preference=device_kind,
        )
        return load_token_fae_decode_context(
            dataset_path=Path(dataset_path),
            fae_checkpoint_path=Path(source.fae_checkpoint_path),
            decode_device=decode_device,
            decode_device_kind=resolved_device_kind,
            jit_decode=jit_decode,
        )

    initial_decode_device_kind, _ = resolve_requested_jax_device(
        coarse_decode_device,
        auto_preference="cpu",
    )
    first_decode_context = _decode_context_factory(
        initial_decode_device_kind,
        bool(initial_decode_device_kind == "gpu"),
    )
    initial_decode_batch_size = resolve_decode_sample_batch_size(
        int(decode_batch_size),
        requested_device=coarse_decode_device,
        auto_cap=AUTO_COARSE_DECODE_BATCH_CAP,
    )
    decode_contexts = {str(initial_decode_device_kind): first_decode_context}
    decode_device_resolved = str(initial_decode_device_kind)
    decode_point_batch_size = resolve_decode_point_batch_size(
        coarse_decode_point_batch_size,
        grid_size=int(first_decode_context.grid_coords.shape[0]),
    )

    def _adaptive_decode_batch(latents: np.ndarray) -> dict[str, Any]:
        nonlocal decode_device_resolved, initial_decode_batch_size, decode_point_batch_size
        result = decode_token_latent_batch(
            latents=np.asarray(latents, dtype=np.float32),
            decode_context_factory=_decode_context_factory,
            grid_coords=np.asarray(first_decode_context.grid_coords, dtype=np.float32),
            requested_device=str(coarse_decode_device),
            auto_preference="cpu",
            sample_batch_size=int(initial_decode_batch_size),
            point_batch_size=int(decode_point_batch_size),
            stage_label="conditional_rollout_token_decode",
            clip_bounds=first_decode_context.clip_bounds,
            resolved_device=decode_device_resolved,
            context_cache=decode_contexts,
            logger=lambda message: print(message, flush=True),
        )
        decode_device_resolved = str(result["resolved_device"])
        initial_decode_batch_size = int(result["sample_batch_size"])
        decode_point_batch_size = int(result["point_batch_size"])
        return result

    runtime_metadata.update(
        {
            "clip_to_dataset_range": bool(first_decode_context.clip_bounds is not None),
            "clip_bounds": (
                list(first_decode_context.clip_bounds)
                if first_decode_context.clip_bounds is not None
                else None
            ),
            "decode_device_requested": str(coarse_decode_device),
            "decode_device_resolved": str(initial_decode_device_kind),
            "decode_point_batch_size_requested": (
                None if coarse_decode_point_batch_size is None else int(coarse_decode_point_batch_size)
            ),
            "decode_jit_enabled": bool(first_decode_context.decode_jit_enabled),
            "decode_batch_size": int(initial_decode_batch_size),
            "decode_point_batch_size": int(decode_point_batch_size),
        }
    )
    root_latents = np.asarray(archive_contract.latent_test_root, dtype=np.float32)

    return CoarseConsistencyRuntime(
        provider="csp_token_dit",
        run_dir=run_dir,
        split=archive_contract.split,
        latent_train=_empty_token_latent_train(root_latents),
        latent_test=_root_only_token_latent_test(root_latents),
        zt=np.asarray(archive_contract.zt, dtype=np.float32),
        time_indices=np.asarray(archive_contract.time_indices, dtype=np.int64),
        decode_latents_to_fields=_make_decode_latents_to_fields(
            decode_fn=first_decode_context.decode_fn,
            grid_coords=np.asarray(first_decode_context.grid_coords, dtype=np.float32),
            transform_info=first_decode_context.transform_info,
            decode_batch_size=int(initial_decode_batch_size),
            clip_bounds=first_decode_context.clip_bounds,
            adaptive_decode_batch_fn=_adaptive_decode_batch,
            runtime_metadata=runtime_metadata,
            stage_label="conditional_rollout_token_decode",
        ),
        sample_interval_latents=_unsupported_runtime_op("sample_interval_latents"),
        sample_full_rollout_knots=_unsupported_runtime_op("sample_full_rollout_knots"),
        sample_full_rollout_knots_for_split=_unsupported_runtime_op("sample_full_rollout_knots_for_split"),
        sample_full_rollout_dense=_unsupported_runtime_op("sample_full_rollout_dense"),
        supports_conditioned_metrics=True,
        metadata=runtime_metadata,
    )


def _load_token_csp_runtime_contract(
    *,
    run_dir: Path,
    dataset_path: Path,
    coarse_decode_device: str = "auto",
    coarse_decode_point_batch_size: int | None = None,
    sampling_max_batch_size: int | None = None,
) -> CoarseConsistencyRuntime:
    from scripts.csp.token_run_context import resolve_token_csp_source_contract

    cfg, source, archive_contract = resolve_token_csp_source_contract(
        run_dir,
        dataset_override=str(dataset_path),
    )
    _resolution, _grid_coords, _transform_info, clip_bounds = load_decode_dataset_metadata(dataset_path)
    resolved_sampling_max_batch_size = _resolve_token_conditioned_sampling_max_batch_size(
        sampling_max_batch_size,
    )
    runtime_metadata = _token_rollout_metadata(
        cfg=cfg,
        source=source,
        clip_bounds=clip_bounds,
        sampling_max_batch_size=int(resolved_sampling_max_batch_size),
    )
    runtime_metadata.update(
        {
            "decode_device_requested": str(coarse_decode_device),
            "decode_point_batch_size_requested": (
                None if coarse_decode_point_batch_size is None else int(coarse_decode_point_batch_size)
            ),
        }
    )
    root_latents = np.asarray(archive_contract.latent_test_root, dtype=np.float32)
    return CoarseConsistencyRuntime(
        provider="csp_token_dit",
        run_dir=run_dir,
        split=archive_contract.split,
        latent_train=_empty_token_latent_train(root_latents),
        latent_test=_root_only_token_latent_test(root_latents),
        zt=np.asarray(archive_contract.zt, dtype=np.float32),
        time_indices=np.asarray(archive_contract.time_indices, dtype=np.int64),
        decode_latents_to_fields=_unsupported_runtime_op("decode_latents_to_fields"),
        sample_interval_latents=_unsupported_runtime_op("sample_interval_latents"),
        sample_full_rollout_knots=_unsupported_runtime_op("sample_full_rollout_knots"),
        sample_full_rollout_knots_for_split=_unsupported_runtime_op("sample_full_rollout_knots_for_split"),
        sample_full_rollout_dense=_unsupported_runtime_op("sample_full_rollout_dense"),
        supports_conditioned_metrics=True,
        metadata=runtime_metadata,
    )


def load_coarse_consistency_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    device: str | torch.device,
    decode_mode: str,
    decode_batch_size: int,
    use_ema: bool,
    coarse_sampling_device: str = "auto",
    coarse_decode_device: str = "auto",
    coarse_decode_point_batch_size: int | None = None,
    sampling_max_batch_size: int | None = None,
    sampling_adjoint: str = "direct",
) -> CoarseConsistencyRuntime:
    run_dir_resolved = Path(run_dir).expanduser().resolve()
    dataset_path_resolved = Path(dataset_path).expanduser().resolve()
    if (run_dir_resolved / "config" / "args.json").exists():
        cfg = json.loads((run_dir_resolved / "config" / "args.json").read_text())
        if str(cfg.get("model_type", "conditional_bridge")) in {
            "conditional_bridge_token_dit",
            "paired_prior_bridge_token_dit",
        }:
            return _load_token_csp_runtime(
                run_dir=run_dir_resolved,
                dataset_path=dataset_path_resolved,
                decode_mode=decode_mode,
                decode_batch_size=decode_batch_size,
                coarse_sampling_device=coarse_sampling_device,
                coarse_decode_device=coarse_decode_device,
                coarse_decode_point_batch_size=coarse_decode_point_batch_size,
                sampling_max_batch_size=sampling_max_batch_size,
                sampling_adjoint=sampling_adjoint,
            )
        return _load_csp_runtime(
            run_dir=run_dir_resolved,
            dataset_path=dataset_path_resolved,
            decode_mode=decode_mode,
            decode_batch_size=decode_batch_size,
        )
    if (run_dir_resolved / "args.txt").exists():
        return _load_latent_msbm_runtime(
            run_dir=run_dir_resolved,
            dataset_path=dataset_path_resolved,
            device=device,
            decode_mode=decode_mode,
            decode_batch_size=decode_batch_size,
            use_ema=use_ema,
        )
    raise FileNotFoundError(
        f"Could not determine a coarse-consistency runtime provider for {run_dir_resolved}."
    )


def load_coarse_consistency_runtime_contract(
    *,
    run_dir: Path,
    dataset_path: Path,
    device: str | torch.device,
    decode_mode: str,
    decode_batch_size: int,
    use_ema: bool,
    coarse_sampling_device: str = "auto",
    coarse_decode_device: str = "auto",
    coarse_decode_point_batch_size: int | None = None,
    sampling_max_batch_size: int | None = None,
) -> CoarseConsistencyRuntime:
    del coarse_sampling_device
    run_dir_resolved = Path(run_dir).expanduser().resolve()
    dataset_path_resolved = Path(dataset_path).expanduser().resolve()
    if (run_dir_resolved / "config" / "args.json").exists():
        cfg = json.loads((run_dir_resolved / "config" / "args.json").read_text())
        if str(cfg.get("model_type", "conditional_bridge")) in {
            "conditional_bridge_token_dit",
            "paired_prior_bridge_token_dit",
        }:
            return _load_token_csp_runtime_contract(
                run_dir=run_dir_resolved,
                dataset_path=dataset_path_resolved,
                coarse_decode_device=coarse_decode_device,
                coarse_decode_point_batch_size=coarse_decode_point_batch_size,
                sampling_max_batch_size=sampling_max_batch_size,
            )
    return load_coarse_consistency_runtime(
        run_dir=run_dir_resolved,
        dataset_path=dataset_path_resolved,
        device=device,
        decode_mode=decode_mode,
        decode_batch_size=decode_batch_size,
        use_ema=use_ema,
        coarse_sampling_device="auto",
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=coarse_decode_point_batch_size,
        sampling_max_batch_size=sampling_max_batch_size,
    )


def load_rollout_latent_cache_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    decode_batch_size: int,
    coarse_sampling_device: str = "auto",
    coarse_decode_device: str = "auto",
    coarse_decode_point_batch_size: int | None = None,
    sampling_max_batch_size: int | None = None,
    sampling_only: bool,
    sampling_adjoint: str = "direct",
) -> CoarseConsistencyRuntime:
    run_dir_resolved = Path(run_dir).expanduser().resolve()
    dataset_path_resolved = Path(dataset_path).expanduser().resolve()
    if (run_dir_resolved / "config" / "args.json").exists():
        cfg = json.loads((run_dir_resolved / "config" / "args.json").read_text())
        if str(cfg.get("model_type", "conditional_bridge")) in {
            "conditional_bridge_token_dit",
            "paired_prior_bridge_token_dit",
        }:
            if sampling_only:
                return _load_token_csp_sampling_only_runtime(
                    run_dir=run_dir_resolved,
                    dataset_path=dataset_path_resolved,
                    coarse_sampling_device=coarse_sampling_device,
                    sampling_max_batch_size=sampling_max_batch_size,
                    sampling_adjoint=sampling_adjoint,
                )
            return _load_token_csp_runtime_contract(
                run_dir=run_dir_resolved,
                dataset_path=dataset_path_resolved,
                coarse_decode_device=coarse_decode_device,
                coarse_decode_point_batch_size=coarse_decode_point_batch_size,
                sampling_max_batch_size=sampling_max_batch_size,
            )
    return (
        load_coarse_consistency_runtime(
            run_dir=run_dir_resolved,
            dataset_path=dataset_path_resolved,
            device="cpu",
            decode_mode="standard",
            decode_batch_size=decode_batch_size,
            use_ema=True,
            coarse_sampling_device=coarse_sampling_device,
            coarse_decode_device=coarse_decode_device,
            coarse_decode_point_batch_size=coarse_decode_point_batch_size,
            sampling_max_batch_size=sampling_max_batch_size,
            sampling_adjoint=sampling_adjoint,
        )
        if sampling_only
        else load_coarse_consistency_runtime_contract(
            run_dir=run_dir_resolved,
            dataset_path=dataset_path_resolved,
            device="cpu",
            decode_mode="standard",
            decode_batch_size=decode_batch_size,
            use_ema=True,
            coarse_sampling_device=coarse_sampling_device,
            coarse_decode_device=coarse_decode_device,
            coarse_decode_point_batch_size=coarse_decode_point_batch_size,
            sampling_max_batch_size=sampling_max_batch_size,
        )
    )


def load_rollout_decoded_cache_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    decode_batch_size: int,
    coarse_decode_device: str = "auto",
    coarse_decode_point_batch_size: int | None = None,
    sampling_max_batch_size: int | None = None,
) -> CoarseConsistencyRuntime:
    run_dir_resolved = Path(run_dir).expanduser().resolve()
    dataset_path_resolved = Path(dataset_path).expanduser().resolve()
    if (run_dir_resolved / "config" / "args.json").exists():
        cfg = json.loads((run_dir_resolved / "config" / "args.json").read_text())
        if str(cfg.get("model_type", "conditional_bridge")) in {
            "conditional_bridge_token_dit",
            "paired_prior_bridge_token_dit",
        }:
            return _load_token_csp_decode_only_runtime(
                run_dir=run_dir_resolved,
                dataset_path=dataset_path_resolved,
                decode_batch_size=decode_batch_size,
                coarse_decode_device=coarse_decode_device,
                coarse_decode_point_batch_size=coarse_decode_point_batch_size,
                sampling_max_batch_size=sampling_max_batch_size,
            )
    return load_coarse_consistency_runtime(
        run_dir=run_dir_resolved,
        dataset_path=dataset_path_resolved,
        device="cpu",
        decode_mode="standard",
        decode_batch_size=decode_batch_size,
        use_ema=True,
        coarse_sampling_device="auto",
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=coarse_decode_point_batch_size,
        sampling_max_batch_size=sampling_max_batch_size,
    )


def split_ground_truth_fields_for_run(
    gt_fields_by_index: dict[int, np.ndarray],
    *,
    split: dict[str, Any],
    time_indices: np.ndarray,
    latent_train: np.ndarray,
    latent_test: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    n_train = int(split["n_train"])
    n_test = int(split["n_test"])
    train_fields: dict[int, np.ndarray] = {}
    test_fields: dict[int, np.ndarray] = {}

    for tidx in np.asarray(time_indices, dtype=np.int64):
        tidx_int = int(tidx)
        full = np.asarray(gt_fields_by_index[tidx_int], dtype=np.float32)
        train_slice = full[:n_train]
        test_slice = full[n_train : n_train + n_test]
        if train_slice.shape[0] != int(latent_train.shape[1]):
            raise ValueError(
                f"Train split mismatch for dataset index {tidx_int}: "
                f"{train_slice.shape[0]} vs latent_train {latent_train.shape[1]}."
            )
        if test_slice.shape[0] != int(latent_test.shape[1]):
            raise ValueError(
                f"Test split mismatch for dataset index {tidx_int}: "
                f"{test_slice.shape[0]} vs latent_test {latent_test.shape[1]}."
            )
        train_fields[tidx_int] = train_slice
        test_fields[tidx_int] = test_slice
    return train_fields, test_fields


def evaluate_conditioned_interval_coarse_consistency_for_runtime(
    *,
    runtime: CoarseConsistencyRuntime,
    test_fields_by_tidx: dict[int, np.ndarray],
    ladder: FilterLadder,
    full_h_schedule: list[float],
    output_dir: Path,
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    relative_eps: float,
    transfer_ridge_lambda: float = 1e-8,
    condition_set: dict[str, Any] | None = None,
    seed_policy: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned interval metrics."
        )

    if condition_set is None:
        rng = np.random.default_rng(int(seed))
        n_test = int(runtime.latent_test.shape[1])
        test_sample_indices = rng.choice(n_test, size=min(int(n_conditions), n_test), replace=False)
        test_sample_indices.sort()
        condition_set = build_interval_condition_batch(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=runtime.time_indices,
            interval_positions=np.arange(max(0, int(runtime.latent_test.shape[0]) - 1), dtype=np.int64),
            pair_labels=[
                make_pair_label(
                    tidx_coarse=int(runtime.time_indices[pair_idx + 1]),
                    tidx_fine=int(runtime.time_indices[pair_idx]),
                    full_H_schedule=full_h_schedule,
                )[0]
                for pair_idx in range(int(runtime.latent_test.shape[0]) - 1)
            ],
        )
    if seed_policy is None:
        seed_policy = build_seed_policy(int(seed))

    cache_payload = build_or_load_interval_decoded_cache(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        full_h_schedule=full_h_schedule,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
    )
    test_sample_indices = np.asarray(cache_payload["test_sample_indices"], dtype=np.int64)
    intervals: dict[str, Any] = {}
    qualitative_examples: dict[str, Any] = {}
    t_count = int(runtime.latent_test.shape[0])
    for pair_idx in range(t_count - 1):
        tidx_fine = int(runtime.time_indices[pair_idx])
        tidx_coarse = int(runtime.time_indices[pair_idx + 1])
        pair_label, h_coarse, h_fine, display_label = make_pair_label(
            tidx_coarse=tidx_coarse,
            tidx_fine=tidx_fine,
            full_H_schedule=full_h_schedule,
        )

        generated_fields = np.asarray(cache_payload["decoded_fine_fields"][pair_idx], dtype=np.float32)
        condition_fields = np.asarray(cache_payload["coarse_targets"][pair_idx], dtype=np.float32)
        recoarsened_fields = np.asarray(
            ladder.transfer_between_H(
                generated_fields.reshape(generated_fields.shape[0] * generated_fields.shape[1], -1),
                source_H=float(h_fine),
                target_H=float(h_coarse),
                ridge_lambda=float(transfer_ridge_lambda),
            ),
            dtype=np.float32,
        ).reshape(generated_fields.shape[0], generated_fields.shape[1], -1)
        summary = aggregate_conditionwise_dirac_statistics(
            recoarsened_fields,
            condition_fields,
            relative_eps=relative_eps,
        )
        summary["coarse_scale_idx"] = int(pair_idx + 1)
        qualitative = select_conditioned_qualitative_examples(
            generated_fields,
            condition_fields,
            filtered_fields=recoarsened_fields,
            relative_eps=relative_eps,
        )
        qualitative["transfer_metadata"] = {
            "source_H": float(h_fine),
            "target_H": float(h_coarse),
            "operator_name": (
                "tran_periodic_tikhonov_transfer"
                if float(transfer_ridge_lambda) > 0.0
                else "tran_periodic_spectral_transfer"
            ),
            "ridge_lambda": float(transfer_ridge_lambda),
        }
        qualitative["test_sample_indices"] = test_sample_indices[
            np.asarray(qualitative["condition_indices"], dtype=np.int64)
        ].astype(np.int64)
        summary["pair_metadata"] = {
            "pair_label": pair_label,
            "display_label": display_label,
            "tidx_coarse": int(tidx_coarse),
            "tidx_fine": int(tidx_fine),
            "H_coarse": float(h_coarse),
            "H_fine": float(h_fine),
            "modeled_marginal_coarse_order": int(pair_idx + 2),
            "modeled_marginal_fine_order": int(pair_idx + 1),
            "modeled_n_marginals": int(t_count),
            "provider": runtime.provider,
            "transfer_operator": (
                "tran_periodic_tikhonov_transfer"
                if float(transfer_ridge_lambda) > 0.0
                else "tran_periodic_spectral_transfer"
            ),
            "ridge_lambda": float(transfer_ridge_lambda),
        }
        intervals[pair_label] = summary
        qualitative_examples[pair_label] = qualitative

    return {
        "n_conditions": int(test_sample_indices.shape[0]),
        "n_realizations_per_condition": int(n_realizations),
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "intervals": intervals,
        "qualitative_examples": qualitative_examples,
    }


def evaluate_conditioned_global_coarse_return_for_runtime(
    *,
    runtime: CoarseConsistencyRuntime,
    test_fields_by_tidx: dict[int, np.ndarray],
    ladder: FilterLadder,
    full_h_schedule: list[float],
    output_dir: Path,
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    relative_eps: float,
    transfer_ridge_lambda: float = 1e-8,
    condition_chunk_size: int | None = None,
    root_rollout_realizations_max: int | None = None,
    condition_set: dict[str, Any] | None = None,
    seed_policy: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned global metrics."
        )

    if condition_set is None:
        rng = np.random.default_rng(int(seed))
        n_test = int(runtime.latent_test.shape[1])
        test_sample_indices = rng.choice(n_test, size=min(int(n_conditions), n_test), replace=False)
        test_sample_indices.sort()
        condition_set = build_root_condition_batch(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=runtime.time_indices,
        )
    if seed_policy is None:
        seed_policy = build_seed_policy(int(seed))

    cache_payload = build_or_load_global_decoded_cache(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=(
            int(root_rollout_realizations_max)
            if root_rollout_realizations_max is not None
            else int(n_realizations)
        ),
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        active_n_conditions=int(n_conditions),
        active_n_realizations=int(n_realizations),
        condition_chunk_size=condition_chunk_size,
    )
    test_sample_indices = np.asarray(cache_payload["test_sample_indices"], dtype=np.int64)
    coarse_tidx = int(runtime.time_indices[-1])
    fine_tidx = int(runtime.time_indices[0])
    coarse_targets = np.asarray(cache_payload["coarse_targets"], dtype=np.float32)
    finest_fields = np.asarray(cache_payload["decoded_finest_fields"], dtype=np.float32)
    coarse_h = float(full_h_schedule[coarse_tidx])
    fine_h = float(full_h_schedule[fine_tidx])
    recoarsened = np.asarray(
        ladder.transfer_between_H(
            finest_fields.reshape(finest_fields.shape[0] * finest_fields.shape[1], -1),
            source_H=fine_h,
            target_H=coarse_h,
            ridge_lambda=float(transfer_ridge_lambda),
        ),
        dtype=np.float32,
    ).reshape(finest_fields.shape[0], finest_fields.shape[1], -1)

    summary = summarize_conditioned_residuals(
        recoarsened - coarse_targets[:, None, :],
        coarse_targets,
        relative_eps=relative_eps,
    )
    qualitative = select_conditioned_qualitative_examples(
        finest_fields,
        coarse_targets,
        filtered_fields=recoarsened,
        relative_eps=relative_eps,
    )
    qualitative["transfer_metadata"] = {
        "source_H": fine_h,
        "target_H": coarse_h,
        "operator_name": (
            "tran_periodic_tikhonov_transfer"
            if float(transfer_ridge_lambda) > 0.0
            else "tran_periodic_spectral_transfer"
        ),
        "ridge_lambda": float(transfer_ridge_lambda),
    }
    qualitative["test_sample_indices"] = test_sample_indices[
        np.asarray(qualitative["condition_indices"], dtype=np.int64)
    ].astype(np.int64)
    summary["n_conditions"] = int(test_sample_indices.shape[0])
    summary["n_realizations_per_condition"] = int(n_realizations)
    summary["test_sample_indices"] = test_sample_indices.astype(np.int64).tolist()
    summary["pair_metadata"] = {
        "display_label": "Conditioned global",
        "tidx_coarse": coarse_tidx,
        "tidx_fine": fine_tidx,
        "H_coarse": coarse_h,
        "H_fine": fine_h,
        "provider": runtime.provider,
        "transfer_operator": (
            "tran_periodic_tikhonov_transfer"
            if float(transfer_ridge_lambda) > 0.0
            else "tran_periodic_spectral_transfer"
        ),
        "ridge_lambda": float(transfer_ridge_lambda),
    }
    return {
        "summary": summary,
        "qualitative_examples": qualitative,
    }


def precompute_conditioned_interval_latent_cache_for_runtime(
    *,
    runtime: CoarseConsistencyRuntime,
    full_h_schedule: list[float],
    output_dir: Path,
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    condition_set: dict[str, Any] | None = None,
    seed_policy: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned interval metrics."
        )
    if condition_set is None:
        rng = np.random.default_rng(int(seed))
        n_test = int(runtime.latent_test.shape[1])
        test_sample_indices = rng.choice(n_test, size=min(int(n_conditions), n_test), replace=False)
        test_sample_indices.sort()
        condition_set = build_interval_condition_batch(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=runtime.time_indices,
            interval_positions=np.arange(max(0, int(runtime.latent_test.shape[0]) - 1), dtype=np.int64),
            pair_labels=[
                make_pair_label(
                    tidx_coarse=int(runtime.time_indices[pair_idx + 1]),
                    tidx_fine=int(runtime.time_indices[pair_idx]),
                    full_H_schedule=full_h_schedule,
                )[0]
                for pair_idx in range(int(runtime.latent_test.shape[0]) - 1)
            ],
        )
    if seed_policy is None:
        seed_policy = build_seed_policy(int(seed))
    return build_or_load_interval_latent_cache(
        runtime=runtime,
        full_h_schedule=full_h_schedule,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
    )


def precompute_conditioned_global_latent_cache_for_runtime(
    *,
    runtime: CoarseConsistencyRuntime,
    output_dir: Path,
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
    root_rollout_realizations_max: int | None = None,
    condition_set: dict[str, Any] | None = None,
    seed_policy: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned global metrics."
        )
    if condition_set is None:
        rng = np.random.default_rng(int(seed))
        n_test = int(runtime.latent_test.shape[1])
        test_sample_indices = rng.choice(n_test, size=min(int(n_conditions), n_test), replace=False)
        test_sample_indices.sort()
        condition_set = build_root_condition_batch(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=runtime.time_indices,
        )
    if seed_policy is None:
        seed_policy = build_seed_policy(int(seed))
    return build_or_load_global_latent_cache(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=(
            int(root_rollout_realizations_max)
            if root_rollout_realizations_max is not None
            else int(n_realizations)
        ),
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )


def precompute_conditioned_interval_decoded_cache_for_runtime(
    *,
    runtime: CoarseConsistencyRuntime,
    test_fields_by_tidx: dict[int, np.ndarray],
    full_h_schedule: list[float],
    output_dir: Path,
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    condition_set: dict[str, Any] | None = None,
    seed_policy: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned interval metrics."
        )
    if condition_set is None:
        rng = np.random.default_rng(int(seed))
        n_test = int(runtime.latent_test.shape[1])
        test_sample_indices = rng.choice(n_test, size=min(int(n_conditions), n_test), replace=False)
        test_sample_indices.sort()
        condition_set = build_interval_condition_batch(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=runtime.time_indices,
            interval_positions=np.arange(max(0, int(runtime.latent_test.shape[0]) - 1), dtype=np.int64),
            pair_labels=[
                make_pair_label(
                    tidx_coarse=int(runtime.time_indices[pair_idx + 1]),
                    tidx_fine=int(runtime.time_indices[pair_idx]),
                    full_H_schedule=full_h_schedule,
                )[0]
                for pair_idx in range(int(runtime.latent_test.shape[0]) - 1)
            ],
        )
    if seed_policy is None:
        seed_policy = build_seed_policy(int(seed))
    return build_or_load_interval_decoded_cache(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        full_h_schedule=full_h_schedule,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
    )


def precompute_conditioned_global_decoded_cache_for_runtime(
    *,
    runtime: CoarseConsistencyRuntime,
    test_fields_by_tidx: dict[int, np.ndarray],
    output_dir: Path,
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
    root_rollout_realizations_max: int | None = None,
    condition_set: dict[str, Any] | None = None,
    seed_policy: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned global metrics."
        )
    if condition_set is None:
        rng = np.random.default_rng(int(seed))
        n_test = int(runtime.latent_test.shape[1])
        test_sample_indices = rng.choice(n_test, size=min(int(n_conditions), n_test), replace=False)
        test_sample_indices.sort()
        condition_set = build_root_condition_batch(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=runtime.time_indices,
        )
    if seed_policy is None:
        seed_policy = build_seed_policy(int(seed))
    return build_or_load_global_decoded_cache(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=(
            int(root_rollout_realizations_max)
            if root_rollout_realizations_max is not None
            else int(n_realizations)
        ),
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        active_n_conditions=int(n_conditions),
        active_n_realizations=int(n_realizations),
        condition_chunk_size=condition_chunk_size,
    )
