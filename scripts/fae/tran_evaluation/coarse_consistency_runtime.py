from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from data.transform_utils import apply_inverse_transform, load_transform_info
from mmsfm.fae.fae_latent_utils import (
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.fae.tran_evaluation.coarse_consistency import (
    evaluate_interval_coarse_consistency,
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
from scripts.fae.tran_evaluation.run_support import (
    parse_key_value_args_file as parse_args_file,
    resolve_existing_path,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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
    supports_conditioned_metrics: bool
    metadata: dict[str, Any]


def _make_decode_latents_to_fields(
    *,
    decode_fn: Any,
    grid_coords: np.ndarray,
    transform_info: dict[str, Any],
    decode_batch_size: int,
    clip_bounds: tuple[float, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    def decode_latents_to_fields(latents: np.ndarray) -> np.ndarray:
        z = np.asarray(latents, dtype=np.float32)
        if z.ndim < 2:
            raise ValueError(
                "Expected latents with shape (N, ...) and an explicit batch axis, "
                f"got {z.shape}."
            )
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
        rollout_batches = []
        for local_idx, test_idx in enumerate(np.asarray(test_sample_indices, dtype=np.int64)):
            z_start = torch.from_numpy(latent_test[-1, int(test_idx)][None, :]).float().to(device)
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
        supports_conditioned_metrics=True,
        metadata={
            "provider": "latent_msbm",
            "decode_mode": str(decode_mode),
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

    supports_conditioned = csp_runtime.model_type == "conditional_bridge" and csp_runtime.condition_mode is not None

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

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(csp_runtime.archive.latent_test[int(interval_idx) + 1, test_indices], dtype=np.float32)
        global_conditions = np.asarray(csp_runtime.archive.latent_test[-1, test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        repeated_global = np.repeat(global_conditions, int(n_realizations), axis=0)
        interval_zt = np.asarray(csp_runtime.archive.zt[int(interval_idx) : int(interval_idx) + 2], dtype=np.float32)
        interval_offset = int(csp_runtime.archive.num_intervals - 1 - int(interval_idx))
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
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This CSP runtime does not support conditioned full-rollout sampling.")

        import jax
        import jax.numpy as jnp
        from csp import sample_conditional_batch

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(csp_runtime.archive.latent_test[-1, test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
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
        supports_conditioned_metrics=supports_conditioned,
        metadata={
            "provider": "csp",
            "model_type": str(csp_runtime.model_type),
            "condition_mode": str(csp_runtime.condition_mode),
            "decode_mode": str(decode_mode),
        },
    )


def _load_token_csp_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    decode_mode: str,
    decode_batch_size: int,
) -> CoarseConsistencyRuntime:
    del decode_mode
    from scripts.csp.token_run_context import (
        load_token_csp_sampling_runtime,
        load_token_fae_decode_context,
    )

    csp_runtime = load_token_csp_sampling_runtime(
        run_dir,
        dataset_override=str(dataset_path),
    )
    if csp_runtime.source.fae_checkpoint_path is None:
        raise ValueError(
            "The token-native CSP run does not record a resolvable FAE checkpoint for conditioned coarse evaluation."
        )
    decode_context = load_token_fae_decode_context(
        dataset_path=Path(dataset_path),
        fae_checkpoint_path=Path(csp_runtime.source.fae_checkpoint_path),
    )
    split = csp_runtime.archive.split
    if not isinstance(split, dict):
        raise ValueError("The token CSP latent archive does not contain split metadata required for held-out evaluation.")

    supports_conditioned = (
        csp_runtime.model_type == "conditional_bridge_token_dit" and csp_runtime.condition_mode is not None
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

        import jax
        import jax.numpy as jnp
        from csp import sample_token_conditional_batch

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
        traj = sample_token_conditional_batch(
            csp_runtime.model,
            jnp.asarray(repeated_conditions, dtype=jnp.float32),
            jnp.asarray(interval_zt, dtype=jnp.float32),
            csp_runtime.sigma_fn,
            float(csp_runtime.dt0),
            jax.random.PRNGKey(int(seed)),
            condition_mode=str(csp_runtime.condition_mode),
            global_condition_batch=jnp.asarray(repeated_global, dtype=jnp.float32),
            interval_offset=interval_offset,
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
        del drift_clip_norm
        if not supports_conditioned:
            raise ValueError("This token-native CSP runtime does not support conditioned full-rollout sampling.")

        import jax
        import jax.numpy as jnp
        from csp import sample_token_conditional_batch

        test_indices = np.asarray(test_sample_indices, dtype=np.int64)
        coarse_conditions = np.asarray(csp_runtime.archive.latent_test[-1, test_indices], dtype=np.float32)
        repeated_conditions = np.repeat(coarse_conditions, int(n_realizations), axis=0)
        traj = sample_token_conditional_batch(
            csp_runtime.model,
            jnp.asarray(repeated_conditions, dtype=jnp.float32),
            jnp.asarray(csp_runtime.archive.zt, dtype=jnp.float32),
            csp_runtime.sigma_fn,
            float(csp_runtime.dt0),
            jax.random.PRNGKey(int(seed)),
            condition_mode=str(csp_runtime.condition_mode),
            global_condition_batch=jnp.asarray(repeated_conditions, dtype=jnp.float32),
            interval_offset=0,
        )
        rollout = np.asarray(traj, dtype=np.float32)
        return rollout.reshape(
            coarse_conditions.shape[0],
            int(n_realizations),
            *rollout.shape[1:],
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
            decode_fn=decode_context.decode_fn,
            grid_coords=np.asarray(decode_context.grid_coords, dtype=np.float32),
            transform_info=decode_context.transform_info,
            decode_batch_size=decode_batch_size,
            clip_bounds=decode_context.clip_bounds,
        ),
        sample_interval_latents=sample_interval_latents,
        sample_full_rollout_knots=sample_full_rollout_knots,
        supports_conditioned_metrics=supports_conditioned,
        metadata={
            "provider": "csp_token_dit",
            "model_type": str(csp_runtime.model_type),
            "condition_mode": str(csp_runtime.condition_mode),
            "decode_mode": "token_native",
        },
    )


def load_coarse_consistency_runtime(
    *,
    run_dir: Path,
    dataset_path: Path,
    device: str | torch.device,
    decode_mode: str,
    decode_batch_size: int,
    use_ema: bool,
) -> CoarseConsistencyRuntime:
    run_dir_resolved = Path(run_dir).expanduser().resolve()
    dataset_path_resolved = Path(dataset_path).expanduser().resolve()
    if (run_dir_resolved / "config" / "args.json").exists():
        cfg = json.loads((run_dir_resolved / "config" / "args.json").read_text())
        if str(cfg.get("model_type", "conditional_bridge")) == "conditional_bridge_token_dit":
            return _load_token_csp_runtime(
                run_dir=run_dir_resolved,
                dataset_path=dataset_path_resolved,
                decode_mode=decode_mode,
                decode_batch_size=decode_batch_size,
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


def _select_condition_indices(
    latent_test: np.ndarray,
    *,
    n_conditions: int,
    seed: int,
) -> np.ndarray:
    n_test = int(latent_test.shape[1])
    rng = np.random.default_rng(int(seed))
    test_sample_indices = rng.choice(n_test, size=min(int(n_conditions), n_test), replace=False)
    test_sample_indices.sort()
    return test_sample_indices.astype(np.int64)


def evaluate_conditioned_interval_coarse_consistency_for_runtime(
    *,
    runtime: CoarseConsistencyRuntime,
    test_fields_by_tidx: dict[int, np.ndarray],
    ladder: FilterLadder,
    full_h_schedule: list[float],
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    relative_eps: float,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned interval metrics."
        )

    test_sample_indices = _select_condition_indices(runtime.latent_test, n_conditions=n_conditions, seed=seed)
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

        condition_fields = np.asarray(test_fields_by_tidx[tidx_coarse][test_sample_indices], dtype=np.float32)
        condition_fields = condition_fields.reshape(condition_fields.shape[0], -1)
        generated_latents = runtime.sample_interval_latents(
            test_sample_indices,
            pair_idx,
            int(n_realizations),
            int(seed),
            drift_clip_norm,
        )
        generated_fields_flat = runtime.decode_latents_to_fields(
            generated_latents.reshape(
                generated_latents.shape[0] * generated_latents.shape[1],
                *generated_latents.shape[2:],
            )
        )
        generated_fields = generated_fields_flat.reshape(
            generated_latents.shape[0],
            generated_latents.shape[1],
            *generated_fields_flat.shape[1:],
        )
        summary = evaluate_interval_coarse_consistency(
            generated_fields,
            condition_fields,
            ladder=ladder,
            coarse_scale_idx=pair_idx + 1,
            relative_eps=relative_eps,
        )
        qualitative = select_conditioned_qualitative_examples(
            generated_fields,
            condition_fields,
            ladder=ladder,
            coarse_scale_idx=pair_idx + 1,
            relative_eps=relative_eps,
        )
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
    n_conditions: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None,
    relative_eps: float,
) -> dict[str, Any]:
    if not runtime.supports_conditioned_metrics:
        raise ValueError(
            f"Runtime provider {runtime.provider!r} does not support conditioned global metrics."
        )

    test_sample_indices = _select_condition_indices(runtime.latent_test, n_conditions=n_conditions, seed=seed)
    coarse_tidx = int(runtime.time_indices[-1])
    fine_tidx = int(runtime.time_indices[0])
    coarse_targets = np.asarray(test_fields_by_tidx[coarse_tidx][test_sample_indices], dtype=np.float32)
    coarse_targets = coarse_targets.reshape(coarse_targets.shape[0], -1)

    rollout_knots = runtime.sample_full_rollout_knots(
        test_sample_indices,
        int(n_realizations),
        int(seed),
        drift_clip_norm,
    )
    finest_latents = rollout_knots[:, :, 0, ...].reshape(
        rollout_knots.shape[0] * rollout_knots.shape[1],
        *rollout_knots.shape[3:],
    )
    finest_fields = runtime.decode_latents_to_fields(finest_latents)
    recoarsened = ladder.filter_at_scale(
        finest_fields.reshape(finest_fields.shape[0], -1),
        int(len(runtime.time_indices) - 1),
    ).reshape(rollout_knots.shape[0], rollout_knots.shape[1], -1)

    summary = summarize_conditioned_residuals(
        recoarsened - coarse_targets[:, None, :],
        coarse_targets,
        relative_eps=relative_eps,
    )
    qualitative = select_conditioned_qualitative_examples(
        finest_fields.reshape(rollout_knots.shape[0], rollout_knots.shape[1], -1),
        coarse_targets,
        filtered_fields=recoarsened,
        relative_eps=relative_eps,
    )
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
        "H_coarse": float(full_h_schedule[coarse_tidx]),
        "H_fine": float(full_h_schedule[fine_tidx]),
        "provider": runtime.provider,
    }
    return {
        "summary": summary,
        "qualitative_examples": qualitative,
    }
