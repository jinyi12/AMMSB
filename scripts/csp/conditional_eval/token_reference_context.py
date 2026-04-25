from __future__ import annotations

"""Scientific context assembly for token-native kNN conditional evaluation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.csp.token_conditional_phases import DEFAULT_TOKEN_CONDITIONAL_SAMPLING_MAX_BATCH_SIZE
from scripts.fae.tran_evaluation.conditional_support import (
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    make_pair_label,
)


def default_corpus_dataset_path(dataset_path: Path) -> Path:
    if dataset_path.stem.endswith("_corpus"):
        return dataset_path
    return dataset_path.with_name(f"{dataset_path.stem}_corpus{dataset_path.suffix}")


def load_token_corpus_latents(
    *,
    corpus_latents_path: Path,
    time_indices: np.ndarray,
    run_dir: Path,
    dataset_path: Path | None,
    load_corpus_latents_fn,
) -> tuple[dict[int, np.ndarray], int]:
    try:
        return load_corpus_latents_fn(corpus_latents_path, time_indices)
    except KeyError as exc:
        requested = [int(tidx) for tidx in np.asarray(time_indices, dtype=np.int64).reshape(-1).tolist()]
        message = (
            "Conditional token-native evaluation requires a corpus latent archive whose "
            f"time_indices match the run contract. Requested time_indices={requested}, "
            f"but {corpus_latents_path} is incompatible. {exc}"
        )
        if dataset_path is not None:
            suggested_corpus_path = default_corpus_dataset_path(dataset_path)
            suggested_output_path = run_dir / "corpus_latents.npz"
            message += (
                "\nRe-encode a compatible archive, for example:\n"
                f"python scripts/fae/tran_evaluation/encode_corpus.py "
                f"--corpus_path {suggested_corpus_path} "
                f"--run_dir {run_dir} "
                f"--output_path {suggested_output_path}"
            )
        raise ValueError(message) from exc


def validate_token_corpus_latents(
    corpus_latents_by_tidx: dict[int, np.ndarray],
    *,
    time_indices: np.ndarray,
    token_shape: tuple[int, int],
    expected_dim: int,
    corpus_latents_path: Path,
    run_dir: Path,
    dataset_path: Path | None,
) -> None:
    for tidx in np.asarray(time_indices, dtype=np.int64).reshape(-1):
        arr = np.asarray(corpus_latents_by_tidx[int(tidx)], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                "Token-native conditional evaluation expects flattened corpus latents with shape "
                f"(N, {expected_dim}), but latents_{int(tidx)} in {corpus_latents_path} has shape {arr.shape}."
            )
        if int(arr.shape[-1]) == expected_dim:
            continue

        suggestion = "Re-encode the corpus against this run contract."
        if dataset_path is not None:
            suggested_corpus_path = default_corpus_dataset_path(dataset_path)
            suggested_output_path = run_dir / "corpus_latents.npz"
            suggestion = (
                "Re-encode the corpus against this run contract, for example:\n"
                f"python scripts/fae/tran_evaluation/encode_corpus.py "
                f"--corpus_path {suggested_corpus_path} "
                f"--run_dir {run_dir} "
                f"--output_path {suggested_output_path}"
            )
        raise ValueError(
            "Token-native conditional evaluation expects flattened transformer corpus latents with feature "
            f"dimension {expected_dim} for token_shape={token_shape}, but {corpus_latents_path} stores "
            f"latents_{int(tidx)} with shape {arr.shape}. This usually means you passed a legacy vector-latent "
            f"corpus archive. {suggestion}"
        )


def subsample_token_corpus_latents(
    corpus_latents_by_tidx: dict[int, np.ndarray],
    *,
    time_indices: np.ndarray,
    max_corpus_samples: int | None,
    seed: int,
) -> tuple[dict[int, np.ndarray], int, np.ndarray | None]:
    first_tidx = int(np.asarray(time_indices, dtype=np.int64).reshape(-1)[0])
    n_corpus = int(np.asarray(corpus_latents_by_tidx[first_tidx]).shape[0])
    if max_corpus_samples is None or int(max_corpus_samples) <= 0 or n_corpus <= int(max_corpus_samples):
        return corpus_latents_by_tidx, n_corpus, None

    rng = np.random.default_rng(int(seed))
    keep_idx = np.sort(rng.choice(n_corpus, size=int(max_corpus_samples), replace=False).astype(np.int64))
    reduced = {
        int(tidx): np.asarray(corpus_latents_by_tidx[int(tidx)][keep_idx], dtype=np.float32)
        for tidx in np.asarray(time_indices, dtype=np.int64).reshape(-1)
    }
    return reduced, int(keep_idx.shape[0]), keep_idx


def prepare_token_reference_context(
    args,
    *,
    repo_root: Path,
    load_token_csp_sampling_runtime_fn,
    load_corpus_latents_fn,
) -> SimpleNamespace:
    run_dir = Path(args.run_dir).expanduser().resolve()
    runtime = load_token_csp_sampling_runtime_fn(
        run_dir,
        latents_override=args.latents_path,
        fae_checkpoint_override=args.fae_checkpoint,
    )
    latents_path = runtime.source.latents_path
    latent_test_tokens = runtime.archive.latent_test
    latent_test_flat = latent_test_tokens.reshape(latent_test_tokens.shape[0], latent_test_tokens.shape[1], -1)
    zt = runtime.archive.zt
    time_indices = runtime.archive.time_indices
    tau_knots = runtime.tau_knots
    token_shape = runtime.archive.token_shape
    t_count, n_test, _latent_dim = latent_test_flat.shape
    if n_test <= 0:
        raise ValueError("No test latents available for token-native CSP conditional evaluation.")

    corpus_latents_path = Path(args.corpus_latents_path).expanduser()
    if not corpus_latents_path.is_absolute():
        corpus_latents_path = (Path(repo_root) / corpus_latents_path).resolve()
    corpus_latents_by_tidx, n_corpus_original = load_token_corpus_latents(
        corpus_latents_path=corpus_latents_path,
        time_indices=time_indices,
        run_dir=run_dir,
        dataset_path=getattr(runtime.source, "dataset_path", None),
        load_corpus_latents_fn=load_corpus_latents_fn,
    )
    validate_token_corpus_latents(
        corpus_latents_by_tidx,
        time_indices=time_indices,
        token_shape=token_shape,
        expected_dim=int(token_shape[0] * token_shape[1]),
        corpus_latents_path=corpus_latents_path,
        run_dir=run_dir,
        dataset_path=getattr(runtime.source, "dataset_path", None),
    )
    corpus_latents_by_tidx, n_corpus, corpus_keep_indices = subsample_token_corpus_latents(
        corpus_latents_by_tidx,
        time_indices=time_indices,
        max_corpus_samples=getattr(args, "max_corpus_samples", None),
        seed=args.seed,
    )
    return SimpleNamespace(
        runtime=runtime,
        latents_path=latents_path,
        latent_test_tokens=latent_test_tokens,
        latent_test_flat=latent_test_flat,
        zt=zt,
        time_indices=time_indices,
        tau_knots=tau_knots,
        token_shape=token_shape,
        t_count=int(t_count),
        n_test=int(n_test),
        corpus_latents_path=corpus_latents_path,
        corpus_latents_by_tidx=corpus_latents_by_tidx,
        n_corpus=int(n_corpus),
        n_corpus_original=int(n_corpus_original),
        corpus_keep_indices=corpus_keep_indices,
        condition_mode=str(runtime.condition_mode),
    )


def resolve_saved_test_sample_indices(
    *,
    args,
    requested_phases: list[str],
    sample_metadata: dict[str, np.ndarray] | None,
    existing_results: dict[str, np.ndarray] | None,
    n_test: int,
    selection_seed: int,
    sample_phase_name: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(selection_seed))
    if sample_metadata is not None and "test_sample_indices" in sample_metadata:
        indices = np.asarray(sample_metadata["test_sample_indices"], dtype=np.int64).reshape(-1)
    elif sample_phase_name in requested_phases or existing_results is None or "test_sample_indices" not in existing_results:
        indices = rng.choice(n_test, size=min(args.n_test_samples, n_test), replace=False)
        indices.sort()
        return indices.astype(np.int64)
    else:
        indices = np.asarray(existing_results["test_sample_indices"], dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError(
            "Existing knn_reference_results.npz does not contain test_sample_indices. "
            "Run the reference-cache stage first: --phases reference_cache."
        )
    if np.any(indices < 0) or np.any(indices >= n_test):
        raise ValueError(f"Saved test_sample_indices are out of range for the current runtime: {indices.tolist()}.")
    return indices


def print_token_reference_header(
    *,
    args,
    run_dir: Path,
    output_dir: Path,
    requested_phases: list[str],
    context: SimpleNamespace,
) -> None:
    print("============================================================", flush=True)
    print("Token-native CSP knn_reference evaluation", flush=True)
    print(f"  run_dir            : {run_dir}", flush=True)
    print(f"  output_dir         : {output_dir}", flush=True)
    print(f"  model_type         : {context.runtime.model_type}", flush=True)
    print(f"  condition_mode     : {context.condition_mode}", flush=True)
    print(f"  conditional_eval_mode : {CHATTERJEE_CONDITIONAL_EVAL_MODE}", flush=True)
    print(f"  stages             : {', '.join(requested_phases) if requested_phases else 'none'}", flush=True)
    print(f"  token_shape        : {context.token_shape}", flush=True)
    print(f"  source_latents     : {context.latents_path}", flush=True)
    print(f"  corpus_latents     : {context.corpus_latents_path}", flush=True)
    print(f"  n_corpus           : {context.n_corpus}", flush=True)
    if context.corpus_keep_indices is not None:
        print(f"  n_corpus_original  : {context.n_corpus_original}", flush=True)
    print(f"  n_test_samples     : {min(args.n_test_samples, context.n_test)}", flush=True)
    print(f"  n_ecmmd_conditions : {min(args.n_test_samples, context.n_test)}", flush=True)
    print(f"  n_realizations     : {args.n_realizations}", flush=True)
    print(f"  k_neighbors        : {args.k_neighbors}", flush=True)
    print(
        "  sampling_max_batch_size : "
        f"{getattr(args, 'sampling_max_batch_size', None) or DEFAULT_TOKEN_CONDITIONAL_SAMPLING_MAX_BATCH_SIZE}",
        flush=True,
    )
    print(f"  skip_reports       : {bool(getattr(args, 'skip_ecmmd', False))}", flush=True)
    print(
        f"  n_plot_conditions  : {max(0, min(args.n_plot_conditions, min(args.n_test_samples, context.n_test)))}",
        flush=True,
    )
    print(f"  plot_value_budget  : {args.plot_value_budget}", flush=True)
    print("============================================================", flush=True)


def initialize_pairwise_result_store(result_store_keys: tuple[str, ...]) -> dict[str, Any]:
    store = {key: {} for key in result_store_keys}
    store["pair_labels"] = []
    return store


def store_pair_array_fields(
    store: dict[str, Any],
    *,
    pair_label: str,
    source: dict[str, object],
    fields: tuple[tuple[str, str, Any], ...],
) -> None:
    for store_key, source_key, dtype in fields:
        store[store_key][pair_label] = np.asarray(source[source_key], dtype=dtype)


def build_pair_metadata(
    *,
    pair_idx: int,
    context: SimpleNamespace,
    full_H_schedule: list[float],
) -> tuple[str, str, dict[str, object], int, int]:
    tidx_fine = int(context.time_indices[pair_idx])
    tidx_coarse = int(context.time_indices[pair_idx + 1])
    pair_label, h_coarse, h_fine, display_label = make_pair_label(
        tidx_coarse=tidx_coarse,
        tidx_fine=tidx_fine,
        full_H_schedule=full_H_schedule,
    )
    metadata = {
        "tidx_fine": int(tidx_fine),
        "tidx_coarse": int(tidx_coarse),
        "H_fine": float(h_fine),
        "H_coarse": float(h_coarse),
        "display_label": str(display_label),
        "modeled_marginal_coarse_order": int(pair_idx + 2),
        "modeled_marginal_fine_order": int(pair_idx + 1),
        "modeled_n_marginals": int(context.t_count),
    }
    return pair_label, str(display_label), metadata, tidx_coarse, tidx_fine
