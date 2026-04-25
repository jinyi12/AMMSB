#!/usr/bin/env python
"""Pairwise latent-geometry comparison for maintained FAE runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmsfm.fae.fae_latent_utils import (  # noqa: E402
    build_fae_from_checkpoint,
    load_fae_checkpoint,
)
from mmsfm.fae.multiscale_dataset_naive import load_training_time_data_naive  # noqa: E402
from scripts.fae.tran_evaluation.latent_geometry import (  # noqa: E402
    LatentGeometryConfig,
    evaluate_latent_geometry,
)
from scripts.fae.tran_evaluation.latent_geometry_model_plots import (  # noqa: E402
    plot_pair_metric_bars,
    plot_pair_time_deltas,
    remove_legacy_pairwise_outputs,
)
from scripts.fae.tran_evaluation.latent_geometry_model_selection import (  # noqa: E402
    RUN_ROLE_BASELINE,
    RUN_ROLE_TREATMENT,
    RegistryRun,
    canonical_pair_runs,
    order_pair_runs,
    registry_run_from_explicit_dir,
)
from scripts.fae.tran_evaluation.latent_geometry_model_summary import (  # noqa: E402
    _compute_pairwise_deltas,
    _summarise_run,
    _summary_sort_key,
    _write_pair_delta_table,
    _write_pair_summary_files,
)
from scripts.fae.tran_evaluation.run_support import (  # noqa: E402
    load_json_dict,
    resolve_data_path_from_args_json,
    resolve_held_out_indices,
    resolve_run_checkpoint,
)
from scripts.images.field_visualization import format_for_paper  # noqa: E402


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() else "_")
    cleaned = "".join(keep).strip("_").lower()
    return cleaned or "run"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latent-geometry robustness for a maintained baseline/treatment FAE pair.",
    )
    parser.add_argument(
        "--run_dir",
        dest="run_dirs",
        action="append",
        default=[],
        help="Expert override: provide exactly two explicit FAE run directories (baseline first, treatment second).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/latent_geometry_transformer_pair",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute latent geometry even if cached per-run results exist.",
    )
    parser.add_argument(
        "--check_existing_metrics",
        action="store_true",
        default=True,
        help="Check for existing per-run latent_geometry_metrics.json artifacts before recomputing.",
    )
    parser.add_argument(
        "--no_check_existing_metrics",
        action="store_false",
        dest="check_existing_metrics",
        help="Disable reuse of existing latent geometry metric artifacts.",
    )
    parser.add_argument(
        "--latent_geom_budget",
        type=str,
        default="standard",
        choices=["light", "standard", "thorough"],
    )
    parser.add_argument("--latent_geom_n_samples", type=int, default=None)
    parser.add_argument("--latent_geom_n_probes", type=int, default=None)
    parser.add_argument("--latent_geom_n_slq_probes", type=int, default=None)
    parser.add_argument("--latent_geom_n_lanczos_steps", type=int, default=None)
    parser.add_argument("--latent_geom_n_hvp_probes", type=int, default=None)
    parser.add_argument("--latent_geom_eps", type=float, default=1e-6)
    parser.add_argument("--latent_geom_near_null_tau", type=float, default=1e-4)
    parser.add_argument(
        "--latent_geom_trace_estimator",
        type=str,
        default="fhutch",
        choices=["fhutch", "hutchpp"],
    )
    parser.add_argument(
        "--latent_geom_split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
    )
    parser.add_argument(
        "--latent_geom_max_samples_per_time",
        type=int,
        default=128,
        help="Max encoded fields per time marginal (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_existing_metrics(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _has_valid_model_metadata(payload: dict[str, Any]) -> bool:
    meta = payload.get("model_metadata", {})
    if not isinstance(meta, dict):
        return False
    required = ("decoder_type", "optimizer", "loss_type", "scale")
    if not all(meta.get(key) not in (None, "") for key in required):
        return False
    return meta.get("regularizer") not in (None, "") or meta.get("prior_flag") not in (None, "")


def _has_current_metric_schema(payload: dict[str, Any]) -> bool:
    gs = payload.get("global_summary", {})
    if isinstance(gs, dict) and "rho_vol_mean_over_time" in gs:
        return True
    per_time = payload.get("per_time", [])
    if isinstance(per_time, list) and per_time:
        first = per_time[0]
        if isinstance(first, dict) and "rho_vol_mean" in first:
            return True
    return False


def _existing_metrics_score(path: Path, payload: dict[str, Any]) -> tuple[int, int, int]:
    score = 0
    if _has_valid_model_metadata(payload):
        score += 100
    if _has_current_metric_schema(payload):
        score += 200
    path_text = str(path)
    if "latent_geometry_transformer_pair" in path_text:
        score += 40
    if "/per_run/" in path_text:
        score += 20
    if "/tran_evaluation/" in path_text:
        score -= 10
    mtime = int(path.stat().st_mtime) if path.exists() else 0
    return score, mtime, -len(path_text)


def _find_existing_metrics_for_run(run_dir: Path) -> Optional[tuple[Path, dict[str, Any]]]:
    direct_candidates = [
        run_dir / "tran_evaluation" / "latent_geometry_metrics.json",
        run_dir / "latent_geometry_eval" / "latent_geometry_metrics.json",
        run_dir / "latent_geometry_metrics.json",
    ]
    matches: list[tuple[tuple[int, int, int], Path, dict[str, Any]]] = []
    for candidate in direct_candidates:
        payload = _load_existing_metrics(candidate)
        if payload is not None and _has_current_metric_schema(payload):
            matches.append((_existing_metrics_score(candidate, payload), candidate, payload))

    results_root = REPO_ROOT / "results"
    if results_root.exists():
        candidate_patterns = (
            "latent_geometry_transformer_pair*/per_run/*/latent_geometry_metrics.json",
            "latent_geometry_*/per_run/*/latent_geometry_metrics.json",
        )
        seen: set[Path] = set()
        for pattern in candidate_patterns:
            for candidate in results_root.glob(pattern):
                if candidate in seen:
                    continue
                seen.add(candidate)
                payload = _load_existing_metrics(candidate)
                if payload is None or not _has_current_metric_schema(payload):
                    continue
                payload_run_dir = str(payload.get("run_dir", "")).strip()
                if payload_run_dir and Path(payload_run_dir).expanduser().resolve() == run_dir:
                    matches.append((_existing_metrics_score(candidate, payload), candidate, payload))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    _score, best_path, best_payload = matches[0]
    return best_path, best_payload


def _normalise_results_for_run(results: dict[str, Any], run: RegistryRun) -> dict[str, Any]:
    payload = dict(results)
    payload["run_dir"] = str(Path(run.best_run_dir).resolve())
    payload["model_metadata"] = asdict(run)
    geom_meta = dict(payload.get("latent_geometry_metadata", {}))
    geom_meta["run_role"] = str(run.run_role)
    geom_meta["decoder_type"] = str(geom_meta.get("decoder_type", "") or run.decoder_type)
    payload["latent_geometry_metadata"] = geom_meta
    return payload


def _load_time_data(
    run_dir: Path,
    *,
    split: str,
    max_samples_per_time: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    args_json = load_json_dict(run_dir / "args.json")
    data_path = resolve_data_path_from_args_json(
        args_json,
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        roots=[Path.cwd()],
    )
    train_ratio_raw = args_json.get("train_ratio", 0.8)
    train_ratio = 0.8 if train_ratio_raw is None else float(train_ratio_raw)

    held_out_indices = resolve_held_out_indices(
        data_path=data_path,
        raw_indices=args_json.get("held_out_indices", ""),
        raw_times=args_json.get("held_out_times", ""),
    )

    time_data = load_training_time_data_naive(
        str(data_path),
        held_out_indices=held_out_indices,
        train_ratio=train_ratio,
        split=split,  # type: ignore[arg-type]
        max_samples=max_samples_per_time,
        seed=seed,
    )
    if not time_data:
        raise ValueError(f"No time marginals available for {run_dir}")

    coords = np.asarray(time_data[0]["x"], dtype=np.float32)
    n_common = min(int(d["u"].shape[0]) for d in time_data)
    if n_common < 1:
        raise ValueError(f"No usable samples for {run_dir}")

    fields_per_time = np.stack(
        [np.asarray(d["u"][:n_common], dtype=np.float32) for d in time_data],
        axis=0,
    )
    dataset_time_indices = np.asarray([int(d["idx"]) for d in time_data], dtype=np.int64)
    return fields_per_time, coords, dataset_time_indices


def _compute_run_latent_geometry(
    run: RegistryRun,
    *,
    cfg: LatentGeometryConfig,
    split: str,
    max_samples_per_time: int,
    seed: int,
) -> dict[str, Any]:
    run_dir = Path(run.best_run_dir)
    checkpoint = resolve_run_checkpoint(run_dir, repo_root=REPO_ROOT, roots=[Path.cwd()])
    fields_per_time, coords, dataset_time_indices = _load_time_data(
        run_dir,
        split=split,
        max_samples_per_time=max_samples_per_time,
        seed=seed,
    )

    ckpt = load_fae_checkpoint(checkpoint)
    autoencoder, params, batch_stats, _meta = build_fae_from_checkpoint(ckpt)
    results = evaluate_latent_geometry(
        autoencoder, params, batch_stats,
        fields_per_time, coords,
        config=cfg,
    )
    results["run_dir"] = str(run_dir)
    results["time_indices"] = dataset_time_indices.tolist()
    results["latent_geom_split"] = split
    results["n_fields_per_time"] = int(fields_per_time.shape[1])
    results["model_metadata"] = asdict(run)
    geom_meta = dict(results.get("latent_geometry_metadata", {}))
    geom_meta["run_role"] = str(run.run_role)
    results["latent_geometry_metadata"] = geom_meta
    return results


def _select_pair_runs(args: argparse.Namespace) -> tuple[RegistryRun, RegistryRun, list[Path]]:
    explicit_run_dirs = [Path(run_raw).resolve() for run_raw in args.run_dirs if str(run_raw).strip()]
    if explicit_run_dirs:
        if len(explicit_run_dirs) != 2:
            raise RuntimeError("Explicit pairwise mode requires exactly two --run_dir arguments.")
        selected = [
            registry_run_from_explicit_dir(
                run_dir,
                repo_root=REPO_ROOT,
                roots=[Path.cwd()],
                run_role=RUN_ROLE_BASELINE if idx == 0 else RUN_ROLE_TREATMENT,
                run_label="Baseline" if idx == 0 else "Treatment",
            )
            for idx, run_dir in enumerate(explicit_run_dirs)
        ]
        baseline, treatment = order_pair_runs(selected)
        return baseline, treatment, explicit_run_dirs

    baseline, treatment = canonical_pair_runs(
        repo_root=REPO_ROOT,
        roots=[Path.cwd()],
    )
    return baseline, treatment, explicit_run_dirs


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_pairwise_outputs(out_dir)

    baseline_run, treatment_run, explicit_run_dirs = _select_pair_runs(args)
    selected = [baseline_run, treatment_run]

    format_for_paper()
    cfg = LatentGeometryConfig.from_preset(
        args.latent_geom_budget,
        seed=args.seed,
    ).with_overrides(
        n_samples=args.latent_geom_n_samples,
        n_probes=args.latent_geom_n_probes,
        n_slq_probes=args.latent_geom_n_slq_probes,
        n_lanczos_steps=args.latent_geom_n_lanczos_steps,
        n_hvp_probes=args.latent_geom_n_hvp_probes,
        eps=args.latent_geom_eps,
        near_null_tau=args.latent_geom_near_null_tau,
        trace_estimator=args.latent_geom_trace_estimator,
    )

    result_by_role: dict[str, dict[str, Any]] = {}
    summary_by_role: dict[str, dict[str, Any]] = {}

    for i, run in enumerate(selected, start=1):
        run_name = _safe_name(run.matrix_cell_id or run.best_run_dir)
        run_cache_dir = per_run_dir / run_name
        run_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = run_cache_dir / "latent_geometry_metrics.json"

        print(f"[{i}/{len(selected)}] {run.run_role}: {run.best_run_dir}")
        cache_payload = _load_existing_metrics(cache_path)
        if (
            cache_payload is not None
            and _has_current_metric_schema(cache_payload)
            and not args.force_recompute
            and (not args.check_existing_metrics or _has_valid_model_metadata(cache_payload))
        ):
            results = cache_payload
        elif args.check_existing_metrics and not args.force_recompute:
            existing = _find_existing_metrics_for_run(Path(run.best_run_dir))
            if existing is not None:
                existing_path, results = existing
                print(f"  Reusing existing latent geometry metrics from {existing_path}")
                results = _normalise_results_for_run(results, run)
                cache_path.write_text(json.dumps(results, indent=2))
            else:
                results = _compute_run_latent_geometry(
                    run,
                    cfg=cfg,
                    split=args.latent_geom_split,
                    max_samples_per_time=args.latent_geom_max_samples_per_time,
                    seed=args.seed,
                )
                results = _normalise_results_for_run(results, run)
                cache_path.write_text(json.dumps(results, indent=2))
        else:
            results = _compute_run_latent_geometry(
                run,
                cfg=cfg,
                split=args.latent_geom_split,
                max_samples_per_time=args.latent_geom_max_samples_per_time,
                seed=args.seed,
            )
            results = _normalise_results_for_run(results, run)
            cache_path.write_text(json.dumps(results, indent=2))

        results = _normalise_results_for_run(results, run)
        summary = _summarise_run(results)
        result_by_role[str(run.run_role)] = results
        summary_by_role[str(run.run_role)] = summary

    summaries = sorted(summary_by_role.values(), key=_summary_sort_key)
    pairwise = _compute_pairwise_deltas(
        result_by_role[RUN_ROLE_BASELINE],
        result_by_role[RUN_ROLE_TREATMENT],
        baseline_summary=summary_by_role[RUN_ROLE_BASELINE],
        treatment_summary=summary_by_role[RUN_ROLE_TREATMENT],
    )

    selection = {
        "selection_mode": "explicit_run_dirs" if explicit_run_dirs else "canonical_transformer_pair",
        "explicit_run_dirs": [str(path) for path in explicit_run_dirs],
        "check_existing_metrics": bool(args.check_existing_metrics),
        "baseline_run_dir": str(Path(baseline_run.best_run_dir).resolve()),
        "treatment_run_dir": str(Path(treatment_run.best_run_dir).resolve()),
    }

    _write_pair_summary_files(
        summaries,
        pairwise=pairwise,
        out_dir=out_dir,
        selection=selection,
    )
    _write_pair_delta_table(pairwise, out_dir=out_dir)
    plot_pair_metric_bars(summaries, out_dir)
    plot_pair_time_deltas(pairwise, out_dir=out_dir)

    print(f"\nSaved pairwise latent-geometry artifacts to {out_dir}")
    print(f"- {out_dir / 'latent_geom_pair_summary.json'}")
    print(f"- {out_dir / 'latent_geom_pair_summary.csv'}")
    print(f"- {out_dir / 'latent_geom_pair_delta_table.md'} / .csv")
    print(f"- {out_dir / 'latent_geom_pair_metric_<metric>.png'} / .pdf")
    print(f"- {out_dir / 'latent_geom_pair_time_delta_<metric>.png'} / .pdf")


if __name__ == "__main__":
    main()
