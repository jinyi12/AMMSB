from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_NAME = "generated_consistency_summary.txt"
METRICS_NAME = "generated_consistency_metrics.json"
ARRAYS_NAME = "generated_consistency_arrays.npz"
MANIFEST_NAME = "generated_consistency_manifest.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def npz_safe_key(label: str) -> str:
    safe = str(label).replace(" ", "_").replace("=", "").replace(">", "to")
    safe = safe.replace("-", "_").replace("/", "_").replace(".", "p")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def load_generated_data_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        payload: dict[str, Any] = {}
        for key in (
            "realizations_phys",
            "realizations_log",
            "trajectory_fields_phys",
            "trajectory_fields_log",
            "trajectory_fields_phys_all",
            "trajectory_fields_log_all",
            "zt",
            "time_indices",
            "trajectory_all_time_indices",
            "sample_indices",
        ):
            if key in data:
                payload[key] = np.asarray(data[key])
        if "resolution" in data:
            payload["resolution"] = int(np.asarray(data["resolution"]).item())
        if "is_realizations" in data:
            payload["is_realizations"] = bool(np.asarray(data["is_realizations"]).item())
        if "decode_mode" in data:
            payload["decode_mode"] = str(np.asarray(data["decode_mode"]).item())
    return payload


def build_global_coarse_targets(
    *,
    gt_coarse_fields: np.ndarray,
    sample_indices: np.ndarray | None,
    default_sample_idx: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_indices is None:
        index = int(default_sample_idx)
        if not (0 <= index < int(gt_coarse_fields.shape[0])):
            raise IndexError(
                f"default_sample_idx={index} is out of range for coarse GT fields "
                f"with shape {gt_coarse_fields.shape}."
            )
        coarse_targets = np.repeat(gt_coarse_fields[index : index + 1], int(n_samples), axis=0)
        group_ids = np.zeros(int(n_samples), dtype=np.int64)
        return coarse_targets.astype(np.float32), group_ids

    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] != int(n_samples):
        raise ValueError(
            f"sample_indices length {indices.shape[0]} does not match n_samples={n_samples}."
        )
    if np.any(indices < 0) or np.any(indices >= int(gt_coarse_fields.shape[0])):
        raise IndexError(
            "sample_indices contain values outside the available coarse GT field range: "
            f"min={int(indices.min())}, max={int(indices.max())}, "
            f"n_fields={int(gt_coarse_fields.shape[0])}."
        )
    return gt_coarse_fields[indices].astype(np.float32), indices


def build_coarse_only_summary(coarse_results: dict[str, Any] | None) -> str:
    lines = ["Tran Evaluation Coarse Consistency Summary", "=" * 42]
    if not coarse_results:
        lines.append("No coarse-consistency results were produced.")
        return "\n".join(lines)

    conditioned_interval = coarse_results.get("conditioned_interval", {})
    if conditioned_interval:
        lines.append("Conditioned interval consistency:")
        for pair_key, item in conditioned_interval.items():
            meta = item.get("pair_metadata", {})
            display = meta.get("display_label", pair_key)
            total_rel = item.get("total_rel", {})
            bias_rel = item.get("bias_rel", {})
            spread_rel = item.get("spread_rel", {})
            lines.append(
                f"  {display}: "
                f"C_rel={float(total_rel.get('mean', float('nan'))):.6f}, "
                f"B_rel={float(bias_rel.get('mean', float('nan'))):.6f}, "
                f"S_rel={float(spread_rel.get('mean', float('nan'))):.6f}"
            )

    conditioned_global = coarse_results.get("conditioned_global_return")
    if conditioned_global is not None:
        lines.append(
            "Conditioned global return: "
            f"C_rel={float(conditioned_global['total_rel']['mean']):.6f}, "
            f"B_rel={float(conditioned_global['bias_rel']['mean']):.6f}, "
            f"S_rel={float(conditioned_global['spread_rel']['mean']):.6f}"
        )

    cache_global = coarse_results.get("cache_global_return")
    if cache_global is not None:
        lines.append(
            "Cache global return: "
            f"C_rel={float(cache_global['total_rel']['mean']):.6f}, "
            f"B_rel={float(cache_global['bias_rel']['mean']):.6f}, "
            f"S_rel={float(cache_global['spread_rel']['mean']):.6f}"
        )

    path_results = coarse_results.get("path_self_consistency")
    if path_results is not None:
        lines.append(
            "Path self-consistency: "
            f"mean_rel={float(path_results['mean_rel_across_intervals']):.6f}, "
            f"mean_stable_rel={float(path_results['mean_stable_relative_across_intervals']):.6f}"
        )

    return "\n".join(lines)


def build_coarse_curves_payload(coarse_results: dict[str, Any]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    conditioned_interval = coarse_results.get("conditioned_interval", {})
    for pair_key, item in conditioned_interval.items():
        key = npz_safe_key(pair_key)
        payload[f"cond_interval_total_rel_{key}"] = np.asarray(
            item["per_condition"]["total_rel"],
            dtype=np.float32,
        )
        payload[f"cond_interval_bias_rel_{key}"] = np.asarray(
            item["per_condition"]["bias_rel"],
            dtype=np.float32,
        )
        payload[f"cond_interval_spread_rel_{key}"] = np.asarray(
            item["per_condition"]["spread_rel"],
            dtype=np.float32,
        )

    conditioned_global = coarse_results.get("conditioned_global_return")
    if conditioned_global is not None:
        payload["cond_global_total_rel"] = np.asarray(conditioned_global["per_condition"]["total_rel"], dtype=np.float32)
        payload["cond_global_bias_rel"] = np.asarray(conditioned_global["per_condition"]["bias_rel"], dtype=np.float32)
        payload["cond_global_spread_rel"] = np.asarray(conditioned_global["per_condition"]["spread_rel"], dtype=np.float32)

    cache_global = coarse_results.get("cache_global_return")
    if cache_global is not None:
        payload["cache_global_total_rel"] = np.asarray(cache_global["per_condition"]["total_rel"], dtype=np.float32)
        payload["cache_global_bias_rel"] = np.asarray(cache_global["per_condition"]["bias_rel"], dtype=np.float32)
        payload["cache_global_spread_rel"] = np.asarray(cache_global["per_condition"]["spread_rel"], dtype=np.float32)

    path_results = coarse_results.get("path_self_consistency")
    if path_results is not None:
        for pair_idx, item in path_results["per_interval"].items():
            payload[f"path_self_rel_interval{pair_idx}"] = np.asarray(item["per_group_rel"], dtype=np.float32)
    return payload


def write_coarse_consistency_artifacts(
    *,
    output_dir: Path,
    summary_text: str,
    metrics_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    curves_payload: dict[str, np.ndarray] | None = None,
) -> dict[str, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(summary_text)

    metrics_path = output_dir / METRICS_NAME
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    arrays_path: Path | None = None
    if curves_payload:
        arrays_path = output_dir / ARRAYS_NAME
        np.savez_compressed(arrays_path, **curves_payload)

    manifest = dict(manifest_payload)
    manifest["summary_path"] = str(summary_path)
    manifest["metrics_path"] = str(metrics_path)
    manifest["arrays_path"] = None if arrays_path is None else str(arrays_path)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "arrays_path": arrays_path,
        "manifest_path": manifest_path,
    }


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, found {type(payload).__name__}.")
    return payload


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _infer_square_resolution(fields: np.ndarray) -> int:
    arr = np.asarray(fields)
    if arr.ndim >= 2 and int(arr.shape[-1]) == int(arr.shape[-2]):
        return int(arr.shape[-1])
    n_points = int(arr.shape[-1])
    resolution = int(round(float(n_points) ** 0.5))
    if resolution * resolution != n_points:
        raise ValueError(
            "Could not infer a square field resolution from saved cache payload with "
            f"terminal shape {arr.shape[-1:]}."
        )
    return resolution


def load_saved_coarse_metrics_payload(output_dir: Path) -> dict[str, Any]:
    metrics_path = Path(output_dir) / METRICS_NAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing saved coarse metrics artifact: {metrics_path}. "
            "Run the coarse evaluation first or omit --plot_only."
        )
    payload = _load_json_dict(metrics_path)
    config = payload.get("config")
    coarse_results = payload.get("coarse_consistency")
    if not isinstance(config, dict):
        raise ValueError(f"Saved metrics payload {metrics_path} is missing a dict-valued 'config' section.")
    if not isinstance(coarse_results, dict):
        raise ValueError(
            f"Saved metrics payload {metrics_path} is missing a dict-valued 'coarse_consistency' section."
        )
    return payload


def _rebuild_interval_qualitative_results(
    *,
    output_dir: Path,
    coarse_results: dict[str, Any],
    ladder,
    relative_eps: float,
) -> dict[str, Any]:
    from scripts.fae.tran_evaluation.coarse_consistency import select_conditioned_qualitative_examples

    cache_path = Path(output_dir) / "cache" / "conditioned_interval.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing saved conditioned-interval cache export: {cache_path}. "
            "The qualitative panel cannot be rebuilt without it."
        )
    cache_payload = _load_npz_payload(cache_path)
    pair_labels = [str(label) for label in np.asarray(cache_payload["pair_labels"]).reshape(-1).tolist()]
    decoded_fine_fields = np.asarray(cache_payload["decoded_fine_fields"], dtype=np.float32)
    coarse_targets = np.asarray(cache_payload["coarse_targets"], dtype=np.float32)
    test_sample_indices = np.asarray(cache_payload["test_sample_indices"], dtype=np.int64)
    interval_results = coarse_results.get("conditioned_interval", {})
    qualitative_results: dict[str, Any] = {}
    for pair_idx, pair_label in enumerate(pair_labels):
        pair_summary = interval_results.get(pair_label)
        if not isinstance(pair_summary, dict):
            continue
        generated_fields = np.asarray(decoded_fine_fields[pair_idx], dtype=np.float32)
        condition_fields = np.asarray(coarse_targets[pair_idx], dtype=np.float32)
        coarse_scale_idx = int(pair_summary.get("coarse_scale_idx", pair_idx + 1))
        pair_meta = pair_summary.get("pair_metadata", {})
        source_h = pair_meta.get("H_fine")
        target_h = pair_meta.get("H_coarse")
        ridge_lambda = float(pair_meta.get("ridge_lambda", 0.0))
        if source_h is not None and target_h is not None:
            recoarsened_fields = np.asarray(
                ladder.transfer_between_H(
                    generated_fields.reshape(generated_fields.shape[0] * generated_fields.shape[1], -1),
                    source_H=float(source_h),
                    target_H=float(target_h),
                    ridge_lambda=ridge_lambda,
                ),
                dtype=np.float32,
            ).reshape(generated_fields.shape[0], generated_fields.shape[1], -1)
        else:
            recoarsened_fields = np.asarray(
                ladder.filter_at_scale(
                    generated_fields.reshape(generated_fields.shape[0] * generated_fields.shape[1], -1),
                    coarse_scale_idx,
                ),
                dtype=np.float32,
            ).reshape(generated_fields.shape[0], generated_fields.shape[1], -1)
        qualitative = select_conditioned_qualitative_examples(
            generated_fields,
            condition_fields,
            filtered_fields=recoarsened_fields,
            relative_eps=float(relative_eps),
        )
        if source_h is not None and target_h is not None:
            qualitative["transfer_metadata"] = {
                "source_H": float(source_h),
                "target_H": float(target_h),
                "operator_name": str(pair_meta.get("transfer_operator", "tran_periodic_spectral_transfer")),
                "ridge_lambda": ridge_lambda,
            }
        qualitative["test_sample_indices"] = test_sample_indices[
            np.asarray(qualitative["condition_indices"], dtype=np.int64)
        ].astype(np.int64)
        qualitative_results[pair_label] = qualitative
    return qualitative_results


def _rebuild_global_qualitative_results(
    *,
    output_dir: Path,
    full_h_schedule: list[float],
    time_indices: list[int],
    coarse_results: dict[str, Any],
    ladder,
    relative_eps: float,
) -> dict[str, Any]:
    from scripts.fae.tran_evaluation.coarse_consistency import select_conditioned_qualitative_examples

    cache_path = Path(output_dir) / "cache" / "conditioned_global.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing saved conditioned-global cache export: {cache_path}. "
            "The qualitative panel cannot be rebuilt without it."
        )
    cache_payload = _load_npz_payload(cache_path)
    finest_fields = np.asarray(cache_payload["decoded_finest_fields"], dtype=np.float32)
    coarse_targets = np.asarray(cache_payload["coarse_targets"], dtype=np.float32)
    test_sample_indices = np.asarray(cache_payload["test_sample_indices"], dtype=np.int64)
    summary_meta = coarse_results.get("conditioned_global_return", {}).get("pair_metadata", {})
    if summary_meta.get("H_fine") is not None and summary_meta.get("H_coarse") is not None:
        source_h = float(summary_meta["H_fine"])
        target_h = float(summary_meta["H_coarse"])
        ridge_lambda = float(summary_meta.get("ridge_lambda", 0.0))
        recoarsened_fields = np.asarray(
            ladder.transfer_between_H(
                finest_fields.reshape(finest_fields.shape[0] * finest_fields.shape[1], -1),
                source_H=source_h,
                target_H=target_h,
                ridge_lambda=ridge_lambda,
            ),
            dtype=np.float32,
        ).reshape(finest_fields.shape[0], finest_fields.shape[1], -1)
    else:
        source_h = float(full_h_schedule[int(time_indices[0])])
        target_h = float(full_h_schedule[int(time_indices[-1])])
        ridge_lambda = 0.0
        recoarsened_fields = np.asarray(
            ladder.filter_at_H(
                finest_fields.reshape(finest_fields.shape[0] * finest_fields.shape[1], -1),
                H=target_h,
            ),
            dtype=np.float32,
        ).reshape(finest_fields.shape[0], finest_fields.shape[1], -1)
    qualitative = select_conditioned_qualitative_examples(
        finest_fields,
        coarse_targets,
        filtered_fields=recoarsened_fields,
        relative_eps=float(relative_eps),
    )
    qualitative["transfer_metadata"] = {
        "source_H": float(source_h),
        "target_H": float(target_h),
        "operator_name": str(summary_meta.get("transfer_operator", "tran_periodic_spectral_transfer")),
        "ridge_lambda": ridge_lambda,
    }
    qualitative["test_sample_indices"] = test_sample_indices[
        np.asarray(qualitative["condition_indices"], dtype=np.int64)
    ].astype(np.int64)
    return qualitative


def load_saved_coarse_report_payload(
    *,
    output_dir: Path,
    l_domain: float,
    relative_eps: float,
    include_interval: bool = True,
    include_global: bool = True,
) -> dict[str, Any]:
    from scripts.fae.tran_evaluation.core import FilterLadder

    payload = load_saved_coarse_metrics_payload(output_dir)
    config = payload["config"]
    coarse_results = payload["coarse_consistency"]
    full_h_schedule = [float(item) for item in config.get("full_H_schedule", [])]
    if not full_h_schedule:
        raise ValueError(
            f"Saved coarse metrics artifact {(Path(output_dir) / METRICS_NAME)} is missing 'config.full_H_schedule'."
        )
    time_indices = [int(item) for item in config.get("time_indices", [])]
    resolution_value = config.get("resolution")
    if resolution_value is None:
        cache_root = Path(output_dir) / "cache"
        if (cache_root / "conditioned_global.npz").exists():
            resolution = _infer_square_resolution(
                _load_npz_payload(cache_root / "conditioned_global.npz")["decoded_finest_fields"]
            )
        elif (cache_root / "conditioned_interval.npz").exists():
            resolution = _infer_square_resolution(
                _load_npz_payload(cache_root / "conditioned_interval.npz")["decoded_fine_fields"]
            )
        else:
            raise FileNotFoundError(
                f"Missing saved coarse cache exports under {cache_root}; cannot infer resolution for plot reuse."
            )
    else:
        resolution = int(resolution_value)
    ladder = FilterLadder(
        H_schedule=full_h_schedule,
        L_domain=float(config.get("l_domain", l_domain)),
        resolution=resolution,
    )
    qualitative_results = {
        "conditioned_interval": {},
        "conditioned_global_return": None,
    }
    if include_interval and coarse_results.get("conditioned_interval"):
        qualitative_results["conditioned_interval"] = _rebuild_interval_qualitative_results(
            output_dir=output_dir,
            coarse_results=coarse_results,
            ladder=ladder,
            relative_eps=float(relative_eps),
        )
    if include_global and coarse_results.get("conditioned_global_return") is not None:
        qualitative_results["conditioned_global_return"] = _rebuild_global_qualitative_results(
            output_dir=output_dir,
            full_h_schedule=full_h_schedule,
            time_indices=time_indices,
            coarse_results=coarse_results,
            ladder=ladder,
            relative_eps=float(relative_eps),
        )
    return {
        "metrics_payload": payload,
        "coarse_results": coarse_results,
        "coarse_qualitative_results": qualitative_results,
        "resolution": int(resolution),
        "summary_path": Path(output_dir) / SUMMARY_NAME,
        "metrics_path": Path(output_dir) / METRICS_NAME,
        "arrays_path": Path(output_dir) / ARRAYS_NAME,
        "manifest_path": Path(output_dir) / MANIFEST_NAME,
    }
