from __future__ import annotations

"""Execution policy helpers for conditional rollout cache and plot stages."""

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.csp.plot_latent_trajectories import plot_latent_trajectory_summary
from scripts.csp.resource_policy import RESOURCE_PROFILE_SHARED_SAFE


TOKEN_ROLLOUT_MODEL_TYPES = {
    "conditional_bridge_token_dit",
    "paired_prior_bridge_token_dit",
}


def is_token_rollout_run(run_dir: Path) -> bool:
    config_path = Path(run_dir).expanduser().resolve() / "config" / "args.json"
    if not config_path.exists():
        return False
    cfg = json.loads(config_path.read_text())
    return str(cfg.get("model_type", "conditional_bridge")) in TOKEN_ROLLOUT_MODEL_TYPES


def resolve_effective_rollout_devices(
    *,
    args: argparse.Namespace,
    resource_policy,
) -> tuple[str, str, str, str, bool]:
    requested_sampling = str(getattr(args, "coarse_sampling_device", "auto"))
    requested_decode = str(getattr(args, "coarse_decode_device", "auto"))
    if bool(getattr(args, "nogpu", False)):
        return requested_sampling, requested_decode, "cpu", "cpu", False
    shared_safe_decode_cpu_defaulted = str(getattr(resource_policy, "profile", "")) == RESOURCE_PROFILE_SHARED_SAFE
    effective_sampling = requested_sampling
    effective_decode = "cpu" if shared_safe_decode_cpu_defaulted else requested_decode
    return requested_sampling, requested_decode, effective_sampling, effective_decode, bool(shared_safe_decode_cpu_defaulted)


def resolve_rollout_latent_cache_dir(*, run_dir: Path, output_dir: Path) -> Path | None:
    candidates = [output_dir.parent / "cache", run_dir / "eval" / "cache"]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "latent_samples.npz").exists() or (resolved / "latent_samples_tokens.npz").exists():
            return resolved
    return None


def resolve_rollout_coarse_split(*, cache_dir: Path, runtime: Any) -> str:
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text())
        cache_split = str(cache_manifest.get("coarse_split", "")).strip()
        if cache_split in {"train", "test"}:
            return cache_split
    runtime_split = str(getattr(runtime, "split", "")).strip()
    if runtime_split in {"train", "test"}:
        return runtime_split
    return "test"


def write_rollout_latent_trajectory_report(
    *,
    run_dir: Path,
    output_dir: Path,
    runtime: Any,
    n_plot_conditions: int,
    seed: int,
    plot_latent_trajectory_summary_fn=plot_latent_trajectory_summary,
) -> dict[str, Any] | None:
    cache_dir = resolve_rollout_latent_cache_dir(run_dir=run_dir, output_dir=output_dir)
    if cache_dir is None:
        return None
    return plot_latent_trajectory_summary_fn(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        coarse_split=resolve_rollout_coarse_split(cache_dir=cache_dir, runtime=runtime),
        max_conditions_per_pair=max(1, int(n_plot_conditions)),
        seed=int(seed),
    )
