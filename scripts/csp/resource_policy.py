from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Sequence


RESOURCE_PROFILE_SHARED_SAFE = "shared_safe"
RESOURCE_PROFILE_BALANCED = "balanced"
RESOURCE_PROFILE_FAST_LOCAL = "fast_local"
RESOURCE_PROFILE_CHOICES = (
    RESOURCE_PROFILE_SHARED_SAFE,
    RESOURCE_PROFILE_BALANCED,
    RESOURCE_PROFILE_FAST_LOCAL,
)

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)

_RESOURCE_PROFILE_DEFAULTS = {
    RESOURCE_PROFILE_SHARED_SAFE: {
        "cpu_threads": 8,
        "cpu_cores": 8,
        "memory_budget_gb": 12.0,
        "condition_chunk_size": 1,
    },
    RESOURCE_PROFILE_BALANCED: {
        "cpu_threads": 16,
        "cpu_cores": 16,
        "memory_budget_gb": 24.0,
        "condition_chunk_size": 2,
    },
    RESOURCE_PROFILE_FAST_LOCAL: {
        "cpu_threads": None,
        "cpu_cores": None,
        "memory_budget_gb": None,
        "condition_chunk_size": None,
    },
}


@dataclass(frozen=True)
class ResourcePolicy:
    profile: str
    cpu_threads: int | None
    cpu_cores: int | None
    memory_budget_gb: float | None
    condition_chunk_size: int | None


def add_resource_policy_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--resource_profile",
        choices=RESOURCE_PROFILE_CHOICES,
        default=RESOURCE_PROFILE_SHARED_SAFE,
        help=(
            "Host resource budget preset. shared_safe is the default shared-machine policy; "
            "fast_local leaves host thread/core/memory caps unset."
        ),
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help="Optional override for CPU worker threads used by BLAS/XLA/Torch CPU kernels.",
    )
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=None,
        help="Optional Linux CPU-affinity cap applied to the current process.",
    )
    parser.add_argument(
        "--memory_budget_gb",
        type=float,
        default=None,
        help="Optional host-memory budget used to shrink conditional rollout chunk sizes.",
    )
    parser.add_argument(
        "--condition_chunk_size",
        type=int,
        default=None,
        help="Optional override for per-condition rollout cache chunking.",
    )
    return parser


def _validate_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0 when provided; received {value!r}.")
    return parsed


def _validate_positive_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0 when provided; received {value!r}.")
    return parsed


def resolve_resource_policy(args: argparse.Namespace) -> ResourcePolicy:
    profile = str(getattr(args, "resource_profile", RESOURCE_PROFILE_SHARED_SAFE))
    defaults = _RESOURCE_PROFILE_DEFAULTS[profile]
    return ResourcePolicy(
        profile=profile,
        cpu_threads=_validate_positive_int(
            "cpu_threads",
            getattr(args, "cpu_threads", None)
            if getattr(args, "cpu_threads", None) is not None
            else defaults["cpu_threads"],
        ),
        cpu_cores=_validate_positive_int(
            "cpu_cores",
            getattr(args, "cpu_cores", None)
            if getattr(args, "cpu_cores", None) is not None
            else defaults["cpu_cores"],
        ),
        memory_budget_gb=_validate_positive_float(
            "memory_budget_gb",
            getattr(args, "memory_budget_gb", None)
            if getattr(args, "memory_budget_gb", None) is not None
            else defaults["memory_budget_gb"],
        ),
        condition_chunk_size=_validate_positive_int(
            "condition_chunk_size",
            getattr(args, "condition_chunk_size", None)
            if getattr(args, "condition_chunk_size", None) is not None
            else defaults["condition_chunk_size"],
        ),
    )


def _startup_policy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nogpu", action="store_true")
    add_resource_policy_args(parser)
    return parser


def append_resource_policy_cli_args(cmd: list[str], *, args: argparse.Namespace) -> list[str]:
    cmd.extend(["--resource_profile", str(getattr(args, "resource_profile", RESOURCE_PROFILE_SHARED_SAFE))])
    if getattr(args, "cpu_threads", None) is not None:
        cmd.extend(["--cpu_threads", str(int(getattr(args, "cpu_threads")))])
    if getattr(args, "cpu_cores", None) is not None:
        cmd.extend(["--cpu_cores", str(int(getattr(args, "cpu_cores")))])
    if getattr(args, "memory_budget_gb", None) is not None:
        cmd.extend(["--memory_budget_gb", str(float(getattr(args, "memory_budget_gb")))])
    if getattr(args, "condition_chunk_size", None) is not None:
        cmd.extend(["--condition_chunk_size", str(int(getattr(args, "condition_chunk_size")))])
    return cmd


def _replace_xla_thread_flags(existing_flags: str, *, cpu_threads: int) -> str:
    tokens = [token for token in shlex.split(str(existing_flags or "")) if token.strip()]
    filtered = [
        token
        for token in tokens
        if not (
            token.startswith("intra_op_parallelism_threads=")
            or token == "--xla_cpu_multi_thread_eigen=false"
            or token == "--xla_cpu_multi_thread_eigen=true"
            or token.startswith("--xla_cpu_multi_thread_eigen=")
        )
    ]
    filtered.extend(
        [
            "--xla_cpu_multi_thread_eigen=false",
            f"intra_op_parallelism_threads={int(cpu_threads)}",
        ]
    )
    return " ".join(filtered)


def _apply_cpu_affinity(cpu_cores: int) -> list[int] | None:
    if sys.platform != "linux":
        return None
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        allowed = sorted(int(core) for core in os.sched_getaffinity(0))
    except OSError:
        return None
    if not allowed:
        return None
    chosen = allowed[: max(1, min(int(cpu_cores), len(allowed)))]
    try:
        os.sched_setaffinity(0, chosen)
    except OSError:
        return None
    return chosen


def apply_startup_resource_policy(
    *,
    policy: ResourcePolicy,
    nogpu: bool,
) -> ResourcePolicy:
    if bool(nogpu):
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if policy.cpu_threads is not None:
        for env_name in _THREAD_ENV_VARS:
            if env_name == "TF_NUM_INTEROP_THREADS":
                os.environ[env_name] = "1"
                continue
            os.environ[env_name] = str(int(policy.cpu_threads))
        os.environ["XLA_FLAGS"] = _replace_xla_thread_flags(
            os.environ.get("XLA_FLAGS", ""),
            cpu_threads=int(policy.cpu_threads),
        )
    if policy.cpu_cores is not None:
        _apply_cpu_affinity(int(policy.cpu_cores))
    return policy


def apply_startup_resource_policy_from_argv(argv: Sequence[str] | None = None) -> ResourcePolicy:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parsed, _unknown = _startup_policy_parser().parse_known_args(raw_args)
    policy = resolve_resource_policy(parsed)
    return apply_startup_resource_policy(policy=policy, nogpu=bool(getattr(parsed, "nogpu", False)))


def apply_torch_thread_policy(policy: ResourcePolicy) -> None:
    if policy.cpu_threads is None:
        return
    import torch

    torch.set_num_threads(int(policy.cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
