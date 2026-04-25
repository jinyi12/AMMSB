from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import scripts.csp.resource_policy as resource_policy_module


def test_resolve_resource_policy_defaults_and_fast_local_override() -> None:
    shared = resource_policy_module.resolve_resource_policy(
        SimpleNamespace(
            resource_profile="shared_safe",
            cpu_threads=None,
            cpu_cores=None,
            memory_budget_gb=None,
            condition_chunk_size=None,
        )
    )
    assert shared.cpu_threads == 8
    assert shared.cpu_cores == 8
    assert shared.memory_budget_gb == pytest.approx(12.0)
    assert shared.condition_chunk_size == 1

    fast_local = resource_policy_module.resolve_resource_policy(
        SimpleNamespace(
            resource_profile="fast_local",
            cpu_threads=6,
            cpu_cores=None,
            memory_budget_gb=None,
            condition_chunk_size=3,
        )
    )
    assert fast_local.cpu_threads == 6
    assert fast_local.cpu_cores is None
    assert fast_local.memory_budget_gb is None
    assert fast_local.condition_chunk_size == 3


def test_apply_startup_resource_policy_sets_env_and_affinity(monkeypatch) -> None:
    recorded: dict[str, object] = {}
    monkeypatch.setattr(resource_policy_module.sys, "platform", "linux")
    monkeypatch.setattr(resource_policy_module.os, "sched_getaffinity", lambda _pid: {2, 4, 6, 8})
    monkeypatch.setattr(
        resource_policy_module.os,
        "sched_setaffinity",
        lambda _pid, cpus: recorded.setdefault("affinity", tuple(sorted(cpus))),
    )
    for name in (
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    policy = resource_policy_module.ResourcePolicy(
        profile="shared_safe",
        cpu_threads=8,
        cpu_cores=2,
        memory_budget_gb=12.0,
        condition_chunk_size=1,
    )
    resource_policy_module.apply_startup_resource_policy(policy=policy, nogpu=True)

    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["TF_NUM_INTRAOP_THREADS"] == "8"
    assert os.environ["TF_NUM_INTEROP_THREADS"] == "1"
    assert "--xla_cpu_multi_thread_eigen=false" in os.environ["XLA_FLAGS"]
    assert "intra_op_parallelism_threads=8" in os.environ["XLA_FLAGS"]
    assert recorded["affinity"] == (2, 4)


def test_apply_startup_resource_policy_replaces_conflicting_xla_thread_flags(monkeypatch) -> None:
    monkeypatch.setattr(resource_policy_module.sys, "platform", "linux")
    monkeypatch.setattr(resource_policy_module.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(resource_policy_module.os, "sched_setaffinity", lambda _pid, cpus: None)
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=64 --other_flag=yes",
    )

    policy = resource_policy_module.ResourcePolicy(
        profile="shared_safe",
        cpu_threads=8,
        cpu_cores=8,
        memory_budget_gb=12.0,
        condition_chunk_size=1,
    )
    resource_policy_module.apply_startup_resource_policy(policy=policy, nogpu=False)

    assert os.environ["XLA_FLAGS"] == (
        "--other_flag=yes --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
    )
