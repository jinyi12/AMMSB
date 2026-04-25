from __future__ import annotations

from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import build_root_condition_batch
from scripts.csp.conditional_eval.population_contract import (
    POPULATION_DEFAULT_CONDITIONS_ID,
    POPULATION_DEFAULT_CONDITIONS_OOD,
    POPULATION_CORR_ID_FALLBACK,
    POPULATION_DEFAULT_DOMAINS,
    POPULATION_SWEEP_TIERS,
)
from scripts.csp.conditional_eval.rollout_targets import select_split_sample_indices


def parse_population_domains(raw_value: str | None) -> list[str]:
    if raw_value is None or str(raw_value).strip() == "":
        return list(POPULATION_DEFAULT_DOMAINS)
    requested = [str(item).strip().lower() for item in str(raw_value).split(",") if str(item).strip()]
    ordered: list[str] = []
    for domain in POPULATION_DEFAULT_DOMAINS:
        if domain in requested and domain not in ordered:
            ordered.append(domain)
    invalid = sorted(set(requested) - set(POPULATION_DEFAULT_DOMAINS))
    if invalid:
        raise ValueError(
            f"Unsupported population domain(s) {invalid}. Expected a subset of {POPULATION_DEFAULT_DOMAINS}."
        )
    return ordered


def population_condition_seed(generation_seed: int, sample_index: int) -> int:
    return int(generation_seed) + int(sample_index)


def _domain_key(domain: str, split: str) -> str:
    return f"{str(domain)}_{str(split)}"


def _split_available_count(runtime, split: str) -> int:
    if str(split) == "train":
        return int(np.asarray(runtime.latent_train).shape[1])
    if str(split) == "test":
        return int(np.asarray(runtime.latent_test).shape[1])
    raise ValueError(f"Unsupported population split {split!r}; expected 'train' or 'test'.")


def _requested_population_budget(
    *,
    domain: str,
    requested_conditions: int,
    n_available: int,
) -> tuple[int, str]:
    requested = max(1, min(int(requested_conditions), int(n_available)))
    if str(domain) == "id" and int(requested_conditions) == POPULATION_DEFAULT_CONDITIONS_ID:
        return min(int(n_available), int(max(requested, POPULATION_CORR_ID_FALLBACK))), "default_id_fallback"
    if str(domain) == "ood" and int(requested_conditions) == POPULATION_DEFAULT_CONDITIONS_OOD:
        return int(n_available), "default_ood_all_available"
    return int(requested), "explicit_request"


def _candidate_tiers(*, budget_conditions: int) -> list[int]:
    tiers = [int(tier) for tier in POPULATION_SWEEP_TIERS if int(tier) <= int(budget_conditions)]
    if int(budget_conditions) not in tiers:
        tiers.append(int(budget_conditions))
    return sorted(set(int(item) for item in tiers))


def _population_target_display_labels(target_specs: list[dict[str, Any]]) -> list[str]:
    return [str(spec["display_label"]) for spec in target_specs]


def _population_recoarsened_display_labels(target_specs: list[dict[str, Any]]) -> list[str]:
    return [
        f"{str(spec['display_label'])} transferred to H={float(spec['H_condition']):g}"
        for spec in target_specs
    ]


def build_population_domain_specs(
    *,
    args,
    runtime,
    target_specs: list[dict[str, Any]],
    seed_policy: dict[str, int],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    domains = parse_population_domains(getattr(args, "population_domains", None))
    for domain_idx, domain in enumerate(domains):
        split = str(
            getattr(
                args,
                "population_id_split" if domain == "id" else "population_ood_split",
            )
        )
        requested_conditions = int(
            getattr(
                args,
                "population_conditions_id" if domain == "id" else "population_conditions_ood",
            )
        )
        n_available = _split_available_count(runtime, split)
        budget_conditions, budget_policy = _requested_population_budget(
            domain=domain,
            requested_conditions=requested_conditions,
            n_available=n_available,
        )
        sample_indices = select_split_sample_indices(
            n_available=int(n_available),
            n_conditions=int(budget_conditions),
            seed=int(seed_policy["condition_selection_seed"]) + 100_000 * int(domain_idx),
            sort_indices=False,
        )
        time_indices = np.asarray(runtime.time_indices, dtype=np.int64)
        condition_set = build_root_condition_batch(
            split=str(split),
            test_sample_indices=sample_indices.astype(np.int64),
            time_indices=time_indices,
            conditioning_time_index=int(time_indices[-1]),
        )
        specs.append(
            {
                "domain": str(domain),
                "split": str(split),
                "domain_key": _domain_key(str(domain), str(split)),
                "domain_index": int(domain_idx),
                "requested_conditions": int(requested_conditions),
                "budget_conditions": int(budget_conditions),
                "budget_policy": str(budget_policy),
                "candidate_tiers": _candidate_tiers(budget_conditions=int(budget_conditions)),
                "condition_set": condition_set,
                "sample_indices": sample_indices.astype(np.int64),
                "target_display_labels": _population_target_display_labels(target_specs),
                "recoarsened_display_labels": _population_recoarsened_display_labels(target_specs),
            }
        )
    return specs
