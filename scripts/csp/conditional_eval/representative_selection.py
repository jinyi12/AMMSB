from __future__ import annotations

import numpy as np


def _candidate_from_sorted(order: np.ndarray, used: set[int]) -> int | None:
    for item in np.asarray(order, dtype=np.int64).tolist():
        if int(item) not in used:
            return int(item)
    return None


def _median_candidates(order: np.ndarray) -> np.ndarray:
    center = len(order) // 2
    candidates: list[int] = []
    for delta in range(len(order)):
        right = center + delta
        left = center - delta - 1
        if right < len(order):
            candidates.append(int(order[right]))
        if left >= 0:
            candidates.append(int(order[left]))
    return np.asarray(candidates, dtype=np.int64)


def _farthest_high_score_candidate(
    *,
    condition_pca: np.ndarray,
    local_scores: np.ndarray,
    used: set[int],
    require_top_quartile: bool,
) -> int | None:
    n_conditions = int(local_scores.shape[0])
    if n_conditions == 0:
        return None
    score_arr = np.asarray(local_scores, dtype=np.float64)
    candidate_pool = np.arange(n_conditions, dtype=np.int64)
    if require_top_quartile and n_conditions > 1:
        threshold = float(np.quantile(score_arr, 0.75))
        candidate_pool = candidate_pool[score_arr >= threshold]
        if candidate_pool.size == 0:
            candidate_pool = np.arange(n_conditions, dtype=np.int64)
    candidate_pool = candidate_pool[[int(idx) not in used for idx in candidate_pool.tolist()]]
    if candidate_pool.size == 0:
        return None
    if not used:
        best_pos = int(np.argmax(score_arr[candidate_pool]))
        return int(candidate_pool[best_pos])
    selected = np.asarray(sorted(used), dtype=np.int64)
    distances = np.linalg.norm(
        condition_pca[candidate_pool, None, :] - condition_pca[selected][None, :, :],
        axis=2,
    )
    min_dist = distances.min(axis=1)
    best_pos = int(np.lexsort((-score_arr[candidate_pool], -min_dist))[-1])
    return int(candidate_pool[best_pos])


def select_representative_conditions(
    *,
    local_scores: np.ndarray,
    condition_pca: np.ndarray,
    n_show: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    scores = np.asarray(local_scores, dtype=np.float64).reshape(-1)
    n_conditions = int(scores.shape[0])
    if n_conditions == 0 or int(n_show) <= 0:
        return np.asarray([], dtype=np.int64), []
    target = int(min(max(1, int(n_show)), n_conditions))
    order = np.argsort(scores)
    rng = np.random.default_rng(int(seed))

    selected: list[int] = []
    roles: list[str] = []
    used: set[int] = set()

    def _append(candidate: int | None, role: str) -> None:
        if candidate is None or int(candidate) in used or len(selected) >= target:
            return
        selected.append(int(candidate))
        roles.append(str(role))
        used.add(int(candidate))

    for role, candidates in (
        ("best", order),
        ("median", _median_candidates(order)),
        ("worst", order[::-1]),
    ):
        _append(_candidate_from_sorted(candidates, used), role)

    _append(
        _farthest_high_score_candidate(
            condition_pca=np.asarray(condition_pca, dtype=np.float64),
            local_scores=scores,
            used=used,
            require_top_quartile=True,
        ),
        "diverse_high",
    )

    if len(selected) < target:
        remaining = [idx for idx in range(n_conditions) if idx not in used]
        if remaining:
            _append(int(rng.choice(np.asarray(remaining, dtype=np.int64))), "random")

    extra_idx = 1
    while len(selected) < target:
        _append(
            _farthest_high_score_candidate(
                condition_pca=np.asarray(condition_pca, dtype=np.float64),
                local_scores=scores,
                used=used,
                require_top_quartile=False,
            ),
            f"extra_{extra_idx}",
        )
        extra_idx += 1
        if len(selected) >= n_conditions:
            break

    return np.asarray(selected[:target], dtype=np.int64), roles[:target]

