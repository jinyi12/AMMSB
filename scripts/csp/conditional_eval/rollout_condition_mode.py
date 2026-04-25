from __future__ import annotations

from typing import Literal, cast


RolloutConditionMode = Literal["chatterjee_knn", "exact_query"]

DEFAULT_ROLLOUT_CONDITION_MODE: RolloutConditionMode = "chatterjee_knn"
EXACT_QUERY_ROLLOUT_CONDITION_MODE: RolloutConditionMode = "exact_query"
CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE: RolloutConditionMode = "chatterjee_knn"


def validate_rollout_condition_mode(value: str | None) -> RolloutConditionMode:
    mode = str(DEFAULT_ROLLOUT_CONDITION_MODE if value in (None, "") else value)
    if mode not in {
        CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
        EXACT_QUERY_ROLLOUT_CONDITION_MODE,
    }:
        raise ValueError(
            "rollout_condition_mode must be one of "
            f"{(CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE, EXACT_QUERY_ROLLOUT_CONDITION_MODE)}, "
            f"got {value!r}."
        )
    return cast(RolloutConditionMode, mode)
