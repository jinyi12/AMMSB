# %%
# %% [markdown]
# # Tran Filter: Direct vs Recursive Filtering
#
# This notebook-style script explores whether switching the current
# coarse-consistency-style direct filtering to recursive filtering produces the
# same fields on the `fae_tran_inclusions_minmax.npz` dataset.
#
# It compares five objects for the modeled scales:
# - stored dataset target at scale `H`
# - direct filtering from `H=0`
# - recursive filtering from `H=0`
# - direct filtering from the modeled native field `H=1`
# - recursive filtering from the modeled native field `H=1`
#
# The most relevant comparison for the current evaluation contract is:
# - direct-from-`H=1` vs recursive-from-`H=1`
#
# The script saves summary metrics and a small figure bundle under
# `notebooks/figures/tran_filter_direct_vs_recursive/`.

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transform_utils import apply_inverse_transform, load_transform_info
from scripts.fae.tran_evaluation.coarse_consistency_eval import build_coarse_eval_scope
from scripts.fae.tran_evaluation.core import FilterLadder
from scripts.images.field_visualization import format_for_paper


# %%
# %% [markdown]
# ## Configuration

# %%
DATASET_PATH = REPO_ROOT / "data" / "fae_tran_inclusions_minmax.npz"
OUTPUT_DIR = REPO_ROOT / "notebooks" / "figures" / "tran_filter_direct_vs_recursive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The current minmax dataset is produced from the default Tran inclusion schedule.
FULL_H_SCHEDULE = [0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
L_DOMAIN = 6.0

# Override from the shell for lighter/faster smoke runs if needed.
N_METRIC_SAMPLES = int(os.environ.get("TRAN_FILTER_EXPLORE_SAMPLES", "16"))
VIZ_SAMPLE_POSITION = int(os.environ.get("TRAN_FILTER_EXPLORE_VIZ_POS", "0"))

format_for_paper()

print(f"Dataset      : {DATASET_PATH}")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"Metric samples: {N_METRIC_SAMPLES}")


# %%
# %% [markdown]
# ## Load A Small Physical-Scale Subset

# %%
def _sorted_marginal_keys(npz_data) -> list[str]:
    return sorted(
        [key for key in npz_data.files if str(key).startswith("raw_marginal_")],
        key=lambda key: float(str(key).replace("raw_marginal_", "")),
    )


def load_dataset_subset(
    dataset_path: Path,
    *,
    sample_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        transform_info = load_transform_info(dataset_npz)
        marginal_keys = _sorted_marginal_keys(dataset_npz)
        resolution = int(np.asarray(dataset_npz["resolution"]).item())
        times = np.asarray(dataset_npz["times"], dtype=np.float32)
        times_normalized = np.asarray(dataset_npz["times_normalized"], dtype=np.float32)
        held_out_indices = [int(value) for value in np.asarray(dataset_npz["held_out_indices"], dtype=np.int32).tolist()]
        scale_mode = str(np.asarray(dataset_npz["scale_mode"]).item())
        data_generator = str(np.asarray(dataset_npz["data_generator"]).item())

        fields_by_index: dict[int, np.ndarray] = {}
        n_total_samples: int | None = None
        selected_indices: np.ndarray | None = None

        for dataset_index, key in enumerate(marginal_keys):
            raw_fields = np.asarray(dataset_npz[key], dtype=np.float32)
            if n_total_samples is None:
                n_total_samples = int(raw_fields.shape[0])
                if sample_indices is None:
                    selected_indices = np.arange(n_total_samples, dtype=np.int64)
                else:
                    selected_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
                    if np.any(selected_indices < 0) or np.any(selected_indices >= n_total_samples):
                        raise IndexError(
                            f"sample_indices must lie in [0, {n_total_samples}), got "
                            f"{selected_indices.min()}..{selected_indices.max()}."
                        )

            assert selected_indices is not None
            subset = np.asarray(raw_fields[selected_indices], dtype=np.float32)
            physical = np.asarray(
                apply_inverse_transform(subset, transform_info),
                dtype=np.float32,
            ).reshape(subset.shape[0], -1)
            fields_by_index[int(dataset_index)] = physical

    assert n_total_samples is not None
    assert selected_indices is not None
    return {
        "fields_by_index": fields_by_index,
        "resolution": int(resolution),
        "times": times,
        "times_normalized": times_normalized,
        "held_out_indices": held_out_indices,
        "scale_mode": scale_mode,
        "data_generator": data_generator,
        "n_total_samples": int(n_total_samples),
        "selected_indices": selected_indices.astype(np.int64),
    }


with np.load(DATASET_PATH, allow_pickle=True) as _probe:
    first_key = _sorted_marginal_keys(_probe)[0]
    n_total_samples = int(np.asarray(_probe[first_key]).shape[0])

metric_sample_indices = np.arange(min(N_METRIC_SAMPLES, n_total_samples), dtype=np.int64)
dataset_subset = load_dataset_subset(DATASET_PATH, sample_indices=metric_sample_indices)

held_out_set = set(dataset_subset["held_out_indices"])
modeled_dataset_indices = [
    int(dataset_index)
    for dataset_index in range(1, len(FULL_H_SCHEDULE))
    if int(dataset_index) not in held_out_set
]
modeled_h_schedule = [float(FULL_H_SCHEDULE[idx]) for idx in modeled_dataset_indices]
dataset_index_to_eval_index = {
    int(dataset_index): int(eval_index)
    for eval_index, dataset_index in enumerate(modeled_dataset_indices)
}

assert len(FULL_H_SCHEDULE) == len(dataset_subset["times"]), (
    "This notebook assumes the current default Tran inclusion schedule. "
    f"Expected {len(FULL_H_SCHEDULE)} marginals, got {len(dataset_subset['times'])}."
)

resolution = int(dataset_subset["resolution"])
full_ladder = FilterLadder(
    H_schedule=list(FULL_H_SCHEDULE),
    L_domain=float(L_DOMAIN),
    resolution=int(resolution),
)
eval_h_schedule, eval_gt_fields, eval_ladder = build_coarse_eval_scope(
    list(FULL_H_SCHEDULE),
    dataset_subset["fields_by_index"],
    modeled_dataset_indices,
    float(L_DOMAIN),
    int(resolution),
)

print(f"Data generator      : {dataset_subset['data_generator']}")
print(f"Scale mode          : {dataset_subset['scale_mode']}")
print(f"Resolution          : {resolution}x{resolution}")
print(f"Held-out indices    : {dataset_subset['held_out_indices']}")
print(f"Modeled indices     : {modeled_dataset_indices}")
print(f"Modeled H schedule  : {modeled_h_schedule}")
print(f"Metric subset size  : {len(metric_sample_indices)}")


# %%
# %% [markdown]
# ## Filtering Helpers

# %%
def direct_filter(
    fields: np.ndarray,
    *,
    ladder: FilterLadder,
    scale_idx: int,
) -> np.ndarray:
    return np.asarray(
        ladder.filter_at_scale(np.asarray(fields, dtype=np.float32).reshape(fields.shape[0], -1), int(scale_idx)),
        dtype=np.float32,
    ).reshape(fields.shape[0], -1)


def recursive_filter(
    fields: np.ndarray,
    *,
    ladder: FilterLadder,
    final_scale_idx: int,
) -> np.ndarray:
    filtered = np.asarray(fields, dtype=np.float32).reshape(fields.shape[0], -1)
    for scale_idx in range(1, int(final_scale_idx) + 1):
        filtered = np.asarray(
            ladder.filter_at_scale(filtered, int(scale_idx)),
            dtype=np.float32,
        ).reshape(filtered.shape[0], -1)
    return filtered


def relative_l2_per_sample(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64).reshape(reference.shape[0], -1)
    est = np.asarray(estimate, dtype=np.float64).reshape(estimate.shape[0], -1)
    numerator = np.linalg.norm(est - ref, axis=1)
    denominator = np.maximum(np.linalg.norm(ref, axis=1), float(eps))
    return np.asarray(numerator / denominator, dtype=np.float64)


def summarise_error(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:
    rel = relative_l2_per_sample(reference, estimate)
    abs_l2 = np.linalg.norm(
        np.asarray(estimate, dtype=np.float64).reshape(estimate.shape[0], -1)
        - np.asarray(reference, dtype=np.float64).reshape(reference.shape[0], -1),
        axis=1,
    )
    return {
        "mean_rel_l2": float(np.mean(rel)),
        "median_rel_l2": float(np.median(rel)),
        "max_rel_l2": float(np.max(rel)),
        "mean_abs_l2": float(np.mean(abs_l2)),
    }


# %%
# %% [markdown]
# ## Aggregate Error Table Across Modeled Scales

# %%
micro_fields = np.asarray(dataset_subset["fields_by_index"][0], dtype=np.float32)
native_h1_fields = np.asarray(dataset_subset["fields_by_index"][modeled_dataset_indices[0]], dtype=np.float32)

summary_rows: list[dict[str, float | int]] = []
field_cache: dict[tuple[str, int], np.ndarray] = {}

for dataset_index in modeled_dataset_indices:
    target_fields = np.asarray(dataset_subset["fields_by_index"][dataset_index], dtype=np.float32)
    h_target = float(FULL_H_SCHEDULE[dataset_index])
    eval_index = int(dataset_index_to_eval_index[dataset_index])

    direct_from_h0 = direct_filter(micro_fields, ladder=full_ladder, scale_idx=int(dataset_index))
    recursive_from_h0 = recursive_filter(micro_fields, ladder=full_ladder, final_scale_idx=int(dataset_index))
    direct_from_h1 = direct_filter(native_h1_fields, ladder=eval_ladder, scale_idx=int(eval_index))
    recursive_from_h1 = recursive_filter(native_h1_fields, ladder=eval_ladder, final_scale_idx=int(eval_index))

    field_cache[("target", dataset_index)] = target_fields
    field_cache[("direct_h0", dataset_index)] = direct_from_h0
    field_cache[("recursive_h0", dataset_index)] = recursive_from_h0
    field_cache[("direct_h1", dataset_index)] = direct_from_h1
    field_cache[("recursive_h1", dataset_index)] = recursive_from_h1

    comparisons = {
        "direct_h0_vs_target": summarise_error(target_fields, direct_from_h0),
        "recursive_h0_vs_target": summarise_error(target_fields, recursive_from_h0),
        "recursive_h0_vs_direct_h0": summarise_error(direct_from_h0, recursive_from_h0),
        "direct_h1_vs_target": summarise_error(target_fields, direct_from_h1),
        "recursive_h1_vs_target": summarise_error(target_fields, recursive_from_h1),
        "direct_h1_vs_recursive_h1": summarise_error(direct_from_h1, recursive_from_h1),
    }
    for comparison_name, metrics in comparisons.items():
        summary_rows.append(
            {
                "dataset_index": int(dataset_index),
                "H_target": float(h_target),
                "comparison": comparison_name,
                **metrics,
            }
        )


def print_summary_table(rows: list[dict[str, float | int]]) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["H_target"]), str(row["comparison"])))
    header = (
        f"{'idx':>3} | {'H_target':>8} | {'comparison':>26} | "
        f"{'mean_rel_l2':>12} | {'median_rel_l2':>14} | {'max_rel_l2':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in ordered:
        print(
            f"{int(row['dataset_index']):>3} | "
            f"{float(row['H_target']):>8.3f} | "
            f"{str(row['comparison']):>26} | "
            f"{float(row['mean_rel_l2']):>12.6e} | "
            f"{float(row['median_rel_l2']):>14.6e} | "
            f"{float(row['max_rel_l2']):>11.6e}"
        )


print_summary_table(summary_rows)
(OUTPUT_DIR / "summary_rows.json").write_text(json.dumps(summary_rows, indent=2))


# %%
# %% [markdown]
# ## Error Curves For The `H=1` Evaluation Variants

# %%
target_h_values = np.asarray([float(FULL_H_SCHEDULE[idx]) for idx in modeled_dataset_indices], dtype=np.float64)
direct_h1_vs_target = np.asarray(
    [
        next(
            row["mean_rel_l2"]
            for row in summary_rows
            if int(row["dataset_index"]) == int(dataset_index) and str(row["comparison"]) == "direct_h1_vs_target"
        )
        for dataset_index in modeled_dataset_indices
    ],
    dtype=np.float64,
)
recursive_h1_vs_target = np.asarray(
    [
        next(
            row["mean_rel_l2"]
            for row in summary_rows
            if int(row["dataset_index"]) == int(dataset_index) and str(row["comparison"]) == "recursive_h1_vs_target"
        )
        for dataset_index in modeled_dataset_indices
    ],
    dtype=np.float64,
)
direct_h1_vs_recursive_h1 = np.asarray(
    [
        next(
            row["mean_rel_l2"]
            for row in summary_rows
            if int(row["dataset_index"]) == int(dataset_index) and str(row["comparison"]) == "direct_h1_vs_recursive_h1"
        )
        for dataset_index in modeled_dataset_indices
    ],
    dtype=np.float64,
)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.plot(target_h_values, direct_h1_vs_target, marker="o", label=r"direct from $H=1$ vs dataset target")
ax.plot(target_h_values, recursive_h1_vs_target, marker="s", label=r"recursive from $H=1$ vs dataset target")
ax.plot(target_h_values, direct_h1_vs_recursive_h1, marker="^", label=r"direct vs recursive from $H=1$")
ax.set_xlabel(r"Target coarse scale $H$")
ax.set_ylabel(r"Mean relative $L^2$")
ax.set_title(r"Filtering From The Modeled Native Field $H=1$")
ax.grid(alpha=0.25)
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "direct_vs_recursive_h1_error_curves.png", dpi=180, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "direct_vs_recursive_h1_error_curves.pdf", bbox_inches="tight")
plt.close(fig)


# %%
# %% [markdown]
# ## Field Panels For A Representative Sample

# %%
def choose_visualisation_targets(indices: list[int]) -> list[int]:
    if len(indices) <= 3:
        return list(indices)
    choices = [indices[1], indices[len(indices) // 2], indices[-1]]
    unique: list[int] = []
    for value in choices:
        if value not in unique:
            unique.append(value)
    return unique


viz_dataset_indices = choose_visualisation_targets(modeled_dataset_indices)
viz_sample_position = max(0, min(int(VIZ_SAMPLE_POSITION), len(metric_sample_indices) - 1))
sample_label = int(metric_sample_indices[viz_sample_position])

row_names = [
    "dataset target",
    "direct from H=1",
    "recursive from H=1",
    "direct - recursive",
]

fig, axes = plt.subplots(
    len(row_names),
    len(viz_dataset_indices),
    figsize=(1.95 * len(viz_dataset_indices) + 0.65, 1.9 * len(row_names) + 0.3),
    squeeze=False,
)

for col_idx, dataset_index in enumerate(viz_dataset_indices):
    target = field_cache[("target", dataset_index)][viz_sample_position].reshape(resolution, resolution)
    direct_h1 = field_cache[("direct_h1", dataset_index)][viz_sample_position].reshape(resolution, resolution)
    recursive_h1 = field_cache[("recursive_h1", dataset_index)][viz_sample_position].reshape(resolution, resolution)
    delta = direct_h1 - recursive_h1
    image_rows = [target, direct_h1, recursive_h1, delta]

    common_vmin = min(float(np.min(target)), float(np.min(direct_h1)), float(np.min(recursive_h1)))
    common_vmax = max(float(np.max(target)), float(np.max(direct_h1)), float(np.max(recursive_h1)))
    delta_lim = max(1e-12, float(np.max(np.abs(delta))))

    for row_idx, image in enumerate(image_rows):
        ax = axes[row_idx, col_idx]
        if row_idx < 3:
            im = ax.imshow(image, origin="lower", cmap="cividis", vmin=common_vmin, vmax=common_vmax)
        else:
            im = ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=-delta_lim, vmax=delta_lim)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(f"H={FULL_H_SCHEDULE[dataset_index]:g}", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(row_names[row_idx], fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

fig.suptitle(f"Sample index {sample_label}: direct vs recursive filtering from H=1", y=1.01, fontsize=11)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "direct_vs_recursive_h1_field_panels.png", dpi=180, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "direct_vs_recursive_h1_field_panels.pdf", bbox_inches="tight")
plt.close(fig)


# %%
# %% [markdown]
# ## Key Takeaways

# %%
for dataset_index in viz_dataset_indices:
    h_target = float(FULL_H_SCHEDULE[dataset_index])
    mismatch = next(
        row for row in summary_rows
        if int(row["dataset_index"]) == int(dataset_index)
        and str(row["comparison"]) == "direct_h1_vs_recursive_h1"
    )
    print(
        f"H={h_target:g}: direct-vs-recursive mean_rel_l2="
        f"{float(mismatch['mean_rel_l2']):.6e}, max_rel_l2={float(mismatch['max_rel_l2']):.6e}"
    )

print("\nSaved outputs:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  - {path.name}")
