#!/usr/bin/env python
"""Evaluate post-filtered field diagnostics in unconditional or conditional mode.

This script isolates the diagnostics behind the manuscript's
"Post-Filtered Consistency Evaluation" section as a companion analysis.

It is not the primary coarse-consistency definition used by
`evaluate.py`. The maintained Phase 1 coarse metric in `evaluate.py`
now evaluates Dirac-target coarse consistency for the sequential
conditional bridge. This script remains useful for the separate question:
after re-filtering generated fine fields, how well do their induced
unconditional or local conditional laws match reference ensembles?

1. Decode/generated fine-scale fields from the backward MSBM model.
2. Reapply the physical filter family to obtain post-filtered fields.
3. Compare those filtered fields against either:
   - unconditional observed marginals at each modeled scale, or
   - a conditional reference ensemble built from corpus k-NN weights given a
     coarse conditioning field.
4. Save one-point PDF and two-point autocorrelation figures plus compact
   JSON/text summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.tran_evaluation.core import (  # noqa: E402
    FilterLadder,
    build_default_H_schedule,
    load_ground_truth,
    load_time_index_mapping,
)
from scripts.fae.tran_evaluation.conditional_support import (  # noqa: E402
    knn_gaussian_weights as _knn_gaussian_weights,
    sample_weighted_rows as _sample_weighted_rows,
)
from scripts.fae.tran_evaluation.first_order import evaluate_first_order_pair  # noqa: E402
from scripts.fae.tran_evaluation.generate import (  # noqa: E402
    generate_backward_realizations,
)
from scripts.fae.tran_evaluation.run_support import parse_key_value_args_file as parse_args_file  # noqa: E402
from scripts.fae.tran_evaluation.report import (  # noqa: E402
    format_for_paper,
    plot_direct_field_correlation,
    plot_direct_field_pdfs,
)
from scripts.fae.tran_evaluation.second_order import (  # noqa: E402
    correlation_lengths,
    ensemble_directional_correlation,
    tran_J_mismatch,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Companion post-filtered field diagnostics for MSBM fields.",
    )
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--mode",
        type=str,
        default="unconditional",
        choices=["unconditional", "conditional"],
        help="Reference to compare against after post-filtering.",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Dataset npz. Defaults to data_path in args.txt.",
    )
    p.add_argument(
        "--corpus_path",
        type=str,
        default="data/fae_tran_inclusions_corpus.npz",
        help="Aligned corpus used for conditional references.",
    )
    p.add_argument(
        "--corpus_latents_path",
        type=str,
        default="data/corpus_latents_ntk_prior.npz",
        help="Encoded corpus latents used for conditional references.",
    )
    p.add_argument("--n_realizations", type=int, default=200)
    p.add_argument(
        "--n_gt_neighbors",
        type=int,
        default=200,
        help="Observed ensemble size for unconditional references.",
    )
    p.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Conditioning sample index used when mode=conditional.",
    )
    p.add_argument("--k_neighbors", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    p.add_argument("--drift_clip_norm", type=float, default=None)
    p.add_argument("--nogpu", action="store_true")
    p.add_argument("--decode_mode", type=str, default="standard", choices=["standard"])
    p.add_argument("--L_domain", type=float, default=6.0)
    p.add_argument("--H_meso_list", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0")
    p.add_argument("--H_macro", type=float, default=6.0)
    p.add_argument("--min_spacing_pixels", type=int, default=4)
    return p.parse_args()


def _load_corpus_references(
    *,
    corpus_path: Path,
    corpus_latents_path: Path,
    time_indices: np.ndarray,
    transform_info: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    corpus_lat = np.load(corpus_latents_path, allow_pickle=True)
    corpus_latents_by_tidx: dict[int, np.ndarray] = {}
    for tidx in time_indices:
        corpus_latents_by_tidx[int(tidx)] = np.asarray(
            corpus_lat[f"latents_{int(tidx)}"], dtype=np.float32,
        )
    corpus_lat.close()

    corpus_ds = np.load(corpus_path, allow_pickle=True)
    corpus_keys = sorted(
        [k for k in corpus_ds.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )
    corpus_fields_by_tidx: dict[int, np.ndarray] = {}
    from data.transform_utils import apply_inverse_transform  # noqa: E402

    for tidx in time_indices:
        raw = corpus_ds[corpus_keys[int(tidx)]].astype(np.float32)
        corpus_fields_by_tidx[int(tidx)] = apply_inverse_transform(raw, transform_info)
    corpus_ds.close()
    return corpus_latents_by_tidx, corpus_fields_by_tidx


def _make_eval_scope(
    *,
    full_H_schedule: list[float],
    gt_fields_by_index: dict[int, np.ndarray],
    modeled_dataset_indices: list[int],
    L_domain: float,
    resolution: int,
    n_gt_neighbors: int,
) -> tuple[list[float], list[float], dict[int, np.ndarray], FilterLadder]:
    eval_H_schedule = [0.0] + [full_H_schedule[idx] for idx in modeled_dataset_indices[1:]]
    label_H_schedule = [full_H_schedule[idx] for idx in modeled_dataset_indices]
    eval_gt_fields = {
        new_idx: gt_fields_by_index[old_idx][:n_gt_neighbors]
        for new_idx, old_idx in enumerate(modeled_dataset_indices)
    }
    eval_ladder = FilterLadder(
        H_schedule=eval_H_schedule,
        L_domain=L_domain,
        resolution=resolution,
    )
    return eval_H_schedule, label_H_schedule, eval_gt_fields, eval_ladder


def _summarize_postfiltered(
    *,
    ref_fields: dict[int, np.ndarray],
    gen_fields: dict[int, np.ndarray],
    label_H_schedule: list[float],
    resolution: int,
    pixel_size: float,
    min_spacing_pixels: int,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for scale in sorted(set(ref_fields.keys()) & set(gen_fields.keys())):
        first = evaluate_first_order_pair(
            ref_fields[scale],
            gen_fields[scale],
            resolution,
            min_spacing_pixels,
        )
        obs_corr = ensemble_directional_correlation(ref_fields[scale], resolution)
        gen_corr = ensemble_directional_correlation(gen_fields[scale], resolution)
        J = tran_J_mismatch(
            obs_corr["R_e1_mean"],
            obs_corr["R_e2_mean"],
            gen_corr["R_e1_mean"],
            gen_corr["R_e2_mean"],
            pixel_size,
        )
        cl_obs_e1 = correlation_lengths(obs_corr["R_e1_mean"], pixel_size)
        cl_obs_e2 = correlation_lengths(obs_corr["R_e2_mean"], pixel_size)
        cl_gen_e1 = correlation_lengths(gen_corr["R_e1_mean"], pixel_size)
        cl_gen_e2 = correlation_lengths(gen_corr["R_e2_mean"], pixel_size)
        summary[str(scale)] = {
            "H": float(label_H_schedule[scale]),
            "w1": first["wasserstein1"],
            "moments": first["moments"],
            "J": J,
            "xi_obs_e1": cl_obs_e1["xi_e"],
            "xi_obs_e2": cl_obs_e2["xi_e"],
            "xi_gen_e1": cl_gen_e1["xi_e"],
            "xi_gen_e2": cl_gen_e2["xi_e"],
        }

    return summary


def _write_summary_text(
    *,
    mode: str,
    scale_summary: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append(f"POST-FILTERED {mode.upper()} CONSISTENCY SUMMARY")
    lines.append("=" * 80)
    lines.append(
        f"{'Scale':>5} | {'H':>6} | {'W1_norm':>10} | {'Mean_RE':>10} | {'Var_RE':>10} | {'J_norm':>10}"
    )
    lines.append(
        f"{'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}"
    )
    for scale in sorted(scale_summary, key=int):
        item = scale_summary[scale]
        w = item["w1"]
        m = item["moments"]["relative_error"]
        J = item["J"]
        lines.append(
            f"{int(scale):>5} | {item['H']:>6.2f} | {w['w1_normalised']:>10.6f} | "
            f"{m['mean']:>10.6f} | {m['variance']:>10.6f} | {J['J_normalised']:>10.6f}"
        )
    lines.append("=" * 80)
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = parse_args_file(run_dir / "args.txt")
    dataset_path = Path(args.dataset_path or train_cfg.get("data_path"))
    gt = load_ground_truth(dataset_path)
    resolution = gt["resolution"]
    pixel_size = args.L_domain / resolution
    time_indices = load_time_index_mapping(run_dir)
    modeled_dataset_indices = [int(idx) for idx in time_indices.tolist()]

    H_meso = [float(x) for x in args.H_meso_list.split(",")]
    full_H_schedule = build_default_H_schedule(H_meso, args.H_macro)
    eval_H_schedule, label_H_schedule, eval_gt_fields, eval_ladder = _make_eval_scope(
        full_H_schedule=full_H_schedule,
        gt_fields_by_index=gt["fields_by_index"],
        modeled_dataset_indices=modeled_dataset_indices,
        L_domain=args.L_domain,
        resolution=resolution,
        n_gt_neighbors=args.n_gt_neighbors,
    )

    sample_mode = "marginal" if args.mode == "unconditional" else "fixed"
    print(f"Mode          : {args.mode}")
    print(f"Run directory : {run_dir}")
    print(f"Dataset       : {dataset_path}")
    print(f"Output        : {out_dir}")
    print(f"time_indices  : {time_indices.tolist()}")

    gen = generate_backward_realizations(
        run_dir=run_dir,
        dataset_npz_path=dataset_path,
        n_realizations=args.n_realizations,
        sample_idx=args.sample_idx,
        sample_mode=sample_mode,
        seed=args.seed,
        use_ema=args.use_ema,
        drift_clip_norm=args.drift_clip_norm,
        device=torch.device("cpu") if args.nogpu else None,
        decode_mode=args.decode_mode,
    )
    gen_filtered = eval_ladder.filter_all_scales(gen["realizations_phys"])

    ref_fields_eval: dict[int, np.ndarray]
    obs_label: str
    basename_pdf: str
    basename_corr: str
    metadata: dict[str, Any] = {
        "mode": args.mode,
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "n_realizations": int(args.n_realizations),
        "sample_idx": int(args.sample_idx),
        "time_indices": time_indices.tolist(),
        "label_H_schedule": label_H_schedule,
        "sample_indices": (
            gen.get("sample_indices").tolist()
            if gen.get("sample_indices") is not None else None
        ),
    }

    if args.mode == "unconditional":
        ref_fields_eval = eval_gt_fields
        obs_label = "Obs"
        basename_pdf = "fig_postfiltered_unconditional_pdfs"
        basename_corr = "fig_postfiltered_unconditional_correlation"
    else:
        lat_npz = np.load(run_dir / "fae_latents.npz", allow_pickle=True)
        latent_train = np.asarray(lat_npz["latent_train"], dtype=np.float32)
        lat_npz.close()
        if args.sample_idx < 0 or args.sample_idx >= latent_train.shape[1]:
            raise ValueError(
                f"sample_idx={args.sample_idx} outside [0, {latent_train.shape[1]})"
            )

        corpus_latents_by_tidx, corpus_fields_by_tidx = _load_corpus_references(
            corpus_path=Path(args.corpus_path),
            corpus_latents_path=Path(args.corpus_latents_path),
            time_indices=time_indices,
            transform_info=gt["transform_info"],
        )

        coarse_tidx = int(time_indices[-1])
        query_z = latent_train[-1, args.sample_idx]
        knn_idx, knn_weights = _knn_gaussian_weights(
            query_z,
            corpus_latents_by_tidx[coarse_tidx],
            args.k_neighbors,
        )
        rng = np.random.default_rng(args.seed)
        ref_fields_eval = {}
        for eval_idx, tidx in enumerate(modeled_dataset_indices):
            ref_fields_eval[eval_idx] = _sample_weighted_rows(
                corpus_fields_by_tidx[int(tidx)],
                knn_idx,
                knn_weights,
                args.n_realizations,
                rng,
            ).astype(np.float32)

        obs_label = "Reference"
        basename_pdf = "fig_postfiltered_conditional_pdfs"
        basename_corr = "fig_postfiltered_conditional_correlation"
        metadata["conditioning_dataset_index"] = coarse_tidx
        metadata["conditioning_H"] = float(full_H_schedule[coarse_tidx])
        metadata["k_neighbors"] = int(args.k_neighbors)

    scale_summary = _summarize_postfiltered(
        ref_fields=ref_fields_eval,
        gen_fields=gen_filtered,
        label_H_schedule=label_H_schedule,
        resolution=resolution,
        pixel_size=pixel_size,
        min_spacing_pixels=args.min_spacing_pixels,
    )

    format_for_paper()
    plot_direct_field_pdfs(
        ref_fields_eval,
        gen_filtered,
        label_H_schedule,
        out_dir,
        min_spacing_pixels=args.min_spacing_pixels,
        obs_label=obs_label,
        basename=basename_pdf,
    )
    plot_direct_field_correlation(
        ref_fields_eval,
        gen_filtered,
        label_H_schedule,
        resolution,
        pixel_size,
        out_dir,
        obs_label=obs_label,
        basename=basename_corr,
    )

    metrics = {
        "config": metadata,
        "scales": scale_summary,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_summary_text(
        mode=args.mode,
        scale_summary=scale_summary,
        out_path=out_dir / "summary.txt",
    )
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
