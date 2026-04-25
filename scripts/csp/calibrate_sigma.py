from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from csp.sigma_calibration import calibrate_flat_latent_sigma
from csp.sigma_calibration import SIGMA_CALIBRATION_METHODS
from csp.sigma_calibration import SIGMA_CALIBRATION_ZT_MODES
from scripts.csp.latent_archive import load_fae_latent_archive
from scripts.csp.token_latent_archive import load_token_fae_latent_archive


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a CSP bridge sigma from a canonical flat FAE latent archive. "
            "The default recommendation is the exact global common-sigma isotropic Brownian-reference MLE."
        ),
    )
    parser.add_argument("--latents_path", type=str, required=True, help="Path to fae_latents.npz.")
    parser.add_argument(
        "--method",
        type=str,
        choices=SIGMA_CALIBRATION_METHODS,
        default="global_mle",
        help="Sigma calibration method. Use knn_legacy to reproduce the old KNN residual heuristic.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.25,
        help="Legacy-only midpoint bridge fraction used by --method knn_legacy.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=32,
        help="Legacy-only K used for the conditional residual KNN estimate.",
    )
    parser.add_argument(
        "--n_probe",
        type=int,
        default=512,
        help="Legacy-only number of conditions probed per interval for the KNN residual estimate.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Legacy-only PRNG seed used for probe subsampling.")
    parser.add_argument(
        "--zt_mode",
        type=str,
        choices=SIGMA_CALIBRATION_ZT_MODES,
        default="archive",
        help="Use the archive zt as-is or override it with uniform retained pseudotime.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the calibration summary as JSON instead of a human-readable report.",
    )
    args = parser.parse_args(argv)
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    legacy_only_flags = {"--kappa", "--k_neighbors", "--n_probe", "--seed"}
    if str(args.method) != "knn_legacy":
        unexpected_flags = [
            flag
            for flag in legacy_only_flags
            if any(token == flag or token.startswith(f"{flag}=") for token in argv_tokens)
        ]
        if unexpected_flags:
            parser.error(
                f"{', '.join(sorted(unexpected_flags))} require --method knn_legacy."
            )
    return args


def _summary_payload(latents_path: Path, summary: object) -> dict[str, object]:
    return {
        "latents_path": str(latents_path),
        "method": str(summary.method),
        "zt_mode": str(summary.zt_mode),
        "uniform_intervals": bool(
            summary.interval_lengths.size <= 1
            or float(summary.interval_lengths.max() - summary.interval_lengths.min()) <= 1e-8
        ),
        "tau_knots": [float(value) for value in summary.tau_knots.tolist()],
        "interval_lengths": [float(value) for value in summary.interval_lengths.tolist()],
        "interval_sample_counts": [int(value) for value in summary.interval_sample_counts.tolist()],
        "latent_dim": int(summary.latent_dim),
        "delta_rms": [float(value) for value in summary.delta_rms.tolist()],
        "conditional_residual_rms": [float(value) for value in summary.conditional_residual_rms.tolist()],
        "sigma_by_delta": [float(value) for value in summary.sigma_by_delta.tolist()],
        "sigma_mle_by_interval": [float(value) for value in summary.sigma_by_delta.tolist()],
        "sigma_by_conditional": [float(value) for value in summary.sigma_by_conditional.tolist()],
        "sigma_sq_mle_by_interval": [float(value) for value in summary.interval_sigma_sq_mle.tolist()],
        "global_sigma_sq_mle": float(summary.global_sigma_sq_mle),
        "global_sigma_mle": float(summary.global_sigma_mle),
        "global_mle_standardized_squared_l2_sum": float(summary.global_mle_standardized_squared_l2_sum),
        "global_mle_sample_weight": float(summary.global_mle_sample_weight),
        "global_sigma_sq_mle_denominator": float(summary.latent_dim) * float(summary.global_mle_sample_weight),
        "pooled_squared_l2_sum": float(summary.pooled_squared_l2_sum),
        "pooled_interval_weight": float(summary.pooled_interval_weight),
        "pooled_sigma_sq_denominator": float(summary.latent_dim) * float(summary.pooled_interval_weight),
        "recommended_constant_sigma": float(summary.recommended_constant_sigma),
        "recommended_constant_sigma_source": str(summary.recommended_constant_sigma_source),
        "reference_constant_sigma_by_delta": float(summary.constant_sigma_by_delta),
        "midpoint_std_by_constant_sigma": [
            float(value) for value in summary.midpoint_std_by_constant_sigma.tolist()
        ],
    }


def _load_calibration_inputs(latents_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        archive = load_fae_latent_archive(latents_path)
        return np.asarray(archive.latent_train, dtype=np.float32), np.asarray(archive.zt, dtype=np.float32), "flat"
    except Exception as flat_error:
        try:
            archive = load_token_fae_latent_archive(latents_path)
        except Exception:
            raise flat_error
        latent_train = np.asarray(archive.latent_train, dtype=np.float32)
        flat_latent_train = latent_train.reshape(latent_train.shape[0], latent_train.shape[1], -1)
        return flat_latent_train, np.asarray(archive.zt, dtype=np.float32), "token_native"


def main() -> None:
    args = _parse_args()
    latents_path = Path(args.latents_path).expanduser().resolve()
    latent_train, zt, archive_format = _load_calibration_inputs(latents_path)
    summary = calibrate_flat_latent_sigma(
        latent_train,
        zt,
        method=str(args.method),
        zt_mode=str(args.zt_mode),
        kappa=float(args.kappa),
        k_neighbors=int(args.k_neighbors),
        n_probe=int(args.n_probe),
        seed=int(args.seed),
    )
    payload = _summary_payload(latents_path, summary)
    payload["archive_format"] = archive_format
    payload["kappa"] = float(args.kappa) if str(args.method) == "knn_legacy" else None

    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return

    print("============================================================", flush=True)
    print("CSP sigma calibration", flush=True)
    print(f"  latents_path              : {latents_path}", flush=True)
    print(f"  archive_format            : {payload['archive_format']}", flush=True)
    print(f"  method                    : {payload['method']}", flush=True)
    print(f"  zt_mode                   : {payload['zt_mode']}", flush=True)
    print(f"  uniform_intervals         : {payload['uniform_intervals']}", flush=True)
    print(f"  tau_knots                 : {payload['tau_knots']}", flush=True)
    print(f"  interval_lengths          : {payload['interval_lengths']}", flush=True)
    print(f"  interval_sample_counts    : {payload['interval_sample_counts']}", flush=True)
    print(f"  latent_dim                : {payload['latent_dim']}", flush=True)
    print(f"  recommended_constant_sigma: {payload['recommended_constant_sigma']:.6g}", flush=True)
    print(f"  recommendation_source     : {payload['recommended_constant_sigma_source']}", flush=True)
    print(f"  global_sigma_sq_mle       : {payload['global_sigma_sq_mle']:.6g}", flush=True)
    print(f"  global_sigma_mle          : {payload['global_sigma_mle']:.6g}", flush=True)
    print(f"  global_mle_sqsum_over_dt  : {payload['global_mle_standardized_squared_l2_sum']:.6g}", flush=True)
    print(f"  global_mle_sample_weight  : {payload['global_mle_sample_weight']:.6g}", flush=True)
    print(f"  global_sigma_denominator  : {payload['global_sigma_sq_mle_denominator']:.6g}", flush=True)
    print(f"  pooled_squared_l2_sum     : {payload['pooled_squared_l2_sum']:.6g}", flush=True)
    print(f"  pooled_interval_weight    : {payload['pooled_interval_weight']:.6g}", flush=True)
    print(f"  pooled_sigma_denominator  : {payload['pooled_sigma_sq_denominator']:.6g}", flush=True)
    print(f"  midpoint_std(constant)    : {payload['midpoint_std_by_constant_sigma']}", flush=True)
    print(f"  delta_rms                 : {payload['delta_rms']}", flush=True)
    print(f"  sigma_mle_by_interval     : {payload['sigma_mle_by_interval']}", flush=True)
    print(f"  sigma_sq_mle_by_interval  : {payload['sigma_sq_mle_by_interval']}", flush=True)
    print(f"  reference_sigma_by_delta  : {payload['reference_constant_sigma_by_delta']:.6g}", flush=True)
    if str(args.method) == "knn_legacy":
        print(f"  kappa                     : {float(args.kappa):.6g}", flush=True)
        print(f"  conditional_residual_rms  : {payload['conditional_residual_rms']}", flush=True)
        print(f"  sigma_by_conditional      : {payload['sigma_by_conditional']}", flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
