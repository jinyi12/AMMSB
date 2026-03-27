from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from csp.sigma_calibration import calibrate_flat_latent_sigma
from csp.sigma_calibration import SIGMA_CALIBRATION_ZT_MODES
from scripts.csp.latent_archive import load_fae_latent_archive


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a CSP bridge sigma from a canonical flat FAE latent archive. "
            "The default recommendation targets conditional generation via a KNN residual scale."
        ),
    )
    parser.add_argument("--latents_path", type=str, required=True, help="Path to fae_latents.npz.")
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.25,
        help="Target midpoint bridge std as a fraction of the empirical latent scale.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=32,
        help="K used for the conditional residual KNN estimate.",
    )
    parser.add_argument(
        "--n_probe",
        type=int,
        default=512,
        help="Number of conditions probed per interval for the KNN residual estimate.",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed used for probe subsampling.")
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
    return parser.parse_args()


def _summary_payload(latents_path: Path, summary: object) -> dict[str, object]:
    return {
        "latents_path": str(latents_path),
        "zt_mode": str(summary.zt_mode),
        "uniform_intervals": bool(
            summary.interval_lengths.size <= 1
            or float(summary.interval_lengths.max() - summary.interval_lengths.min()) <= 1e-8
        ),
        "tau_knots": [float(value) for value in summary.tau_knots.tolist()],
        "interval_lengths": [float(value) for value in summary.interval_lengths.tolist()],
        "delta_rms": [float(value) for value in summary.delta_rms.tolist()],
        "conditional_residual_rms": [float(value) for value in summary.conditional_residual_rms.tolist()],
        "sigma_by_delta": [float(value) for value in summary.sigma_by_delta.tolist()],
        "sigma_by_conditional": [float(value) for value in summary.sigma_by_conditional.tolist()],
        "recommended_constant_sigma": float(summary.constant_sigma_by_conditional),
        "reference_constant_sigma_by_delta": float(summary.constant_sigma_by_delta),
        "midpoint_std_by_constant_sigma": [
            float(value) for value in summary.midpoint_std_by_constant_sigma.tolist()
        ],
    }


def main() -> None:
    args = _parse_args()
    latents_path = Path(args.latents_path).expanduser().resolve()
    archive = load_fae_latent_archive(latents_path)
    summary = calibrate_flat_latent_sigma(
        archive.latent_train,
        archive.zt,
        zt_mode=str(args.zt_mode),
        kappa=float(args.kappa),
        k_neighbors=int(args.k_neighbors),
        n_probe=int(args.n_probe),
        seed=int(args.seed),
    )
    payload = _summary_payload(latents_path, summary)

    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return

    print("============================================================", flush=True)
    print("CSP sigma calibration", flush=True)
    print(f"  latents_path              : {latents_path}", flush=True)
    print(f"  zt_mode                   : {payload['zt_mode']}", flush=True)
    print(f"  uniform_intervals         : {payload['uniform_intervals']}", flush=True)
    print(f"  tau_knots                 : {payload['tau_knots']}", flush=True)
    print(f"  interval_lengths          : {payload['interval_lengths']}", flush=True)
    print(f"  kappa                     : {float(args.kappa):.6g}", flush=True)
    print(f"  conditional_residual_rms  : {payload['conditional_residual_rms']}", flush=True)
    print(f"  sigma_by_conditional      : {payload['sigma_by_conditional']}", flush=True)
    print(f"  recommended_constant_sigma: {payload['recommended_constant_sigma']:.6g}", flush=True)
    print(f"  midpoint_std(constant)    : {payload['midpoint_std_by_constant_sigma']}", flush=True)
    print(f"  delta_rms                 : {payload['delta_rms']}", flush=True)
    print(f"  sigma_by_delta            : {payload['sigma_by_delta']}", flush=True)
    print(f"  reference_sigma_by_delta  : {payload['reference_constant_sigma_by_delta']:.6g}", flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
