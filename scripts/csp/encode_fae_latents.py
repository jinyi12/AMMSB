from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.latent_archive_from_fae import (
    ARCHIVE_ZT_MODES,
    build_latent_archive_from_fae,
    write_latent_archive_from_fae_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a multiscale FAE dataset into the canonical CSP latent archive. "
            "Transformer token latents are stored as flattened transport vectors."
        ),
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the multiscale dataset npz.")
    parser.add_argument("--fae_checkpoint", type=str, required=True, help="Path to the FAE checkpoint pkl.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/csp/manual_run/fae_latents.npz",
        help="Where to write the canonical latent archive.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Defaults to <output_path stem>_manifest.json.",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=64,
        help="Batch size used while encoding time marginals.",
    )
    parser.add_argument(
        "--max_samples_per_time",
        type=int,
        default=None,
        help="Optional cap on samples per time marginal before the train/test split.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="Optional train split ratio override. Defaults to the FAE checkpoint metadata.",
    )
    parser.add_argument(
        "--held_out_indices",
        type=str,
        default="",
        help="Optional comma-separated dataset time indices to exclude.",
    )
    parser.add_argument(
        "--held_out_times",
        type=str,
        default="",
        help="Optional comma-separated normalized times to exclude.",
    )
    parser.add_argument(
        "--time_dist_mode",
        type=str,
        choices=("zt", "uniform"),
        default="zt",
        help="Compatibility time-grid mode for the optional t_dists archive field.",
    )
    parser.add_argument(
        "--zt_mode",
        type=str,
        choices=ARCHIVE_ZT_MODES,
        default="retained_times",
        help="How to place retained training marginals on the archive zt grid.",
    )
    parser.add_argument(
        "--t_scale",
        type=float,
        default=1.0,
        help="Compatibility time-grid scale for the optional t_dists archive field.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else output_path.with_name(f"{output_path.stem}_manifest.json")
    )

    manifest = build_latent_archive_from_fae(
        dataset_path=Path(args.data_path),
        fae_checkpoint_path=Path(args.fae_checkpoint),
        output_path=output_path,
        encode_batch_size=int(args.encode_batch_size),
        max_samples_per_time=args.max_samples_per_time,
        train_ratio=args.train_ratio,
        held_out_indices_raw=str(args.held_out_indices),
        held_out_times_raw=str(args.held_out_times),
        time_dist_mode=str(args.time_dist_mode),
        t_scale=float(args.t_scale),
        zt_mode=str(args.zt_mode),
    )
    write_latent_archive_from_fae_manifest(manifest_path, manifest)

    print("============================================================", flush=True)
    print("CSP FAE latent archive", flush=True)
    print(f"  dataset         : {manifest['dataset_path']}", flush=True)
    print(f"  fae_checkpoint  : {manifest['fae_checkpoint_path']}", flush=True)
    print(f"  output_path     : {manifest['output_path']}", flush=True)
    print(f"  latent_train    : {tuple(manifest['latent_train_shape'])}", flush=True)
    print(f"  latent_test     : {tuple(manifest['latent_test_shape'])}", flush=True)
    print(
        f"  transport       : {manifest['transport_info']['transport_latent_format']}",
        flush=True,
    )
    print(f"  manifest_path   : {manifest_path}", flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
