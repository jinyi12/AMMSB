from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.token_latent_archive_from_fae import (
    build_token_latent_archive_from_fae,
    write_token_latent_archive_from_fae_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a token-native transformer FAE latent archive and train the token-native "
            "paired-prior CSP bridge."
        ),
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the multiscale dataset npz.")
    parser.add_argument("--fae_checkpoint", type=str, required=True, help="Path to the transformer FAE checkpoint pkl.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/csp/paired_prior_bridge_token_dit/manual_run",
        help="Output run directory for token-native paired-prior CSP artifacts.",
    )
    parser.add_argument(
        "--latents_path",
        type=str,
        default=None,
        help="Defaults to <outdir>/fae_token_latents.npz.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Defaults to <outdir>/config/fae_token_latents_manifest.json.",
    )
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--max_samples_per_time", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--held_out_indices", type=str, default="")
    parser.add_argument("--held_out_times", type=str, default="")
    parser.add_argument("--time_dist_mode", type=str, choices=("zt", "uniform"), default="zt")
    parser.add_argument("--t_scale", type=float, default=1.0)
    parser.add_argument("--skip_encode_if_exists", action="store_true")

    parser.add_argument("--dit_hidden_dim", type=int, default=256)
    parser.add_argument("--dit_n_layers", type=int, default=3)
    parser.add_argument("--dit_num_heads", type=int, default=4)
    parser.add_argument("--dit_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dit_time_emb_dim", type=int, default=32)
    parser.add_argument(
        "--token_conditioning",
        type=str,
        choices=("set_conditioned_memory", "slotwise_additive", "mixed_sequence"),
        default="set_conditioned_memory",
    )
    parser.add_argument("--delta_v", type=float, default=1.0)
    parser.add_argument("--theta_trim", type=float, default=0.05)
    parser.add_argument("--dt0", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    latents_path = (
        Path(args.latents_path).expanduser().resolve()
        if args.latents_path is not None
        else outdir / "fae_token_latents.npz"
    )
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else outdir / "config" / "fae_token_latents_manifest.json"
    )

    if not args.skip_encode_if_exists or not latents_path.exists():
        manifest = build_token_latent_archive_from_fae(
            dataset_path=Path(args.data_path),
            fae_checkpoint_path=Path(args.fae_checkpoint),
            output_path=latents_path,
            encode_batch_size=int(args.encode_batch_size),
            max_samples_per_time=args.max_samples_per_time,
            train_ratio=args.train_ratio,
            held_out_indices_raw=str(args.held_out_indices),
            held_out_times_raw=str(args.held_out_times),
            time_dist_mode=str(args.time_dist_mode),
            t_scale=float(args.t_scale),
        )
        write_token_latent_archive_from_fae_manifest(manifest_path, manifest)
    else:
        print(f"Reusing existing token latent archive: {latents_path}", flush=True)

    cmd = [
        sys.executable,
        "scripts/csp/train_csp_token_paired_prior.py",
        "--latents_path",
        str(latents_path),
        "--source_dataset_path",
        str(Path(args.data_path).expanduser().resolve()),
        "--fae_checkpoint",
        str(Path(args.fae_checkpoint).expanduser().resolve()),
        "--outdir",
        str(outdir),
        "--dit_hidden_dim",
        str(int(args.dit_hidden_dim)),
        "--dit_n_layers",
        str(int(args.dit_n_layers)),
        "--dit_num_heads",
        str(int(args.dit_num_heads)),
        "--dit_mlp_ratio",
        str(float(args.dit_mlp_ratio)),
        "--dit_time_emb_dim",
        str(int(args.dit_time_emb_dim)),
        "--token_conditioning",
        str(args.token_conditioning),
        "--delta_v",
        str(float(args.delta_v)),
        "--theta_trim",
        str(float(args.theta_trim)),
        "--dt0",
        str(float(args.dt0)),
        "--lr",
        str(float(args.lr)),
        "--num_steps",
        str(int(args.num_steps)),
        "--batch_size",
        str(int(args.batch_size)),
        "--sample_count",
        str(int(args.sample_count)),
        "--log_every",
        str(int(args.log_every)),
        "--seed",
        str(int(args.seed)),
    ]

    print("============================================================", flush=True)
    print("Token-native paired-prior CSP training from FAE assets", flush=True)
    print(f"  dataset         : {Path(args.data_path).expanduser().resolve()}", flush=True)
    print(f"  fae_checkpoint  : {Path(args.fae_checkpoint).expanduser().resolve()}", flush=True)
    print(f"  latents_path    : {latents_path}", flush=True)
    print(f"  outdir          : {outdir}", flush=True)
    print("============================================================", flush=True)
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
