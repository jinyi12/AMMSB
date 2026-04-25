from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmsfm.fae.joint_csp_support import export_joint_fae_csp
from scripts.csp.latent_archive_from_fae import ARCHIVE_ZT_MODES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a saved joint FiLM + SIGReg + latent-CSP checkpoint into a standard flat CSP run.",
    )
    parser.add_argument("--fae-checkpoint", type=str, required=True, help="Path to a saved joint FAE checkpoint.")
    parser.add_argument("--outdir", type=str, default=None, help="Export directory. Defaults to <fae_run>/joint_csp.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional dataset path override.")
    parser.add_argument("--encode-batch-size", type=int, default=32, help="Batch size used when rebuilding fae_latents.npz.")
    parser.add_argument(
        "--max-samples-per-time",
        type=int,
        default=0,
        help="Optional cap on encoded samples per retained time (0 disables the cap).",
    )
    parser.add_argument("--train-ratio", type=float, default=None, help="Optional train split override.")
    parser.add_argument("--held-out-indices", type=str, default="", help="Optional held-out index override.")
    parser.add_argument("--held-out-times", type=str, default="", help="Optional held-out time override.")
    parser.add_argument(
        "--time-dist-mode",
        type=str,
        choices=["zt", "uniform"],
        default="zt",
        help="Compatibility time grid mode for the optional archive t_dists field.",
    )
    parser.add_argument("--t-scale", type=float, default=1.0, help="Compatibility time-grid scale for t_dists.")
    parser.add_argument(
        "--zt-mode",
        type=str,
        choices=ARCHIVE_ZT_MODES,
        default="retained_times",
        help="How to place retained training marginals on the archive zt grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_manifest = export_joint_fae_csp(
        fae_checkpoint_path=args.fae_checkpoint,
        outdir=args.outdir,
        dataset_path=args.data_path,
        encode_batch_size=int(args.encode_batch_size),
        max_samples_per_time=(
            None if int(args.max_samples_per_time) <= 0 else int(args.max_samples_per_time)
        ),
        train_ratio=args.train_ratio,
        held_out_indices_raw=args.held_out_indices,
        held_out_times_raw=args.held_out_times,
        time_dist_mode=args.time_dist_mode,
        t_scale=float(args.t_scale),
        zt_mode=args.zt_mode,
        checkpoint_preference="explicit_path",
    )
    print("Joint FAE -> CSP export", flush=True)
    print(f"  Output dir         : {export_manifest['outdir']}", flush=True)
    print(f"  CSP config         : {export_manifest['config_path']}", flush=True)
    print(f"  Conditional bridge : {export_manifest['conditional_bridge_path']}", flush=True)
    print(f"  Latents archive    : {export_manifest['latents_path']}", flush=True)


if __name__ == "__main__":
    main()
