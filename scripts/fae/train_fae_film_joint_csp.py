"""Train a FiLM FAE with a jointly optimized latent CSP bridge and no SIGReg."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmsfm.fae.joint_csp_support import (
    JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE,
    export_joint_fae_csp,
    resolve_joint_fae_checkpoint,
    setup_vector_joint_csp_training,
)
from mmsfm.fae.standard_training_flow import run_standard_training
from mmsfm.fae.standard_training_support import (
    build_film_joint_csp_parser as build_parser,
    build_standard_autoencoder,
    select_film_joint_csp_run_metadata,
    validate_film_joint_csp_args as validate_args,
)

__all__ = ["build_parser", "validate_args", "main"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    if getattr(args, "init_checkpoint", None):
        print(f"Initialization mode: checkpoint_init ({args.init_checkpoint})")
    else:
        print("Initialization mode: fresh")
    architecture_name, wandb_name_prefix, wandb_tags = select_film_joint_csp_run_metadata()
    result = run_standard_training(
        args,
        build_autoencoder_fn=build_standard_autoencoder,
        architecture_name=architecture_name,
        wandb_name_prefix=wandb_name_prefix,
        wandb_tags=wandb_tags,
        setup_fn=setup_vector_joint_csp_training,
    )

    if args.skip_joint_csp_export:
        return

    fae_run_dir = Path(result["output_dir"]).expanduser().resolve()
    fae_checkpoint_path = resolve_joint_fae_checkpoint(
        fae_run_dir,
        preference=JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE,
    )
    export_manifest = export_joint_fae_csp(
        fae_checkpoint_path=fae_checkpoint_path,
        outdir=args.joint_csp_export_dir,
        dataset_path=args.data_path,
        encode_batch_size=max(1, int(args.batch_size)),
        train_ratio=float(args.train_ratio),
        held_out_indices_raw=str(getattr(args, "held_out_indices", "")),
        held_out_times_raw=str(getattr(args, "held_out_times", "")),
        checkpoint_preference=JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE,
    )
    print(f"Exported joint CSP run to: {export_manifest['outdir']}")


if __name__ == "__main__":
    main()
