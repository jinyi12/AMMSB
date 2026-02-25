"""Train Multi-marginal Schrödinger Bridge Matching (MSBM) in a pretrained FAE latent space.

This file is intentionally kept lean as a CLI entrypoint. The implementation
lives in `train_latent_msbm_impl.py`. Shared FAE checkpoint/codec helpers live
in `fae_latent_utils.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make repo importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Backwards-compatible re-exports (historically imported from this module).
from scripts.fae.fae_naive.fae_latent_utils import (  # noqa: F401,E402
    NoopTimeModule as _NoopTimeModule,
    build_attention_fae_from_checkpoint as _build_attention_fae_from_checkpoint,
    decode_latent_knots_to_fields as _decode_latent_knots_to_fields,
    load_fae_checkpoint as _load_fae_checkpoint,
    make_fae_apply_fns as _make_fae_apply_fns,
)


def main() -> None:
    from scripts.fae.fae_naive.train_latent_msbm_impl import main as _main  # noqa: E402

    _main()


if __name__ == "__main__":
    main()

