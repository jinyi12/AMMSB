from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.resource_policy import apply_startup_resource_policy_from_argv

apply_startup_resource_policy_from_argv()

from scripts.csp.conditional_rollout_runtime import (
    build_latent_cache_parser,
    run_conditional_rollout_latent_cache,
)


def main() -> None:
    parser = build_latent_cache_parser(
        description="Build the conditional rollout latent cache and reference cache.",
    )
    run_conditional_rollout_latent_cache(parser.parse_args())


if __name__ == "__main__":
    main()
