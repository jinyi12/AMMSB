#!/usr/bin/env bash
# Historical wrapper retained as a signpost only.
#
# The active latent-geometry workflow is now the maintained transformer pair:
#   scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh
#
# The publication-era latent128 FiLM comparison depended on the removed
# multi-run chain/effect-table surface. It is no longer supported by the
# active compare_latent_geometry_models.py entrypoint.

set -euo pipefail

cat <<'EOF'
Historical wrapper: scripts/fae/experiments/evaluate_latent_geometry_latent128.sh

The maintained latent-geometry workflow is the canonical transformer pair:
  bash scripts/fae/experiments/evaluate_latent_geometry_transformer_pair.sh

The old latent128 publication comparison used a historical multi-run geometry
surface that is no longer implemented in the active tree.

If you need those latent128 publication figures, use a historical revision or
the publication-era documentation as reference only.
EOF

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exit 0
fi

exit 1
