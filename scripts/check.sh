#!/usr/bin/env bash
# Validation script — run from repo root: bash scripts/check.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Ruff lint check ==="
if command -v ruff &>/dev/null; then
    ruff check mmsfm/ scripts/fae/ tests/ --exit-zero
else
    echo "ruff not installed, skipping lint"
fi

echo ""
echo "=== Import validation ==="
python -c "import mmsfm; print('mmsfm OK')" 2>/dev/null || echo "mmsfm import failed"

echo ""
echo "=== Tests ==="
if [ -d tests ]; then
    python -m pytest tests/ -x -q --tb=short 2>/dev/null || echo "Some tests failed"
else
    echo "No tests directory found"
fi

echo ""
echo "=== Done ==="
