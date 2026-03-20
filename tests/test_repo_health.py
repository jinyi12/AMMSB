from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.repo_health import check_make_targets, check_required_files, collect_markdown_link_errors


def test_check_required_files_reports_missing_targets(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Demo\n")

    missing = check_required_files(repo_root)

    assert "AGENTS.md" in missing
    assert "README.md" not in missing


def test_check_make_targets_reports_missing_targets(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "Makefile").write_text("setup:\n\t@echo setup\nlint:\n\t@echo lint\n")

    missing = check_make_targets(repo_root)

    assert "setup" not in missing
    assert "lint" not in missing
    assert "check" in missing


def test_collect_markdown_link_errors_ignores_urls_and_anchors(tmp_path):
    repo_root = tmp_path / "repo"
    docs = repo_root / "docs"
    docs.mkdir(parents=True)
    (docs / "target.md").write_text("# Target\n")
    (docs / "index.md").write_text(
        "[ok](target.md)\n[missing](missing.md)\n[web](https://example.com)\n[anchor](#top)\n"
    )

    errors = collect_markdown_link_errors(repo_root, ["docs/index.md"])

    assert errors == ["docs/index.md -> missing.md"]
