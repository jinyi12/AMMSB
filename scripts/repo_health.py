#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from scripts.harness_inventory import collect_file_line_counts, collect_repeated_function_names
except ModuleNotFoundError:
    from harness_inventory import collect_file_line_counts, collect_repeated_function_names

REQUIRED_FILES = (
    "AGENTS.md",
    "Makefile",
    "README.md",
    "docs/architecture.md",
    "docs/conventions.md",
    "docs/evaluation_pipeline.md",
    "docs/experiments/index.md",
    "docs/index.md",
    "docs/latent_geometry_formulation.md",
    "docs/latent_geometry_plotting.md",
    "docs/publication_fae_figures_execution.md",
    "docs/publication_figures.md",
    "docs/publication_latent128_msbm_execution.md",
    "docs/runbooks/index.md",
    "docs/runbooks/bootstrap.md",
    "docs/runbooks/csp.md",
    "docs/runbooks/repo-health.md",
    "docs/runbooks/skills.md",
    "docs/runbooks/tran-evaluation.md",
    "scripts/install_repo_skills.py",
    "scripts/refactor_hotspots.py",
    "scripts/repo_health.py",
    ".github/workflows/harness.yml",
)
REQUIRED_MAKE_TARGETS = (
    "setup",
    "install-local",
    "install-csp",
    "install-skills",
    "lint",
    "check",
    "repo-health",
    "hotspots",
    "test",
    "test-harness",
    "test-tran-eval",
    "test-csp",
    "smoke-tran-eval",
    "smoke-csp",
)
LINK_CHECK_DOCS = (
    "README.md",
    "AGENTS.md",
    "docs/architecture.md",
    "docs/conventions.md",
    "docs/evaluation_pipeline.md",
    "docs/experiments/index.md",
    "docs/index.md",
    "docs/latent_geometry_formulation.md",
    "docs/latent_geometry_plotting.md",
    "docs/publication_fae_figures_execution.md",
    "docs/publication_figures.md",
    "docs/publication_latent128_msbm_execution.md",
    "docs/runbooks/index.md",
    "docs/runbooks/bootstrap.md",
    "docs/runbooks/csp.md",
    "docs/runbooks/repo-health.md",
    "docs/runbooks/skills.md",
    "docs/runbooks/tran-evaluation.md",
)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
URL_SCHEMES = ("http://", "https://", "mailto:")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the MMSFM repository harness surfaces.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root. Defaults to the MMSFM checkout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when required harness files or links are missing.",
    )
    parser.add_argument(
        "--hotspot-threshold",
        type=int,
        default=900,
        help="Warn when active Python files exceed this many lines.",
    )
    return parser.parse_args()


def check_required_files(repo_root: Path) -> list[str]:
    """Return required harness files that are missing."""
    required = [*REQUIRED_FILES, *discover_required_skill_files(repo_root)]
    return [path for path in required if not (repo_root / path).exists()]


def discover_required_skill_files(repo_root: Path) -> list[str]:
    """Return required repo-local skill files for every discovered skill."""
    skills_root = repo_root / ".codex" / "skills"
    if not skills_root.exists():
        return []

    required: list[str] = []
    for skill_md in sorted(skills_root.glob("*/SKILL.md")):
        skill_dir = skill_md.parent
        required.append(skill_md.relative_to(repo_root).as_posix())
        required.append((skill_dir / "agents" / "openai.yaml").relative_to(repo_root).as_posix())
    return required


def check_make_targets(repo_root: Path) -> list[str]:
    """Return required Makefile targets that are missing."""
    makefile = repo_root / "Makefile"
    if not makefile.exists():
        return list(REQUIRED_MAKE_TARGETS)
    text = makefile.read_text()
    missing: list[str] = []
    for target in REQUIRED_MAKE_TARGETS:
        if re.search(rf"(?m)^{re.escape(target)}\s*:", text) is None:
            missing.append(target)
    return missing


def collect_markdown_link_errors(repo_root: Path, doc_paths: list[str]) -> list[str]:
    """Return broken relative markdown links in the selected docs."""
    errors: list[str] = []
    for doc in doc_paths:
        path = repo_root / doc
        if not path.exists():
            continue
        text = path.read_text()
        for raw_target in MARKDOWN_LINK_RE.findall(text):
            if raw_target.startswith(URL_SCHEMES) or raw_target.startswith("#"):
                continue
            target = raw_target.split("#", 1)[0]
            if not target or target.startswith("file://"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{doc} -> {raw_target}")
    return errors


def _render_errors(errors: list[str], *, header: str) -> None:
    print(header)
    if not errors:
        print("  none")
        return
    for item in errors:
        print(f"  - {item}")


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()

    missing_files = check_required_files(repo_root)
    missing_targets = check_make_targets(repo_root)
    broken_links = collect_markdown_link_errors(repo_root, list(LINK_CHECK_DOCS))

    hotspot_stats = collect_file_line_counts(repo_root, min_lines=args.hotspot_threshold)
    repeated = collect_repeated_function_names(repo_root)

    print("Repository Harness Health")
    print("=" * 80)
    print(f"Repo root: {repo_root}")
    print("")
    _render_errors(missing_files, header="Missing required harness files")
    print("")
    _render_errors(missing_targets, header="Missing Makefile targets")
    print("")
    _render_errors(broken_links, header="Broken relative markdown links")
    print("")
    print("Large active Python files")
    if not hotspot_stats:
        print("  none")
    else:
        for stat in hotspot_stats[:10]:
            print(f"  - {stat.path.as_posix()} ({stat.line_count} lines)")
    print("")
    print("Repeated top-level helper names")
    if not repeated:
        print("  none")
    else:
        for item in repeated[:10]:
            print(f"  - {item.name} ({len(item.occurrences)} files)")

    has_errors = bool(missing_files or missing_targets or broken_links)
    if args.strict and has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
