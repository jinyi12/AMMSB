from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

ACTIVE_PYTHON_ROOTS = ("mmsfm", "csp", "data", "scripts", "tests")
PROTECTED_SEGMENTS = {
    ".conda",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "ECMMD-CondTwoSamp",
    "MSBM",
    "archive",
    "functional_autoencoders",
    "results",
    "spacetime-geometry",
    "wandb",
}


@dataclass(frozen=True)
class FileStat:
    path: Path
    line_count: int


@dataclass(frozen=True)
class RepeatedFunction:
    name: str
    occurrences: tuple[str, ...]


def _is_protected_path(path: Path) -> bool:
    return any(part in PROTECTED_SEGMENTS for part in path.parts)


def iter_active_python_files(
    repo_root: Path,
    *,
    roots: Sequence[str] | None = None,
) -> Iterator[Path]:
    """Yield Python files from active repo surfaces, excluding protected trees."""
    for root_name in roots or ACTIVE_PYTHON_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if _is_protected_path(path):
                continue
            yield path


def count_lines(path: Path) -> int:
    """Return the logical line count for a text file."""
    return len(path.read_text().splitlines())


def collect_file_line_counts(
    repo_root: Path,
    *,
    roots: Sequence[str] | None = None,
    min_lines: int = 0,
) -> list[FileStat]:
    """Return active Python files sorted by descending line count."""
    stats = [
        FileStat(path=path.relative_to(repo_root), line_count=count_lines(path))
        for path in iter_active_python_files(repo_root, roots=roots)
    ]
    stats = [stat for stat in stats if stat.line_count >= int(min_lines)]
    return sorted(stats, key=lambda stat: (-stat.line_count, stat.path.as_posix()))


def collect_repeated_function_names(
    repo_root: Path,
    *,
    roots: Sequence[str] | None = None,
    min_occurrences: int = 2,
) -> list[RepeatedFunction]:
    """Return top-level function names repeated across active modules."""
    seen: dict[str, list[str]] = {}
    for path in iter_active_python_files(repo_root, roots=roots):
        rel_path = path.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen.setdefault(node.name, []).append(rel_path)

    repeated = [
        RepeatedFunction(name=name, occurrences=tuple(sorted(paths)))
        for name, paths in seen.items()
        if len(paths) >= int(min_occurrences)
    ]
    return sorted(repeated, key=lambda item: (-len(item.occurrences), item.name))


def relative_paths(paths: Iterable[Path], repo_root: Path) -> list[str]:
    """Render paths relative to the repository root."""
    return [path.relative_to(repo_root).as_posix() for path in paths]
