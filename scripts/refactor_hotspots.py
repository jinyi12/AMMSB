#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.harness_inventory import collect_file_line_counts, collect_repeated_function_names
except ModuleNotFoundError:
    from harness_inventory import collect_file_line_counts, collect_repeated_function_names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report large active Python files and repeated top-level helper names.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root. Defaults to the MMSFM checkout.",
    )
    parser.add_argument(
        "--top-files",
        type=int,
        default=20,
        help="Number of file hotspots to print.",
    )
    parser.add_argument(
        "--top-functions",
        type=int,
        default=20,
        help="Number of repeated helper names to print.",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=200,
        help="Minimum line count for file hotspots.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    file_stats = collect_file_line_counts(repo_root, min_lines=args.min_lines)
    repeated = collect_repeated_function_names(repo_root)

    payload = {
        "repo_root": str(repo_root),
        "min_lines": int(args.min_lines),
        "file_hotspots": [
            {"path": stat.path.as_posix(), "line_count": stat.line_count}
            for stat in file_stats[: args.top_files]
        ],
        "repeated_functions": [
            {"name": item.name, "occurrences": list(item.occurrences)}
            for item in repeated[: args.top_functions]
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Harness Refactor Hotspots")
    print("=" * 80)
    print("")
    print(f"Repo root: {repo_root}")
    print(f"Minimum file size: {args.min_lines} lines")
    print("")
    print("Largest active Python files")
    print("-" * 80)
    for stat in payload["file_hotspots"]:
        print(f"{stat['line_count']:6d}  {stat['path']}")

    print("")
    print("Repeated top-level helper names")
    print("-" * 80)
    for item in payload["repeated_functions"]:
        print(f"{item['name']} ({len(item['occurrences'])} files)")
        for path in item["occurrences"]:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
