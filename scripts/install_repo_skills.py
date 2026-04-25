#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def discover_repo_skills(repo_root: Path) -> list[Path]:
    """Return repo-local skill directories that contain a SKILL.md file."""
    skills_root = repo_root / ".codex" / "skills"
    if not skills_root.exists():
        return []
    return sorted(path.parent for path in skills_root.glob("*/SKILL.md"))


def default_global_skills_root() -> Path:
    """Return the global Codex skill root used for auto-discovery."""
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "skills"
    return Path.home() / ".codex" / "skills"


def backup_existing_path(path: Path) -> Path:
    """Move an existing path to a numbered backup name and return the backup path."""
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.bak.{index}")
        if not candidate.exists():
            path.rename(candidate)
            return candidate
        index += 1


def install_skill(
    source: Path,
    destination_root: Path,
    *,
    mode: str,
    backup_existing: bool,
) -> tuple[str, Path | None]:
    """Install one repo skill via symlink or copy."""
    destination = destination_root / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return "unchanged", None
        destination.unlink()
        backup = None
    elif destination.exists():
        if backup_existing:
            backup = backup_existing_path(destination)
        else:
            raise FileExistsError(f"Destination already exists: {destination}")
    else:
        backup = None

    if mode == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(source, destination)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return "installed", backup


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Install repo-local Codex skills into the global skill discovery directory.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root containing .codex/skills.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=default_global_skills_root(),
        help="Global Codex skill root. Defaults to $CODEX_HOME/skills or ~/.codex/skills.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Install mode. Symlink keeps the repo as the source of truth.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Fail instead of backing up conflicting existing skills.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install plan without changing anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    destination_root = args.destination_root.expanduser().resolve()
    skills = discover_repo_skills(repo_root)

    print(f"Repo skills root: {repo_root / '.codex' / 'skills'}")
    print(f"Global skills root: {destination_root}")
    print(f"Install mode: {args.mode}")
    print("")

    if not skills:
        print("No repo-local skills found.")
        return 0

    for skill_dir in skills:
        print(f"- {skill_dir.name}: {skill_dir} -> {destination_root / skill_dir.name}")
    if args.dry_run:
        return 0

    print("")
    for skill_dir in skills:
        status, backup = install_skill(
            skill_dir,
            destination_root,
            mode=args.mode,
            backup_existing=not args.no_backup,
        )
        if status == "unchanged":
            print(f"unchanged: {skill_dir.name}")
            continue
        if backup is not None:
            print(f"backed up existing {skill_dir.name} to {backup}")
        print(f"installed: {skill_dir.name}")

    print("")
    print("Installed repo-local skills into the global discovery directory.")
    print("Start a new Codex session to let the runtime discover them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
