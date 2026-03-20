#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillInstall:
    name: str
    source: Path
    destination: Path


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


def build_install_plan(repo_root: Path, destination_root: Path) -> list[SkillInstall]:
    """Map repo-local skills to their global installation destinations."""
    return [
        SkillInstall(name=skill_dir.name, source=skill_dir, destination=destination_root / skill_dir.name)
        for skill_dir in discover_repo_skills(repo_root)
    ]


def backup_existing_path(path: Path) -> Path:
    """Move an existing path to a numbered backup name and return the backup path."""
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.bak.{index}")
        if not candidate.exists():
            path.rename(candidate)
            return candidate
        index += 1


def install_skill(install: SkillInstall, *, mode: str, backup_existing: bool) -> tuple[str, Path | None]:
    """Install one repo skill via symlink or copy."""
    install.destination.parent.mkdir(parents=True, exist_ok=True)

    if install.destination.is_symlink():
        if install.destination.resolve() == install.source.resolve():
            return "unchanged", None
        install.destination.unlink()
        backup = None
    elif install.destination.exists():
        if backup_existing:
            backup = backup_existing_path(install.destination)
        else:
            raise FileExistsError(f"Destination already exists: {install.destination}")
    else:
        backup = None

    if mode == "symlink":
        install.destination.symlink_to(install.source.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(install.source, install.destination)
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
    plan = build_install_plan(repo_root, destination_root)

    print(f"Repo skills root: {repo_root / '.codex' / 'skills'}")
    print(f"Global skills root: {destination_root}")
    print(f"Install mode: {args.mode}")
    print("")

    if not plan:
        print("No repo-local skills found.")
        return 0

    for install in plan:
        print(f"- {install.name}: {install.source} -> {install.destination}")
    if args.dry_run:
        return 0

    print("")
    for install in plan:
        status, backup = install_skill(
            install,
            mode=args.mode,
            backup_existing=not args.no_backup,
        )
        if status == "unchanged":
            print(f"unchanged: {install.name}")
            continue
        if backup is not None:
            print(f"backed up existing {install.name} to {backup}")
        print(f"installed: {install.name}")

    print("")
    print("Installed repo-local skills into the global discovery directory.")
    print("Start a new Codex session to let the runtime discover them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
