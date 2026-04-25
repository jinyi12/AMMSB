from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.install_repo_skills import (
    backup_existing_path,
    default_global_skills_root,
    discover_repo_skills,
    install_skill,
)


def test_discover_repo_skills_finds_skill_directories(tmp_path):
    repo_root = tmp_path / "repo"
    skill_dir = repo_root / ".codex" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: demo\n---\n")

    skills = discover_repo_skills(repo_root)

    assert skills == [skill_dir]


def test_install_skill_creates_symlink_and_backs_up_existing_directory(tmp_path):
    source = tmp_path / "repo" / ".codex" / "skills" / "demo-skill"
    destination_root = tmp_path / "global"
    destination = destination_root / "demo-skill"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("---\nname: demo-skill\ndescription: demo\n---\n")
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("old")

    status, backup = install_skill(source, destination_root, mode="symlink", backup_existing=True)

    assert status == "installed"
    assert backup is not None
    assert backup.exists()
    assert destination.is_symlink()
    assert destination.resolve() == source.resolve()


def test_backup_existing_path_uses_numbered_suffix(tmp_path):
    path = tmp_path / "skill"
    path.mkdir()

    backup = backup_existing_path(path)

    assert backup.name == "skill.bak.1"
    assert backup.exists()
    assert not path.exists()


def test_default_global_skills_root_uses_home(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert default_global_skills_root() == tmp_path / ".codex" / "skills"
