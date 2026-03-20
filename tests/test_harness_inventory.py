from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.harness_inventory import collect_file_line_counts, collect_repeated_function_names


def test_collect_file_line_counts_skips_protected_directories(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "mmsfm").mkdir(parents=True)
    (repo_root / "archive").mkdir()
    (repo_root / "mmsfm" / "core.py").write_text("a = 1\nb = 2\n")
    (repo_root / "archive" / "legacy.py").write_text("x = 1\n" * 50)

    stats = collect_file_line_counts(repo_root)

    assert [stat.path.as_posix() for stat in stats] == ["mmsfm/core.py"]


def test_collect_repeated_function_names_reports_top_level_duplicates(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "a.py").write_text("def shared():\n    return 1\n")
    (repo_root / "scripts" / "b.py").write_text("def shared():\n    return 2\n")
    (repo_root / "scripts" / "c.py").write_text("def unique():\n    return 3\n")

    repeated = collect_repeated_function_names(repo_root)

    assert len(repeated) == 1
    assert repeated[0].name == "shared"
    assert repeated[0].occurrences == ("scripts/a.py", "scripts/b.py")
