from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmsfm.wandb_compat import wandb as package_wandb
from scripts.wandb_compat import wandb as script_wandb


def test_script_wandb_compat_reuses_package_wrapper():
    assert script_wandb is package_wandb
    assert hasattr(script_wandb, "init")
    assert hasattr(script_wandb, "Api")
    assert hasattr(script_wandb, "Image")
