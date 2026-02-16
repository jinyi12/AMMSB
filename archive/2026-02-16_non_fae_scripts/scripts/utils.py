"""Compatibility shim for legacy scripts in this archive.

The original training scripts assumed they were run from the repo's `scripts/`
directory (so `import utils` resolved to `scripts/utils.py`). After archiving,
we keep those scripts runnable by re-exporting the active implementation.
"""

from __future__ import annotations

from scripts.utils import *  # noqa: F403

