"""Copy ``data/scraped/extracted/*.txt`` into ``data/scraped/originals/*.txt``.

Run once before deploying the Pipeline Sync demo. The copied snapshots are
used as the "Reset to original" source — let the user edit a document, run
the incremental sync, then snap back to the pristine version without
re-crawling the live site.

Usage::

    python -m scripts.snapshot_originals
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import settings


def main() -> None:
    src = Path(settings.scraped_dir)
    dst = Path(src.parent, "originals")
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for f in sorted(src.glob("*.txt")):
        target = dst / f.name
        shutil.copy2(f, target)
        copied += 1
    print(f"snapshot complete: {copied} files in {dst}")


if __name__ == "__main__":
    main()
