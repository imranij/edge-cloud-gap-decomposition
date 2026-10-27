"""Extract every zip archive of the repository into the folders it was packed from.

The per-split outputs are stored as zip archives to keep the repository small. Each archive holds paths
relative to the repository root, so extracting all of them restores the layout that the scripts expect.
Files that already exist are overwritten.

    python tools/unpack_archives.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    archives = sorted(p for p in ROOT.rglob("*.zip") if ".git" not in p.parts)
    for a in archives:
        with zipfile.ZipFile(a) as z:
            for name in z.namelist():
                target = (ROOT / name).resolve()
                if ROOT not in target.parents:
                    raise SystemExit(f"{a}: entry {name} points outside the repository")
            z.extractall(ROOT)
            print(f"{a.relative_to(ROOT)}: {len(z.namelist())} files")


if __name__ == "__main__":
    main()
