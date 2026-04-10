"""
create_symlinks.py — Flatten vdb/ into two symlink directories.

Creates:
  notebooks/data/originals/   — symlinks to all *-picture-*.jpg files
  notebooks/data/predictions/ — symlinks to all *-prediction-*.jpg files

Files are named: {camera}__{original_filename}
No data is copied — symlinks point to the original files in vdb/.

Usage:
    python3.8 scripts/create_symlinks.py
"""

from pathlib import Path

ROOT     = Path(__file__).parent.parent
VDB      = ROOT / "vdb"
ORIG_OUT = ROOT / "notebooks/data/originals"
PRED_OUT = ROOT / "notebooks/data/predictions"

ORIG_OUT.mkdir(parents=True, exist_ok=True)
PRED_OUT.mkdir(parents=True, exist_ok=True)

orig_count = pred_count = skip = 0

for jpg in VDB.rglob("*.jpg"):
    if "-picture-" in jpg.name:
        dest_dir = ORIG_OUT
        orig_count += 1
    elif "-prediction-" in jpg.name:
        dest_dir = PRED_OUT
        pred_count += 1
    else:
        skip += 1
        continue

    cam = jpg.relative_to(VDB).parts[0]
    link = dest_dir / f"{cam}__{jpg.name}"

    if not link.exists():
        link.symlink_to(jpg.resolve())

print(f"Originals   : {orig_count}")
print(f"Predictions : {pred_count}")
print(f"Skipped     : {skip}")
print(f"Done — symlinks in notebooks/data/originals/ and notebooks/data/predictions/")
