"""Cleanup script — remove training artifacts from previous failed runs.

Targets:
  - logs/distillation/     (vis PNGs, TensorBoard events)
  - checkpoints/distillation/  (old .pt checkpoints from 1008px runs, ~2.1GB)

Preserves:
  - All source code (.py)
  - Original datasets (data/sa1b, data/sa_v, data/sa_co)
  - outputs/baseline/ (teacher baseline outputs)
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Directories to completely remove and recreate empty
DIRS_TO_CLEAN = [
    BASE_DIR / "logs" / "distillation",
    BASE_DIR / "checkpoints" / "distillation",
]

# Safety: directories that must NEVER be touched
PROTECTED = {
    BASE_DIR / "data",
    BASE_DIR / "models",
    BASE_DIR / "distillation",
}


def cleanup():
    total_freed = 0

    for d in DIRS_TO_CLEAN:
        # Safety check
        resolved = d.resolve()
        if any(resolved == p.resolve() or str(resolved).startswith(str(p.resolve())) is False
               for p in PROTECTED
               if resolved == p.resolve()):
            print(f"  SKIP (protected): {d}")
            continue

        if not d.exists():
            print(f"  SKIP (not found): {d}")
            continue

        # Calculate size before deletion
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        total_freed += size

        # List what will be deleted
        files = list(d.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        print(f"  DELETE: {d}  ({file_count} files, {size / 1024 / 1024:.1f} MB)")

        for f in sorted(files):
            if f.is_file():
                print(f"    - {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

        shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Recreated empty: {d}")

    print(f"\nTotal freed: {total_freed / 1024 / 1024:.1f} MB ({total_freed / 1024 / 1024 / 1024:.2f} GB)")


if __name__ == "__main__":
    print("=" * 50)
    print("  SAM3 M4 — Cleanup Training Artifacts")
    print("=" * 50)
    cleanup()
    print("\nDone. Ready for fresh training run.")
