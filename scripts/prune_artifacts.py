#!/usr/bin/env python3
"""Prune large training artifacts to reclaim disk space (safe-by-default).

Default behavior is DRY-RUN: prints what would be deleted and how much space
you would reclaim. Use `--apply` to actually delete.

This is intended to be run AFTER you've confirmed the final student checkpoint
and any export artifacts you want to keep.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


RE_STEP = re.compile(r"_step(\d+)\.pt$")


@dataclass(frozen=True)
class Candidate:
    path: Path
    size_bytes: int


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n_f = n / 1024.0
        if n_f < 1024.0:
            return f"{n_f:.2f} {unit}"
        n = int(n_f)
    return f"{n} B"


def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def _parse_step(p: Path) -> int:
    m = RE_STEP.search(p.name)
    return int(m.group(1)) if m else -1


def _latest_n(paths: Iterable[Path], n: int) -> List[Path]:
    items = sorted(paths, key=lambda p: (_parse_step(p), p.name))
    return items[-n:] if n > 0 else []


def _collect_distillation_ckpts(ckpt_dir: Path) -> Tuple[List[Path], List[Path]]:
    p1 = sorted(ckpt_dir.glob("phase1_*.pt"))
    p2 = sorted(ckpt_dir.glob("phase2_*.pt"))
    return p1, p2


def _collect_video_ckpts(ckpt_dir: Path) -> List[Path]:
    return sorted(ckpt_dir.glob("video_*.pt"))


def _collect_vis_pngs(vis_dir: Path) -> List[Path]:
    return sorted(vis_dir.glob("*.png"))


def _collect_tensorboard_events(log_root: Path) -> List[Path]:
    # SummaryWriter typically writes events.* under logs/distillation/phase{n}/
    return sorted(log_root.rglob("events.*"))


def _collect_cache_pt(cache_dir: Path) -> List[Path]:
    return sorted(cache_dir.glob("*.pt"))


def _to_candidates(paths: Iterable[Path]) -> List[Candidate]:
    out: List[Candidate] = []
    for p in paths:
        if p.is_file():
            out.append(Candidate(path=p, size_bytes=_file_size(p)))
    return out


def _confirm_targets_exist(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Some target paths do not exist:\n  " + "\n  ".join(missing))


def _delete_files(files: List[Candidate], *, apply: bool) -> int:
    total = 0
    for c in files:
        total += c.size_bytes
        if apply:
            try:
                c.path.unlink()
            except FileNotFoundError:
                pass
    return total


def _delete_dir_contents(dir_path: Path, *, apply: bool) -> int:
    if not dir_path.exists():
        return 0
    total = sum(p.stat().st_size for p in dir_path.rglob("*") if p.is_file())
    if apply:
        for p in sorted(dir_path.iterdir()):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)  # py3.8+: ok
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune training artifacts to reclaim disk space")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    parser.add_argument("--keep-last-n", type=int, default=1, help="Keep last N checkpoints per group")

    parser.add_argument("--prune-distillation-ckpt", action="store_true", help="Prune checkpoints/distillation/*.pt")
    parser.add_argument("--prune-video-ckpt", action="store_true", help="Prune checkpoints/video_distillation/*.pt")
    parser.add_argument("--prune-vis", action="store_true", help="Delete logs/distillation/vis/*.png")
    parser.add_argument("--prune-tensorboard", action="store_true", help="Delete TensorBoard events.* under logs/distillation/")
    parser.add_argument("--prune-cache", action="store_true", help="Delete data/sa_v/cached_features/*.pt")

    parser.add_argument("--keep", action="append", default=[], help="Extra file path to always keep (repeatable)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    ckpt_distill = root / "checkpoints" / "distillation"
    ckpt_video = root / "checkpoints" / "video_distillation"
    vis_dir = root / "logs" / "distillation" / "vis"
    log_root = root / "logs" / "distillation"
    cache_dir = root / "data" / "sa_v" / "cached_features"

    apply = bool(args.apply)
    keep_last_n = max(int(args.keep_last_n), 0)

    keep_set = {str(Path(p).resolve()) for p in args.keep}

    planned_deletes: List[Candidate] = []
    planned_keep: List[Path] = []

    if args.prune_distillation_ckpt:
        p1, p2 = _collect_distillation_ckpts(ckpt_distill)
        keep_p1 = _latest_n(p1, keep_last_n)
        keep_p2 = _latest_n(p2, keep_last_n)
        planned_keep.extend(keep_p1 + keep_p2)

        del_p1 = [p for p in p1 if p not in keep_p1]
        del_p2 = [p for p in p2 if p not in keep_p2]
        planned_deletes.extend(_to_candidates(del_p1 + del_p2))

    if args.prune_video_ckpt:
        vids = _collect_video_ckpts(ckpt_video)
        keep_v = _latest_n(vids, keep_last_n)
        planned_keep.extend(keep_v)
        planned_deletes.extend(_to_candidates([p for p in vids if p not in keep_v]))

    if args.prune_vis:
        planned_deletes.extend(_to_candidates(_collect_vis_pngs(vis_dir)))

    if args.prune_tensorboard:
        planned_deletes.extend(_to_candidates(_collect_tensorboard_events(log_root)))

    if args.prune_cache:
        planned_deletes.extend(_to_candidates(_collect_cache_pt(cache_dir)))

    # Remove explicitly kept paths from deletion set.
    filtered: List[Candidate] = []
    for c in planned_deletes:
        if str(c.path.resolve()) in keep_set:
            planned_keep.append(c.path)
            continue
        filtered.append(c)
    planned_deletes = filtered

    # Print plan.
    print("=" * 72)
    print("Prune plan" + (" (APPLY)" if apply else " (DRY-RUN)"))
    print("=" * 72)

    if planned_keep:
        uniq_keep = []
        seen = set()
        for p in planned_keep:
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                uniq_keep.append(p)
        uniq_keep = sorted(uniq_keep)
        print("\nKEEP:")
        for p in uniq_keep:
            if p.exists() and p.is_file():
                print(f"  - {p}  ({_fmt_bytes(_file_size(p))})")
            else:
                print(f"  - {p}")

    if not planned_deletes:
        print("\nNothing selected for deletion. (Did you pass any --prune-* flags?)")
        return 0

    total_bytes = sum(c.size_bytes for c in planned_deletes)
    print("\nDELETE:")
    for c in sorted(planned_deletes, key=lambda x: (-x.size_bytes, str(x.path))):
        print(f"  - {c.path}  ({_fmt_bytes(c.size_bytes)})")

    print("\nReclaimable:", _fmt_bytes(total_bytes))

    if not apply:
        print("\nDry-run complete. Re-run with --apply to delete.")
        return 0

    # Execute deletions.
    freed = _delete_files(planned_deletes, apply=True)
    print("\nDeleted. Freed:", _fmt_bytes(freed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

