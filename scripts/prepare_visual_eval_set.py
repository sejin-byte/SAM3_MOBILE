#!/usr/bin/env python3
"""Prepare a diverse image manifest for visual QA (CPU-only).

Selects SA-1B images by object-count buckets so visual checks include:
- sparse scenes
- medium complexity scenes
- dense scenes

Output is a plain text manifest (one relative image path per line).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def evenly_pick(items: List[Tuple[Path, int, float]], n: int) -> List[Tuple[Path, int, float]]:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return items
    if n == 1:
        return [items[len(items) // 2]]

    step = (len(items) - 1) / float(n - 1)
    out = []
    for i in range(n):
        idx = int(round(i * step))
        out.append(items[idx])
    return out


def bucket_name(num_objects: int) -> str:
    if num_objects <= 5:
        return "sparse"
    if num_objects <= 20:
        return "medium"
    return "dense"


def resolve_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare visual-eval image manifest")
    parser.add_argument("--sa1b-dir", default="data/sa1b", help="SA-1B root directory")
    parser.add_argument("--output", default="configs/visual_eval/image_manifest_phase2.txt")
    parser.add_argument("--num-images", type=int, default=24, help="Number of SA-1B images to select")
    parser.add_argument("--scan-limit", type=int, default=3000,
                        help="Max JSON files to inspect (subsampled evenly)")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Include outputs/baseline/test_image.png as first entry")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sa1b_dir = (repo_root / args.sa1b_dir).resolve()
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_json = sorted(sa1b_dir.glob("sa_*.json"))
    if not all_json:
        raise SystemExit(f"No SA-1B json files found in {sa1b_dir}")

    if args.scan_limit > 0 and len(all_json) > args.scan_limit:
        stride = max(len(all_json) // args.scan_limit, 1)
        scan_list = all_json[::stride][:args.scan_limit]
    else:
        scan_list = all_json

    sparse: List[Tuple[Path, int, float]] = []
    medium: List[Tuple[Path, int, float]] = []
    dense: List[Tuple[Path, int, float]] = []

    for json_path in scan_list:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        anns = data.get("annotations", [])
        num_obj = len(anns)
        area_sum = 0.0
        for ann in anns:
            area = ann.get("area", 0.0)
            if isinstance(area, (int, float)):
                area_sum += float(max(area, 0.0))

        jpg = json_path.with_suffix(".jpg")
        if not jpg.exists():
            continue

        item = (jpg, num_obj, area_sum)
        b = bucket_name(num_obj)
        if b == "sparse":
            sparse.append(item)
        elif b == "medium":
            medium.append(item)
        else:
            dense.append(item)

    # Sort inside buckets for stable diverse sampling.
    sparse.sort(key=lambda x: (x[1], x[2], x[0].name))
    medium.sort(key=lambda x: (x[1], x[2], x[0].name))
    dense.sort(key=lambda x: (x[1], x[2], x[0].name))

    n_total = max(int(args.num_images), 3)
    n_sparse = max(n_total // 3, 1)
    n_medium = max(n_total // 3, 1)
    n_dense = max(n_total - n_sparse - n_medium, 1)

    picked: List[Tuple[Path, int, float]] = []
    picked.extend(evenly_pick(sparse, n_sparse))
    picked.extend(evenly_pick(medium, n_medium))
    picked.extend(evenly_pick(dense, n_dense))

    # If any bucket lacked enough samples, top-up from remaining pool.
    if len(picked) < n_total:
        pool = sparse + medium + dense
        seen = {p[0] for p in picked}
        for item in pool:
            if item[0] in seen:
                continue
            picked.append(item)
            seen.add(item[0])
            if len(picked) >= n_total:
                break

    # Keep deterministic output order: sparse -> medium -> dense by filename.
    picked.sort(key=lambda x: (bucket_name(x[1]), x[0].name))

    lines: List[str] = []
    lines.append("# Visual Eval Image Manifest")
    lines.append(f"# source_sa1b_dir: {resolve_relative(sa1b_dir, repo_root)}")
    lines.append(f"# scanned_json: {len(scan_list)} / {len(all_json)}")
    lines.append(f"# selected_images: {len(picked)}")
    lines.append("# format: relative_image_path")

    if args.include_baseline:
        baseline = repo_root / "outputs" / "baseline" / "test_image.png"
        if baseline.exists():
            lines.append(resolve_relative(baseline, repo_root))

    for jpg, num_obj, area_sum in picked:
        lines.append(resolve_relative(jpg, repo_root))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote_manifest={resolve_relative(output_path, repo_root)}")
    print(f"selected_images={len(picked)}")
    print(f"bucket_counts sparse={len(sparse)} medium={len(medium)} dense={len(dense)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

