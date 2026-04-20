"""
Build a flat `logs/phase4_merged/` for `analyze_results.py`:

- From `phase_4_logs/zip/**/logs/`: all `p4_*_sync_*` and `p4_*_async_*` CSVs
  (s0/sA/sB, all seeds). Omits sweep tags and omits hybrid rows from old exports.
- From `threshold_folders/**/logs/` (recursive): `p4_sA_hybrid_*` and `p4_sB_hybrid_*`
  only (post-fix Kaggle hybrid). Does not copy `p4_s0_hybrid_*` or `sweep_*`.

Later copies win for the same basename (so re-run is deterministic if sources change).

Usage
─────
  python experiments/merge_phase4_compare.py
  python experiments/merge_phase4_compare.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PREFIX = "hybrid_resnet18_cifar10_nw2_"
KIND_SUFFIXES = ("_ps.csv", "_staleness.csv", "_switches.csv", "_worker0.csv", "_worker1.csv")

RE_P4_SYNC_ASYNC = re.compile(r"^p4_s[0AB]_(sync|async)_s\d+$")
RE_P4_AB_HYBRID = re.compile(r"^p4_s[AB]_hybrid_s\d+$")

# Prefer logs from the artifact folder that matches the scenario (avoids stray duplicates,
# e.g. `artifacts_s0_seed42/logs` also containing `p4_sB_*` copies).
_SCENARIO_PATH_HINTS = {
    "s0": ("artifacts_s0_", "chunkc", "artifacts_after_chunkc"),
    "sA": ("artifacts_sa", "chunka", "artifacts_after_chunka"),
    "sB": ("artifacts_sb", "chunkb", "artifacts_after_chunkb"),
}


def _scenario_from_tag(tag: str) -> str:
    parts = tag.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "p4" else ""


def _source_path_score(tag: str, src_path: str) -> int:
    sc = _scenario_from_tag(tag)
    hints = _SCENARIO_PATH_HINTS.get(sc, ())
    p = src_path.replace("\\", "/").lower()
    return sum(1 for h in hints if h.lower() in p)


def _pick_best_source(sources: List[str], tag: str) -> str:
    if len(sources) == 1:
        return sources[0]
    ranked = sorted(
        sources,
        key=lambda sp: (_source_path_score(tag, sp), sp),
        reverse=True,
    )
    return ranked[0]


def _tag_from_basename(fname: str) -> Optional[str]:
    if not fname.startswith(PREFIX) or not fname.lower().endswith(".csv"):
        return None
    for suf in KIND_SUFFIXES:
        if fname.endswith(suf):
            return fname[len(PREFIX) : -len(suf)]
    return None


def _iter_csv_files(root: str):
    if not os.path.isdir(root):
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                yield os.path.join(dirpath, fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Phase 4 logs for scenario comparison table.")
    parser.add_argument(
        "--old-root",
        default="phase_4_logs/zip",
        help="Root to walk for legacy sync/async logs (default: phase_4_logs/zip).",
    )
    parser.add_argument(
        "--new-root",
        default="threshold_folders",
        help="Root to walk for new sA/sB hybrid logs (default: threshold_folders).",
    )
    parser.add_argument(
        "--dest",
        default="logs/phase4_merged",
        help="Flat destination relative to project root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    old_root = os.path.join(_PROJECT_ROOT, args.old_root)
    new_root = os.path.join(_PROJECT_ROOT, args.new_root)
    dest_dir = os.path.join(_PROJECT_ROOT, args.dest)

    if not args.dry_run:
        os.makedirs(dest_dir, exist_ok=True)

    by_basename: Dict[str, List[Tuple[str, str, str]]] = {}

    def _add_plan(source_label: str, tag: str, src: str) -> None:
        fn = os.path.basename(src)
        by_basename.setdefault(fn, []).append((source_label, tag, src))

    for src in _iter_csv_files(old_root):
        fn = os.path.basename(src)
        tag = _tag_from_basename(fn)
        if tag is None or not RE_P4_SYNC_ASYNC.match(tag):
            continue
        _add_plan("old", tag, src)

    for src in _iter_csv_files(new_root):
        fn = os.path.basename(src)
        tag = _tag_from_basename(fn)
        if tag is None or not RE_P4_AB_HYBRID.match(tag):
            continue
        _add_plan("new", tag, src)

    resolved: List[Tuple[str, str, str]] = []
    for fn, entries in sorted(by_basename.items()):
        tag = entries[0][1]
        src = _pick_best_source([e[2] for e in entries], tag)
        label = next(e[0] for e in entries if e[2] == src)
        resolved.append((label, tag, src))

    copied = 0
    print(f"Destination: {dest_dir}")
    print(f"Planned copies: {len(resolved)} (after de-duplicating basenames)\n")

    for source_label, tag, src in resolved:
        dst = os.path.join(dest_dir, os.path.basename(src))
        if args.dry_run:
            print(f"  [{source_label}] {tag}: {os.path.basename(src)}")
            copied += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

    if args.dry_run:
        print(f"\nDry-run: {copied} file(s); no writes.")
    else:
        print(f"\nCopied {copied} file(s) to {dest_dir}")
        print(
            "Next: python experiments/analyze_results.py "
            f"--log-dir {args.dest} --tta-target 40.0 --output docs/phase4_results_tables.md"
        )


if __name__ == "__main__":
    main()
