"""
Merge Phase 4 CSV logs from multiple Kaggle export folders into one directory.

Scans each immediate child of --source (e.g. phase_4_logs/zip/artifacts_*)
for a ``logs/`` subfolder, copies every ``*.csv`` into --dest, and skips files
whose basename already exists (first copy wins).

Usage
─────
  python experiments/merge_phase4_logs.py
  python experiments/merge_phase4_logs.py --dry-run
  python experiments/merge_phase4_logs.py --dest logs/my_merged

Then run analysis:

  python experiments/analyze_results.py --log-dir ./logs/phase4_kaggle_merged --tta-target 40.0 \\
      --output docs/phase4_results_tables.md
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _iter_artifact_log_dirs(source_root: str) -> List[Tuple[str, str]]:
    """Return list of (artifact_name, logs_path) for each .../artifact/logs directory."""
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(source_root):
        return out
    for name in sorted(os.listdir(source_root)):
        base = os.path.join(source_root, name)
        if not os.path.isdir(base):
            continue
        logs_path = os.path.join(base, "logs")
        if os.path.isdir(logs_path):
            out.append((name, logs_path))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Phase 4 logs from Kaggle zip exports into one folder."
    )
    parser.add_argument(
        "--source",
        default="phase_4_logs/zip",
        help="Root containing artifact folders (each may contain logs/). "
        "Default: phase_4_logs/zip (relative to project root).",
    )
    parser.add_argument(
        "--dest",
        default="logs/phase4_kaggle_merged",
        help="Destination directory for merged CSVs (relative to project root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying files.",
    )
    args = parser.parse_args()

    source_root = os.path.join(_PROJECT_ROOT, args.source)
    dest_dir = os.path.join(_PROJECT_ROOT, args.dest)

    pairs = _iter_artifact_log_dirs(source_root)
    if not pairs:
        print(f"No .../logs directories found under: {source_root}")
        sys.exit(1)

    copied = 0
    skipped_dup = 0
    skipped_non_csv = 0

    if not args.dry_run:
        os.makedirs(dest_dir, exist_ok=True)

    print(f"Source: {source_root}")
    print(f"Destination: {dest_dir}")
    print(f"Artifact folders with logs: {len(pairs)}\n")

    for artifact_name, logs_path in pairs:
        for fname in sorted(os.listdir(logs_path)):
            src = os.path.join(logs_path, fname)
            if not os.path.isfile(src):
                continue
            if not fname.lower().endswith(".csv"):
                skipped_non_csv += 1
                continue
            dst = os.path.join(dest_dir, fname)
            if os.path.exists(dst):
                print(f"  SKIP (exists): {fname}  [from {artifact_name} skipped]")
                skipped_dup += 1
                continue
            if args.dry_run:
                print(f"  COPY: {fname}  <= {artifact_name}/logs/")
                copied += 1
            else:
                shutil.copy2(src, dst)
                print(f"  COPY: {fname}  <= {artifact_name}/logs/")
                copied += 1

    print()
    print(f"Copied: {copied}")
    print(f"Skipped (duplicate basename): {skipped_dup}")
    if skipped_non_csv:
        print(f"Ignored (non-.csv in logs): {skipped_non_csv}")

    if args.dry_run:
        print("\nDry-run only; no files written.")
    else:
        print(f"\nMerged logs ready at: {dest_dir}")
        print(
            "Next: python experiments/analyze_results.py "
            f'--log-dir ./{args.dest} --tta-target 40.0 --output docs/phase4_results_tables.md'
        )


if __name__ == "__main__":
    main()
