"""
Build pdc_project_kaggle.zip for Kaggle upload.

Kaggle bootstrap expects extract → /kaggle/working/pdc_project/{src,configs,...}.
PowerShell Compress-Archive without a parent folder breaks that (flat src/ at zip root).

Usage (from repo root):
  python experiments/package_kaggle_zip.py
"""

from __future__ import annotations

import os
import zipfile

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ZIP_NAME = "pdc_project_kaggle.zip"
PREFIX = "pdc_project"


def _add_tree(z: zipfile.ZipFile, abs_dir: str) -> None:
    for root, dirs, files in os.walk(abs_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.endswith(".egg-info")]
        for fname in files:
            if fname.endswith(".pyc"):
                continue
            path = os.path.join(root, fname)
            arc = os.path.join(PREFIX, os.path.relpath(path, _PROJECT_ROOT))
            z.write(path, arc.replace("\\", "/"))


def main() -> None:
    out_path = os.path.join(_PROJECT_ROOT, ZIP_NAME)
    if os.path.isfile(out_path):
        os.remove(out_path)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("src", "configs", "experiments"):
            d = os.path.join(_PROJECT_ROOT, name)
            if not os.path.isdir(d):
                raise FileNotFoundError(d)
            _add_tree(z, d)
        req = os.path.join(_PROJECT_ROOT, "requirements.txt")
        z.write(req, f"{PREFIX}/requirements.txt")

    n = os.path.getsize(out_path)
    print(f"Wrote {out_path} ({n // 1024} KB)")
    print("Upload this file to Kaggle; extract cell should find .../pdc_project/src/...")


if __name__ == "__main__":
    main()
