"""
Logging utilities shared across all training phases.

CSVLogger   – appends one row per epoch to a CSV file (rank-0 only).
setup_logger – returns a stdlib Logger with rank-aware formatting.

CSV schema (Phase 1):
    epoch, train_loss, val_loss, val_acc_top1, val_acc_top5,
    epoch_time_seconds, throughput_imgs_per_sec, world_size, batch_size
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
#  Field names that every Phase-1 CSV row must contain                        #
# --------------------------------------------------------------------------- #

CSV_FIELDNAMES: List[str] = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc_top1",
    "val_acc_top5",
    "epoch_time_seconds",
    "throughput_imgs_per_sec",
    "world_size",
    "batch_size",
]


# --------------------------------------------------------------------------- #
#  Console logger                                                              #
# --------------------------------------------------------------------------- #

def setup_logger(rank: int, name: str = "ddp_train") -> logging.Logger:
    """
    Returns a stdlib Logger for the given process rank.

    Only rank-0 logs at INFO level; all other ranks log only WARNINGS and
    above so the console is not flooded in large-scale runs.
    """
    logger = logging.getLogger(f"{name}.rank{rank}")

    # Avoid adding duplicate handlers if this is called more than once
    if not logger.handlers:
        handler   = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=f"[R{rank}] %(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    return logger


# --------------------------------------------------------------------------- #
#  CSV logger                                                                  #
# --------------------------------------------------------------------------- #

class CSVLogger:
    """
    Thread-safe (single-process) CSV logger.

    Opens (or creates) a CSV file and writes one row per call to `.log()`.
    The header is written only when the file is first created.

    Example:
        logger = CSVLogger("logs/run.csv", CSV_FIELDNAMES)
        logger.log({"epoch": 1, "train_loss": 1.23, ...})
    """

    def __init__(self, filepath: str, fieldnames: List[str]) -> None:
        self.filepath   = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Write header on fresh file
        file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
        mode = "a" if file_exists else "w"
        with open(filepath, mode, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """Append one row.  Missing keys are written as empty strings."""
        with open(self.filepath, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)
