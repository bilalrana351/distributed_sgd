"""
Lightweight metric helpers used by every training phase.

AverageMeter  – running mean of a scalar (loss, accuracy, time, …)
compute_accuracy – vectorised top-k accuracy from logits
compute_throughput – images/sec across the whole distributed group
"""

from __future__ import annotations

from typing import List, Tuple

import torch


class AverageMeter:
    """
    Tracks the running mean of a single scalar value.

    Example:
        loss_meter = AverageMeter("loss")
        for batch in loader:
            loss = criterion(model(x), y)
            loss_meter.update(loss.item(), n=x.size(0))
        print(loss_meter.avg)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: avg={self.avg:.4f}, n={self.count})"


def compute_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> List[float]:
    """
    Compute top-k classification accuracy (%) for each k in `topk`.

    Args:
        output: Raw logits of shape (N, C).
        target: Ground-truth class indices of shape (N,).
        topk:   Which top-k values to compute. e.g. (1, 5).

    Returns:
        List of float percentages, one per k value.

    Note:
        For CIFAR-10 (C=10), top-5 is always ≈ 100 % and is only included
        for logging consistency with CIFAR-100 runs.
    """
    with torch.no_grad():
        maxk      = max(topk)
        batch_sz  = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
        pred    = pred.t()                                               # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))           # (maxk, N)

        results: List[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100.0 / batch_sz).item())

        return results


def compute_throughput(
    batch_size: int,
    world_size: int,
    epoch_time_seconds: float,
    num_batches: int,
) -> float:
    """
    Total images processed per second across **all** workers in one epoch.

    Formula:
        throughput = (batch_size × world_size × num_batches) / epoch_time
    """
    if epoch_time_seconds <= 0:
        return 0.0
    total_images = batch_size * world_size * num_batches
    return total_images / epoch_time_seconds
