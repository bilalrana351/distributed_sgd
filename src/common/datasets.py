"""
CIFAR-10 / CIFAR-100 data loaders with optional DistributedSampler support.

Usage (distributed):
    train_loader, val_loader, train_sampler = get_dataloaders(
        dataset_name="cifar10",
        batch_size=128,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    # Must call train_sampler.set_epoch(epoch) at the start of every epoch
    # so shuffling is different across epochs.

Usage (single process):
    train_loader, val_loader, _ = get_dataloaders("cifar10", batch_size=128)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Per-channel mean and std computed on the full training split
_CIFAR_STATS: dict[str, dict] = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
        "cls": torchvision.datasets.CIFAR10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "cls": torchvision.datasets.CIFAR100,
    },
}


def _get_transforms(dataset_name: str, train: bool) -> T.Compose:
    stats = _CIFAR_STATS[dataset_name]
    normalize = T.Normalize(stats["mean"], stats["std"])
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([T.ToTensor(), normalize])


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    data_root: str = "./data",
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """
    Returns (train_loader, val_loader, train_sampler).

    train_sampler is None when not distributed.
    When distributed, caller must do: train_sampler.set_epoch(epoch) each epoch.
    """
    name = dataset_name.lower()
    if name not in _CIFAR_STATS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(_CIFAR_STATS)}")

    DatasetCls = _CIFAR_STATS[name]["cls"]

    train_ds = DatasetCls(root=data_root, train=True,  download=True, transform=_get_transforms(name, train=True))
    val_ds   = DatasetCls(root=data_root, train=False, download=True, transform=_get_transforms(name, train=False))

    train_sampler: Optional[DistributedSampler] = None
    val_sampler:   Optional[DistributedSampler] = None

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True,  drop_last=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),   # mutually exclusive with sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,                    # keeps consistent batch sizes across ranks
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,         # no grad → can use larger batch
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, train_sampler


def get_num_classes(dataset_name: str) -> int:
    """Returns 10 for CIFAR-10, 100 for CIFAR-100."""
    return _CIFAR_STATS[dataset_name.lower()]["num_classes"]
