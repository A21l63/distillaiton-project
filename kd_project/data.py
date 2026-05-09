from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_transforms(train: bool = True):
    """
    Return transforms for CIFAR-10.

    For train:
        basic augmentation may be used.
    For test:
        deterministic transforms.

    Keep this simple.
    """
    # TODO: define torchvision transforms for train/test
    ...


def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for CIFAR-10.

    Returns:
        train_loader, test_loader
    """
    # TODO: create CIFAR-10 train dataset
    # TODO: create CIFAR-10 test dataset
    # TODO: create DataLoader objects
    ...
