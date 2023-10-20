from typing import Optional

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils.data

from src.models.resnet import get_resnet20
from src.dataloader.dataloader import *


# Model
def get_cifar10_model(use_cuda=False):
    return get_resnet20(use_cuda=use_cuda)


# Dataset
def get_cifar10_data(root_dir, train=True):
    cifar10_stats = {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    }
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
            ]
        )
    dataset = datasets.CIFAR10(root=root_dir,
                               train=train,
                               download=False,
                               transform=transform)
    return dataset


# Dataloader
def get_train_loader(root_dir, n_workers,  alpha=1.0, batch_size=32, noniid=False):
    dataset = get_cifar10_data(root_dir=root_dir, train=True)
    if not noniid:
        loader = iidLoader(size=n_workers, dataset=dataset, bsz=batch_size)
    else:
        loader = dirichletLoader(size=n_workers, dataset=dataset, alpha=alpha, bsz=batch_size)
    return loader


def get_test_loader(root_dir, batch_size):
    dataset = get_cifar10_data(root_dir=root_dir, train=False)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
