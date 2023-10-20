from typing import Optional

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.utils.data
from torch import nn
import torch.nn.functional as F

from src.dataloader.dataloader import iidLoader, dirichletLoader


class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Model
def get_fmnist_model():
    return CNNMNIST()


# Dataset
def get_fmnist_data(root_dir, train=True):
    fmnist_stats = {
        "mean": (0.1307,),
        "std": (0.3081,),
    }
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(fmnist_stats['mean'], fmnist_stats['std']),
        ]
    )
    dataset = datasets.FashionMNIST(root=root_dir,
                                    train=train,
                                    download=False,
                                    transform=transform)
    return dataset


# Dataloader
def get_train_loader(root_dir, n_workers, alpha=1.0, batch_size=32, noniid=False):
    dataset = get_fmnist_data(root_dir=root_dir, train=True)
    if not noniid:
        loader = iidLoader(size=n_workers, dataset=dataset, bsz=batch_size)
    else:
        loader = dirichletLoader(size=n_workers, dataset=dataset, alpha=alpha, bsz=batch_size)
    return loader


def get_test_loader(root_dir, batch_size):
    dataset = get_fmnist_data(root_dir=root_dir, train=False)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
