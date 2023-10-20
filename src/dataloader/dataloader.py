from __future__ import print_function

import math

import numpy as np
import torch
from collections import defaultdict
np.random.seed(0)


class Partition(torch.utils.data.Dataset):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.classes = 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        data_idx = self.index[i]
        return self.data[data_idx]


class customDataLoader():
    """ Virtual class: load a particular partition of dataset"""

    def __init__(self, size, dataset, bsz):
        '''
        size: number of paritions in the loader
        dataset: pytorch dataset
        bsz: batch size of the data loader
        '''
        self.size = size
        self.dataset = dataset
        self.classes = np.unique(dataset.targets).tolist()
        self.stats = defaultdict(list)
        self.bsz = bsz
        self.partition_list = self.getPartitions()
        # print(self.stats)
        num_unique_items = len(np.unique(np.concatenate(self.partition_list)))
        if len(dataset) != num_unique_items:
            print(
                f"Number of unique items in partitions ({num_unique_items}) is not equal to the size of dataset ({len(dataset)}), some data may not be included")

    def getPartitions(self):
        raise NotImplementedError()

    def __len__(self):
        return self.size

    def __getitem__(self, rank):
        assert rank < self.size, 'partition index should be smaller than the size of the partition'
        partition = Partition(self.dataset, self.partition_list[rank])
        partition.classes = self.classes
        train_set = torch.utils.data.DataLoader(partition, batch_size=int(self.bsz), shuffle=True,
                                                )  # drop last since some network requires batchnorm
        return train_set


class iidLoader(customDataLoader):
    def __init__(self, size, dataset, bsz=128):
        super(iidLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        data_len = len(self.dataset)
        indexes = [x for x in range(0, data_len)]
        np.random.shuffle(indexes)
        # fractions of data in each partition
        partition_sizes = [1.0 / self.size for _ in range(self.size)]

        partition_list = []
        for frac in partition_sizes:
            part_len = int(frac * data_len)
            partition_list.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        return partition_list


class byLabelLoader(customDataLoader):
    def __init__(self, size, dataset, bsz=128):
        super(byLabelLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        data_len = len(self.dataset)

        partition_list = []
        self.labels = np.unique(self.dataset.targets).tolist()
        label = self.dataset.targets
        label = torch.tensor(np.array(label))
        for i in self.labels:
            label_iloc = (label == i).nonzero(as_tuple=False).squeeze().tolist()
            partition_list.append(label_iloc)
        return partition_list


class dirichletLoader(customDataLoader):
    def __init__(self, size, dataset, alpha=0.9, bsz=128):
        # alpha is used in getPartition,
        # and getPartition is used in parent constructor
        # hence need to initialize alpha first
        self.alpha = alpha
        super(dirichletLoader, self).__init__(size, dataset, bsz)

    # def getPartitions(self):
    #     data_len = len(self.dataset)
    #
    #     partition_list = [[] for j in range(self.size)]
    #     self.labels = np.unique(self.dataset.targets).tolist()
    #     label = self.dataset.targets
    #     label = torch.tensor(np.array(label))
    #     # for reproducibility
    #     np.random.seed(1)
    #     for i in self.labels:
    #         label_iloc = (label == i).nonzero(as_tuple=False).squeeze().numpy()
    #         np.random.shuffle(label_iloc)
    #         p = np.random.dirichlet([self.alpha] * self.size)
    #         # choose which partition a data is assigned to
    #         assignment = np.random.choice(range(self.size), size=len(label_iloc), p=p.tolist())
    #         part_list = [(label_iloc[(assignment == k)]).tolist() for k in range(self.size)]
    #         for j in range(self.size):
    #             partition_list[j] += part_list[j]
    #             self.stats[i].append(len(part_list[j]))
    #     return partition_list
    def getPartitions(self):
        data_per_device = math.floor(len(self.dataset)/self.size)
        users_idxs = [[] for i in range(self.size)]  # Index dictionary for devices

        # Combine indexes and labels for sorting
        idxs_labels = np.vstack((np.arange(len(self.dataset)), np.array(self.dataset.targets)))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :].tolist()

        niid_data_per_device = int(data_per_device*self.alpha)

        # Distribute non-IID data
        for i in range(self.size):
            users_idxs[i] = idxs[i*niid_data_per_device:(i+1)*niid_data_per_device]

        # Still have some data
        if self.size*niid_data_per_device < len(self.dataset):
            # Filter distributed data
            idxs = idxs[self.size*niid_data_per_device:]
            # Randomize data after sorting
            np.random.shuffle(idxs)

            remaining_data_per_device = data_per_device-niid_data_per_device

            # Distribute IID data
            for i in range(self.size):
                users_idxs[i].extend(idxs[i*remaining_data_per_device:(i+1)*remaining_data_per_device])
        return users_idxs


if __name__ == '__main__':
    from torchvision import datasets, transforms

    dataset = datasets.MNIST('../../datasets/mnist',
                             train=True,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    # loader = iidLoader(10, dataset)
    # print(f"\nInitialized {len(loader)} loaders, each with batch size {loader.bsz}.\
    # \nThe size of dataset in each loader are:")
    # print([len(loader[i].dataset) for i in range(len(loader))])
    # print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")
    #
    # loader = byLabelLoader(10, dataset)
    # print(f"\nInitialized {len(loader)} loaders, each with batch size {loader.bsz}.\
    # \nThe size of dataset in each loader are:")
    # print([len(loader[i].dataset) for i in range(len(loader))])
    # print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    loader = dirichletLoader(50, dataset, alpha=1.0)
    print(f"\nInitialized {len(loader)} loaders, each with batch size {loader.bsz}.\
    \nThe size of dataset in each loader are:")
    print([len(loader[i].dataset) for i in range(len(loader))])
    print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")


