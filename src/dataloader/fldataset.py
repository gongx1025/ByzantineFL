import logging
import os
import pickle
from abc import ABC
from functools import partial
from typing import Optional, Generator
import numpy as np
import torch
from sklearn.utils import shuffle
from .customdataset import CustomTensorDataset
from ..utils import set_random_seed

logger = logging.getLogger(__name__)


class FLDataset(ABC):
    def __init__(
            self,
            train_set=None,
            test_set=None,
            data_root: str = "./datasets",
            iid: Optional[bool] = True,
            alpha: Optional[float] = 1.0,
            num_clients: Optional[int] = 50,
            num_classes: Optional[int] = 10,
            seed=0,
            train_data=None,
            test_data=None,
            train_bs: Optional[int] = 32,
            train_transform=None,
            test_transform=None,
    ):
        self.train_transform = train_transform
        self.test_transform = test_transform
        if train_data:
            self.train_data = train_data
            self.test_data = test_data
            self.train_bs = train_bs
            self._preprocess()
            return

        self.num_classes = num_classes
        self.train_bs = train_bs

        train_user_ids, train_dataset = self._generate_datasets(
            train_set,
            iid=iid,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
        )

        self.max_batchs = len(train_dataset["0"]["y"]) // train_bs
        self._preprocess(train_user_ids, train_dataset, test_set)

    def __reduce__(self):
        deserializer = FLDataset
        return (
            partial(
                deserializer,
                train_data=self.train_data,
                test_data=self.test_data,
                train_bs=self.train_bs,
                train_transform=self.train_transform,
                test_transform=self.test_transform,
            ),
            (),
        )

    def _generate_datasets(
            self, train_set, iid=True, alpha=1.0, num_clients=50, seed=0
    ):
        x_train, y_train = train_set

        x_train = x_train.astype("float32") / 255.0
        np.random.seed(seed)
        x_train, y_train = shuffle(x_train, y_train)

        train_user_ids = [str(id) for id in range(num_clients)]

        if iid:
            x_train_splits = np.split(x_train, num_clients)
            y_train_splits = np.split(y_train, num_clients)
        else:
            print("generating non-iid data")
            min_size = 0
            # self.num_classes = 10
            N = y_train.shape[0]
            client_dataidx_map = {}

            while min_size < 10:
                proportion_list = []
                idx_batch = [[] for _ in range(num_clients)]
                for k in range(self.num_classes):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

                    proportions = np.array(
                        [
                            p * (len(idx_j) < N / num_clients)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    proportion_list.append(proportions)
            x_train_splits, y_train_splits = [], []
            for j in range(num_clients):
                np.random.shuffle(idx_batch[j])
                client_dataidx_map[j] = idx_batch[j]
                x_train_splits.append(x_train[idx_batch[j], :])
                y_train_splits.append(y_train[idx_batch[j]])

        train_dataset = {}
        for id, index in zip(train_user_ids, range(num_clients)):
            train_dataset[id] = {
                "x": x_train_splits[index],
                "y": y_train_splits[index].flatten(),
            }
        return train_user_ids, train_dataset

    def _preprocess_train_data(
            self, data, labels, batch_size, seed=0
    ) -> (torch.Tensor, torch.LongTensor):
        i = 0
        # The following line is needed for reproducing the randomness of transforms.
        set_random_seed(seed)

        idx = np.random.permutation(len(labels))
        data, labels = data[idx], labels[idx]

        while True:
            if i * batch_size >= len(labels):
                i = 0
                idx = np.random.permutation(len(labels))
                data, labels = data[idx], labels[idx]

                continue
            else:
                X = data[i * batch_size: (i + 1) * batch_size, :]
                y = labels[i * batch_size: (i + 1) * batch_size]
                i += 1
                X = torch.Tensor(X)
                if self.train_transform:
                    X = self.train_transform(X)
                yield X, torch.LongTensor(y)

    def _preprocess_test_data(
            self,
            data,
            labels,
    ) -> CustomTensorDataset:
        tensor_x = torch.Tensor(data)  # transform to torch tensor
        tensor_y = torch.LongTensor(labels)
        return CustomTensorDataset(
            tensor_x, tensor_y, transform_list=self.test_transform
        )

    def _preprocess(self, train_user_ids, train_dataset, test_set):
        self._train_dls = {}
        self._test_dls = {}
        for idx, u_id in enumerate(train_user_ids):
            self._train_dls[u_id] = self._preprocess_train_data(
                data=np.array(train_dataset[u_id]["x"]),
                labels=np.array(train_dataset[u_id]["y"]),
                batch_size=self.train_bs,
            )

        self.test_dataset = self._preprocess_test_data(test_set[0], test_set[1])

    def get_clients(self):
        return self.train_data.keys()

    def get_train_loader(self, u_id: str) -> Generator:
        """
        Get the local dataset of given user `id`.
        Args:
            u_id (str): user id.

        Returns: the `generator` of dataset for the given `u_id`.
        """
        return self._train_dls[u_id]

    def get_train_data(self, u_id, num_batches):
        data = [next(self._train_dls[u_id]) for _ in range(num_batches)]
        return data

    def get_test_dataloader(self, test_batch_size=128, shuffle=None, drop_last=True, **kwargs):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs
        )

