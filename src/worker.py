import logging
from collections import defaultdict
from typing import Optional, Callable

import torch


class BaseWorker(object):
    _is_byzantine: bool = False

    def __init__(
            self,
            id,
            model,
            data_loader,
            local_round: Optional[int] = 1,
            optimizer: Optional[torch.optim.Optimizer] = torch.optim.SGD,
            device: Optional[torch.device] = torch.device("cpu"),
            loss_func=torch.nn.CrossEntropyLoss(),

    ):
        self._id = id
        self._state = defaultdict(dict)
        self.data_loader = data_loader
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        self.device = device
        self.loss_func = loss_func
        self.running = {}
        self.metrics = {}
        self.local_round = local_round
        self.optimizer = optimizer
        self.model = model
        self._is_trusted: bool = False

    def id(self):
        return self._id

    def is_byzantine(self):
        r"""Return a boolean value specifying if the client is Byzantine."""
        return self._is_byzantine

    def is_trusted(self):
        return self._is_trusted

    def trust(self, trusted=True) -> None:
        r"""Trusts the client as an honest participant. This property is useful
        for trust-based algorithms.

        Args:
            trusted: Boolean; whether the client is trusted; default to True.
        """
        self._is_trusted = trusted

    def is_train(self):
        return True

    def add_metric(
            self, name: str, callback: Callable[[torch.Tensor, torch.Tensor], float]
    ):
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    def train_epoch_start(self) -> None:
        self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model.train()

    def on_train_batch_begin(self, data, target):
        """Called at the beginning of a training batch in `train_global_model`
        methods.

        Subclasses should override for any actions to run.

        Args:
            data: input of the batch data.
            target: target of the batch data.
        """
        return data, target

    def compute_gradient(self):
        # self.model.to(self.device)
        results = {}
        data, target = self.running["train_loader_iterator"].__next__()
        data, target = data.to(self.device), target.to(self.device)
        data, target = self.on_train_batch_begin(data=data, target=target)
        self.optimizer.zero_grad()
        output = self.model(data)
        # Clamp loss value to avoid possible 'Nan' gradient with some
        # attack types.
        loss = torch.clamp(self.loss_func(output, target), 0, 1e6)
        loss.backward()
        self._save_grad()

        self.running["data"] = data
        self.running["target"] = target

        results["loss"] = loss.item()
        results["length"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)
        return results

    def get_gradient(self) -> torch.Tensor:
        return self._get_saved_grad()

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self._state[p]
                param_state["saved_grad"] = torch.clone(p.grad).detach()

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self._state[p]
                layer_gradients.append(param_state["saved_grad"].data.view(-1))
        return torch.cat(layer_gradients)

    def get_data_size(self) -> int:
        return len(self.data_loader.dataset)

    def __str__(self):
        return f"BaseWorker"


class MomentumWorker(BaseWorker):
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self._state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                else:
                    param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self._state[p]
                layer_gradients.append(param_state["momentum_buffer"].data.view(-1))
        return torch.cat(layer_gradients)


class ByzantineWorker(BaseWorker):
    _is_byzantine: bool = True

    # def __init__(self, *args, **kwargs):
    #     super(ByzantineWorker).__init__(*args, **kwargs)

    def configure(self, simulator):
        # call configure after defining DistribtuedSimulator
        self.simulator = simulator
        simulator.register_omniscient_callback(self.omniscient_callback)

    def omniscient_callback(self):
        pass

    def __str__(self):
        return f"ByzantineWorker"
