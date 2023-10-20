import torch


class BaseServer(object):
    def __init__(self, model, aggregator, optimizer):
        self.model = model
        self.aggregator = aggregator
        self.optimizer = optimizer

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        It should be called before assigning pseudo-gradient.

        Args:
            set_to_none: See `Pytorch documentation <https://pytorch.org/docs/s
            table/generated/torch.optim.Optimizer.zero_grad.html>`_.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_model(self):
        return self.model

    def global_update(self, gradients):
        gradient = self.aggregator(gradients)
        self.zero_grad()
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # for p in self.model.parameters():
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                p.grad.data = x.clone().detach()
                beg = end
        self.optimizer.step()
