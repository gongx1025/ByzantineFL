import torch

from src.worker import ByzantineWorker

'''
    dev_type ='std' 
    threshold=10.0,
    threshold_diff=1e-5,
'''


class MinMaxWorker(ByzantineWorker):
    def __init__(
            self,
            num_byzantine: int,
            dev_type="std",
            threshold=10.0,
            threshold_diff=1e-5,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dev_type = dev_type
        self.threshold = threshold
        self.threshold_diff = threshold_diff
        self.num_byzantine = num_byzantine

    def get_gradient(self) -> torch.Tensor:
        return self._gradient

    def is_train(self):
        return False

    def omniscient_callback(self):
        all_updates = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                all_updates.append(w.get_gradient())
        all_updates = torch.stack(all_updates)
        # all_updates = torch.stack(
        #     list(map(lambda w: w.get_gradient(), self.simulator.workers))
        # )
        model_re = torch.mean(all_updates, 0)

        if self.dev_type == "sign":
            deviation = torch.sign(model_re)
        elif self.dev_type == "unit_vec":
            deviation = model_re / torch.norm(model_re)
        elif self.dev_type == "std":
            deviation = torch.std(all_updates, 0)

        lambda_ = torch.Tensor([self.threshold]).to(all_updates.device)

        threshold_diff = self.threshold_diff
        lamda_fail = lambda_
        lamda_succ = 0

        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lambda_) > threshold_diff:
            mal_update = (model_re - lambda_ * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)

            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lambda_
                lambda_ = lambda_ + lamda_fail / 2
            else:
                lambda_ = lambda_ - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)

        self._gradient = mal_update


'''
MIN-MAX attack
'''


def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).float().cuda()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update
