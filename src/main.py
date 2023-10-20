import argparse
import torch
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))

from torch.nn.modules.loss import CrossEntropyLoss

from aggregators.bulyan import Bulyan
from aggregators.coordinatewise_median import CM
from aggregators.mean import Mean
from aggregators.multikrum import MultiKrum
from aggregators.trimmed_mean import TM
from aggregators.fltrust import Fltrust
from aggregators.dnc import Dnc
from attackers.alieworker import ALittleIsEnoughWorker
from attackers.labelflippingworker import LabelflippingWorker
from attackers.minmaxworker import MinMaxWorker
from attackers.noiseworker import NoiseWorker
from attackers.randomworker import RandomWorker
from server import BaseServer
from simulator import TrainSimulator, EvalSimulator
from utils import top1_accuracy, initialize_logger
from worker import MomentumWorker
from task import cifar10, tiny_imagenet, fmnist, mnist

parser = argparse.ArgumentParser()

#  /*************************Basic Setting*************************/
parser.add_argument("--agg", type=str, default="rfedfw")
parser.add_argument("--attack", type=str, default="ALIE")
parser.add_argument("--global_rounds", type=int, default=100)
parser.add_argument("--local_round", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_workers", type=int, default=50)
parser.add_argument("--n_byz", type=int, default=10)
parser.add_argument("--trusted_id", type=str, default=40)
parser.add_argument("--use_cuda", action="store_true", default=True)
parser.add_argument("--noniid", default=False)
parser.add_argument("--alpha", type=float, default=0.2, help="percentage of non-iid data per worker, value in [0, 1]")
parser.add_argument("--datasets", type=str, default="tiny-imagenet")
parser.add_argument("--factor", type=float, default=3.0)

args = parser.parse_args()


#  /*************************OutFolder Setting*************************/
def initial_outfolder(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
    DATA_DIR = ROOT_DIR + f"datasets/{args.datasets}/"
    EXP_DIR = ROOT_DIR + f"outputs/{args.datasets}/"

    if not args.noniid:
        LOG_DIR = (
                EXP_DIR
                + f"f{args.n_byz}_w{args.n_workers}_{args.attack}_{args.agg}_m{args.momentum}_iid_seed{args.seed}_fa{args.factor}"
        )
    else:
        LOG_DIR = (
                EXP_DIR
                + f"f{args.n_byz}_w{args.n_workers}_{args.attack}_{args.agg}_m{args.momentum}_non-iid_s{args.alpha}_seed{args.seed}_fa{args.factor}"
        )
    return DATA_DIR, LOG_DIR


#  /*************************Server Aggregator*************************/
def _get_aggregator():
    if args.agg == "mean":
        return Mean()
    if args.agg == "median":
        return CM()
    if args.agg == "krum":
        return MultiKrum(n=args.n_workers, f=args.n_byz, m=1)
    if args.agg == "tm":
        return TM(b=args.n_byz)
    if args.agg == "bulyan":
        return Bulyan(n_byzantine=args.n_byz)
    if args.agg == "fltrust":
        return Fltrust()
    if args.agg == "dnc":
        return Dnc(num_byzantine=args.n_byz)

    raise NotImplementedError(args.agg)


#  /*************************Initialize Workers*************************/
def initialize_worker(trainer, model, train_loader, worker_rank, optimizer, loss_func, device):
    # NOTE: The first N_BYZ nodes are Byzantine
    if worker_rank < args.n_byz:
        if args.attack == "LF":
            return LabelflippingWorker(
                id=str(worker_rank),
                model=model,
                data_loader=train_loader[worker_rank],
                local_round=args.local_round,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
            )

        if args.attack == "RA":
            attacker = RandomWorker(
                id=str(worker_rank),
                model=model,
                data_loader=train_loader[worker_rank],
                local_round=args.local_round,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
            )
            attacker.configure(trainer)
            return attacker

        if args.attack == "NO":
            attacker = NoiseWorker(
                id=str(worker_rank),
                model=model,
                data_loader=train_loader[worker_rank],
                local_round=args.local_round,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
            )
            attacker.configure(trainer)
            return attacker

        if args.attack == "ALIE":
            attacker = ALittleIsEnoughWorker(
                id=str(worker_rank),
                model=model,
                data_loader=train_loader[worker_rank],
                local_round=args.local_round,
                n=args.n_workers,
                m=args.n_byz,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
            )
            attacker.configure(trainer)
            return attacker

        if args.attack == "MinMax":
            attacker = MinMaxWorker(
                id=str(worker_rank),
                model=model,
                data_loader=train_loader[worker_rank],
                local_round=args.local_round,
                num_byzantine=args.n_byz,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
            )
            attacker.configure(trainer)
            return attacker

        raise NotImplementedError(f"No such attack {args.attack}")
    return MomentumWorker(
        id=str(worker_rank),
        model=model,
        data_loader=train_loader[worker_rank],
        local_round=args.local_round,
        optimizer=optimizer,
        loss_func=loss_func,
        device=device,
        momentum=args.momentum
    )


#  /************************* Main *************************/
def run(task="cifar10"):
    DATA_DIR, LOG_DIR = initial_outfolder(args)
    initialize_logger(LOG_DIR)
    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")
    kwargs = {"pin_memory": True} if args.use_cuda else {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = None
    train_loader = None
    test_loader = None

    if task == "cifar10":
        model = cifar10.get_cifar10_model(use_cuda=args.use_cuda).to(device)
        train_loader = cifar10.get_train_loader(root_dir=DATA_DIR, n_workers=args.n_workers,
                                                alpha=args.alpha, batch_size=args.batch_size,
                                                noniid=args.noniid)
        test_loader = cifar10.get_test_loader(root_dir=DATA_DIR,
                                              batch_size=args.test_batch_size)
    elif task == "fmnist":
        model = fmnist.get_fmnist_model().to(device)
        train_loader = fmnist.get_train_loader(root_dir=DATA_DIR, n_workers=args.n_workers,
                                               alpha=args.alpha, batch_size=args.batch_size,
                                               noniid=args.noniid)
        test_loader = fmnist.get_test_loader(root_dir=DATA_DIR,
                                             batch_size=args.test_batch_size)
    elif task == "mnist":
        model = mnist.get_mnist_model().to(device)
        train_loader = mnist.get_train_loader(root_dir=DATA_DIR, n_workers=args.n_workers,
                                              alpha=args.alpha, batch_size=args.batch_size,
                                              noniid=args.noniid)
        test_loader = mnist.get_test_loader(root_dir=DATA_DIR,
                                            batch_size=args.test_batch_size)
    elif task == "tiny-imagenet":
        model = tiny_imagenet.get_tinyimg_model(use_cuda=args.use_cuda).to(device)
        train_loader = tiny_imagenet.get_train_loader(root_dir=DATA_DIR, n_workers=args.n_workers,
                                                      alpha=args.alpha, batch_size=args.batch_size,
                                                      noniid=args.noniid)
        test_loader = tiny_imagenet.get_test_loader(root_dir=DATA_DIR,
                                                    batch_size=args.test_batch_size)

    loss_func = CrossEntropyLoss().to(device)
    # train_layer = [p for p in model.parameters() if p.requires_grad == True]
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(train_layer, lr=args.lr)
    metrics = {"top1": top1_accuracy}

    server = BaseServer(model=model, aggregator=_get_aggregator(), optimizer=optimizer)

    # 训练器
    trainer = TrainSimulator(
        server=server,
        metrics=metrics,
        use_cuda=args.use_cuda,
        max_batches_per_epoch=9999,
        log_interval=10,
        agg=args.agg
    )

    # 测试器
    evaluator = EvalSimulator(
        server=server,
        data_loader=test_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda
    )

    # 学习率调节
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer=optimizer, milestones=[100], gamma=args.lr
    # )

    # 初始化客户端
    for worker_rank in range(args.n_workers):
        worker = initialize_worker(
            trainer=trainer,
            model=model,
            train_loader=train_loader,
            worker_rank=worker_rank,
            optimizer=optimizer,
            loss_func=loss_func,
            device=device
        )
        trainer.add_worker(worker)
    if args.trusted_id:
        trainer.set_trusted_clients([args.trusted_id])

    for epoch in range(1, args.global_rounds + 1):
        trainer.train(epoch)
        evaluator.test(epoch)
        # scheduler.step()
        # print(f"E={epoch}; Learning rate = {scheduler.get_last_lr()[0]:}")


if __name__ == '__main__':
    tasks = ["mnist", "fmnist", "cifar10", "tiny-imagenet"]
    aggs = ["mean", "median", "krum", "tm", "bulyan", "fltrust", "dnc"]
    atks = ["NA", "RA", "NO", "LF", "ALIE", "MinMax"]

    # for task in tasks:
    #     for atk in atks:
    #         for agg in aggs:
    #             args.agg = agg
    #             args.attack = atk
    #             if task == "tiny-imagenet":
    #                 args.lr = 0.05
    #                 args.global_rounds = 100
    #             if atk == "NA":
    #                 args.n_byz = 0
    #             else:
    #                 args.n_byz = 10
    #             args.datasets = task
    #             run(task=task)

    # 计算时间
    # for task in tasks:
    #     for atk in atks:
    #         for agg in aggs:
    #             args.agg = agg
    #             args.attack = atk
    #             args.global_rounds = 50
    #             args.datasets = task
    #             if atk == "NA":
    #                 args.n_byz = 0
    #             else:
    #                 args.n_byz = 10
    #             run(task=task)
    # 不同节点数量的训练时间
    # workers = [10, 20, 30, 40]
    # for task in tasks:
    #     for atk in atks:
    #         for agg in aggs:
    #             for w in workers:
    #                 args.agg = agg
    #                 args.attack = atk
    #                 args.global_rounds = 50
    #                 args.datasets = task
    #                 args.n_workers = w
    #                 if atk == "NA":
    #                     args.n_byz = 0
    #                 else:
    #                     args.n_byz = 10
    #                 run(task=task)

    # 训练收敛性
    # for task in tasks:
    #     for atk in atks:
    #         for agg in aggs:
    #             args.agg = agg
    #             args.attack = atk
    #             args.global_rounds = 500
    #             args.datasets = task
    #             if atk == "NA":
    #                 args.n_byz = 0
    #             else:
    #                 args.n_byz = 10
    #             run(task=task)

    # non-iid
    alphas = [0.2, 0.4]
    for task in tasks:
        for agg in aggs:
            for num in alphas:
                for atk in atks:
                    if agg == "dnc" and num == 0.2:
                        continue
                    args.noniid = True
                    args.alpha = num
                    args.agg = agg
                    args.attack = atk
                    args.datasets = task
                    try:
                        run(task=task)
                    except Exception:
                        break
