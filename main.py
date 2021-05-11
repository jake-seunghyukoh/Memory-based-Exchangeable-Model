import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import numpy as np

import argparse
import warnings
import time
import csv
import os
import sys

sys.path.append("/home/ubuntu/MEDIAR-Lab/MEDIAR-Lung-Cancer-Classifier/Model")

from train import train
from validate import validate
from dataset import PatchDataset
from utils import *
from model import MEM

import pdb

best_acc = 0


def main(config):
    print(config)
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if config.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    num_gpus_per_node = torch.cuda.device_count()
    root_dir = "/home/ubuntu/MEDIAR-Lab/MEDIAR-Lung-Cancer-Classifier/"
    config.log_train = f"{root_dir}result/train_log/train_{time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))}_mag-{config.Mag}.csv"
    config.log_valid = f"{root_dir}result/valid_log/valid_{time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))}_mag-{config.Mag}.csv"

    model_params = f"{root_dir}checkpoints/model_{time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))}_mag-{config.Mag}.pth"

    with open(config.log_train, "w") as f:
        f_writer = csv.writer(f, lineterminator="\n")
        csv_header = ["epoch", "slide_name", "class_loss", "domain_loss"]
        f_writer.writerow(csv_header)

    with open(config.log_valid, "w") as f:
        f_writer = csv.writer(f, lineterminator="\n")
        csv_header = [
            "epoch",
            "slide_name",
            "class_label",
            "class_hat",
            "class_softmax",
        ]
        f_writer.writerow(csv_header)

    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = num_gpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        label = load_label("/home/ubuntu/TCGA/", "metadata3.json")
        mp.spawn(
            main_worker,
            nprocs=num_gpus_per_node,
            args=(num_gpus_per_node, config, label, model_params),
        )
    else:
        # Simply call main_worker function
        label = load_label("/home/ubuntu/TCGA/", "metadata3.json")
        main_worker(config.gpu, num_gpus_per_node, config, label, model_params)


def main_worker(gpu, num_gpus_per_node, config, label, model_params):
    global best_acc
    config.gpu = gpu
    if config.gpu is not None:
        print(f"Use GPU: {config.gpu}")

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            """For multiprocessing distributed training, rank needs to be the
            global rank among all the processes
            """
            config.rank = config.rank * num_gpus_per_node + gpu
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=config.rank,
        )
    # create model
    model = MEM(debug=False)

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif config.distributed:
        """For multiprocessing distributed, DistributedDataParallel constructor
        should always set the single device scope, otherwise,
        DistributedDataParallel will use all available devices.
        """
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            """When using a single GPU per process and per
            DistributedDataParallel, we need to divide the batch size
            ourselves based on the total number of GPUs we have
            """
            config.batch_size = int(config.batch_size / num_gpus_per_node)
            config.workers = int(
                (config.workers + num_gpus_per_node - 1) / num_gpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.gpu]
            )
        else:
            model.cuda()
            """DistributedDataParallel will divide and allocate batch_size to all
            available GPUs if device_ids are not set
            """
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        """Data Parallel will divide and allocate batch_size to all available GPUs"""
        if config.arch.startswith("alexnet") or config.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(config.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            print(f"=> loading checkpoint {config.resume}")
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu
                loc = f"cuda:{config.gpu}"
                checkpoint = torch.load(config.resume, map_location=loc)
            config.start_epoch = checkpoint["epoch"]
            best_acc = checkpoint["best_acc"]
            if config.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                best_acc = best_acc.to(config.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                f"=> loaded checkpoint {config.resume} (epcoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at {config.resume}")

    cudnn.benchmark = True

    # Data loading code
    train_dataset = PatchDataset(
        config.data, label, config, mag=config.Mag, mode="train"
    )

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        PatchDataset(config.data, label, config, mag=config.Mag, mode="valid"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
    )

    if config.evaluate:
        validate(val_loader, model, criterion, config)
        return

    count = 0
    tolerance = 15

    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, config)
        lr_DA = adjust_DA_learning_rate(epoch, config)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config, lr_DA)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, config)

        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        if is_best:
            count = 0
        else:
            count += 1

        best_acc = max(acc, best_acc)

        if not config.multiprocessing_distributed or (
            config.multiprocessing_distributed and config.rank % num_gpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                model_params,
            )
        if count >= tolerance:
            print(f"Stop training at epoch: {epoch}")
            break


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # model_names = sorted(
    #     name
    #     for name in models.__dict__
    #     if name.islower()
    #     and not name.startswith("__")
    #     and callable(models.__dict__[name])
    # )

    parser = argparse.ArgumentParser(description="MEDIAR Lung Cancer Classifier")

    parser.add_argument(
        "data",
        default="/home/ubuntu/TCGA/images/",
        metavar="DIR",
        help="path to dataset",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="Vgg",
        metavar="N",
        help="name of the model",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--da",
        "--DArate",
        default=0.001,
        type=float,
        metavar="DALR",
        help="initial domain adversarial learning rate",
        dest="da",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:8000",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    # image option
    parser.add_argument("--Mag", default=20, type=int, help="Magnification")
    parser.add_argument(
        "--num_patches", default=50, type=int, help="the number of patches in each bags"
    )
    args = parser.parse_args()

    main(args)