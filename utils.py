import torch
import torch.nn as nn
import numpy as np
import shutil
import json
import copy


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth.tar",
    root_dir="/home/ubuntu/MEDIAR-Lab/MEDIAR-Lung-Cancer-Classifier/checkpoints/",
    model_name="ResNet",
):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, root_dir + model_name + "_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_DA_learning_rate(epoch, config):
    p = ((epoch + 1) / config.epochs) * config.da
    lr = (2 / (1 + np.exp(-10 * p))) - 1
    return lr


def load_label(root_dir, file_name):
    with open(root_dir + file_name) as f:
        meta = json.loads(f.read())
        meta = dict((m["file_name"].replace(".svs", ""), m) for m in meta)

    label = []
    count_Normal = 0
    count_LUSC = 0
    count_LUAD = 0
    for key in meta:
        cancer_type = meta[key]["cancer_type"]
        sample_type = meta[key]["sample_type"]
        if "Normal" in sample_type:
            label.append(0)
            count_Normal += 1
        elif cancer_type == "LUSC":
            label.append(1)
            count_LUSC += 1
        elif cancer_type == "LUAD":
            label.append(2)
            count_LUAD += 1
        else:
            assert False

    count_total = count_LUSC + count_LUAD + count_Normal
    label = np.array(label)

    print(
        f"\nTotal images: {count_total}, LUSC: {count_LUSC}, LUAD: {count_LUAD}, Normal: {count_Normal}\n"
    )

    return label


def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])