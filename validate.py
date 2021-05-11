import torch

import time
import csv

from utils import *


def validate(val_loader, model, criterion, config):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, acc], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, class_label, domain_label, slide_name, patch_idx) in enumerate(
            val_loader
        ):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)

            if torch.cuda.is_available():
                class_label = class_label.cuda(config.gpu, non_blocking=True)
                domain_label = domain_label.cuda(config.gpu, non_blocking=True)

            # compute output
            class_prob, domain_prob, attn = model(images, 0)

            class_loss = criterion(class_prob, class_label)
            domain_loss = criterion(
                domain_prob.unsqueeze(0).transpose(1, 2), domain_label
            )
            loss = class_loss + domain_loss

            # measure accuracy and record loss
            _acc = accuracy(class_prob, class_label)
            losses.update(loss.item(), images.size(0))
            acc.update(_acc[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if not config.multiprocessing_distributed or (
            #     config.multiprocessing_distributed and config.rank % 2 == 0
            # ):
            #     with open(config.log_valid, "a") as f:
            #         # TODO: epoch 기록, 기록 되는 데이터 유형 재정리
            #         f_writer = csv.writer(f, lineterminator="\n")
            #         for i in range(len(class_label)):
            #             # [bag name, ground truth, predicted label] + [y_prob[1], y_prob[2]]
            #             slideid_tlabel_plabel = [
            #                 slide_name[0][: -len("_files")],
            #                 class_label[i].item(),
            #                 torch.argmax(class_prob[i]),
            #             ] + class_prob[i].tolist()
            #
            #             f_writer.writerow(slideid_tlabel_plabel)
            #             f_writer.writerow(patch_idx[i].tolist())  # write instance
            #
            #             # reduce 1st dim [1,100] -> [100]
            #             attention_weights = attn.squeeze(0)
            #             attention_weights_list = attention_weights.tolist()
            #
            #             # write attention weights for each instance
            #             f_writer.writerow(attention_weights_list)

            if i % config.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(" * Acc@1 {acc.avg:.3f}".format(acc=acc))

    return acc.avg