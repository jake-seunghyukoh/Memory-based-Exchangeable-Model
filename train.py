import time

from utils import *


def train(train_loader, model, criterion, optimizer, epoch, args, lr_DA):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, class_label, domain_label, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        if torch.cuda.is_available():
            class_label = class_label.cuda(args.gpu, non_blocking=True)
            domain_label = domain_label.cuda(args.gpu, non_blocking=True)

        # compute output
        class_prob = model(images)

        class_loss = criterion(class_prob, class_label)
        loss = class_loss

        # measure accuracy and record loss
        _acc = accuracy(class_prob, class_label)
        losses.update(loss.item(), images.size(0))
        acc.update(_acc[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)