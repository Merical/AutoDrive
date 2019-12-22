import numpy
import os
import shutil
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def train(train_loader, model, criterion, optimizer, epoch, cuda):
    """Train the model on Training Set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        #topk = (1,5) if labels >= 100 else (1,) # TO FIX
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o,target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data.item(), input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.data.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))


def validate(val_loader, model, criterion, cuda):
    """Validate the model on Validation Set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # Evaluate all the validation set
    for i, (input, target) in enumerate(val_loader):
        if cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        # print ("Output: ", output)
        #topk = (1,5) if labels >= 100 else (1,) # TODO: add more topk evaluation
        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o,target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data.item(), input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % 20 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   i, len(val_loader), batch_time=batch_time,
                   loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))

    print(' * Prec@1 {top1}'
          .format(top1=numpy.asscalar(top1.avg.cpu().numpy())))
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res