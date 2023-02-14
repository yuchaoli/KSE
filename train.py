# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import torch.nn as nn
import argparse
import importlib
import copy
from tensorboardX import SummaryWriter
import pdb

from utils import base, models

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# parser = argparse.ArgumentParser(description='KSE Experiments')
# parser.add_argument('--dataset', dest='dataset', help='training dataset', default='cifar10', type=str)
# parser.add_argument('--net', dest='net', help='training network', default='resnet56', type=str)
# parser.add_argument('--pretrained', dest='pretrained', help='whether use pretrained model', default=False, type=bool)
# parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint dir', default=None, type=str)
# parser.add_argument('--train_dir', dest='train_dir', help='training data dir', default="tmp", type=str)
# parser.add_argument('--save_best', dest='save_best', help='whether only save best model', default=True, type=bool)

# parser.add_argument('--train_batch_size', dest='train_batch_size', help='training batch size', default=64, type=int)
# parser.add_argument('--test_batch_size', dest='test_batch_size', help='test batch size', default=50, type=int)
# parser.add_argument('--gpus', dest='gpus', help='gpu id',default=[0], type=int,nargs='+')

# parser.add_argument('--learning_rate', dest='learning_rate', help='learning rate', default=0.01, type=float)
# parser.add_argument('--momentum', dest='momentum', help='momentum', default=0.9, type=float)
# parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay', default=1e-5, type=float)
# parser.add_argument('--epochs', dest='epochs', help='epochs', default=200, type=int)
# parser.add_argument('--schedule', dest='schedule', help='Decrease learning rate',default=[100, 150],type=int,nargs='+')
# parser.add_argument('--gamma', dest='gamma', help='gamma', default=0.1, type=float)

# parser.add_argument('--G', dest='G', help='G', default=None, type=int)
# parser.add_argument('--T', dest='T', help='T', default=None, type=int)

# args = parser.parse_args()


# opt = {
#     'dataset': 'cifar10',
#     'net': 'densenet40',
#     'pretrained': False, # True,
#     'checkpoint': None, # 'pth/densenet40.pth',
#     'train_dir': 'tmp/densenet40_NONE',
#     'save_best': True,
#     'train_batch_size': 128,
#     'test_batch_size': 50,
#     'gpus': [0],
#     'learning_rate': 0.01,
#     'momentum': 0.9,
#     'weight_decay': 1e-5,
#     'epochs': 2, #200,
#     'schedule': [100],
#     'gamma': 0.1,
#     'G': None, # 5,
#     'T': None # 0
# }

opt = {
    'dataset': 'cifar10',
    'net': 'densenet40',
    'pretrained': True,
    'checkpoint': 'pth/densenet40.pth',
    'train_dir': 'tmp/densenet40_TEST',
    'save_best': True,
    'train_batch_size': 128,
    'test_batch_size': 50,
    'gpus': [0],
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'epochs': 2,
    'schedule': [100],
    'gamma': 0.1,
    'G': 5,
    'T': 0
}

args = dotdict(opt)

if __name__ == "__main__":
    model = importlib.import_module("model.model_deploy").__dict__[args.net](args.pretrained, args.checkpoint)

    train_loader, test_loader = importlib.import_module("dataset."+args.dataset).__dict__["load_data"](
        args.train_batch_size, args.test_batch_size)

    writer = SummaryWriter(args.train_dir)

    # KSE
    models.KSE(model, args.G, args.T)

    # network forward init
    models.forward_init(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    if torch.cuda.is_available():
        criterion = criterion.cuda(args.gpus[0])
        model = model.cuda(args.gpus[0])

    if len(args.gpus) != 1:
        model = torch.nn.DataParallel(model, args.gpus).cuda()

    if len(args.gpus) != 1:
        model_tmp = copy.deepcopy(model.module)
    else:
        model_tmp = copy.deepcopy(model)
    models.save(model_tmp)
    torch.save(model_tmp.state_dict(), args.train_dir + "/model.pth")

    if torch.__version__ == "0.3.1":
        train = base.train_3
        validate = base.validate_3
    else:
        train = base.train_4
        validate = base.validate_4

    best_acc = 0
    for i in range(args.epochs):
        train(train_loader, model, criterion, optimizer, i, writer)
        top1_acc, top5_acc = validate(test_loader, model, criterion)
        lr_scheduler.step()

        if len(args.gpus) != 1:
            model_tmp = copy.deepcopy(model.module)
        else:
            model_tmp = copy.deepcopy(model)
        models.save(model_tmp)

        if args.save_best:
            if best_acc < top1_acc:
                torch.save(model_tmp.state_dict(), args.train_dir+"/model_best.pth")
                best_acc = top1_acc
        else:
            torch.save(model_tmp.state_dict(), args.train_dir + "/model_"+str(i)+".pth")

        writer.add_scalar('val-acc', top1_acc, i)
        writer.add_scalar('val-top5-acc', top5_acc, i)

    print("best acc: {:.2f}".format(best_acc))


