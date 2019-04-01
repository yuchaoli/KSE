# coding: utf-8

import torch
import torch.nn as nn
import argparse
import importlib
from tensorboardX import SummaryWriter
import pdb

from utils import base, models

parser = argparse.ArgumentParser(description='KSE Experiments')
parser.add_argument('--dataset', dest='dataset', help='training dataset', default='cifar10', type=str)
parser.add_argument('--net', dest='net', help='training network', default='resnet56', type=str)
parser.add_argument('--pretrained', dest='pretrained', help='whether use pretrained model', default=False, type=bool)
parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint dir', default=None, type=str)

parser.add_argument('--train_batch_size', dest='train_batch_size', help='training batch size', default=64, type=int)
parser.add_argument('--test_batch_size', dest='test_batch_size', help='test batch size', default=50, type=int)

parser.add_argument('--G', dest='G', help='G', default=4, type=int)
parser.add_argument('--T', dest='T', help='T', default=1, type=int)

args = parser.parse_args()

if __name__ == "__main__":
    model = importlib.import_module("model.model_deploy").__dict__[args.net](args.pretrained, args.checkpoint)
    train_loader, test_loader = importlib.import_module("dataset."+args.dataset).__dict__["load_data"](
        args.train_batch_size, args.test_batch_size)

    criterion = nn.CrossEntropyLoss()

    # create architecture base on mask
    models.create_arch(model, args.G, args.T)
    # load clusters and indexs
    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    # tranform indexs
    models.load(model)
    # generate auxiliary variable
    models.forward_init(model)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    if torch.__version__ == "0.3.1":
        top1_acc, top5_acc = base.validate_3(test_loader, model, criterion)
    else:
        top1_acc, top5_acc = base.validate_4(test_loader, model, criterion)

