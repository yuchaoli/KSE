# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import torch.nn as nn

import yaml
import argparse
import importlib
import copy
from tensorboardX import SummaryWriter
import pdb
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from utils import base, models
from models.yolo import Model
from utils.general import check_dataset
from utils.utils import Conv2d_KSE

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

opt = {
    'dataset': 'cifar10',
    'net': 'densenet40',
    'pretrained': True,
    'checkpoint': 'pth/densenet40.pth',
    'train_dir': 'tmp/densenet40_TEST',
    'save_best': True,
    'train_batch_size': 128,
    'test_batch_size': 128,
    'gpus': [0],
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'epochs': 2,
    'schedule': [100],
    'gamma': 0.1,
    'G': 1,
    'T': 0
}

args = dotdict(opt)

batch_size = 16
num_workers = 4
epochs = 2
img_size = [640, 640]

data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'
weights = './yolov7-tiny.pt'
# cfg = 'cfg/training/yolov7-tiny.yaml'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = device.type != 'cpu'
plots = True

if __name__ == "__main__":
    # check data
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)

    train_path = data_dict['train']
    test_path = data_dict['val']

    nc = int(data_dict['nc'])
    names = data_dict['names']

    # load model
    ckpt = torch.load(weights, map_location=device) # = {'model', 'optimizer', 'training_results', 'epoch'}
    state_dict = ckpt['model'].float().state_dict()
    model = Model(ckpt['model'].yaml, nc=nc, anchors=hyp.get('anchors')).to(device)
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict
    
    # KSE
    n_kse_layers = sum(isinstance(module, Conv2d_KSE) for module in model.modules())
    models.KSE(model, args.G, args.T, n_kse_layers)
    print('KSE finished')

    # network forward init
    a, b = models.forward_init(model)

    input = torch.randn(2, 3, 32, 32).cuda()
    model.cuda()
    y = model(input)
    print(y)

    print('{:.2f}% of channels remaining'.format(a/b*100))
    
    # summary(model, (128, 3, 32, 32))
    input = torch.randn(2, 3, 32, 32).cuda()
    flops = FlopCountAnalysis(model, input)
    print('Total FLOPs: {}M'.format(round(flops.by_module()[''] / 1e6)))
    print(parameter_count_table(model))
    exit()

    train_loader, test_loader = importlib.import_module("dataset."+args.dataset).__dict__["load_data"](
        args.train_batch_size, args.test_batch_size)

    writer = SummaryWriter(args.train_dir)

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


