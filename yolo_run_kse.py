# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import torch.nn as nn

import yaml
import copy
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from pathlib import Path

from yolo_test import test
from utils import models
from models.yolo import Model
from utils.general import check_dataset, check_img_size, colorstr, labels_to_class_weights
from utils.utils import Conv2d_KSE
from utils.datasets import create_dataloader

# validation params
batch_size = 16
num_workers = 4
img_size = [640, 640]

# YOLO params
data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'
weights = './yolov7-tiny.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
plots = True

# KSE params
G = 3
T = 0

save_dir = Path('tmp/yolo_kse')
os.makedirs(save_dir, exist_ok=True)

show_flop_params = False

if __name__ == "__main__":
    # load model
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)
    
    nc = int(data_dict['nc'])
    names = data_dict['names']
    ckpt = torch.load(weights, map_location=device) # = {'model', 'optimizer', 'training_results', 'epoch'}
    state_dict = ckpt['model'].float().state_dict()
    model = Model(ckpt['model'].yaml, nc=int(data_dict['nc']), anchors=hyp.get('anchors')).to(device)
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict
    
    # KSE
    n_kse_layers = sum(isinstance(module, Conv2d_KSE) for module in model.modules())
    models.KSE(model, G, T, n_kse_layers)
    print('KSE finished')

    # network forward init
    model.to(device)
    a, b = models.forward_init(model)
    print('{:.2f}% of channels remaining'.format(a/b*100))
    
    # FLOPs + parameter estimation/calculation
    if show_flop_params:
        input = torch.randn(2, 3, 32, 32).to(device)
        flops = FlopCountAnalysis(model, input)
        print('Total FLOPs: {}M'.format(round(flops.by_module()[''] / 1e6)))
        print(parameter_count_table(model))

    # save model (so that yolo_train_kse.py can import it and train/finetune)
    model_tmp = copy.deepcopy(model)
    models.save(model_tmp)    
    torch.save(model_tmp.state_dict(), os.path.join(save_dir, 'model1.pth'))

    # run validation (make sure accuracy is not completely lost)
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl
    imgsz, imgsz_test = [check_img_size(x, gs) for x in img_size]
    
    test_path = data_dict['val']
    testloader, dataset = create_dataloader(test_path, imgsz_test, batch_size * 2, gs,  # testloader
                                       hyp=hyp, rect=True, 
                                       workers=num_workers,
                                       pad=0.5, prefix=colorstr('val: '))
    # model params
    # hyp['box'] *= 3. / nl  # scale to layers
    # hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    # hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    # hyp['label_smoothing'] = 0.0
    # model.nc = nc  # attach number of classes to model
    # model.hyp = hyp  # attach hyperparameters to model
    # model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # run validation
    results, _, _ = test(
        data_dict,
        batch_size=batch_size,
        imgsz=imgsz_test,
        conf_thres=0.001,
        iou_thres=0.7,
        model=model,
        dataloader=testloader,
        save_dir=save_dir,
        save_json=True,
        plots=False,
        is_coco=True
    )

    # save model (so that yolo_train_kse.py can import it and train/finetune)
    model_tmp = copy.deepcopy(model)
    models.save(model_tmp)    
    torch.save(model_tmp.state_dict(), os.path.join(save_dir, 'model2.pth'))