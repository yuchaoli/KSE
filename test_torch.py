# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch

import numpy as np
import yaml
from pathlib import Path

from utils_yolo.test import test

from utils_yolo.general import check_dataset, init_seeds, fitness, increment_path
from utils_yolo.finetune import *


# files / dirs
model_save_location = 'best.pth'
save_dir = Path(increment_path(Path('tmp/yolo_kse/testing'), exist_ok=False))

data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'
yolo_struct = './data/yolov7_tiny_struct.yaml'

# KSE params
G = 3
T = 0

# params
batch_size = 5
num_workers = 4
img_size = 640

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = device.type != 'cpu'

    os.makedirs(save_dir, exist_ok=True) # inside __main__!!!
    init_seeds(1)
    
    # open data info
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    with open(yolo_struct) as f:
        yolo_struct = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)

    # load model
    nc = int(data_dict['nc'])   # number of classes
    model = load_model(yolo_struct, nc, hyp.get('anchors'), model_save_location, G, T, device)

    # load data
    dataloader = create_dataloader(data_dict['val'], img_size, batch_size, 32, 
                                            hyp=hyp, rect=False, 
                                            workers=num_workers,
                                            pad=0.5, prefix=colorstr('val: '))[0]
    nb = len(dataloader)        # number of batches

    # run validation
    maps = np.zeros(nc) # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = test(
        data_dict,
        batch_size=batch_size,
        imgsz=img_size,
        model=model,
        dataloader=dataloader,
        save_dir=save_dir,
        plots=False,
        is_coco=True,
        save_json=True,
        iou_thres=0.65
    )[0]
    f = fitness(np.array(results).reshape(1, -1))
    print('Fitness:', f)
