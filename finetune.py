# TODO 
# what happens with qc=0? is num filters or inputs changed?
# implement attempt_load(file, device)

# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

from tqdm import tqdm
import numpy as np
import yaml
import copy
from pathlib import Path

from utils_yolo.test import test
from utils_kse import models
from models.yolo import Model
from utils_yolo.general import check_dataset, check_img_size, colorstr, labels_to_class_weights, init_seeds, fitness, \
    increment_path
from utils_kse.utils import Conv2d_KSE
from utils_yolo.datasets import create_dataloader
from utils_yolo.autoanchor import check_anchors
from utils_yolo.loss import ComputeLoss

def create_param_groups(model):
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
            # print('pg2', k)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
            # print('pg0', k)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
            if hasattr(v, 'full_weight') and isinstance(v.full_weight, nn.Parameter):
                pg1.append(v.full_weight)
        # for yolo-tiny, below statements are all False
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)
    return pg0, pg1, pg2

def create_optimizer(model, hyp):
    pg0, pg1, pg2 = create_param_groups(model)
    assert len(pg0) == 55 and len(pg1) == 58*2 and len(pg2) == 58, 'Found {}, {}, {}'.format(len(pg0), len(pg1), len(pg2))
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2
    return optimizer

def load_model(yolo_struct, nc, anchors, kse_weights, G, T, device):
    model = Model(yolo_struct, nc=nc, anchors=anchors).to(device)
    state_dict = torch.load(kse_weights, map_location=device)
    model.load_state_dict(state_dict, strict=False)     # for Conv2d_KSE: loads 'mask' from state_dict, ignores rest
    models.create_arch(model, G, T)                     # creates 'full_weight', 'clusters_{i}', 'indexs_{i}'
    model.load_state_dict(state_dict, strict=False)     # for Conv2d_KSE: loads 'mask', 'full_weight', 'clusters_{i}', 'indexs_{i}'
    del state_dict
    models.load(model)
    models.forward_init(model)
    return model.to(device)

def load_data(model, img_size, data_dict, batch_size, hyp, num_workers):
    gs = max(int(model.stride.max()), 32)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in img_size]
    dataloader, dataset = create_dataloader(data_dict['train'], imgsz, batch_size, gs, hyp=hyp,
                                            augment=True,  workers=num_workers, prefix=colorstr('train: '))
    testloader = create_dataloader(data_dict['val'], imgsz_test, batch_size, gs, 
                                        hyp=hyp, rect=True, 
                                        workers=num_workers,
                                        pad=0.5, prefix=colorstr('val: '))[0]
    
    # update hyp
    nl = model.model[-1].nl
    nc = data_dict['nc']
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0

    # update model
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = data_dict['names']

    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    return imgsz_test, dataloader, dataset, testloader, hyp, model
    
validate_before_training = False

# files / dirs
model_save_location = 'tmp/yolo_kse/model_G3_T0_bias.pth'
save_dir = Path(increment_path(Path('tmp/yolo_kse/training'), exist_ok=False))
wdir = save_dir / 'weights'
last = wdir / 'last.pth'
best = wdir / 'best.pth'
results_file = save_dir / 'results.txt'

data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'
yolo_struct = './data/yolov7_tiny_struct.yaml'

# KSE params
G = 3
T = 0

# training params
batch_size = 16
num_workers = 4
img_size = [640, 640]
nbs = 64 # nominal batch size
accumulate = max(round(nbs / batch_size), 1)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda = device.type != 'cpu'

    os.makedirs(wdir, exist_ok=True) # inside __main__!!!
    init_seeds(1)
    
    # check data
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
    imgsz_test, dataloader, dataset, testloader, hyp, model = load_data(model, img_size, data_dict, batch_size, hyp, num_workers)
    nb = len(dataloader)        # number of batches

    # optimizer
    optimizer = create_optimizer(model, hyp)

    # scaler + loss
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class

    # run validation before training
    maps = np.zeros(nc) # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    best_fitness = 0
    if validate_before_training:
        results, _, _ = test(
            data_dict,
            batch_size=batch_size,
            imgsz=imgsz_test,
            conf_thres=0.001,
            iou_thres=0.7,
            model=model,
            dataloader=testloader,
            save_dir=save_dir,
            compute_loss=compute_loss,
            plots=False,
            is_coco=True
        )
        best_fitness = fitness(np.array(results).reshape(1, -1))

    # train loop
    start_epoch = 0
    best_fitness = 0.0
    epochs = 2
    print(('%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels'))
    for epoch in range(epochs):
        model.train()

        mloss = torch.zeros(4, device=device)
        optimizer.zero_grad()

        pbar = tqdm(enumerate(dataloader), total=nb)
        ix = 0
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # forward pass
            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
            
            # backprop
            scaler.scale(loss).backward()

            # optimize (every nominal batch size)
            if ni % accumulate == 0:
                ix+=1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # print
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate / 60 if rate and pbar.total else 0
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 5) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0])
            pbar.set_description(s)

            if ix == 10:
                break
        # end batch

        # validation
        results, maps, times = test(
            data_dict,
            batch_size=batch_size * 2,
            imgsz=imgsz_test,
            model=model,
            dataloader=testloader,
            save_dir=save_dir,
            compute_loss=compute_loss,
            is_coco=True
        )

        # write results to file
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

        # update best fitness
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi
            print('Found higher fitness:', best_fitness)

        # save last + best
        model_tmp = copy.deepcopy(model)
        models.save(model_tmp)    
        torch.save(model_tmp.state_dict(), last)
        if best_fitness == fi:
            torch.save(model_tmp.state_dict(), best)
    # end batch
    
    # test best.pth
    # for m in (last, best) if best.exists() else (last):  # speed, mAP tests
    #     results, _, _ = test(
    #         data_dict,
    #         batch_size=batch_size * 2,
    #         imgsz=imgsz_test,
    #         conf_thres=0.001,
    #         iou_thres=0.7,
    #         model=attempt_load(m, device),
    #         dataloader=testloader,
    #         save_dir=save_dir,
    #         save_json=True,
    #         plots=False,
    #         is_coco=True
    #     )
    
    torch.cuda.empty_cache()
#
