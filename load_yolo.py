import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.cuda import amp

import yaml
import numpy as np
import time
import os
import math
import logging
import random
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from threading import Thread

from yolo_test import test
from models.yolo import Model
from models.experimental import attempt_load
from utils.general import check_dataset, check_img_size, colorstr, labels_to_class_weights, fitness, \
    increment_path, init_seeds
from utils.datasets import create_dataloader
from utils.autoanchor import check_anchors
from utils.loss import ComputeLoss
from utils.torch_utils import ModelEMA
from utils.plots import plot_images, plot_results

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # opt = {}
    # opt['data'] = './data/coco.yaml'
    # opt['weights'] = './yolov7-tiny.pt' # 'yolov7.pt'
    # opt['batch_size'] = 4
    # opt['img_size'] = 640
    # opt['conf_thres'] = 0.001
    # opt['iou_thres'] = 0.65
    # opt['save_json'] = True
    # opt['single_cls'] = False
    # opt['augment'] = False
    # opt['verbose'] = False
    # opt['half_precision'] = False
    # opt['device'] = '0'
    # opt['project'] = 'runs/test'
    # opt['name'] = 'tiny'
    # opt['task'] = 'test'

    # test(
    #     opt['data'],
    #     opt['weights'],
    #     opt['batch_size'],
    #     opt['img_size'],
    #     opt['conf_thres'],
    #     opt['iou_thres'],
    #     opt['save_json'],
    #     opt['single_cls'],
    #     opt['augment'],
    #     opt['verbose'],
    #     half_precision=opt['half_precision'],
    #     opt=dotdict(opt)
    # )

    # exit()

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
    
    init_seeds(1)

    # directories
    project = 'runs/training'
    name = 'debugging'
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

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
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # load model
    ckpt = torch.load(weights, map_location=device) # = {'model', 'optimizer', 'training_results', 'epoch'}
    state_dict = ckpt['model'].float().state_dict()
    model = Model(ckpt['model'].yaml, nc=nc, anchors=hyp.get('anchors')).to(device)
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict

    # 
    nbs = 64    # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)

    # create optimizer parameter groups
    # pg0: no decay; pg1: normal weights, weight decay; pg2: biases, no decay
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
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

    assert len(pg0) == 55 and len(pg1) == 58 and len(pg2) == 58, 'Found {}, {}, {}'.format(len(pg0), len(pg1), len(pg2))

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2

    ema = ModelEMA(model)

    start_epoch = 0
    best_fitness = 0.0

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in img_size]  # verify imgsz are gs-multiples
    print(imgsz, imgsz_test)
    # data loaders
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs,hyp=hyp,
                                            augment=True,  workers=num_workers, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, data, nc - 1)

    
    testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs,  # testloader
                                       hyp=hyp, rect=True, 
                                       workers=num_workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]
    
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])

    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    model.float()

    # model params
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # start training
    t0 = time.time()
    # nw = 1000   # number of warmup iterations
    maps = np.zeros(nc) # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    torch.save(model, wdir / 'init.pt')


    print('Test before training:')
    results, maps, times = test(
        data_dict,
        batch_size=batch_size * 2,
        imgsz=imgsz_test,
        model=ema.ema,
        dataloader=testloader,
        save_dir=save_dir,
        verbose=False,
        plots=False,
        compute_loss=compute_loss,
        is_coco=True
    )
    fi = fitness(np.array(results).reshape(1, -1))
    print('Start fitness:', fi)
    for epoch in range(start_epoch, epochs):
        model.train()

        mloss = torch.zeros(4, device=device)

        pbar = tqdm(enumerate(dataloader), total=nb)
        optimizer.zero_grad()

        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        ix = 0
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch # number integrated batches (since train start)
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
                if ema:
                    ema.update(model)
            
            # print
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate / 60 if rate and pbar.total else 0
            # remaining_str = '{}:{:02d}'.format(math.floor(remaining / 60), int(remaining % 60))
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # plot
            if plots and ni < 10:
                f = save_dir / f'train_batch{ni}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

            # break training epoch short to test entire code
            if ix == 20:
                break
        # end batch

        final_epoch = epoch + 1 == epochs
        
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        # test results
        results, maps, times = test(
            data_dict,
            batch_size=batch_size * 2,
            imgsz=imgsz_test,
            model=ema.ema,
            dataloader=testloader,
            save_dir=save_dir,
            verbose=nc < 50 and final_epoch,
            plots=final_epoch,
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

        # save model
        ckpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'training_results': results_file.read_text(),
            'model': deepcopy(model),
            'ema': deepcopy(ema.ema),
            'updates': ema.updates,
            'optimizer': optimizer.state_dict()
        }

        # Save last, best and delete
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        if (best_fitness == fi) and (epoch >= 200):
            torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
        if epoch == 0:
            torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
        elif ((epoch+1) % 25) == 0:
            torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
        elif epoch >= (epochs-5):
            torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
        del ckpt

        # end epoch
    # end training

    # plot
    if plots:
        plot_results(save_dir=save_dir)
    
    # test best.pt
    for m in (last, best) if best.exists() else (last):  # speed, mAP tests
        results, _, _ = test(
            data_dict,
            batch_size=batch_size * 2,
            imgsz=imgsz_test,
            conf_thres=0.001,
            iou_thres=0.7,
            model=attempt_load(m, device),
            dataloader=testloader,
            save_dir=save_dir,
            save_json=True,
            plots=False,
            is_coco=True
        )
    
    torch.cuda.empty_cache()

# TODO / not implemented:
#   - !! warmup / scheduler (when finetuning)
#   - logger

# losses increase after training, testing declines rapidly
# - check if training from scratch works
# - check if lower learning rate helps

