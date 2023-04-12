import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import ExponentialLR

import random

from utils_yolo.general import check_img_size, colorstr, labels_to_class_weights
from utils_yolo.datasets import create_dataloader
from utils_yolo.autoanchor import check_anchors
from models.yolo import Model
from utils_kse import models



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
    scheduler = ExponentialLR(optimizer, gamma=hyp['lr_gamma'])
    return optimizer, scheduler

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

def load_data(model, img_size, data_dict, batch_size, hyp, num_workers, device, augment=True):
    gs = max(int(model.stride.max()), 32)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in img_size]
    dataloader, dataset = create_dataloader(data_dict['train'], imgsz, batch_size, gs, hyp=hyp,
                                            augment=augment,  workers=num_workers, prefix=colorstr('train: '))
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

###################################################################
# Failed attempt to torchify augmentation step
# Does not speedup training on Kaggle
# Keeping it here for future reference
###################################################################
def transform(imgs, targets, hyp, device):
    imgs = imgs.to(device, non_blocking=True)
    r = torch.FloatTensor(3).uniform_(-1, 1) * [hyp['hsv_h'], hyp['hsv_s'], hyp['hsv_v']] + [0, 1, 1]
    imgs = TF.adjust_hue(imgs, r[0])
    imgs = TF.adjust_saturation(imgs, r[1])
    imgs = TF.adjust_brightness(imgs, r[2])

    if random.random() < hyp['flipud']:
        imgs = torch.flipud(imgs)
        targets[:, 2] = 1 - targets[:, 2]
    if random.random() < hyp['fliplr']:
        imgs = torch.fliplr(imgs)
        targets[:, 1] = 1 - targets[:, 1]

    return imgs, targets.to(device)