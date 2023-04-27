import argparse
import sys
import time
import warnings
import yaml

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models.common as common
from models.experimental import attempt_load, End2End
from activations import Hardswish, SiLU
from general import set_logging, check_img_size
from torch_utils import select_device
from add_nms import RegisterNMS

from utils_yolo.finetune import *

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'
yolo_struct = './data/yolov7_tiny_struct.yaml'

G = 3
T = 0

if __name__ == '__main__':
    t = time.time()
    # arguments
    opt = dotdict({
        'weights': 'best_N18.pth',
        'onnx_name': 'best_480_640.onnx',
        'img_size': [480, 640],
        # 'img_size': [640, 640],
        'batch_size': 5,
        'dynamic': False,
        'dynamic_batch': False,
        'grid': True,
        'end2end': True,
        'max_wh': 640,
        'topk_all': 100,
        # 'topk_all': 1000,
        'iou_thres': 0.65,
        'conf_thres': 0.35,
        # 'conf_thres': 0.001,
        'simplify': False,
        'include_nms': False,
        'fp16': False,
        'int8': False
    })

    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    with open(yolo_struct) as f:
        yolo_struct = yaml.load(f, Loader=yaml.SafeLoader)


    # Load PyTorch model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nc = int(data_dict['nc']) 
    model = load_model(yolo_struct, nc, hyp.get('anchors'), opt.weights, G, T, device)
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.onnx_name if opt.onnx_name else opt.weights.replace('.pth', '.onnx')  # filename
        model.eval()
        output_names = ['classes', 'boxes'] if y is None else ['output']
        dynamic_axes = None
        if opt.dynamic:
            dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                'output': {0: 'batch', 2: 'y', 3: 'x'}}
        if opt.dynamic_batch:
            opt.batch_size = 'batch'
            dynamic_axes = {
                'images': {
                    0: 'batch',
                }, }
            if opt.end2end and opt.max_wh is None:
                output_axes = {
                    'num_dets': {0: 'batch'},
                    'det_boxes': {0: 'batch'},
                    'det_scores': {0: 'batch'},
                    'det_classes': {0: 'batch'},
                }
            else:
                output_axes = {
                    'output': {0: 'batch'},
                }
            dynamic_axes.update(output_axes)
        if opt.grid:
            if opt.end2end:
                print('\nStarting export end2end onnx model for %s...' % ('TensorRT' if opt.max_wh is None else 'onnxruntime'))
                model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels))
                if opt.end2end and opt.max_wh is None:
                    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                                opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            else:
                model.model[-1].concat = True

        torch.onnx.export(model, img, f, verbose=False, opset_version=16, input_names=['images'],
                            output_names=output_names,
                            dynamic_axes=dynamic_axes,
                            do_constant_folding=True)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        with open('onnx_readable.txt', 'w') as file:
            file.write(onnx.helper.printable_graph(onnx_model.graph))

        if opt.end2end and opt.max_wh is None:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        if opt.simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model, skip_shape_inference=True)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(onnx_model,f)
        print('ONNX export success, saved as %s' % f)

        if opt.include_nms:
            print('Registering NMS plugin for ONNX...')
            mo = RegisterNMS(f)
            mo.register_nms()
            mo.save(f)

    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
