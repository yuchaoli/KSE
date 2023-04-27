import argparse
import json
import os
from pathlib import Path
from threading import Thread
import sys

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import numpy as np
import torch
import yaml
from tqdm import tqdm
import onnxruntime

from utils_yolo.datasets import create_dataloader
from utils_yolo.general import coco80_to_coco91_class, check_dataset, fitness, \
    box_iou, scale_coords, xyxy2xywh, xywh2xyxy, increment_path, colorstr
from utils_yolo.metrics import ap_per_class, ConfusionMatrix
from utils_yolo.plots import plot_images, output_to_target
from utils_yolo.torch_utils import time_synchronized


# NOTE: pred[4:6] = [category_id, conf] i.s.o. [conf, category_id]

if __name__ == '__main__':
    # Initialize/load model and set device
    training = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    onnx_file = 'test.onnx'
    data = './data/coco.yaml'
    hyp = './data/hyp.scratch.tiny.yaml'

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_file, providers=providers)

    save_json = True
    save_txt = False
    plots = False

    save_dir = Path(increment_path(Path('tmp/yolo_kse/testing'), exist_ok=False))
    os.makedirs(save_dir, exist_ok=True)

    # open data info
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data_dict)

    img_size = 640
    batch_size = 5
    num_workers = 4

    # load validation dataset
    dataloader = create_dataloader(data_dict['val'], img_size, batch_size, 32, 
                                            hyp=hyp, rect=False, 
                                            workers=num_workers,
                                            pad=0.5, prefix=colorstr('val: '))[0]

    # Configure
    is_coco = True
    nc =  int(data_dict['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(data_dict['names'])}
    coco91class = coco80_to_coco91_class()
    s = ('%12s' * 4) % ('P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # img = img.to(device, non_blocking=True)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        ort_inputs = { ort_session.get_inputs()[0].name: np.array(img) }

        with torch.no_grad():
            # Run model
            # try:
            #     import matplotlib.pyplot as plt
            #     fig, axs = plt.subplots(2,3)
            #     axs[0,0].imshow(img[0].permute(1,2,0))
            #     axs[0,1].imshow(img[1].permute(1,2,0))
            #     axs[0,2].imshow(img[2].permute(1,2,0))
            #     axs[1,0].imshow(img[3].permute(1,2,0))
            #     axs[1,1].imshow(img[4].permute(1,2,0))
            #     # plt.imshow(img[0].permute(1,2,0))
            #     plt.show()
            # except Exception as e:
            #     print(e)

            
            t = time_synchronized()
            ort_outs = torch.Tensor(ort_session.run(None, ort_inputs)[0]).to(device)
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            t = time_synchronized()
            t1 += time_synchronized() - t

        # Statistics per image
        for si in range(batch_size):
            pred = ort_outs[ort_outs[:, 0] == si, 1:]
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                    'category_id': coco91class[int(p[4])] if is_coco else int(p[4]),
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[5], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 4]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 4].cpu(), tcls))

        # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # # Print results
    pf = '%12.3g' * 4  # print format
    print(pf % (mp, mr, map50, map))

    # # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))


    # Save JSON
    if save_json and len(jdict):
        anno_json = data_dict['anno_json']
        pred_json = save_dir / 'predictions.json'  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(str(pred_json))  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

# maps = np.zeros(nc) + map
# for i, c in enumerate(ap_class):
#     maps[c] = ap[i]
# # if 'eval' in locals():
# #     return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, eval.stats
# print((mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t)
    f = fitness(np.array((mp, mr, map50, map, *(loss.cpu() / len(dataloader)))).reshape(1, -1))
    print('Fitness:', f)