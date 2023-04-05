import torch

import yaml
from pathlib import Path

from utils_yolo.finetune import load_model
from utils_kse.models import calculate_sparsity

def calculate_total_sparsity(model, save=''):
    l = calculate_sparsity(model)
    num_kse, num_full, _, _ = list(zip(*l))

    if isinstance(save, Path):
        with open(save, 'w') as f:
            for (_, _, cluster_num, group_size) in l:
                f.write('{:15s} {}\n'.format(str(cluster_num), group_size))
    
    print('Total sparsity: {:.2f}%'.format((sum(num_kse) / sum(num_full)) * 100))

# variables
model_save_location = 'best.pth'
G = 3
T = 0

data = './data/coco.yaml'
hyp = './data/hyp.scratch.tiny.yaml'
yolo_struct = './data/yolov7_tiny_struct.yaml'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# open data
with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
with open(data) as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)
with open(yolo_struct) as f:
    yolo_struct = yaml.load(f, Loader=yaml.SafeLoader)

# load model
nc = int(data_dict['nc'])
model = load_model(yolo_struct, nc, hyp.get('anchors'), model_save_location, G, T, device)

# find sparsity
calculate_total_sparsity(model, save=Path('output.txt'))