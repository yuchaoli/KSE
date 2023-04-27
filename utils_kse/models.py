# coding: utf-8
from tqdm import tqdm

import torch
import torch.nn as nn

from utils_kse.utils import Conv2d_KSE
from models.common import Conv_KSE, Concat

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def KSE(model, G=None, T=None, n=None, pbar=None):
    root = pbar is None
    if pbar is None:
        pbar = tqdm(total=n)
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                pbar.set_description(str(child))
                child.KSE_torch(G=G, T=T)
                pbar.update(1)
        else:
            KSE(child, G=G, T=T, pbar=pbar)
    
    if root:
        pbar.close()


def forward_init(model):
    n_remaining, n_total = 0, 0
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                a, b = child.forward_init()
                n_remaining += a
                n_total += b
        else:
            a, b = forward_init(child)
            n_remaining += a
            n_total += b
    return n_remaining, n_total

def add_state_dict_name_conv(model):
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_KSE) or isinstance(module, nn.BatchNorm2d):
            module.name = name
            if module.name == 'model.0.bn':
                module = nn.BatchNorm2d(28)

def set_output_channels(model, l):
    for i in range(len(l)-1):
        item = model.get_submodule(l[i])
        next_item = model.get_submodule(l[i+1])
        
        if hasattr(next_item, 'keep_cin'):
            item.keep_cout = next_item.keep_cin

def remove_unused_output_filters2(model, state_dict):
    # add_state_dict_name_conv(model)
    l_conv = get_all_conv(model)
    # set_output_channels(model, l_conv)

    for m in model.model:
        print(m)
    
    # update conv
    for name in l_conv:
        layer_name = name.replace('.conv', '')
        layer = model.get_submodule(layer_name)
        item = model.get_submodule(name)
        cout = len(item.keep_cout)
        item.full_weight = nn.Parameter(
            torch.Tensor(cout, item.group_size[-1], item.kernel_size, item.kernel_size), 
            requires_grad=True)
        
        w_name = name + '.full_weight'
        state_dict[w_name] = state_dict[w_name][item.keep_cout, :]

        # update corresponding bn
        bn_name = name.replace('conv', 'bn')
        layer.bn = nn.BatchNorm2d(cout, 1e-3, 0.03)
        for key in ['.'.join([bn_name, i]) for i in ['weight', 'bias', 'running_mean', 'running_var']]:
            state_dict[key] = state_dict[key][item.keep_cout]
    
    # print(item.full_weight.shape, state_dict[w_name].shape, item.keep_cout, model.get_submodule(bn_name))
    # print([len(item.r_cout) for item in l[:-1]])
    # print(l[0].r_cout)

    return state_dict

def get_all_conv(model, conv=[]):
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_KSE):
            conv.append(name)
    return conv

def get_conv_names(model):
    l = []
    for name, module in model.named_modules():
        print(name, type(module))
        if isinstance(module, Conv2d_KSE):
            print(name)
            l.append(name)
    return l

def set_next_nodes(model):
    # init index of next node
    for idx, node in enumerate(model.model[:-1]):
        assert idx == node.i
        node.n = []
    # add index of next node
    for idx, node in enumerate(model.model):
        if idx == 0:
            continue
        if isinstance(node.f, int):
            if node.f < 0:
                model.model[idx + node.f].n.append(node.i)
            else:
                model.model[node.f].n.append(node.i)
        elif isinstance(node.f, list):
            for f in node.f:
                if f < 0:
                    model.model[idx + f].n.append(node.i)
                else:
                    model.model[f].n.append(node.i)
    
    # for idx, node in enumerate(model.model[:-1]):
    #     for n in node.n:
    #         if isinstance(model.model[n], Concat):
    #             node.n.remove(n)
    #             node.n.extend(model.model[n].n)


def remove_unused_output_filters(model, state_dict):
    set_next_nodes(model)

    # iterate over nodes
    for idx, node in enumerate(model.model[:-1]):
        if isinstance(node, Concat):
            print('concat:', node.f)

        # ignore multiple next nodes for now
        if len(node.n) > 1:
            continue
        
        next_node = model.model[node.n[0]]
        # ignore everything except Conv_KSE
        if not isinstance(node, Conv_KSE) or not isinstance(next_node, Conv_KSE):
            continue


        cout = next_node.conv.keep_cin
        conv = node.conv
        # change parameter
        conv.full_weight = nn.Parameter(
            torch.Tensor(len(cout), conv.group_size[-1], conv.kernel_size, conv.kernel_size), 
            requires_grad=True)
        node.bn = nn.BatchNorm2d(len(cout), 1e-3, 0.03)

        # change state_dict
        key = 'model.{}.conv.full_weight'.format(idx)
        state_dict[key] = state_dict[key][cout, :]
        keys = ['model.{}.bn.{}'.format(idx, k) for k in ['weight', 'bias', 'running_mean', 'running_var']]
        for key in keys:
            state_dict[key] = state_dict[key][cout]
        print(state_dict[key].shape)

        # print(conv.full_weight.shape, state_dict[])
        # print(idx, node.n)

    return state_dict


def create_arch(model, G=None, T=None):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.create_arch(G=G, T=T)
        else:
            create_arch(child, G=G, T=T)

def create_arch2(model, G=None, T=None):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.create_arch2(G=G, T=T)
        else:
            create_arch2(child, G=G, T=T)

def load(model):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.load()
        else:
            load(child)


def save(model):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.save()
                # print(child, "save finish!")
        else:
            save(child)

def calculate_sparsity(model, l=[]):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                l.append(child.calculate_sparsity())
        else:
            calculate_sparsity(child, l=l)
    return l
