# coding: utf-8

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def KSE(model, G=None, T=None):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.KSE(G=G, T=T)
                print(child, "KSE finish!")
        else:
            KSE(child, G=G, T=T)


def forward_init(model):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.forward_init()
                print(child, "forward_init finish!")
        else:
            forward_init(child)


def create_arch(model, G=None, T=None):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.create_arch(G=G, T=T)
                print(child, "create arch finish!")
        else:
            create_arch(child, G=G, T=T)


def load(model):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in ["Conv2d_KSE"]:
                child.load()
                print(child, "load finish!")
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