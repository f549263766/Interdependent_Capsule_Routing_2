"""
@author: QYZ
@time: 2021/09/04
@file: tools.py
@describe: This file aims to record some functions.
"""
import os
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import prettytable as pt
import torch
import torch.nn as nn
from torch.utils.data import Subset

DATA_NAME = ["MNIST", "CIFAR10", "SVHN", "FashionMNIST", "smallNORB", "MultiMNIST", "affNIST"]
AZIMUTH = ['00', '02', '04', '34', '32', '30']
ELEVATION = ['00', '01', '02']


def split_dataset(train_data, random_indices=False):
    """
    This function aims to split dataset into train and validation set.
    Args:
        train_data: the divided input data set.
        random_indices: Whether to use randomly splitting.
    Return:
        train and validation set after splitting.
    """
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    if random_indices:
        np.random.seed(2021)
        np.random.shuffle(indices)

    train_idx = indices[split:]
    valid_idx = indices[:split]

    train_set = Subset(train_data, train_idx)
    valid_set = Subset(train_data, valid_idx)
    return train_set, valid_set


def one_hot_encode(target, length):
    """
    Converts batches of class indices to classes of one-hot vectors.
    Args:
        target: the input labels for transform to one-hot vector.
        length: the length of classification.
    Return:
        one_hot_vec
    """
    batch_s = np.size(target, 0)
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def clearing_unnecessary_parameters(args):
    """
    Clearing Unnecessary Parameters.
    Args:
        args: Namespace parameters.
    Returns:
        Namespace parameters after unnecessary parameters are cleared.
    """
    dict_args = vars(args).copy()
    for arg in dict_args:
        if getattr(args, arg) is not None:
            pass
        else:
            delattr(args, arg)
    return args


def setting_seed(seed=2021):
    """
    This function aims to fix a random seed before training.
    Args:
        seed: the input of setting seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imshow(img, channels=False, name="affNIST.png"):
    """
    Show the picture on the compute.
    Args:
        img: The input image for showing.
        channels: Enter whether the image shape contains the number of channels.
        name: Whether to save the picture in result dir.
    """
    path = "../datasets/images/"
    if not os.path.exists(path):
        os.makedirs(path)
    if channels:
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    else:
        plt.imshow(img)
    if name is None:
        pass
    else:
        plt.savefig(path + name)
    plt.show()


def colorstr(*str_input):
    """
    Adding color to the string.
    Args:
        *str_input: Any string and size of color or thickness.
    Returns:
        The transformed string.
    """
    *args, string = str_input if len(str_input) > 1 else ('blue', 'bold', str_input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def tab_print(**kwargs):
    """
    Output of the input parameters in a table
    Args:
        **kwargs: Parameter dictionary.
    """
    if len(kwargs.keys()) <= 0:
        raise ValueError("Please input correct configs, such as parameters!")
    else:
        configs = kwargs.get("configs", None)
        if configs:
            tb_configs = pt.PrettyTable()
            tb_configs.field_names = [colorstr("red", "bold", "Parameter"), colorstr("red", "bold", "Value")]
            for key in configs:
                tb_configs.add_row([colorstr("green", key), colorstr(configs[key])])
            tb_configs.junction_char = '╬'
            tb_configs.horizontal_char = '═'
            tb_configs.vertical_char = '║'
            print(tb_configs)


def show_process(step="STEP-1:", process=" INPUT THE REQUIRED PARAMETERS FOR THE PROJECT"):
    """
    Print the current process status.
    Args:
        step: Number of current processes.
        process: Current process status.
    """
    num_remaining = 87 - len(step + process)
    if num_remaining % 2 == 0:
        front = "║" + " " * ((num_remaining - 2) // 2)
        tail = " " * ((num_remaining - 2) // 2) + "║"
    else:
        front = "║" + " " * ((num_remaining - 2) // 2)
        tail = " " * ((num_remaining - 2) // 2 + 1) + "║"
    num_step = step
    process = process
    middle = front + colorstr("red", "bold", num_step) + colorstr("yellow", "bold", process) + tail
    top = ""
    bottom = ""
    for i in range(87):
        if i == 0:
            top += "╔"
            bottom += "╚"
        elif i == 86:
            top += "╕"
            bottom += "╝"
        else:
            top += "═"
            bottom += "═"
    total = top + "\n" + middle + "\n" + bottom
    print(total)


def model_summary(model, input_size, batch_size=-1, device="cuda"):
    """
    Print the overall structure of the model and the number of parameters, similar in style to keras.
    Args:
        model: Select the corresponding model.
        input_size: Tensor shapes for model inputs.
        batch_size: Training input batch size.
        device: Equipment where the model is located.
    """

    def register_hook(module):

        def hook(modules, inputs, output):
            class_name = str(modules.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(inputs[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [batch_size] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(modules, "weight") and hasattr(modules.weight, "size"):
                params += torch.prod(torch.LongTensor(list(modules.weight.size())))
                summary[m_key]["trainable"] = modules.weight.requires_grad
            if hasattr(modules, "bias") and hasattr(modules.bias, "size"):
                params += torch.prod(torch.LongTensor(list(modules.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print(colorstr("[*] The model structure is as follows "))
    tb_model = pt.PrettyTable()
    tb_model.field_names = [colorstr("red", "bold", "Layer (type)"),
                            colorstr("red", "bold", "Output Shape"),
                            colorstr("red", "bold", "Param #")]
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        tb_model.add_row([colorstr("yellow", layer),
                          colorstr(str(summary[layer]["output_shape"])),
                          colorstr("magenta", "{0:,}".format(summary[layer]["nb_params"]))])

        total_params += summary[layer]["nb_params"]
        try:
            total_output += np.product(summary[layer]["output_shape"])
        except TypeError:
            for i in range(len(summary[layer]["output_shape"])):
                total_output += np.prod(summary[layer]["output_shape"][i], dtype=np.float64)
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"]:
                    trainable_params += summary[layer]["nb_params"]
    tb_model.junction_char = '╬'
    tb_model.horizontal_char = '═'
    tb_model.vertical_char = '║'
    print(tb_model)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print(colorstr("[*] Total params: ") + colorstr("red", "bold", "{0:,}".format(total_params)))
    print(colorstr("[*] Trainable params: ") + colorstr("red", "bold", "{0:,}".format(trainable_params)))
    print(colorstr("[*] Non-trainable params: ") + colorstr("red", "bold",
                                                            "{0:,}".format(total_params - trainable_params)))
    print(colorstr("[*] Input size (MB): ") + colorstr("red", "bold", "{0:,}".format(total_input_size)))
    print(
        colorstr("[*] Forward/backward pass size (MB): ") + colorstr("red", "bold",
                                                                     "{:.2f}".format(total_output_size)))
    print(colorstr("[*] Params size (MB): ") + colorstr("red", "bold", "{:.2f}".format(total_params_size)))
    print(colorstr("[*] Estimated Total Size (MB): ") + colorstr("red", "bold", "{:.2f}".format(total_size)))
