"""
@author: QYZ
@time: 2021/12/08
@file: loss_function.py
@describe: This file aims to design loss function.
"""
import math

import torch

from utils.tools import one_hot_encode


def spread_loss(pred, y, global_step, device):
    """
    In order to make the training less sensitive to the initialization and
    hyper-parameters of the model, we use “spread loss” to directly maximize the
    gap between the activation of the target class (a_t) and the activation of the
    other classes. If the activation of a wrong class, a_i, is closer than the
    margin, m, to at then it is penalized by the squared distance to the margin.
    :param device: Current operational outfit.
    :param global_step: Current number of train step.
    :param pred: Predicted for each class (batch_size, num_class).
    :param y: Index of class (batch).
    :return: Mean loss for entire batch (scalar)
    """
    # Set the m value
    m = 0.2 + 0.79 / (1 + math.exp(-(min(10.0, global_step / 50000.0 - 4))))
    # Get the number of classes
    num_class = pred.shape[1]
    # Transform label to one hot label (batch, num_class)
    y = one_hot_encode(target=y, length=num_class)
    # Reshape pred shape to (batch, 1, num_class)
    pred = pred.unsqueeze(dim=1)
    # Reshape label shape to (batch, num_class, 1)
    y = y.unsqueeze(dim=2).to(device)
    # Get target label at (batch, 1, 1)
    at = torch.matmul(pred, y)
    # Compute spread loss (batch, 1, num_class)
    loss = torch.pow((m - (at - pred)).clamp(0), 2)
    # Remove target weight (batch)
    loss = torch.matmul(loss, 1. - y)

    return torch.sum(loss)


def total_loss(pred, y, global_step, device):
    """

    :param device:
    :param pred:
    :param y:
    :param global_step:
    :return:
    """
    s_loss = spread_loss(pred, y, global_step, device)

    loss = s_loss

    return loss


if __name__ == "__main__":
    predict = torch.rand(5, 5)
    print(predict)
    label = torch.ones(5)
    label = label.type(torch.LongTensor)
    dev = torch.device("cpu")
    losses = total_loss(predict, label, 0, dev)
    print(losses)