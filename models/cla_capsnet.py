"""
@author: QYZ
@time: 2021/11/26
@file: cla_capsnet.py
@describe: This file aims to design a Interdependent Capsule Network.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils.tools import model_summary

from models.cla_routing import InterdependentRouting2D
from models.capsule_layers import PrimaryCapsules2D, ConvCapsules2D


class CLACapsNet(nn.Module):
    """
    Designing a capsule network based on reciprocal correlation consistent routing.
    """

    def __init__(self, args):
        super(CLACapsNet, self).__init__()
        # Initial the setting of parameter
        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P * self.P]))
        self.A, self.B, self.C, self.D = args.arch[:-1]
        self.n_classes = args.num_classes
        self.in_channels = args.in_channels
        self.feature = args.feature
        self.feature_dim = args.feature_dim

        # Setting up the feature extraction convolution layer
        # In tensor shape [Batch, 1, 32, 32] → Convolution Layer with BatchNormalize 2d
        # Out tensor shape → [Batch, A, 14, 14]
        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.A,
                                kernel_size=5,
                                stride=2,
                                bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        self.BN_1 = nn.BatchNorm2d(self.A)

        # Setting up the Primary Capsule Layer
        # In tensor shape [Batch, A, 14, 14] → Primary Convolution Layer
        # Out tensor shape: pose → [Batch, B, P, P, 14, 14] with BatchNormalize 3d
        # Out tensor shape: activation → [Batch, B, 14, 14] with BatchNormalize 2d and sigmoid
        # or out tensor shape: activation → [Batch, B, V, 14, 14] with BatchNormalize 3d
        self.PrimaryCaps = PrimaryCapsules2D(in_channels=self.A,
                                             out_caps=self.B,
                                             kernel_size=1,
                                             stride=1,
                                             pose_dim=self.P,
                                             feature=self.feature,
                                             feature_dim=self.feature_dim)

        # Setting up the Convolution Capsule Layer 1
        # In tensor shape: pose → [Batch, B, P, P, 14, 14]
        # In tensor shape: activation → [Batch, B, 14, 14]
        # Or in tensor shape: feature → [Batch, B, V, 14, 14]
        # Out tensor shape: pose → [Batch, B, C, P*P, 1, 6, 6, 3, 3]
        # Out tensor shape: activation → [Batch, C, 1, 1, 1, 6, 6, 3, 3]
        # Or out tensor shape: feature → [Batch, B, C, V, 1, 6, 6, 3, 3]
        self.ConvCaps_1 = ConvCapsules2D(in_caps=self.B,
                                         out_caps=self.C,
                                         kernel_size=3,
                                         stride=2,
                                         pose_dim=self.P,
                                         attention=True,
                                         feature=self.feature,
                                         feature_dim=self.feature_dim,
                                         in_h=14)

        # Setting up the Interdependent Agreement Routing 1
        # In tensor shape: pose → [Batch, B, C, P*P, 1, 6, 6, 3, 3]
        # In tensor shape: activation → [Batch, C, 1, 1, 1, 6, 6, 3, 3]
        # In tensor shape: feature → [Batch, B, C, V, 1, 6, 6, 3, 3]
        # Out tensor shape: pose → [Batch, C, P, P, 6, 6]
        # Out tensor shape: activation → [Batch, C, 6, 6]
        # Or out tensor shape: feature → [Batch, C, V, 6, 6]
        self.ConvRouting_1 = InterdependentRouting2D(in_caps=self.B,
                                                     out_caps=self.C,
                                                     kernel_size=3,
                                                     stride=2,
                                                     pose_dim=self.P,
                                                     class_caps=False)

        # Setting up the Convolution Capsule Layer 2
        # In tensor shape: pose → [Batch, C, P, P, 6, 6]
        # In tensor shape: activation → [Batch, C, 6, 6]
        # Or in tensor shape: feature → [Batch, C, V, 6, 6]
        # Out tensor shape: pose → [Batch, C, D, P*P, 1, 4, 4, 3, 3]
        # Out tensor shape: activation → [Batch, D, 1, 1, 1, 4, 4, 3, 3]
        # Or out tensor shape: feature → [Batch, C, D, V, 1, 4, 4, 3, 3]
        self.ConvCaps_2 = ConvCapsules2D(in_caps=self.C,
                                         out_caps=self.D,
                                         kernel_size=3,
                                         stride=1,
                                         pose_dim=self.P,
                                         attention=True,
                                         feature=self.feature,
                                         in_h=6)

        # Setting up the Interdependent Agreement Routing 2
        # In tensor shape: pose → [Batch, C, D, P*P, 1, 4, 4, 3, 3]
        # In tensor shape: activation → [Batch, D, 1, 1, 1, 4, 4, 3, 3]
        # Or in tensor shape: feature → [Batch, C, D, V, 1, 4, 4, 3, 3]
        # Out tensor shape: pose → [Batch, D, P, P, 4, 4]
        # Out tensor shape: activation → [Batch, D, 4, 4]
        # Or out tensor shape: feature → [Batch, D, V, 4, 4]
        self.ConvRouting_2 = InterdependentRouting2D(in_caps=self.C,
                                                     out_caps=self.D,
                                                     kernel_size=3,
                                                     stride=1,
                                                     pose_dim=self.P,
                                                     class_caps=False)

        # Setting up the Class Convolution Capsule Layer
        # In tensor shape: pose → [Batch, D, P, P, 4, 4]
        # In tensor shape: activation → [Batch, D, 4, 4]
        # Or in tensor shape: feature → [Batch, D, V, 4, 4]
        # Out tensor shape: pose → [Batch, D, E, P*P, 1, 1, 1, 4, 4]
        # Out tensor shape: activation → [Batch, D, 1, 1, 1, 1, 1, 4, 4]
        # Or out tensor shape: feature → [Batch, D, E, V, 1, 1, 1, 4, 4]
        self.ClassCaps = ConvCapsules2D(in_caps=self.D,
                                        out_caps=self.n_classes,
                                        kernel_size=1,
                                        stride=1,
                                        pose_dim=self.P,
                                        share_W_ij=True,
                                        coor_add=True,
                                        attention=True,
                                        feature=self.feature,
                                        in_h=4)

        # Setting up the Class Interdependent Agreement Routing
        # In tensor shape: pose → [Batch, D, E, P*P, 1, 1, 1, 4, 4]
        # In tensor shape: activation → [Batch, D, 1, 1, 1, 1, 1, 4, 4]
        # Or in tensor shape: feature → [Batch, D, E, V, 1, 1, 1, 4, 4]
        # Out tensor shape: pose → [Batch, E, P, P]
        # Out tensor shape: activation → [Batch, E]
        # Or out tensor shape: feature → [Batch, E, V]
        self.ClassRouting = InterdependentRouting2D(in_caps=self.D,
                                                    out_caps=self.n_classes,
                                                    kernel_size=4,
                                                    stride=1,
                                                    pose_dim=self.P,
                                                    class_caps=True)

    def forward(self, x):
        # In tensor shape: [Batch, A, H, W]
        # Out tensor shape: [Batch, A, F, F]
        x = F.relu(self.BN_1(self.Conv_1(x)))

        # In tensor shape: [Batch, A, F, F]
        # Out tensor shape: pose → [Batch, B, P, P, F, F]
        # Out tensor shape: activation → [Batch, B, F, F]
        a, v, f = self.PrimaryCaps(x)

        # In tensor shape: pose → [Batch, B, P, P, F, F]
        # In tensor shape: activation → [Batch, B, F, F]
        # Out tensor shape: pose → [Batch, B, C, P*P, 1, F, F, K, K]
        # Out tensor shape: activation → [Batch, C, 1, 1, 1, F, F, K, K]
        a, v, f = self.ConvCaps_1(a, v, f)

        # In tensor shape: pose → [Batch, B, C, P*P, 1, F, F, K, K]
        # In tensor shape: activation → [Batch, C, 1, 1, 1, F, F, K, K]
        # Out tensor shape: pose → [Batch, C, P, P, F, F]
        # Out tensor shape: activation → [Batch, C, F, F]
        a, v, f = self.ConvRouting_1(a, v, f)

        # In tensor shape: pose → [Batch, C, P, P, F, F]
        # In tensor shape: activation → [Batch, C, F, F]
        # Out tensor shape: pose → [Batch, C, D, P*P, 1, F, F, K, K]
        # Out tensor shape: activation → [Batch, D, 1, 1, 1, F, F, K, K]
        a, v, f = self.ConvCaps_2(a, v, f)

        # In tensor shape: pose → [Batch, C, D, P*P, 1, F, F, K, K]
        # In tensor shape: activation → [Batch, D, 1, 1, 1, F, F, K, K]
        # Out tensor shape: pose → [Batch, D, P, P, F, F]
        # Out tensor shape: activation → [Batch, D, F, F]
        a, v, f = self.ConvRouting_2(a, v, f)

        # In tensor shape: pose → [Batch, D, P, P, F, F]
        # In tensor shape: activation → [Batch, D, F, F]
        # Out tensor shape: pose → [Batch, D, E, P*P, 1, F, F, K, K]
        # Out tensor shape: activation → [Batch, D, 1, 1, 1, F, F, K, K]
        a, v, f = self.ClassCaps(a, v, f)

        # In tensor shape: pose → [Batch, D, E, P*P, 1, F, F, K, K]
        # In tensor shape: activation → [Batch, D, 1, 1, 1, F, F, K, K]
        # Out tensor shape: pose → [Batch, n_classes, P, P]
        # Out tensor shape: activation → [Batch, n_classes]
        y_hat, v, f = self.ClassRouting(a, v, f)

        return y_hat


if __name__ == "__main__":
    torch.cuda.set_device(1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_dim", type=int, default=3,
                        help="Dimensions of the gestalt matrix")
    parser.add_argument("--arch", nargs='+', type=int, default=[32, 8, 8, 8, 5],
                        help="Number of output channels per capsule layer")
    parser.add_argument("--in_channels", type=int, default=2,
                        help="Enter the width of the image size")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes")
    parser.add_argument("--feature", type=bool, default=True,
                        help="Whether to use feature")
    parser.add_argument("--feature_dim", type=int, default=16,
                        help="Number of classes")
    configs = parser.parse_args()
    model = CLACapsNet(configs)
    model = model.to(device)
    model_summary(model, (2, 32, 32), 64, device="cuda")
    for param in model.state_dict():
        p_name = param.split('.')[-2] + '.' + param.split('.')[-1]
        if p_name[:2] != 'BN':  # don't print batch norm layers
            print('{:>25} {:>27} {:>15}'.format(
                p_name,
                str(list(model.state_dict()[param].squeeze().size())),
                '{0:,}'.format(np.product(list(model.state_dict()[param].size())))))
    print('-' * 70)
    print('Total params: {:,}'.format(
        sum(p.numel() for p in model.parameters())))
