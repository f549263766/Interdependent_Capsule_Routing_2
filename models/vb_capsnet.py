"""
@author: QYZ
@time: 2021/11/26
@file: vb_capsnet.py
@describe: This file aims to design to Variational-Capsule-Network.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils.tools import model_summary

from models.vb_routing import VariationalBayesRouting2d
from models.capsule_layers import PrimaryCapsules2D, ConvCapsules2D


class CapsuleNet(nn.Module):
    """
    Example: Simple 3 layer CapsNet
    """

    def __init__(self, args):
        super(CapsuleNet, self).__init__()

        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P * self.P]))
        self.A, self.B, self.C, self.D = args.arch[:-1]
        self.n_classes = args.num_classes = args.arch[-1]
        self.in_channels = args.in_channels

        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.A,
                                kernel_size=5, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        self.BN_1 = nn.BatchNorm2d(self.A)

        self.PrimaryCaps = PrimaryCapsules2D(in_channels=self.A, out_caps=self.B,
                                             kernel_size=1, stride=1, pose_dim=self.P)

        self.ConvCaps_1 = ConvCapsules2D(in_caps=self.B, out_caps=self.C,
                                         kernel_size=3, stride=2, pose_dim=self.P)

        self.ConvRouting_1 = VariationalBayesRouting2d(in_caps=self.B, out_caps=self.C,
                                                       kernel_size=3, stride=2, pose_dim=self.P,
                                                       cov='diag', iter=args.routing_iter,
                                                       alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
                                                       Psi0=torch.eye(self.PP), nu0=self.PP + 1)

        self.ConvCaps_2 = ConvCapsules2D(in_caps=self.C, out_caps=self.D,
                                         kernel_size=3, stride=1, pose_dim=self.P)

        self.ConvRouting_2 = VariationalBayesRouting2d(in_caps=self.C, out_caps=self.D,
                                                       kernel_size=3, stride=1, pose_dim=self.P,
                                                       cov='diag', iter=args.routing_iter,
                                                       alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
                                                       Psi0=torch.eye(self.PP), nu0=self.PP + 1)

        self.ClassCaps = ConvCapsules2D(in_caps=self.D, out_caps=self.n_classes,
                                        kernel_size=1, stride=1, pose_dim=self.P, share_W_ij=True, coor_add=True)

        self.ClassRouting = VariationalBayesRouting2d(in_caps=self.D, out_caps=self.n_classes,
                                                      kernel_size=4, stride=1, pose_dim=self.P,
                                                      # adjust final kernel_size K depending on input H/W,
                                                      # for H=W=32, K=4.
                                                      cov='diag', iter=args.routing_iter,
                                                      alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
                                                      Psi0=torch.eye(self.PP), nu0=self.PP + 1, class_caps=True)

    def forward(self, x):
        # Out ← [?, A, F, F]
        x = F.relu(self.BN_1(self.Conv_1(x)))

        # Out ← a [?, B, F, F], v [?, B, P, P, F, F]
        a, v = self.PrimaryCaps(x)

        # Out ← a [?, B, 1, 1, 1, F, F, K, K], v [?, B, C, P*P, 1, F, F, K, K]
        a, v = self.ConvCaps_1(a, v, None)

        # Out ← a [?, C, F, F], v [?, C, P, P, F, F]
        a, v = self.ConvRouting_1(a, v)

        # Out ← a [?, C, 1, 1, 1, F, F, K, K], v [?, C, D, P*P, 1, F, F, K, K]
        a, v = self.ConvCaps_2(a, v, None)

        # Out ← a [?, D, F, F], v [?, D, P, P, F, F]
        a, v = self.ConvRouting_2(a, v)

        # Out ← a [?, D, 1, 1, 1, F, F, K, K], v [?, D, n_classes, P*P, 1, F, F, K, K]
        a, v = self.ClassCaps(a, v, None)

        # Out ← yhat [?, n_classes], v [?, n_classes, P, P]
        yhat, v = self.ClassRouting(a, v)

        return yhat


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_dim", type=int, default=4,
                        help="Dimensions of the gestalt matrix")
    parser.add_argument("--routing_iter", type=int, default=3,
                        help="Number of routing algorithm iterations")
    parser.add_argument("--arch", nargs='+', type=int, default=[64, 16, 32, 32, 5],
                        help="Number of output channels per capsule layer")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Enter the width of the image size")
    configs = parser.parse_args()
    model = CapsuleNet(configs)
    model = model.to(device)
    model_summary(model, (1, 32, 32), 2, device="cuda")
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
