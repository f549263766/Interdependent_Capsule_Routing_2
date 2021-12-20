"""
@author: QYZ
@time: 2021/11/26
@file: cla_routing.py
@describe: This file aims to design Interdependent Agreement Routing.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def reduce_icaps(x):
    return x.sum(dim=(1, -2, -1), keepdim=True)


class InterdependentRouting2D(nn.Module):
    """
    Interdependent agreement routing based on the composition of the capsules, i.e.
    applying spatial and channel attention mechanisms to the capsule pose and capsule
    feature vectors respectively to ensure their equivariance and invariance,
    and computing Interdependent and generating the properties of the advanced capsules separately.
    """

    def __init__(self, in_caps, out_caps, pose_dim, kernel_size, stride,
                 class_caps=False):
        super(InterdependentRouting2D, self).__init__()
        # Initial the setting of parameter
        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.D = np.max([2, self.P * self.P])
        self.K = kernel_size
        self.S = stride
        # Setting diag/full matrix
        self.class_caps = class_caps
        self.n_classes = out_caps if class_caps else None
        # Setting BatchNorm Layer for pose and activation
        self.BN_v = nn.BatchNorm3d(self.C, affine=False)
        self.BN_f = nn.BatchNorm3d(self.C, affine=False)
        self.BN_a = nn.BatchNorm2d(self.C, affine=False)

    # In tensor shape: pose → [Batch, B, C, P*P, 1, F, F, K, K]
    # In tensor shape: activation → [Batch, C, 1, 1, 1, F, F, K, K]
    # In tensor shape: feature → [Batch, B, C, V, 1, F, F, K, K]
    def forward(self, a_i, V_ji, F_ij):
        # Input capsule (B) votes feature map size (K)
        F_i = a_i.shape[-2:]
        # Output capsule (C) feature map size (F)
        F_o = a_i.shape[-4:-2]
        # Total num of lower level capsules
        N = self.B * F_i[0] * F_i[1]

        # Calculating the reciprocal correlation of pose → [Batch, 1, C, P*P, F, F, 1, 1]
        S_j = (.5 / np.sqrt(V_ji.shape[1])) * (torch.pow(reduce_icaps(V_ji), 2) -
                                               reduce_icaps(torch.pow(V_ji, 2)))
        # Squeeze dimension → [Batch, C, P*P, F, F]
        S_j = S_j.squeeze()
        # Normalized Pose Matrix → [Batch, C, P*P, F, F]
        S_j2 = torch.sum(S_j ** 2, dim=2, keepdim=True)
        pose_j = S_j / (torch.sqrt(S_j2) + 1e-8)
        # Calculating the reciprocal correlation of feature → [Batch, 1, C, V, F, F, 1, 1]
        F_j = (.5 / np.sqrt(F_ij.shape[1])) * (torch.pow(reduce_icaps(F_ij), 2) -
                                               reduce_icaps(torch.pow(F_ij, 2)))
        # Squeeze dimension → [Batch, C, V, F, F]
        F_j = F_j.squeeze()
        # Normalized Feature Vector → [Batch, C, V, F, F]
        F_j2 = torch.sum(F_j ** 2, dim=2, keepdim=True)
        feature_j = F_j / (torch.sqrt(F_j2) + 1e-8)

        # Calculating activation values  → [Batch, C, F, F]
        a_j = (S_j.sum(dim=2) + F_j.sum(dim=2)) / V_ji.shape[1]
        # So BN works in the class caps layer
        if self.class_caps:
            # Out tensor shape: activation → [Batch, C, 1, 1]
            a_j = a_j[..., None, None]
            # Out tensor shape: pose → [Batch, C, P*P, 1, 1]
            pose_j = pose_j[..., None, None]
            # Out tensor shape: feature → [Batch, C, V, 1, 1]
            feature_j = feature_j[..., None, None]

        # BN work in pose
        pose_j = self.BN_v(pose_j)
        # Reshape the pose → [Batch, C, P, P, F, F]
        pose_j = pose_j.reshape(-1, self.C, self.P, self.P, *F_o)
        # BN work in feature
        feature_j = self.BN_f(feature_j)
        # Sigmoid work in activation → [Batch, C, F, F]
        a_j = torch.sigmoid(self.BN_a(a_j))

        return a_j.squeeze(), pose_j.squeeze(), feature_j.squeeze()


if __name__ == "__main__":
    a_in = torch.zeros((2, 32, 1, 1, 1, 6, 6, 3, 3))
    b_in = torch.zeros((2, 16, 32, 16, 1, 6, 6, 3, 3))
    c_in = torch.zeros((2, 16, 32, 4, 1, 6, 6, 3, 3))
    model = InterdependentRouting2D(in_caps=16,
                                    out_caps=32,
                                    kernel_size=3,
                                    stride=2,
                                    pose_dim=4,
                                    )
    c_out = model(a_in, b_in, c_in)
    print(c_out[0].shape)
    print(c_out[1].shape)
    print(c_out[2].shape)
