"""
@author: QYZ
@time: 2021/11/26
@file: capsule_layers.py
@describe: This file aims to design some layers in Capsule Networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PrimaryCapsules2D(nn.Module):
    """
    Design the Primary Capsules Layers for Matrix Capsule Network.
    Args:
        in_channels: the input channels of tensors from above layers.
        out_caps: the numbers of output capsules in Primary Capsules Layers.
        pose_dim: the dimension of pose matrix.
        kernel_size: the kernel size of convolution 2D layers.
        stride: the stride of convolution 2D layers.
        padding=0: the padding of convolution 2D layers.
        weight_init: the parameters of weight initialization.
        feature Whether to use feature vector.
    Return:
        A Primary Capsules Layers Module.
    """

    def __init__(self, in_channels, out_caps, pose_dim, kernel_size,
                 stride, padding=0, weight_init='xavier_uniform', feature=False, feature_dim=4):
        super(PrimaryCapsules2D, self).__init__()
        # Initialize the __init__ parameters for input parameters
        self.A = in_channels
        self.B = out_caps
        self.P = pose_dim
        self.K = kernel_size
        self.S = stride
        self.padding = padding
        self.feature = feature
        self.feature_dim = feature_dim

        # Initialize the pose matrix of capsules → [B * P * P, A, K, K]
        w_kernel = torch.empty(self.B * self.P * self.P, self.A, self.K, self.K)
        # Initialize the activations of capsules → [B, A, K, K]
        a_kernel = torch.empty(self.B, self.A, self.K, self.K)
        # Initialize the feature vector of capsules → [B, V, A, K, K]
        f_kernel = torch.empty(self.B * self.feature_dim, self.A, self.K, self.K)

        # Initialize the weight for pose and activation
        if weight_init == 'kaiming_normal':
            nn.init.kaiming_normal_(w_kernel)
            nn.init.kaiming_normal_(a_kernel)
            nn.init.kaiming_normal_(f_kernel)
        elif weight_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w_kernel)
            nn.init.kaiming_uniform_(a_kernel)
            nn.init.kaiming_uniform_(f_kernel)
        elif weight_init == 'xavier_normal':
            nn.init.xavier_normal_(w_kernel)
            nn.init.xavier_normal_(a_kernel)
            nn.init.xavier_normal_(f_kernel)
        elif weight_init == 'xavier_uniform':
            nn.init.xavier_uniform_(w_kernel)
            nn.init.xavier_uniform_(a_kernel)
            nn.init.xavier_uniform_(f_kernel)
        else:
            NotImplementedError('{} not implemented.'.format(weight_init))

        # Adding BatchNorm layers
        self.BN_a = nn.BatchNorm2d(self.B, affine=True)
        self.BN_p = nn.BatchNorm3d(self.B, affine=True)
        self.BN_f = nn.BatchNorm3d(self.B, affine=True)

        if self.feature:
            # Initialize the convolutional kernel → [B*(P*P+1+V), A, K, K]
            self.weight = nn.Parameter(torch.cat([w_kernel, a_kernel, f_kernel], dim=0))
        else:
            # Initialize the convolutional kernel → [B*(P*P+1), A, K, K]
            self.weight = nn.Parameter(torch.cat([w_kernel, a_kernel], dim=0))
            del f_kernel
            del self.BN_f

    def forward(self, x):
        if self.feature:
            # Conv2d
            # In tensor shape [Batch, A, F, F] → conv operation self.weight kernel, B*(P*P+1+V) output channels
            # Out tensor shape → [Batch, B*(P*P+1+V), F', F']
            x = F.conv2d(x, weight=self.weight, stride=self.S, padding=self.padding)

            # Split pose and activation from tensor
            # In tensor shape [Batch, B*(P*P+1+V), F', F'] → split tensor in dimension 1
            # Out tensor shape → ([Batch, B*P*P, F', F'], [Batch, B, F', F'], [Batch, B*V, F', F'])
            poses, activations, feature = torch.split(x, [self.B * self.P * self.P, self.B, self.B * self.feature_dim],
                                                      dim=1)

            # Adding BatchNorm operation for pose matrix
            # In tensor shape [Batch, B*P*P, F', F'] → Reshape and BatchNorm operation
            # Out tensor shape → [Batch, B, P*P, F', F']
            poses = self.BN_p(poses.reshape(-1, self.B, self.P * self.P, *x.shape[2:]))

            # Reshape pose matrix
            # In tensor shape [Batch, B, P*P, F', F'] → Reshape operation
            # Out tensor shape → [Batch, B, P, P, F', F']
            poses = poses.reshape(-1, self.B, self.P, self.P, *x.shape[2:])

            # Adding BatchNorm operation and sigmoid function for activation
            # In tensor shape [Batch, B, F', F'] → BatchNorm and sigmoid operation
            # Out tensor shape → [Batch, B, F', F'])
            activations = torch.sigmoid(self.BN_a(activations))

            # Adding BatchNorm operation for feature vector
            # In tensor shape [Batch, B*V, F', F'] → Reshape and BatchNorm operation
            # Out tensor shape → [Batch, B, V, F', F']
            feature = self.BN_p(feature.reshape(-1, self.B, self.feature_dim, *x.shape[2:]))

            return activations, poses, feature
        else:
            # Conv2d
            # In tensor shape [Batch, A, F, F] → conv operation self.weight kernel, B*(P*P+1) output channels
            # Out tensor shape → [Batch, B*(P*P+1), F', F']
            x = F.conv2d(x, weight=self.weight, stride=self.S, padding=self.padding)

            # Split pose and activation from tensor
            # In tensor shape [Batch, B*(P*P+1), F', F'] → split tensor in dimension 1
            # Out tensor shape → ([Batch, B*P*P, F', F'], [Batch, B, F', F'])
            poses, activations = torch.split(x, [self.B * self.P * self.P, self.B], dim=1)

            # Adding BatchNorm operation for pose matrix
            # In tensor shape [Batch, B*P*P, F', F'] → Reshape and BatchNorm operation
            # Out tensor shape → [Batch, B, P*P, F', F']
            poses = self.BN_p(poses.reshape(-1, self.B, self.P * self.P, *x.shape[2:]))

            # Reshape pose matrix
            # In tensor shape [Batch, B, P*P, F', F'] → Reshape operation
            # Out tensor shape → [Batch, B, P, P, F', F']
            poses = poses.reshape(-1, self.B, self.P, self.P, *x.shape[2:])

            # Adding BatchNorm operation and sigmoid function for activation
            # In tensor shape [Batch, B, F', F'] → BatchNorm and sigmoid operation
            # Out tensor shape → [Batch, B, F', F'])
            activations = torch.sigmoid(self.BN_a(activations))

            return activations, poses


class ConvCapsules2D(nn.Module):
    """
    Convolutional Capsule Layer for Matrix Capsule Network.
    Args:
        in_caps: the number of input capsules.
        out_caps: the number of output capsules.
        pose_dim: the dimension of pose matrix.
        kernel_size: the kernel size of convolution 2D layers.
        stride: the stride of convolution 2D layers.
        padding=0: the padding of convolution 2D layers.
        weight_init: Which method to use to initialise the weights.
        share_W_ij: Whether to share weight parameters.
        coor_add: Coordinate addition for connecting the last convolutional capsule layer to the final layer.
        attention: Whether to use the attention mechanism.
    Return:
        A Convolution Capsules Layers Module.
    """

    def __init__(self, in_caps, out_caps, pose_dim, kernel_size, stride, padding=0,
                 weight_init='xavier_uniform', share_W_ij=False, coor_add=False,
                 attention=False, feature=False, feature_dim=16, in_h=14):
        super(ConvCapsules2D, self).__init__()
        # Initialize the __init__ parameters for input parameters
        self.B = in_caps
        self.C = out_caps
        self.P = pose_dim
        self.PP = np.max([2, self.P * self.P])
        self.K = kernel_size
        self.S = stride
        self.padding = padding
        # Share the transformation matrices across (F*F)
        self.share_W_ij = share_W_ij
        # Embed coordinates
        self.coor_add = coor_add
        # Whether to use attention
        self.attention = attention
        # Whether to use feature vector
        self.feature = feature
        self.feature_dim = feature_dim
        self.in_h = in_h

        # Initialize W_ij shape → [1, B, C, 1, P, P, 1, 1, K, K]
        self.W_ij = torch.empty(1, self.B, self.C, 1, self.P, self.P, 1, 1, self.K, self.K)
        # Initialize w_ij for feature shape → [1, B, C, 1, 1, 1, 1, K, K]
        self.W_f = torch.empty(1, self.B, self.C, 1, 1, 1, 1, self.K, self.K)
        # Initialize 3D spatial attention matrix → [1, B, 1, 1, 1]
        self.w_attention = torch.empty(self.B, self.B, 1, 3, 3)
        # Initialize 3D channel attention matrix → [B, B, 1, 3, 3]
        self.f_attention = torch.empty(self.B, self.B, 1, 3, 3)
        # Initialize q 3D Linear layer's weight → [B, B]
        self.q_weight = torch.empty(self.B, self.B)
        # Initialize q 3D Linear layer's bias → [B]
        self.q_bias = torch.empty(self.B)
        # Initialize k 3D Linear layer's weight → [B, B]
        self.k_weight = torch.empty(self.B, self.B)
        # Initialize k 3D Linear layer's bias → [B]
        self.k_bias = torch.empty(self.B)
        # Initialize v 3D Linear layer's weight → [B, B]
        self.v_weight = torch.empty(self.B, self.B)
        # Initialize v 3D Linear layer's bias → [B]
        self.v_bias = torch.empty(self.B)
        # Initialize o 3D Linear layer's weight → [B, B]
        self.o_weight = torch.empty(self.B, self.B)
        # Initialize o 3D Linear layer's bias → [B]
        self.o_bias = torch.empty(self.B)
        # Initialize o 3D Linear layer's weight → [B, B]
        self.o_weight_f = torch.empty(self.in_h * self.in_h, self.in_h * self.in_h)
        # Initialize o 3D Linear layer's bias → [B]
        self.o_bias_f = torch.empty(self.in_h * self.in_h)
        # Initialize linear
        nn.init.normal_(self.q_weight, std=0.001)
        nn.init.constant_(self.q_bias, 0)
        nn.init.normal_(self.v_weight, std=0.001)
        nn.init.constant_(self.v_bias, 0)
        nn.init.normal_(self.k_weight, std=0.001)
        nn.init.constant_(self.k_bias, 0)
        nn.init.normal_(self.o_weight, std=0.001)
        nn.init.constant_(self.o_bias, 0)
        nn.init.normal_(self.o_weight_f, std=0.001)
        nn.init.constant_(self.o_bias_f, 0)

        # Initialize the weight in convolution
        if weight_init.split('_')[0] == 'xavier':
            # in_caps types * receptive field size
            fan_in = self.B * self.K * self.K * self.PP
            # out_caps types * receptive field size
            fan_out = self.C * self.K * self.K * self.PP
            # in_feature types * receptive field size
            fea_in = self.B * self.K * self.K
            # out_caps types * receptive field size
            fea_out = self.C * self.K * self.K
            # Calculate variance and upper and lower limits
            std = np.sqrt(2. / (fan_in + fan_out))
            bound = np.sqrt(3.) * std
            std_f = np.sqrt(2. / (fea_in + fea_out))
            bound_f = np.sqrt(3.) * std_f

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std), requires_grad=True)
                self.W_f = nn.Parameter(self.W_f.normal_(0, std_f), requires_grad=True)
                nn.init.xavier_normal_(self.w_attention)
                nn.init.xavier_normal_(self.f_attention)
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound), requires_grad=True)
                self.W_f = nn.Parameter(self.W_f.uniform_(-bound_f, bound_f), requires_grad=True)
                nn.init.xavier_uniform_(self.w_attention)
                nn.init.xavier_uniform_(self.f_attention)
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init.split('_')[0] == 'kaiming':
            # fan_in preserves magnitude of the variance of the weights in the forward pass.
            fan_in = self.B * self.K * self.K * self.PP  # in_caps types * receptive field size
            # fan_out has same affect as fan_in for backward pass.
            # fan_out = self.C * self.K*self.K * self.PP # out_caps types * receptive field size
            # in_feature preserves magnitude of the variance of the weights in the forward pass.
            fea_in = self.B * self.K * self.K
            # Calculate variance and upper and lower limits
            std = np.sqrt(2.) / np.sqrt(fan_in)
            bound = np.sqrt(3.) * std
            std_f = np.sqrt(2.) / np.sqrt(fea_in)
            bound_f = np.sqrt(3.) * std_f

            if weight_init.split('_')[1] == 'normal':
                self.W_ij = nn.Parameter(self.W_ij.normal_(0, std), requires_grad=True)
                self.W_f = nn.Parameter(self.W_f.normal_(0, std_f), requires_grad=True)
                nn.init.kaiming_normal_(self.w_attention)
                nn.init.kaiming_normal_(self.f_attention)
            elif weight_init.split('_')[1] == 'uniform':
                self.W_ij = nn.Parameter(self.W_ij.uniform_(-bound, bound), requires_grad=True)
                self.W_f = nn.Parameter(self.W_f.uniform_(-bound_f, bound_f), requires_grad=True)
                nn.init.kaiming_uniform_(self.w_attention)
                nn.init.kaiming_uniform_(self.f_attention)
            else:
                raise NotImplementedError('{} not implemented.'.format(weight_init))

        elif weight_init == 'noisy_identity' and self.PP > 2:
            b = 0.01  # U(0,b)
            # Out → [1, B, C, 1, P, P, 1, 1, K, K]
            self.W_ij = nn.Parameter(
                torch.clamp(.1 * torch.eye(self.P, self.P).repeat(1, self.B, self.C, 1, 1, 1, self.K, self.K, 1, 1) +
                            torch.empty(1, self.B, self.C, 1, 1, 1, self.K, self.K, self.P, self.P).uniform_(0, b),
                            max=1).permute(0, 1, 2, 3, -2, -1, 4, 5, 6, 7), requires_grad=True)
            # Out → [1, B, C, V_i, V_o, 1, 1, K, K]
            self.W_f = nn.Parameter(self.W_f.uniform_(0, b), requires_grad=True)
        else:
            raise NotImplementedError('{} not implemented.'.format(weight_init))

        if self.padding != 0:
            if isinstance(self.padding, int):
                self.padding = [self.padding] * 4

        # Removal of redundant parameters
        if not self.attention:
            del self.w_attention
            del self.f_attention

    def attention_capsule(self, poses, features):
        """
        The properties are enhanced by using channel attention and spatial attention on features and poses,
        respectively.
        :param poses: Capsule pose → [Batch, B, P, P, F, F]
        :param features: Capsule feature → [Batch, B, V, F, F]
        :return: Capsule pose and feature matrices with spatial attention and channel attention, respectively.
        """
        """
        +--------------------------------------------------------------------------------------------+
        |                        VERSION1: ATTENTIVE FEATURE AGGREGATION                             |
        +--------------------------------------------------------------------------------------------+
        """
        # Initialize 3D spatial attention matrix → [1, B, 1, 1, 1]
        # self.w_attention = torch.empty(1, self.B, 1, 1, 1)
        # self.f_attention = torch.empty(self.B, self.B, 1, 1, 1)
        # Initialize 3D channel attention matrix → [1, B, 1, 1, 1]
        # Generating capsule space attention → [Batch, 1, P*P, F, F]
        # attention_pose = F.conv3d(poses.view(poses.shape[0], self.B, self.P * self.P, *poses.shape[4:]),
        #                           weight=self.w_attention.cuda(),
        #                           stride=1)
        # attention_pose = torch.sigmoid(attention_pose).view(poses.shape[0], 1, self.P, self.P, *poses.shape[4:])
        #
        # # Generating capsule channel attention → [Batch, B, V, 1, 1]
        # Max_pool_feature = nn.MaxPool3d(kernel_size=(1, *features.shape[3:]), stride=(1, 1, 1))
        # max_attention_feature = Max_pool_feature(features)
        # max_attention_feature = F.conv3d(max_attention_feature,
        #                                  weight=self.f_attention.cuda(),
        #                                  stride=1)
        # Avg_pool_feature = nn.AvgPool3d(kernel_size=(1, *features.shape[3:]), stride=(1, 1, 1))
        # avg_attention_feature = Avg_pool_feature(features)
        # avg_attention_feature = F.conv3d(avg_attention_feature,
        #                                  weight=self.f_attention.cuda(),
        #                                  stride=1)
        # attention_feature = torch.sigmoid(max_attention_feature + avg_attention_feature)
        """
        +--------------------------------------------------------------------------------------------+
        |                                          VERSION3: DANet                                   |
        +--------------------------------------------------------------------------------------------+
        """
        # Generating capsule space attention → [Batch, B, P*P, F*F]
        attention_pose = F.conv3d(poses.view(poses.shape[0], self.B, self.P * self.P, *poses.shape[4:]),
                                  weight=self.w_attention.cuda(),
                                  stride=1,
                                  padding=(0, 1, 1)).view(poses.shape[0], self.B, self.P * self.P, -1)
        # Pose shape → [Batch, P*P, F*F, B]
        attention_pose = attention_pose.permute(0, 2, 3, 1)
        # Q pose shape → [Batch, P*P, F*F, B]
        attention_pose_q = F.linear(attention_pose, weight=self.q_weight.cuda(), bias=self.q_bias.cuda())
        # K pose shape → [Batch, P*P, B, F*F]
        attention_pose_k = F.linear(attention_pose, weight=self.q_weight.cuda(), bias=self.q_bias.cuda()).permute(0, 1,
                                                                                                                  3, 2)
        # V pose shape → [Batch, P*P, F*F, B]
        attention_pose_v = F.linear(attention_pose, weight=self.q_weight.cuda(), bias=self.q_bias.cuda())
        # Attention pose → [Batch, P*P, F*F, F*F]
        attention_pose_o = torch.matmul(attention_pose_q, attention_pose_k) / np.sqrt(self.B)
        attention_pose_o = torch.softmax(attention_pose_o, dim=-1)
        # Attention pose → [Batch, P*P, F*F, B]
        attention_pose_o = torch.matmul(attention_pose_o, attention_pose_v)
        # Attention pose → [Batch, P*P, F*F, B]
        attention_pose_o = F.linear(attention_pose_o, weight=self.o_weight.cuda(), bias=self.o_bias.cuda())
        # Attention pose → [Batch, B, P*P, F*F]
        attention_pose = attention_pose.permute(0, 3, 1, 2) + attention_pose_o.permute(0, 3, 1, 2)
        # Attention pose → [Batch, B, P, P, F, F]
        attention_pose = attention_pose.view(poses.shape[0], self.B, self.P, self.P, *poses.shape[4:])

        # Generating capsule channel attention → [Batch, B, V, F*F]
        attention_feature = F.conv3d(features,
                                     weight=self.f_attention.cuda(),
                                     stride=1,
                                     padding=(0, 1, 1)).view(features.shape[0], self.B, features.shape[2], -1)
        # Q feature shape → [Batch, V, B, F*F]
        attention_feature_q = attention_feature.permute(0, 2, 1, 3)
        # K feature shape → [Batch, V, F*F, B]
        attention_feature_k = attention_feature.permute(0, 2, 3, 1)
        # V feature shape → [Batch, V, B, F*F]
        attention_feature_v = attention_feature.permute(0, 2, 1, 3)
        # Attention feature → [Batch, V, B, B]
        attention_feature_o = torch.matmul(attention_feature_q, attention_feature_k) / np.sqrt(self.in_h * self.in_h)
        attention_feature_o = torch.softmax(attention_feature_o, dim=-1)
        # Attention feature → [Batch, V, B, F*F]
        attention_feature_o = torch.matmul(attention_feature_o, attention_feature_v)
        # Attention feature → [Batch, V, B, F*F]
        attention_feature_o = F.linear(attention_feature_o, weight=self.o_weight_f.cuda(), bias=self.o_bias_f.cuda())
        # Attention feature → [Batch, B, V, F*F]
        attention_feature = attention_feature + attention_feature_o.permute(0, 2, 1, 3)
        # Attention feature → [Batch, B, V, F, F]
        attention_feature = attention_feature.view(features.shape[0], self.B, self.feature_dim, *features.shape[3:])

        return attention_pose, attention_feature

    def forward(self, activations, poses, features):
        if self.feature:
            # Padding tensor
            # In tensor shape ([Batch, B, F, F], [Batch, B, P, P, F, F], [Batch, B, V, F, F]) → Padding operation
            # Out tensor shape → ([Batch, B, F+2*padding, F+2*padding], [Batch, B, P, P, F+2*padding, F+2*padding])
            if self.padding != 0:
                activations = F.pad(activations, self.padding)  # [1,1,1,1]
                poses = F.pad(poses, self.padding + [0] * 4)  # [0,0,1,1,1,1]
                features = F.pad(features, self.padding + [0] * 4)  # [0,0,1,1,1,1]

            # Share the matrices over (F*F), if class caps layer
            if self.share_W_ij:
                # out_caps (C) feature map size
                self.K = poses.shape[-1]

            # Getting feature map size
            Caps_F = (poses.shape[-1] - self.K) // self.S + 1

            if self.attention:
                # Generating the attention matrix
                attention_pose, attention_feature = self.attention_capsule(poses, features)
                # Out tensor shape → [Batch, B, P, P, F, F]
                poses = poses * attention_pose
                # Out tensor shape → [Batch, B, V, F, F]
                features = features * attention_feature

            # Unfold operation
            # In tensor shape [Batch, B, P, P, F, F] → Unfold operation in dimension 5 then 4
            # Out tensor shape → [Batch, B, P, P, F', F', K, K]
            poses = poses.unfold(4, size=self.K, step=self.S).unfold(5, size=self.K, step=self.S)

            # Unsqueeze operation
            # In tensor shape [Batch, B, P, P, F', F', K, K] → Unsqueeze operation in dimension 5 then 2
            # Out tensor shape → [Batch, B, 1, P, P, 1, F', F', K, K]
            poses = poses.unsqueeze(2).unsqueeze(5)

            # Unfold operation
            # In tensor shape [Batch, B, V, F, F] → Unfold operation in dimension 5 then 4
            # Out tensor shape → [Batch, B, V, F', F', K, K]
            features = features.unfold(3, size=self.K, step=self.S).unfold(4, size=self.K, step=self.S)

            # Unsqueeze operation
            # In tensor shape [Batch, B, V, F', F', K, K] → Unsqueeze operation in dimension 5 then 2
            # Out tensor shape → [Batch, B, 1, V, 1, F', F', K, K]
            features = features.unsqueeze(2).unsqueeze(4)

            # Unfold operation
            # In tensor shape [Batch, B, F, F] → Unfold operation in dimension 3 then 2
            # Out tensor shape → [Batch, B, F', F', K, K]
            activations = activations.unfold(2, size=self.K, step=self.S).unfold(3, size=self.K, step=self.S)

            # Reshape operation
            # In tensor shape [Batch, B, F', F', K, K] → Reshape operation
            # Out tensor shape → [Batch, B, 1, 1, 1, F', F', K, K]
            activations = activations.reshape(-1, self.B, 1, 1, 1, *activations.shape[2:4], self.K, self.K)

            # Product generation votes
            # In tensor shape → ([Batch, B, 1, P, P, 1, F', F', K, K] * [1, B, C, 1, P, P, 1, 1, K, K])
            # Out tensor shape → [Batch, B, C, P, P, F', F', K, K]
            V_ji = (poses * self.W_ij).sum(dim=4)  # matmul equiv.

            # Reshape operation
            # In tensor shape [Batch, B, C, P, P, F', F', K, K] → Reshape operation
            # Out tensor shape → [Batch, B, C, P*P, 1, F', F', K, K]
            V_ji = V_ji.reshape(-1, self.B, self.C, self.P * self.P, 1, *V_ji.shape[-4:-2], self.K, self.K)

            # Product generation votes
            # In tensor shape → ([Batch, B, 1, V, 1, F', F', K, K] * [1, B, C, 1, 1, 1, 1, K, K])
            # Out tensor shape → [Batch, B, C, V, F', F', K, K]
            F_ji = (features * self.W_f).sum(dim=4)  # matmul equiv.

            # Reshape operation
            # In tensor shape [Batch, B, C, V, F', F', K, K] → Reshape operation
            # Out tensor shape → [Batch, B, C, V, 1, F', F', K, K]
            F_ji = F_ji.reshape(-1, self.B, self.C, self.feature_dim, 1, *F_ji.shape[-4:-2], self.K, self.K)

            if self.coor_add:
                # If class caps layer (feature map size = 1)
                if V_ji.shape[-1] == 1:
                    Caps_F = self.K  # 1->4

                # Coordinates = torch.arange(Caps_F, dtype=torch.float32) / Caps_F
                coordinates = torch.arange(Caps_F, dtype=torch.float32).add(1.) / (Caps_F * 10)
                # Coordinates for pose
                i_vals = torch.zeros(self.P * self.P, Caps_F, 1).cuda()
                j_vals = torch.zeros(self.P * self.P, 1, Caps_F).cuda()
                i_vals[self.P - 1, :, 0] = coordinates
                j_vals[2 * self.P - 1, 0, :] = coordinates
                # Coordinates for feature
                i_vals_f = torch.zeros(self.feature_dim, Caps_F, 1).cuda()
                j_vals_f = torch.zeros(self.feature_dim, 1, Caps_F).cuda()
                i_vals_f[self.feature_dim // 2 - 1, :, 0] = coordinates
                j_vals_f[self.feature_dim - 1, 0, :] = coordinates

                if V_ji.shape[-1] == 1:  # if class caps layer
                    # Out → [Batch, B, C, P*P, 1, 1, 1, K=F, K=F] (class caps)
                    V_ji = V_ji + (i_vals + j_vals).reshape(1, 1, 1, self.P * self.P, 1, 1, 1, Caps_F, Caps_F)
                    # Out → [Batch, B, C, V, 1, 1, 1, K=F, K=F] (class caps)
                    F_ji = F_ji + (i_vals_f + j_vals_f).reshape(1, 1, 1, self.feature_dim, 1, 1, 1, Caps_F, Caps_F)
                    return activations, V_ji, F_ji

                # Out → [Batch, B, C, P*P, 1, F, F, K, K]
                V_ji = V_ji + (i_vals + j_vals).reshape(1, 1, 1, self.P * self.P, 1, Caps_F, Caps_F, 1, 1)
                # Out → [Batch, B, C, V, 1, F, F, K, K]
                F_ji = F_ji + (i_vals_f + j_vals_f).reshape(1, 1, 1, self.feature_dim, 1, Caps_F, Caps_F, 1, 1)
            return activations, V_ji, F_ji
        else:
            # Padding tensor
            # In tensor shape ([Batch, B, F, F], [Batch, B, P, P, F, F]) → Padding operation
            # Out tensor shape → ([Batch, B, F+2*padding, F+2*padding], [Batch, B, P, P, F+2*padding, F+2*padding])
            if self.padding != 0:
                activations = F.pad(activations, self.padding)  # [1,1,1,1]
                poses = F.pad(poses, self.padding + [0] * 4)  # [0,0,1,1,1,1]

            # Share the matrices over (F*F), if class caps layer
            if self.share_W_ij:
                # out_caps (C) feature map size
                self.K = poses.shape[-1]

            # Getting feature map size
            Caps_F = (poses.shape[-1] - self.K) // self.S + 1

            if self.attention:
                # conv_pose = nn.Conv3d(poses.shape[1], 1, kernel_size=(1, 1, 1), stride=1)
                attention_pose = F.conv3d(poses.view(poses.shape[0], self.B, -1, poses.shape[-2], poses.shape[-1]),
                                          weight=self.w_attention.cuda(),
                                          stride=1)
                attention_pose = torch.sigmoid(attention_pose).view(poses.shape[0], 1, self.P, self.P, poses.shape[-2],
                                                                    poses.shape[-1])
                poses = poses * attention_pose

            # Unfold operation
            # In tensor shape [Batch, B, P, P, F, F] → Unfold operation in dimension 5 then 4
            # Out tensor shape → [Batch, B, P, P, F', F', K, K]
            poses = poses.unfold(4, size=self.K, step=self.S).unfold(5, size=self.K, step=self.S)

            # Unsqueeze operation
            # In tensor shape [Batch, B, P, P, F', F', K, K] → Unsqueeze operation in dimension 5 then 2
            # Out tensor shape → [Batch, B, 1, P, P, 1, F', F', K, K]
            poses = poses.unsqueeze(2).unsqueeze(5)

            # Unfold operation
            # In tensor shape [Batch, B, F, F] → Unfold operation in dimension 3 then 2
            # Out tensor shape → [Batch, B, F', F', K, K]
            activations = activations.unfold(2, size=self.K, step=self.S).unfold(3, size=self.K, step=self.S)

            # Reshape operation
            # In tensor shape [Batch, B, F', F', K, K] → Reshape operation
            # Out tensor shape → [Batch, B, 1, 1, 1, F', F', K, K]
            activations = activations.reshape(-1, self.B, 1, 1, 1, *activations.shape[2:4], self.K, self.K)

            # Product generation votes
            # In tensor shape → ([Batch, B, 1, P, P, 1, F', F', K, K] * [1, B, C, 1, P, P, 1, 1, K, K])
            # Out tensor shape → [Batch, B, C, P, P, F', F', K, K]
            V_ji = (poses * self.W_ij).sum(dim=4)  # matmul equiv.

            # Reshape operation
            # In tensor shape [Batch, B, C, P, P, F', F', K, K] → Reshape operation
            # Out tensor shape → [Batch, B, C, P*P, 1, F', F', K, K]
            V_ji = V_ji.reshape(-1, self.B, self.C, self.P * self.P, 1, *V_ji.shape[-4:-2], self.K, self.K)

            if self.coor_add:
                # if class caps layer (feature map size = 1)
                if V_ji.shape[-1] == 1:
                    Caps_F = self.K  # 1->4

                # coordinates = torch.arange(Caps_F, dtype=torch.float32) / Caps_F
                coordinates = torch.arange(Caps_F, dtype=torch.float32).add(1.) / (Caps_F * 10)
                i_vals = torch.zeros(self.P * self.P, Caps_F, 1).cuda()
                j_vals = torch.zeros(self.P * self.P, 1, Caps_F).cuda()
                i_vals[self.P - 1, :, 0] = coordinates
                j_vals[2 * self.P - 1, 0, :] = coordinates

                if V_ji.shape[-1] == 1:  # if class caps layer
                    # Out → [Batch, B, C, P*P, 1, 1, 1, K=F, K=F] (class caps)
                    V_ji = V_ji + (i_vals + j_vals).reshape(1, 1, 1, self.P * self.P, 1, 1, 1, Caps_F, Caps_F)
                    return activations, V_ji

                # Out → [Batch, B, C, P*P, 1, F, F, K, K]
                V_ji = V_ji + (i_vals + j_vals).reshape(1, 1, 1, self.P * self.P, 1, Caps_F, Caps_F, 1, 1)

            return activations, V_ji
