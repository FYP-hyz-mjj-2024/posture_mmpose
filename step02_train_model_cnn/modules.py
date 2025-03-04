# Basics
from typing import Union, Tuple, Optional

# Essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss


# Local


class ResPool3d(nn.Module):
    """
    Residual Pooling. Uses max-pooling to get max voxels in each local chunk.
    Then, instead of leaving others out, ResPool3d adds the max-pooled voxels
    back to the original 3d channel.
    :param kernel_size: Size of the chunk kernel of max-pool.
    :param stride: Number of steps to jump over when max-pool kernel run through.
    :param padding: Edge padding of each channel.
    :param dilation: Dilation.  (Doesn't matter here.)
    :param ceil_mode: Ceiling mode. (Doesn't matter here.)
    """

    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation', 'ceil_mode']
    ceil_mode: bool

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Optional[Union[int, Tuple[int, ...]]] = None,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        _max_pooled, indices = F.max_pool3d(input, self.kernel_size,
                                            self.stride, self.padding,
                                            self.dilation, ceil_mode=self.ceil_mode,
                                            return_indices=True)

        output_shape = input.shape

        # Initialize a blank tensor "canvas" with the same shape as input.
        output = torch.zeros(output_shape, dtype=input.dtype).to('cuda')

        # Scatter the emphasized value into the blank "canvas".
        output = output.view(-1).scatter_(0, indices.view(-1), _max_pooled.view(-1)).view(output_shape)

        # Add the emphasized values into the original input tensor.
        return input + output


class FocalLoss(nn.Module):
    """
    Alpha-balanced variant of focal loss of one epoch.
    Reference: https://arxiv.org/pdf/1708.02002
    :param alpha: Linear parameter.
    :param gamma: Exponent parameter.
    :param reduction: Reduction method: mean or sum of all samples in this batch.
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.reduction not in ['mean', 'sum']:
            raise ValueError("Reduction method must be 'mean' or 'sum'.")

    def forward(self, inputs, labels):
        probs = torch.softmax(inputs, dim=-1)                           # get softmax of 2 classes
        pt = probs.gather(dim=-1, index=labels.unsqueeze(1))            # get p_t of groud-truth label class
        loss = (self.alpha * ((1 - pt) ** self.gamma)) * (-torch.log(pt))     # calculate focal loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class L2Regularization(nn.Module):
    """
    L2 Regularization. Prevents gradient explosion.
    :param l2_lambda: Lambda of the L2 regularization, penalize factor.
    """
    def __init__(self, l2_lambda=0.0001):
        super(L2Regularization, self).__init__()
        self.l2_lambda = l2_lambda

    def forward(self, _model):
        _device = next(_model.parameters()).device

        # Loss: Square sum of all weights.
        l2_reg = torch.tensor(0., device=_device)
        for param in _model.parameters():
            l2_reg += torch.sum(param ** 2)

        return self.l2_lambda * l2_reg


class MCLoss(nn.Module):
    """
    MCLoss stands for Mean Cross-entropy loss. It is a weighted combination of
    BCE Loss, Focal Loss and L2 Regularization.
    """
    def __init__(self, w1=0.6, w2=0.3, w3=0.1, focal_alpha=0.25, focal_gamma=2, l2_lambda=0.0001):
        super(MCLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.l2 = L2Regularization(l2_lambda)

    def forward(self, outputs, labels, mlp3d_instance):
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(outputs, labels)

        # Focal Loss
        focal_loss = self.focal(outputs, labels)

        # L2 Regularization
        l2_reg = self.l2(mlp3d_instance)

        # Total Loss
        total_loss = self.w1 * ce_loss + self.w2 * focal_loss + self.w3 * l2_reg
        return total_loss


class MLP3d(nn.Module):
    def __init__(self, input_channel_num, output_class_num):
        super(MLP3d, self).__init__()
        self.k = (3, 5, 5)          # Convolution3D kernel size
        self.m_k = 2                # MaxPool3D kernel stride
        self.activation = nn.SiLU()

        self.conv_layers = nn.Sequential(
            # Conv 1: C=2 -> C=8
            nn.Conv3d(in_channels=input_channel_num, out_channels=8, kernel_size=self.k, padding='same'),
            nn.BatchNorm3d(num_features=8),
            self.activation,

            # Conv 2: C=8 -> C=16
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=self.k, padding='same'),
            nn.BatchNorm3d(num_features=16),
            self.activation,

            # Conv 3: C=16 -> C=32
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=self.k, padding='same'),
            nn.BatchNorm3d(num_features=32),
            self.activation,

            # Residual Max-Pooling
            ResPool3d(kernel_size=self.m_k, stride=self.m_k, padding=0),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=29568, out_features=7392),
            self.activation,

            nn.Linear(in_features=7392, out_features=1848),
            self.activation,

            nn.Linear(in_features=1848, out_features=256),
            self.activation,

            nn.Linear(in_features=256, out_features=output_class_num)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 29568
        x = self.fc_layers(x)
        return x