# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" C3D network."""

import math
from typing import Tuple, Union

from mindspore import nn
from mindvideo.models.layers.dropout_dense import DropoutDense
from mindvideo.models.layers.c3d_backbone import C3DBackbone
from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = ['C3D']


@ClassFactory.register(ModuleType.MODEL)
class C3D(nn.Cell):
    """
    TODO: introduction c3d network.

    Args:
        in_d: Depth of input data, it can be considered as frame number of a video. Default: 16.
        in_h: Height of input frames. Default: 112.
        in_w: Width of input frames. Default: 112.
        in_channel(int): Number of channel of input data. Default: 3.
        kernel_size(Union[int, Tuple[int]]): Kernel size for every conv3d layer in C3D.
            Default: (3, 3, 3).
        head_channel(Tuple[int]): Hidden size of multi-dense-layer head. Default: [4096, 4096].
        num_classes(int): Number of classes, it is the size of classfication score for every sample,
            i.e. :math:`CLASSES_{out}`. Default: 400.
        keep_prob(Tuple[int]): Probability of dropout for multi-dense-layer head, the number of probabilities equals
            the number of dense layers.
        pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded
            from network. If `False`, it will create a c3d model with uniform initialization for weight and bias.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvideo.models import C3D
        >>>
        >>> net = C3D(16, 128, 128)
        >>> x = ms.Tensor(np.ones([1, 3, 16, 128, 128]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 400)

    About c3d:

    TODO: c3d introduction.

    Citation:

    .. code-block::

        TODO: c3d Citation.
    """

    def __init__(self,
                 in_d: int = 16,
                 in_h: int = 112,
                 in_w: int = 112,
                 in_channel: int = 3,
                 kernel_size: Union[int, Tuple[int]] = (3, 3, 3),
                 head_channel: Union[int, Tuple[int]] = (4096, 4096),
                 num_classes: int = 400,
                 keep_prob: Union[float, Tuple[float]] = (0.5, 0.5, 1.0)):
        super().__init__()
        last_d = math.ceil(in_d / 16)
        last_h = math.ceil((math.ceil(in_h / 16) + 1) / 2)
        last_w = math.ceil((math.ceil(in_w / 16) + 1) / 2)
        backbone_output_channel = 512 * last_d * last_h * last_w

        # backbone
        self.backbone = C3DBackbone(in_channel=in_channel,
                                    kernel_size=kernel_size)
        # flatten
        self.flatten = nn.Flatten()

        # classifier
        activations = ('relu', 'relu', None)
        if isinstance(head_channel, int):
            head_channel = (head_channel,)
        if isinstance(float, int):
            keep_prob = (keep_prob,)
        head_channel = list(head_channel)
        head_channel.insert(0, backbone_output_channel)
        head_channel.append(num_classes)
        dense_layers = []
        for i in range(len(head_channel)-1):
            dense_layers.append(DropoutDense(head_channel[i],
                                             head_channel[i+1],
                                             activation=activations[i],
                                             keep_prob=keep_prob[i]))
        self.classifier = nn.SequentialCell(dense_layers)

    def construct(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
