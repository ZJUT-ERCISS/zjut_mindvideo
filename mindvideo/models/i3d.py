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
""" I3D network."""

from mindspore import nn
from mindspore import ops
from typing import Union, List, Tuple


from mindvideo.models.layers.unit3d import Unit3D
from mindvideo.models.layers.avgpool3d import AvgPool3D
from mindvideo.models.builder import build_layer
from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = ['I3D']

class AvgPooling3D(nn.Cell):
    """
    A module of average pooling for 3D video features.

    Args:
        kernel_size(Union[int, List[int], Tuple[int]]): The size of kernel window used to take the
            average value, Default: (1, 1, 1).
        strides(Union[int, List[int], Tuple[int]]): The distance of kernel moving. Default: (1, 1, 1).

    Inputs:
        x(Tensor): The input Tensor.

    Returns:
        Tensor, the pooled Tensor.
    """

    def __init__(self,
                 kernel_size: Union[int, List[int], Tuple[int]] = (1, 1, 1),
                 strides: Union[int, List[int], Tuple[int]] = (1, 1, 1),
                 ) -> None:
        super(AvgPooling3D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kernel_size = tuple(kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        strides = tuple(strides)

        self.pool = AvgPool3D(kernel_size, strides)

    def construct(self, x):
        x = self.pool(x)
        return x

@ClassFactory.register(ModuleType.LAYER)
class Inception3dModule(nn.Cell):
    """
    Inception3dModule definition.

    Args:
        in_channels (int):  The number of channels of input frame images.
        out_channels (int): The number of channels of output frame images.

    Returns:
        Tensor, output tensor.

    Examples:
        Inception3dModule(in_channels=3, out_channels=3)
    """

    def __init__(self, in_channels, out_channels):
        super(Inception3dModule, self).__init__()
        self.cat = ops.Concat(axis=1)
        self.b0 = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=(1, 1, 1))
        self.b1a = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[1],
            kernel_size=(1, 1, 1))
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            out_channels=out_channels[2],
            kernel_size=(3, 3, 3))
        self.b2a = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[3],
            kernel_size=(1, 1, 1))
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            out_channels=out_channels[4],
            kernel_size=(3, 3, 3))
        self.b3a = ops.MaxPool3D(
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            pad_mode="same")
        self.b3b = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[5],
            kernel_size=(1, 1, 1))

    def construct(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return self.cat((b0, b1, b2, b3))


@ClassFactory.register(ModuleType.LAYER)
class InceptionI3d(nn.Cell):
    """
    InceptionI3d architecture. TODO: i3d Inception backbone just in 3d?what about 2d. and two steam.

    Args:
        in_channels (int): The number of channels of input frame images(default 3).
    Returns:
        Tensor, output tensor.

    Examples:
        >>> InceptionI3d(in_channels=3)
    """

    def __init__(self, in_channels=3):

        super(InceptionI3d, self).__init__()

        self.conv3d_1a_7x7 = Unit3D(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2))
        self.maxpool3d_2a_3x3 = ops.MaxPool3D(
            kernel_size=(1, 3, 3),
            strides=(1, 2, 2),
            pad_mode="same")

        self.conv3d_2b_1x1 = Unit3D(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1, 1))

        self.conv3d_2c_3x3 = Unit3D(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3, 3))

        self.maxpool3d_3a_3x3 = ops.MaxPool3D(
            kernel_size=(1, 3, 3),
            strides=(1, 2, 2),
            pad_mode="same")

        self.mixed_3b = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 192,
                "out_channels": [64, 96, 128, 16, 32, 32]})

        self.mixed_3c = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 256,
                "out_channels": [128, 128, 192, 32, 96, 64]})

        self.maxpool3d_4a_3x3 = ops.MaxPool3D(
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            pad_mode="same")

        self.mixed_4b = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 128 + 192 + 96 + 64,
                "out_channels": [192, 96, 208, 16, 48, 64]})

        self.mixed_4c = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 192 + 208 + 48 + 64,
                "out_channels": [160, 112, 224, 24, 64, 64]})

        self.mixed_4d = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 160 + 224 + 64 + 64,
                "out_channels": [128, 128, 256, 24, 64, 64]})

        self.mixed_4e = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 128 + 256 + 64 + 64,
                "out_channels": [112, 144, 288, 32, 64, 64]})

        self.mixed_4f = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 112 + 288 + 64 + 64,
                "out_channels": [256, 160, 320, 32, 128, 128]})

        self.maxpool3d_5a_2x2 = ops.MaxPool3D(
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            pad_mode="same")

        self.mixed_5b = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 256 + 320 + 128 + 128,
                "out_channels": [256, 160, 320, 32, 128, 128]})

        self.mixed_5c = build_layer(
            {
                "type": "Inception3dModule",
                "in_channels": 256 + 320 + 128 + 128,
                "out_channels": [384, 192, 384, 48, 128, 128]})

        self.mean_op = ops.ReduceMean(keep_dims=True)
        self.concat_op = ops.Concat(axis=2)
        self.stridedslice_op = ops.StridedSlice()

    def construct(self, x):
        """Average pooling 3D construct."""
        x = self.conv3d_1a_7x7(x)
        x = self.maxpool3d_2a_3x3(x)
        x = self.conv3d_2b_1x1(x)
        x = self.conv3d_2c_3x3(x) 
        x = self.maxpool3d_3a_3x3(x)
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3d_4a_3x3(x)
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool3d_5a_2x2(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        return x


@ClassFactory.register(ModuleType.LAYER)
class I3dHead(nn.Cell):
    """
    I3dHead definition

    Args:
        in_channels: Input channel.
        num_classes (int): The number of classes .
        dropout_keep_prob (float): A float value of prob.

    Returns:
        Tensor, output tensor.

    Examples:
        I3dHead(in_channels=2048, num_classes=400, dropout_keep_prob=0.5)
    """

    def __init__(self, in_channels, num_classes=400, dropout_keep_prob=0.5):
        super(I3dHead, self).__init__()
        self._num_classes = num_classes
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=in_channels,
            out_channels=self._num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            norm=None,
            has_bias=True)
        self.mean_op = ops.ReduceMean()
        self.squeeze = ops.Squeeze(3)

    def construct(self, x):
        x = self.logits(self.dropout(x))
        x = self.squeeze(self.squeeze(x))
        x = self.mean_op(x, 2)
        return x


@ClassFactory.register(ModuleType.MODEL)
class I3D(nn.Cell):
    """
    TODO: introduction i3d network.

    Args:
        in_channel(int): Number of channel of input data. Default: 3.
        num_classes(int): Number of classes, it is the size of classfication score for every sample,
            i.e. :math:`CLASSES_{out}`. Default: 400.
        keep_prob(float): Probability of dropout for multi-dense-layer head, the number of probabilities equals
            the number of dense layers. Default: 0.5.
        pooling_keep_dim: whether to keep dim when pooling. Default: True.
        pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded
            from network. If `False`, it will create a i3d model with uniform initialization for weight and bias. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.mindvideo.models import i3d
        >>>
        >>> net = i3d()
        >>> x = ms.Tensor(np.ones([1, 3, 32, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 400)

    About i3d:

    TODO: i3d introduction.

    Citation:

    .. code-block::

        TODO: i3d Citation.
    """

    def __init__(self,
                 in_channel: int = 3,
                 num_classes: int = 400,
                 keep_prob: float = 0.5,
                 #pooling_keep_dim: bool = True,
                 backbone_output_channel=1024):
        super(I3D, self).__init__()

        self.backbone = InceptionI3d(in_channels=in_channel)
        #self.neck = ops.AvgPool3D(kernel_size=(2,7,7),strides=(1,1,1))
        self.neck = AvgPooling3D(kernel_size=(2,7,7))
        #self.neck = ops.ReduceMean(keep_dims=pooling_keep_dim)
        self.head = I3dHead(in_channels=backbone_output_channel,
                            num_classes=num_classes,
                            dropout_keep_prob=keep_prob)

    def construct(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x
