# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Blocks for classification."""

from typing import Optional

from mindspore import nn


class ConvNormActivation(nn.Cell):
    """
    Convolution/Depthwise fused with normalization and activation blocks definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.BatchNorm2d.
        activation (nn.Cell, optional): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> conv = ConvNormActivation(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 has_bias: bool = False
                 ) -> None:
        super(ConvNormActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                pad_mode='pad',
                padding=padding,
                group=groups,
                has_bias=has_bias
            )
        ]

        if norm:
            layers.append(norm(out_planes))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class Conv2dNormResAct(nn.Cell):
    """
    Convolution/Depthwise fused with normalization and activation blocks definition.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
        stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
        padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the input.
        residual (bool): Whether the input value needs to be added.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> conv = Conv2dNormResAct(16, 256, kernel_size=1, stride=1, padding=0)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, residual=False):
        super(Conv2dNormResAct, self).__init__()
        self.conv_block = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, "pad", padding, has_bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def construct(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dTransPadBN(nn.Cell):
    """
    Convolution/Depthwise fused with normalization and activation blocks definition.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
        stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
        padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the input.
        output_padding (int): The number of padding of the output.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> conv = Conv2dTransPadBN(16, 256, kernel_size=1, stride=1, padding=0)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(Conv2dTransPadBN, self).__init__()
        if output_padding == 1:
            self.conv_block = nn.SequentialCell(
                nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride, "pad", padding, has_bias=True),
                nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT"),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv_block = nn.SequentialCell(
                nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride, "pad", padding, has_bias=True),
                nn.BatchNorm2d(out_channels)
            )
        self.act = nn.ReLU()

    def construct(self, x):
        out = self.conv_block(x)
        return self.act(out)