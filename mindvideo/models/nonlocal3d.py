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
""" nonlocal3d network."""

import math
from typing import List, Optional, Tuple

import mindspore
from mindspore import ops, nn

from mindvideo.models.layers.resnet3d import ResNet3D, ResidualBlockBase3D, ResidualBlock3D
from mindvideo.models.layers.maxpool3dwithpad import Maxpool3DwithPad
from mindvideo.models.layers.adaptiveavgpool3d import AdaptiveAvgPool3D
from mindvideo.models.layers.dropout_dense import DropoutDense
from mindvideo.models.layers.inflate_conv3d import Inflate3D
from mindvideo.models.layers.unit3d import Unit3D
from mindvideo.models.layers.maxpool3d import MaxPool3D
from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = [
    'NonLocalBlockND',
    'NLInflateBlockBase3D',
    'NLInflateBlock3D',
    'NLInflateResNet3D',
    'NLResInflate3D50',
    'nonlocal3d'
]


class NonLocalBlockND(nn.Cell):
    r"""
    Classification backbone for nonlocal.
    Implementation of Non-Local Block with 4 different pairwise functions.

    Applies Non-Local Block over 5D input (a mini-batch of 3D inputs with additional channel dimension).
    .. math::
        embedded_gaussian:
        f(x_i, x_j)=e^{\theta(x_i)^{T} \phi(x_j)}.
        gaussian:
        f(x_i, x_j)=e^{{x_i}^{T} {x_j}}.
        concatenation:
        f(x_i, x_j)=\{ReLU}({w_f}^{T}[\theta(x_i), \phi(x_j)]).
        dot_product:
        f(x_i, x_j)=\theta(x_i)^{T} \phi(x_j).

    Args:
        in_channels (int): original channel size.
        inter_channels (int): channel size inside the block if not specified reduced to half.
        mode: 4 mode to choose (gaussian, embedded, dot, and concatenation).
        bn_layer: whether to add batch norm.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = nn.NonLocalBlockND(in_channels=3, bn_layer=bn_layer)
        >>> x = zeros((2, 3, 8, 20, 20), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (2, 3, 8, 20, 20)
        """

    def __init__(
            self,
            in_channels,
            inter_channels=None,
            mode='embedded',
            sub_sample=True,
            bn_layer=True):

        super(NonLocalBlockND, self).__init__()

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenation']:
            raise ValueError(
                '`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenation`')
        self.mode = mode
        self.transpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        self.tile = ops.Tile()
        self.concat_op = ops.Concat(1)
        self.zeros = ops.Zeros()
        self.softmax = ops.Softmax(axis=-1)

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv3d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           has_bias=True
                           )

        if bn_layer:
            self.w = nn.SequentialCell(
                nn.Conv3d(in_channels=self.inter_channels,
                          out_channels=self.in_channels,
                          kernel_size=1
                          ),
                nn.BatchNorm3d(self.in_channels)
            )
        else:
            self.w = nn.Conv3d(in_channels=self.inter_channels,
                               out_channels=self.in_channels,
                               kernel_size=1
                               )
        if self.mode in ["embedded", "dot", "concatenation"]:
            self.theta = nn.Conv3d(in_channels=self.in_channels,
                                   out_channels=self.inter_channels,
                                   kernel_size=1,
                                   has_bias=True
                                   )
            self.phi = nn.Conv3d(in_channels=self.in_channels,
                                 out_channels=self.inter_channels,
                                 kernel_size=1,
                                 has_bias=True
                                 )
        if self.mode == "concatenation":
            self.concat_project = nn.SequentialCell(
                nn.Conv2d(
                    self.inter_channels * 2,
                    out_channels=1,
                    kernel_size=1,
                    pad_mode='same',
                    has_bias=False),
                nn.ReLU()
            )

        if sub_sample:
            max_pool_layer = MaxPool3D(
                kernel_size=(1, 2, 2), strides=(1, 2, 2))
            self.g = nn.SequentialCell(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.SequentialCell(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer

    def construct(self, x):
        "nonlocalblock construct."
        batch_size = x.shape[0]
        g_x = self.g(x).view((batch_size, self.inter_channels, -1))
        input_perm = (0, 2, 1)
        g_x = self.transpose(g_x, input_perm)
        f = self.zeros((1, 1, 1), mindspore.float32)
        if self.mode == "gaussian":
            theta_x = x.view((batch_size, self.in_channels, -1))
            theta_x = self.transpose(theta_x, input_perm)
            phi_x = x.view(batch_size, self.in_channels, -1)
            f = self.batmatmul(theta_x, phi_x)
        elif self.mode in ["embedded", "dot"]:
            theta_x = self.theta(x).view((batch_size, self.inter_channels, -1))
            theta_x = self.transpose(theta_x, input_perm)
            phi_x = self.phi(x).view((batch_size, self.inter_channels, -1))
            f = self.batmatmul(theta_x, phi_x)
        elif self.mode == "concatenation":
            theta_x = self.theta(x).view(
                (batch_size, self.inter_channels, -1, 1))
            phi_x = self.phi(x).view((batch_size, self.inter_channels, 1, -1))
            h = theta_x.shape[2]
            w = phi_x.shape[3]
            theta_x = self.tile(theta_x, (1, 1, 1, w))
            phi_x = self.tile(phi_x, (1, 1, h, 1))
            concat_feature = self.concat_op((theta_x, phi_x))
            f = self.concat_project(concat_feature)
            b, _, h, w = f.shape
            f = f.view((b, h, w))
        f_div_c = self.zeros((1, 1, 1), mindspore.float32)
        if self.mode in ["gaussian", "embedded"]:
            f_div_c = self.softmax(f)
        elif self.mode in ["dot", "concatenation"]:
            n = f.shape[-1]
            f_div_c = f / n
        y = self.batmatmul(f_div_c, g_x)
        y = self.transpose(y, input_perm)
        y = y.view((batch_size,
                    self.inter_channels,
                    x.shape[2],
                    x.shape[3],
                    x.shape[4]))
        w_y = self.w(y)
        z = x + w_y
        return z


class NLInflateBlockBase3D(ResidualBlockBase3D):
    """
    ResNet residual block base definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlockBase3D(3, 256, stride=2)
    """

    expansion: int = 1

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 non_local: bool = False,
                 non_local_mode: str = 'dot',
                 **kwargs
                 ) -> None:

        assert group != 1 or base_width == 64, "NLInflateBlockBase3D only supports groups=1 and base_width=64"
        super(NLInflateBlockBase3D, self).__init__(in_channel=in_channel,
                                                   out_channel=out_channel,
                                                   conv12=conv12,
                                                   norm=norm,
                                                   down_sample=down_sample,
                                                   **kwargs)
        self.non_local = non_local
        if self.non_local:
            in_channels = out_channel * self.expansion
            self.non_local_block = NonLocalBlockND(
                in_channels, mode=non_local_mode)

    def construct(self, x):
        """NLInflateBlockBase3D construct."""
        identity = x

        out = self.conv12(x)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        if self.non_local:
            out = self.non_local_block(out)
        return out


class NLInflateBlock3D(ResidualBlock3D):
    """
    ResNet3D residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the second convolutional layer. Default: 1.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import ResidualBlock
        >>> ResidualBlock(3, 256, stride=2)
    """

    expansion: int = 4

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 non_local: bool = False,
                 non_local_mode: str = 'dot',
                 **kwargs
                 ) -> None:
        super(NLInflateBlock3D, self).__init__(in_channel=in_channel,
                                               out_channel=out_channel,
                                               mid_channel=out_channel,
                                               conv12=conv12,
                                               group=group,
                                               norm=norm,
                                               activation=[nn.ReLU, nn.ReLU],
                                               down_sample=down_sample,
                                               **kwargs)
        # conv3d doesn't support group>1 now at 1.6.1 version
        out_channel = int(out_channel * (base_width / 64.0)) * group
        self.non_local = non_local
        if self.non_local:
            in_channels = out_channel * self.expansion
            self.non_local_block = NonLocalBlockND(
                in_channels, mode=non_local_mode)

    def construct(self, x):
        """NLInflateBlock3D construct."""
        identity = x

        out = self.conv12(x)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)
        if self.non_local:
            out = self.non_local_block(out)
        return out


class NLInflateResNet3D(ResNet3D):
    """Inflate3D with ResNet3D backbone and non local block.

    Args:
        block (Optional[nn.Cell]): THe block for network.
        layer_nums (list): The numbers of block in different layers.
        norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.
        stage_strides: Stride size for ResNet3D convolutional layer.
        non_local: Determine whether to apply nonlocal block in this block.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Returns:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.mindvideo.models.backbones.nonlocal3d import ResNetI3D, ResNetI3DResidualBlock
        >>> net = ResNet(ResNetI3DResidualBlock, [3, 4, 6, 3])
        >>> x = ms.Tensor(np.ones([1, 3, 32, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 2048, 16, 7, 7)
    """

    def __init__(self,
                 block: Optional[nn.Cell],
                 layer_nums: Tuple[int],
                 stage_channels: Tuple[int] = (64, 128, 256, 512),
                 stage_strides: Tuple[int] = ((1, 1, 1),
                                              (1, 2, 2),
                                              (1, 2, 2),
                                              (1, 2, 2)),
                 down_sample: Optional[nn.Cell] = Unit3D,
                 inflate: Tuple[Tuple[int]] = ((1, 1, 1),
                                               (1, 0, 1, 0),
                                               (1, 0, 1, 0, 1, 0),
                                               (0, 1, 0)),
                 non_local: Tuple[Tuple[int]] = ((0, 0, 0),
                                                 (0, 1, 0, 1),
                                                 (0, 1, 0, 1, 0, 1),
                                                 (0, 0, 0)),
                 **kwargs
                 ):
        super(NLInflateResNet3D, self).__init__(block=block,
                                                layer_nums=layer_nums,
                                                stage_channels=stage_channels,
                                                stage_strides=stage_strides,
                                                down_sample=down_sample
                                                )
        self.in_channels = stage_channels[0]
        self.conv1 = Unit3D(3, stage_channels[0], kernel_size=(
            5, 7, 7), stride=(1, 2, 2), norm=self.norm)
        self.maxpool = Maxpool3DwithPad(kernel_size=(
            1, 3, 3), padding=(0, 0, 1, 1, 1, 1), strides=(1, 2, 2))
        self.pool2 = ops.MaxPool3D(kernel_size=(2, 1, 1), strides=(2, 1, 1))
        self.layer1 = self._make_layer(
            block,
            stage_channels[0],
            layer_nums[0],
            stride=tuple(stage_strides[0]),
            norm=self.norm,
            inflate=inflate[0],
            non_local=non_local[0],
            **kwargs)
        self.layer2 = self._make_layer(
            block,
            stage_channels[1],
            layer_nums[1],
            stride=tuple(stage_strides[1]),
            norm=self.norm,
            inflate=inflate[1],
            non_local=non_local[1],
            **kwargs)
        self.layer3 = self._make_layer(
            block,
            stage_channels[2],
            layer_nums[2],
            stride=tuple(stage_strides[2]),
            norm=self.norm,
            inflate=inflate[2],
            non_local=non_local[2],
            **kwargs)
        self.layer4 = self._make_layer(
            block,
            stage_channels[3],
            layer_nums[3],
            stride=tuple(stage_strides[3]),
            norm=self.norm,
            inflate=inflate[3],
            non_local=non_local[3],
            **kwargs)

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.pool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class NLResInflate3D50(NLInflateResNet3D):
    """
    The class of ResNet50 uses the registration mechanism to register, need to use the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(NLResInflate3D50, self).__init__(
            NLInflateBlock3D, [3, 4, 6, 3], **kwargs)


@ClassFactory.register(ModuleType.MODEL)
class nonlocal3d(nn.Cell):
    """
    nonlocal3d model

    Xiaolong Wang.
    "Non-local Neural Networks."
    https://arxiv.org/pdf/1711.07971v3

    Args:
        in_d: Depth of input data, it can be considered as frame number of a video. Default: 32.
        in_h: Height of input frames. Default: 224.
        in_w: Width of input frames. Default: 224.
        num_classes(int): Number of classes, it is the size of classfication score for every sample,
            i.e. :math:`CLASSES_{out}`. Default: 400.
        pooling_keep_dim: whether to keep dim when pooling. Default: True.
        keep_prob(float): Probability of dropout for multi-dense-layer head, the number of probabilities equals
            the number of dense layers.
        pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded
            from network. If `False`, it will create a nonlocal3d model with uniform initialization for weight and bias.
        backbone: Bcxkbone of nonlocal3d.
        avg_pool: Avgpooling and flatten.
        head: LinearClsHead architecture.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindvision.mindvideo.models import nonlocal3d
        >>>
        >>> net = nonlocal3d()
        >>> x = Tensor(np.random.randn(1, 3, 32, 224, 224).astype(np.float32))
        >>> output = net(x)
        >>> print(output.shape)
        (1, 400)
    """

    def __init__(self,
                 in_d: int = 32,
                 in_h: int = 224,
                 in_w: int = 224,
                 num_classes: int = 400,
                 keep_prob: float = 0.5,
                 backbone: Optional[nn.Cell] = NLResInflate3D50,
                 avg_pool: Optional[nn.Cell] = AdaptiveAvgPool3D,
                 flatten: Optional[nn.Cell] = nn.Flatten,
                 head: Optional[nn.Cell] = DropoutDense
                 ):

        super(nonlocal3d,self).__init__()

        last_d = math.ceil(in_d / 32)
        last_h = math.ceil((math.ceil(in_h / 32) + 1) / 4)
        last_w = math.ceil((math.ceil(in_w / 32) + 1) / 4)
        backbone_output_channel = 512 * last_d * last_h * last_w

        self.backbone = backbone()
        self.avg_pool = avg_pool((1, 1, 1))
        self.flatten = flatten()
        self.head = head(input_channel=backbone_output_channel,
                         out_channel=num_classes,
                         keep_prob=keep_prob)

    def construct(self, x):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.head(x)

        return x
