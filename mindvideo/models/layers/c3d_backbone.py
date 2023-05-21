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
"""C3D backbone."""

from mindspore import nn

from mindvideo.utils.class_factory import ClassFactory, ModuleType
from mindvideo.models.layers import MaxPool3D, Maxpool3DwithPad


@ClassFactory.register(ModuleType.MODEL)
class C3DBackbone(nn.Cell):
    """
    C3D backbone. It works when the of input data is in the shape of :math:`(B, C, T, H, W)`.

    Args:
        in_channel(int): Number of input data. Default: 3.
        kernel_size(Union[int, Tuple[int]]): Kernel size for every conv3d layer in C3D.
            Default: (3, 3, 3).

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> data = Tensor(np.random.randn(8, 2, 16, 112, 112), dtype=mindspore.float32)
        >>> model = C3D(in_channel = 2)

        >>> predict = model(data)
        >>> print(predict.shape)

    """

    def __init__(self, in_channel=3, kernel_size=(3, 3, 3)):
        super(C3DBackbone, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kernel_size = tuple(kernel_size)
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=kernel_size,
                               padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool1 = MaxPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2), pad_mode='same')

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=kernel_size,
                               padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool2 = MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='same')

        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=kernel_size,
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=kernel_size,
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool3 = MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='same')

        self.conv4a = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=kernel_size,
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.conv4b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.pool4 = MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='same')

        self.conv5a = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        self.conv5b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=kernel_size,
                                padding=(1, 1, 1, 1, 1, 1), pad_mode='pad', has_bias=True)
        # self.pool5 = MaxPool3D(kernel_size=(2, 2, 2), strides=(2, 2, 2), pad_mode='pad',
        #                        pad_list=(0, 0, 1, 0, 1, 0))
        self.pool5 = Maxpool3DwithPad(kernel_size=(2, 2, 2), strides=(2, 2, 2),
                                      padding=(0, 0, 1, 0, 1, 0))

        self.relu = nn.ReLU()

    def construct(self, x):
        """C3D network construct."""
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        return x
