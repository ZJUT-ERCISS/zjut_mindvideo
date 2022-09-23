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
"""Max pooling 3D with padding."""

from mindspore import nn
from mindspore import ops

__all__ = ['Maxpool3DwithPad']

class Maxpool3DwithPad(nn.Cell):
    """
    3D max pooling with padding operation..

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
                is an int number that represents depth, height and width of the kernel, or a tuple
                of three int numbers that represent depth, height and width respectively. Default: 1.
        padding (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings
            of head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six
            integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2],
            pad[3], pad[4] and pad[5] correspondingly.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            not only the depth, height of movement but also the width of movement,, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value of pad mode is "same" or "valid" or "SYMMETRIC".
            Default: "SYMMETRIC".


    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 kernel_size,
                 padding,
                 strides=1,
                 pad_mode='SYMMETRIC',
                 ):
        super(Maxpool3DwithPad, self).__init__()
        self.maxpool3d = ops.MaxPool3D(kernel_size=kernel_size,
                                       strides=strides)
        self.padding = ((0, 0), 
                        (padding[0], padding[1]),
                        (padding[2], padding[3]),
                        (padding[4], padding[5]))
        self.pad = nn.Pad(self.padding, mode=pad_mode)
        self.reshape = ops.Reshape()

    def construct(self, x):
        b, c, t, h, w = x.shape
        x = self.reshape(x, (b*c, t, h, w))
        x = self.pad(x)
        x = self.reshape(x, (b, c, x.shape[1], x.shape[2], x.shape[3]))
        x = self.maxpool3d(x)
        return x
