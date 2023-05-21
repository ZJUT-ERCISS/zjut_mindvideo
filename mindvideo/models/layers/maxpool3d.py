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
"""Max pooling 3D."""

from mindspore import nn
from mindspore import ops

from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = ['MaxPool3D']


@ClassFactory.register(ModuleType.GENERAL)
class MaxPool3D(nn.Cell):
    r"""
    3D max pooling operation.

    Applies a 3D max pooling over an input Tensor which can be regarded as a composition of 3D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})`, MaxPool outputs
    regional maximum in the :math:`(D_{in}, H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (d_{ker}, h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1, s_2)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents depth, height and width of the kernel, or a tuple
            of three int numbers that represent depth, height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

            - pad: Implicit paddings on both sides of the input in depth, height, width. The number of "pad" will
              be padded to the input Tensor borders. "pad" must be greater than or equal to 0.

        pad_list (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings
            of head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six
            integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2],
            pad[3], pad[4] and pad[5] correspondingly.
        ceil_mode (bool): Whether to use ceil instead of floor to calculate output shape. Only effective in "pad" mode.
            When "pad_mode" is "pad" and "ceil_mode" is "None", "ceil_mode" will be set as "False". Default: None.
        data_format (str) : The optional value for data format. Currently only support 'NCDHW'. Default: 'NCDHW'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
          Data type must be float16 or float32.

    Outputs:
        Tensor, with shape :math:`(N, C, D_{out}, H_{out}, W_{out})`. Has the data type with `x`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither an int not a tuple.
        TypeError: If `pad_mode` or `data_format` is not a string.
        ValueError: If numbers in `kernel_size` or `strides` are not positive.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `pad_mode` is 'same' or 'valid', 'ceil_mode' is not None.
        ValueError: If `kernel_size` or `strides` is a tuple whose length is not equal to 3.
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)), mindspore.float32)
        >>> max_pool3d = MaxPool3D(kernel_size=2, strides=1, pad_mode="valid")
        >>> output = max_pool3d(x)
        >>> print(output)
        ... [[[[[10. 11.]]]
        ...  [[[22. 23.]]]]]
    """

    def __init__(self,
                 kernel_size=1,
                 strides=1,
                 pad_mode="VALID",
                 pad_list=0,
                 ceil_mode=None,
                 data_format="NCDHW"):
        """Initialize MaxPool1d."""
        # TODO: why not just just used ops.MaxPool3D?
        super(MaxPool3D, self).__init__()
        self.maxpool3d = ops.MaxPool3D(kernel_size=kernel_size,
                                       strides=strides,
                                       pad_mode=pad_mode,
                                       pad_list=pad_list,
                                       ceil_mode=ceil_mode,
                                       data_format=data_format)

    def construct(self, x):
        """Max pooling 3D construct."""
        return self.maxpool3d(x)
