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
"""Roll 3D."""

from mindspore import nn
from mindspore import ops

__all__ = ['Roll3D']


class Roll3D(nn.Cell):
    """
    Roll Tensors of shape (B, D, H, W, C).
    TODO: Compare to torch.roll there is a dim left. and where is the dim?

    Args:
        shift (tuple[int]): shift size for target rolling.

    Inputs:
        Tensor of shape (B, D, H, W, C).

    Outputs:
        Rolled Tensor.
    """

    def __init__(self, shift):
        super().__init__()
        self.shift = shift
        self.concat_1 = ops.Concat(axis=1)
        self.concat_2 = ops.Concat(axis=2)
        self.concat_3 = ops.Concat(axis=3)

    def construct(self, x):
        """Construct a Roll3D ops."""
        if self.shift[0] != 0:
            x = self.concat_1(
                (x[:, -self.shift[0]:, :, :],
                 x[:, :-self.shift[0], :, :]))
        if self.shift[1] != 0:
            x = self.concat_2(
                (x[:, :, -self.shift[1]:, :],
                 x[:, :, :-self.shift[1], :]))
        if self.shift[2] != 0:
            x = self.concat_3(
                (x[:, :, :, -self.shift[2]:],
                 x[:, :, :, :-self.shift[2]]))
        return x
