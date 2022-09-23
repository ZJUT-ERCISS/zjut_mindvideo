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
"""utils."""
from mindspore import ops
from mindspore import Tensor


def round_width(width, multiplier, min_width=8, divisor=8):
    """
    Round width of filters based on width multiplier
    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int): the minimum width after multiplication.
        divisor (int): the new width should be dividable by divisor.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_width = min_width or divisor
    width_out = max(
        min_width, int(width + divisor / 2) // divisor * divisor
    )
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False):
    """Stochastic Depth per sample."""
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    uniform_real_op = ops.UniformReal()
    mask = keep_prob + uniform_real_op(shape)
    floor = ops.Floor()
    mask = floor(mask)
    div = ops.Div()
    return div(x, keep_prob) * mask
