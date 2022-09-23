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
"""vistr mlp"""
from mindspore import nn, ops
from mindspore.common.initializer import HeUniform
from msvideo.utils.init_weight import UniformBias


class MLP(nn.Cell):
    r""" Very simple multi-layer perceptron (also called FFN)
    Args:
        input_dim(int): The number of channels in the input space.
        hidden_dim(int): The number of extra channels
        output_dim(int): The number of channels in the output space.
        num_layers(int): The number of layers in the mlp
    Return:
        tensor, one tensor
    Examples:
        >>> mlp = MLP(384, 384, 4, 3)
        >>> x = mindspore.ops.Ones()((6, 1, 360, 384), mindspore.float32)
        >>> out = mlp(x)
        >>> print(out.shape)
        (6, 1, 360, 4)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([
            nn.Dense(n, k, weight_init=HeUniform(),
                     bias_init=UniformBias([k, n]))
            for n, k in zip([input_dim] + h, h + [output_dim])
        ])

    def construct(self, x):
        """construct"""
        for i, layer in enumerate(self.layers):
            x = ops.ReLU()(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
