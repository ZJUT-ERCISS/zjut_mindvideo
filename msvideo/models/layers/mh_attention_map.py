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
"""vistr MHAttentionMap"""
from typing import Optional
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.ops import stop_gradient
from mindspore.common.initializer import initializer, XavierUniform, Zero


class MHAttentionMsp(nn.Cell):
    r"""This is a 2D attention module, which only returns the attention softmax (no multiplication by value)
    Args:
        query_dim(int): The number of channels in input sequence.
        hidden_dim(int): The number of channels in output sequence.
        num_heads(int): parallel attention heads.
        dropout(float):The dropout rate.Default: 0.0.
        bias(bool): Whether the Conv layer has a bias parameter. Default: True.
    Returns:
        Tensor
    Examples:
        >>> attention = MHAttentionMsp(384, 384, 8)
        >>> ones = mindspore.ops.Ones()
        >>> q = ones((1, 10, 384), mindspore.float32)
        >>> k = ones((1, 384, 10, 17), mindspore.float32)
        >>> mask = ones((1, 10, 17), mindspore.bool_)
        >>> out = attention(q, k, mask)
        >>> print(out.shape)
        (1, 10, 8, 10, 17)
    """

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.reshape = ops.Reshape()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(1-dropout)
        self.softmax = ops.Softmax(axis=-1)
        self.einsum = ops.Einsum(equation="bqnc,bnchw->bqnhw")
        self.cast = ops.Cast()
        self.transpose = ops.Transpose()
        self.reduce = ops.ReduceSum(keep_dims=True)
        # self.conv2d = nn.Conv2d(query_dim, hidden_dim, 1, pad_mode='pad',
        #                         has_bias=True, bias_init='zeros')
        self.conv2d = nn.Conv2d(query_dim, hidden_dim, (1, 1), has_bias=True)

        self.q_linear = nn.Dense(query_dim, hidden_dim, has_bias=bias)
        # self.k_linear = nn.Dense(query_dim, hidden_dim, has_bias=bias)

        self.q_linear.bias = initializer(Zero(), [hidden_dim])
        # self.k_linear.bias = initializer(Zero(), [hidden_dim])

        self.q_linear.weight = initializer(
            XavierUniform(), self.q_linear.weight.shape, mindspore.float32)
        # self.k_linear.weight = initializer(
        #     XavierUniform(), self.k_linear.weight.shape, mindspore.float32)
        self.normalize_fact = (hidden_dim / self.num_heads) ** -0.5

    def construct(self, q, k, mask: Optional[Tensor] = None):
        """construct attention map"""
        q = self.q_linear(q)
        k = self.conv2d(k)
        qh = q.view((q.shape[0], q.shape[1], self.num_heads,
                     self.hidden_dim // self.num_heads))
        kh = k.view((k.shape[0], self.num_heads, self.hidden_dim // self.num_heads,
                     k.shape[-2], k.shape[-1]))

        # 用矩阵计算来实现einsum的效果
        # qh_weights = qh * self.normalize_fact
        # qh_weights = qh_weights.expand_dims(0)
        # kh_weights = self.transpose(kh, (3, 4, 0, 1, 2))

        # weights = qh_weights * kh_weights
        # weights = self.transpose(weights, (4, 2, 3, 0, 1))
        # weights = self.reduce(weights, 0)

        weights = self.einsum((qh * self.normalize_fact, kh))
        shape = weights.shape

        mask = self.cast(mask, mindspore.bool_)
        if mask is not None:
            weights[0].masked_fill(mask.expand_dims(1), float("-inf"))
        weights_fla = weights.reshape(weights.shape[0], weights.shape[1],
                                      weights.shape[2]*weights.shape[3]*weights.shape[4])
        weights = self.softmax(weights_fla)
        weights = self.reshape(self.softmax(weights_fla), shape)
        weights = self.dropout(weights)
        # weights = stop_gradient(weights)
        return weights
