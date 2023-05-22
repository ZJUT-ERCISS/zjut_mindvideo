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
"""vistr multihead_attention"""
import mindspore
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, HeUniform


def linear(input_arr, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Args:
        Input: :math:`(N, *, in_features)` N is the batch size, `*` means any number of
          additional dimensions
        Weight: :math:`(out_features, in_features)`
        Bias: :math:`(out_features)`
        Output: :math:`(N, *, out_features)`
    Returns:
        tensor
    """
    if input_arr.ndim == 2 and bias is not None:
        # fused op is marginally faster
        ret = ops.BatchMatMul()(input_arr, weight.T) + bias
    else:
        output = ops.matmul(input_arr, weight.T)
        if bias is not None:
            output += bias
        ret = output
    return ret


class MultiheadAttention(nn.Cell):
    r"""multi head attention
    Args:
        embed_dim(int): total dimension of the model
        num_heads(int): parallel attention heads
        dropout(float): a Dropout layer on attn_output_weights.Default:0.
    Returns:
        tensor
    """

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cast = ops.Cast()

        self.in_proj_weight = Parameter(initializer('xavier_uniform',
                                                    [3 * embed_dim, embed_dim],
                                                    mstype.float32))
        self.in_proj_bias = Parameter(initializer('zeros',
                                                  [3*embed_dim],
                                                  mstype.float32))

        self.out_proj = nn.Dense(embed_dim, embed_dim,
                                 weight_init=HeUniform())
        self.drop = nn.Dropout(1 - dropout)

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  key_padding_mask: Tensor):
        """construct MultiheadAttention"""
        tgt_len, bsz, embed_dim = query.shape
        scaling = self.head_dim ** -0.5

        q_in_proj_weight = self.in_proj_weight[:self.embed_dim, :]
        k_in_proj_weight = self.in_proj_weight[self.embed_dim:2 *
                                               self.embed_dim, :]
        v_in_proj_weight = self.in_proj_weight[2 *
                                               self.embed_dim:3*self.embed_dim, :]
        q_in_proj_bias = self.in_proj_bias[:self.embed_dim]
        k_in_proj_bias = self.in_proj_bias[self.embed_dim:2*self.embed_dim]
        v_in_proj_bias = self.in_proj_bias[2*self.embed_dim:3*self.embed_dim]
        q = linear(query, q_in_proj_weight, q_in_proj_bias)
        k = linear(key, k_in_proj_weight, k_in_proj_bias)
        v = linear(value, v_in_proj_weight, v_in_proj_bias)

        q = q * scaling

        q = q.view(tgt_len, bsz * self.num_heads,
                   self.head_dim).transpose(1, 0, 2)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        src_len = k.shape[1]

        attn_output_weights = ops.BatchMatMul()(q, k.transpose(0, 2, 1))

        if key_padding_mask is not None:
            key_padding_mask = ops.ExpandDims()(ops.ExpandDims()(key_padding_mask, 1), 2)
            broadcast_to = ops.BroadcastTo((-1, self.num_heads, -1, -1))
            key_padding_mask = broadcast_to(key_padding_mask)
            key_padding_mask = key_padding_mask.reshape(self.num_heads, 1, src_len)
            key_padding_mask = self.cast(key_padding_mask, mindspore.float32)
            attn_output_weights = attn_output_weights + key_padding_mask

        attn_output_weights = ops.Softmax(axis=-1)(attn_output_weights)
        attn_output_weights = self.drop(attn_output_weights)

        attn_output = ops.BatchMatMul()(attn_output_weights, v)
        attn_output = attn_output.transpose(
            1, 0, 2).view(tgt_len, bsz, embed_dim)
        attn_output = linear(
            attn_output, self.out_proj.weight, self.out_proj.bias)
        return attn_output
