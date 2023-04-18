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
"""vistr encoder"""
from mindspore import nn, ops
from mindspore.common.initializer import HeUniform
from msvideo.utils.init_weight import UniformBias
from msvideo.models.layers.multihead_attention import MultiheadAttention


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return ops.ReLU()
    if activation == "gelu":
        return ops.GeLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def with_pos_embed(tensor, pos):
    """with pos embed"""
    return tensor if pos is None else tensor + pos


class TransformerEncoder(nn.Cell):
    r"""transformer encoder is a stack of N encoder layers
    Args:
        encoder_layers: an list of TransformerEncoderlayer class's instance
        norm: the layer normalization component
    Inputs:
        src: the sequence to encoder
        src_key_padding_mask: the mask for the src key per batch
        pos: the sequence's encoder position
    Outputs:
        Tensor
    Examples:
        >>> encoder_layer = TransformerEncoderLayer(d_model=384, nhead=6)
        >>> transformer_encoder = TransformerEncoder(encoder_layer)
        >>> src = ops.ones()((100,1,384), mindspore.float32)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layers, norm=None):
        super().__init__()
        self.layers = encoder_layers
        self.norm = norm

    def construct(self, src, src_key_padding_mask=None, pos=None):
        """construct"""
        output = src

        for layer in self.layers:
            output = layer(
                output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        # TODO 不计算梯度
        # output = ops.stop_gradient(output)
        return output


class TransformerEncoderLayer(nn.Cell):
    r"""transformer encoder layer is made up of self-attn and feedforward network.
    Args:
        d_model(int): the number of expected features in the input
        nhead(int): the number of heads in the multiheadattention models
        dim_feedfroward(int): the dimension of the feedforward network model.Default=2048
        dropout(float): the dropout value.Default=0.1
        activation(str): the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default="relu"
        normalize_before(bool): done normalize before decoderlayer.Default:False
    Inputs:
        src: the sequence to encoder
        src_key_padding_mask: the mask for the src key per batch
        pos: the sequence's encoder position
    Outputs:
        Tensor
    Examples:
        >>> encoder_layer = TransformerEncoderLayer(d_model=384, nhead=6)
        >>> src = ops.ones()((100,1,384), mindspore.float32)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward,
                                weight_init=HeUniform(),
                                bias_init=UniformBias([dim_feedforward, d_model]))
        self.linear2 = nn.Dense(dim_feedforward, d_model,
                                weight_init=HeUniform(),
                                bias_init=UniformBias([d_model, dim_feedforward]))

        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.activation = _get_activation_fn(activation)
        self.drop0 = nn.Dropout(1 - dropout)
        self.drop1 = nn.Dropout(1 - dropout)
        self.drop2 = nn.Dropout(1 - dropout)
        self.normalize_before = normalize_before

    def construct(self, src, src_key_padding_mask=None, pos=None):
        """construct"""
        q = k = with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, key_padding_mask=src_key_padding_mask)
        src = src + self.drop0(src2)
        src = self.norm1(src)

        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.drop1(src2)
        src2 = self.linear2(src2)

        src = src + self.drop2(src2)
        if not self.normalize_before:
            src = self.norm2(src)
        return src
