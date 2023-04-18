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
"""vistr decoder"""
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


class TransformerDecoder(nn.Cell):
    r"""transformer decoder is a stack of N decoder layers
    Args:
        decoder_layers(nn.cell):an instance of the TransformerDecoderLayer() class
        norm(nn.cell):the layer normalization component (optional).Default=None
        return_intermediate(bool):return intermediate result.Default=False
    Inputs:
        tgt(tensor): the sequence to the decoder
        memory(tensor): the sequence from the last layer of the encoder
        tgt_key_padding_mask(tensor): the mask for the tgt keys per batch
        memory_key_padding_mask(tensor): he mask for the memory keys per batch
        pos(tensor): memory's encoded position
        query_pos(tensor): tgt's encoded position
    Outputs:
        Tensor
    Example:
        >>> decoder_layer = TransformerDecoderLayer( 384, 6, 2048, 0.1, "relu", False)
        >>> decoder = TransformerDecoder( decoder_layer)
        >>> tgt = ops.ones()((360, 1, 384), mindspore.float32)
        >>> memory = ops.ones()((6120, 1, 384), mindspore.float32)
        >>> out = decoder(tgt, memory)
    """

    def __init__(self, decoder_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = decoder_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def construct(self, tgt, memory,
                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                  pos=None, query_pos=None):
        """construct decoder"""
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            output = ops.Stack()(intermediate)
            return output
        output = ops.ExpandDims()(output, 0)
        # TODO 不计算梯度
        # output = ops.stop_gradient(output)
        return output


class TransformerDecoderLayer(nn.Cell):
    r"""transformer decoder layer is made up of self-attn and feedforward network
    Args:
        d_model(int): the number of expected features in the input
        nhead(int): the number of heads in the multiheadattention models
        dim_feedfroward(int): the dimension of the feedforward network model.Default=2048
        dropout(float): the dropout value.Default=0.1
        activation(str): the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default="relu"
        normalize_before(bool): done normalize before decoderlayer.Default:False
    Inputs:
        tgt(tensor): the sequence to the decoder
        memory(tensor): the sequence from the last layer of the encoder
        tgt_key_padding_mask(tensor): the mask for the tgt keys per batch
        memory_key_padding_mask(tensor): he mask for the memory keys per batch
        pos(tensor): memory's encoded position
        query_pos(tensor): tgt's encoded position
    Outputs:
        Tensor
    Example:
        >>> decoder_layer = TransformerDecoderLayer(d_model=384, nhead=6)
        >>> tgt = ops.ones()((360, 1, 384), mindspore.float32)
        >>> memory = ops.ones()((6120, 1, 384), mindspore.float32)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward,
                                weight_init=HeUniform(),
                                bias_init=UniformBias([dim_feedforward, d_model]))
        self.linear2 = nn.Dense(dim_feedforward, d_model,
                                weight_init=HeUniform(),
                                bias_init=UniformBias([d_model, dim_feedforward]))

        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm3 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.drop0 = nn.Dropout(1 - dropout)
        self.drop1 = nn.Dropout(1 - dropout)
        self.drop2 = nn.Dropout(1 - dropout)
        self.drop3 = nn.Dropout(1 - dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def construct(self, tgt, memory,
                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                  pos=None, query_pos=None):
        """construct decoder layer"""
        q = k = with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.drop0(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=with_pos_embed(tgt, query_pos),
                                   key=with_pos_embed(memory, pos),
                                   value=memory,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.drop1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.drop2(self.activation(self.linear1(tgt))))

        tgt = tgt + self.drop3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt
