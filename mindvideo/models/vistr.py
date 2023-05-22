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
"""VisTR."""
import math
import mindspore as msp
from mindspore import nn, ops, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import HeUniform, initializer

from mindvideo.utils.init_weight import UniformBias
from mindvideo.models.layers import resnet
from mindvideo.models.layers.mlp import MLP
from mindvideo.models.layers.vistr_encoder import TransformerEncoder, TransformerEncoderLayer
from mindvideo.models.layers.vistr_decoder import TransformerDecoder, TransformerDecoderLayer
from mindvideo.models.layers import maskheadsmallconv
from mindvideo.models.layers import mh_attention_map
from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = ['VistrCom', 'GroupNorm3d']


class GroupNorm3d(nn.Cell):
    """
    modify from mindspore.nn.GroupNorm, add depth
    """
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, gamma_init='ones', beta_init='zeros'):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        if num_channels % num_groups != 0:
            raise ValueError(f"For '{self.cls_name}', the 'num_channels' should be divided by 'num_groups', "
                             f"but got 'num_channels': {num_channels}, 'num_groups': {num_groups}.")
        self.eps = eps
        self.affine = affine

        self.gamma = Parameter(initializer(
            gamma_init, num_channels), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_channels), name="beta", requires_grad=affine)
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.sqrt = ops.Sqrt()

    def _cal_output(self, x):
        """calculate groupnorm output"""
        batch, channel, depth, height, width = self.shape(x)
        x = self.reshape(x, (batch, self.num_groups, -1))
        mean = self.reduce_mean(x, 2)
        var = self.reduce_sum(self.square(x - mean), 2) / \
            (channel * depth * height * width / self.num_groups)
        std = self.sqrt(var + self.eps)
        x = (x - mean) / std
        x = self.reshape(x, (batch, channel, depth, height, width))
        output = x * self.reshape(self.gamma, (-1, 1, 1, 1)) + \
            self.reshape(self.beta, (-1, 1, 1, 1))
        return output

    def construct(self, x):
        output = self._cal_output(x)
        return output

@ClassFactory.register(ModuleType.MODEL)
class VistrCom(nn.Cell):
    """
    Vistr Architecture.
    """

    def __init__(self,
                 name: str = 'ResNet50',
                 train_embeding: bool = True,
                 num_queries: int = 360,
                 num_pos_feats: int = 64,
                 num_frames: int = 36,
                 temperature: int = 10000,
                 normalize: bool = True,
                 scale: float = None,
                 hidden_dim: int = 384,
                 d_model: int = 384,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: int = 0.1,
                 activation: str = "relu",
                 normalize_before: bool = False,
                 return_intermediate_dec: bool = True,
                 aux_loss: bool = True,
                 num_class: int = 41):
        super().__init__()
        # input constant used in construct
        self.num_queries = num_queries
        self.num_frames = num_frames
        self.aux_loss = aux_loss
        num_pos_feats = hidden_dim // 3
        self.normalize = normalize
        dim_t = [temperature ** (2 * (i // 2) / num_pos_feats) for i in range(num_pos_feats)]
        self.dim_t = Tensor(dim_t, msp.float32)

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # embed init
        if name == "ResNet101":
            embeding = resnet.ResNet101()
        if name == "ResNet50":
            embeding = resnet.ResNet50()

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1,
                                    pad_mode='valid', has_bias=True)

        for params in embeding.get_parameters():
            if (not train_embeding or 'layer2' not in params.name and
                    'layer3' not in params.name and
                    'layer4' not in params.name):
                params.requires_grad = False
            if 'beta' in params.name:
                params.requires_grad = False
            if 'gamma' in params.name:
                params.requires_grad = False
        self.embed1 = nn.SequentialCell([embeding.conv1,
                                         embeding.pad,
                                         embeding.max_pool,
                                         embeding.layer1])
        self.embed2 = embeding.layer2
        self.embed3 = embeding.layer3
        self.embed4 = embeding.layer4
        # self.embed1.recompute()
        # self.embed2.recompute()
        # self.embed3.recompute()
        # self.embed4.recompute()

        # backbone init

        hidden_dim = d_model
        # encoder
        encoder_layers = nn.CellList([
            TransformerEncoderLayer(d_model,
                                    nhead,
                                    dim_feedforward,
                                    dropout,
                                    activation,
                                    normalize_before)
            for _ in range(num_encoder_layers)
        ])
        encoder_norm = nn.LayerNorm([d_model], epsilon=1e-5) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, encoder_norm)
        # decoder
        decoder_layers = nn.CellList([TransformerDecoderLayer(d_model,
                                                              nhead,
                                                              dim_feedforward,
                                                              dropout,
                                                              activation,
                                                              normalize_before)
                                      for _ in range(num_decoder_layers)])
        decoder_norm = nn.LayerNorm([d_model], epsilon=1e-5)
        self.decoder = TransformerDecoder(decoder_layers,
                                          decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        # embed
        self.class_embed = nn.Dense(hidden_dim,
                                    num_class+1,
                                    weight_init=HeUniform(),
                                    bias_init=UniformBias([num_class+1, hidden_dim]))
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.encoder.recompute()
        # self.decoder.recompute()

        # head init
        hidden_dim = d_model

        self.bbox_attention = mh_attention_map.MHAttentionMsp(hidden_dim,
                                                              hidden_dim,
                                                              nhead,
                                                              dropout=0.0)
        self.mask_head = maskheadsmallconv.MaskHeadSmallConv(hidden_dim + nhead,
                                                             [1024, 512, 256],
                                                             hidden_dim)
        self.insmask_head = nn.SequentialCell([nn.Conv3d(24, 12, 5,
                                                         pad_mode='pad',
                                                         has_bias=True,
                                                         padding=2),
                                               GroupNorm3d(4, 12),
                                               nn.ReLU(),
                                               nn.Conv3d(12, 12, 5,
                                                         pad_mode='pad',
                                                         has_bias=True,
                                                         padding=2),
                                               GroupNorm3d(4, 12),
                                               nn.ReLU(),
                                               nn.Conv3d(12, 12, 5,
                                                         pad_mode='pad',
                                                         has_bias=True,
                                                         padding=2),
                                               GroupNorm3d(4, 12),
                                               nn.ReLU(),
                                               nn.Conv3d(12, 1, 1,
                                                         pad_mode='pad',
                                                         has_bias=True)
                                               ])
        # self.bbox_attention.recompute()
        # self.mask_head.recompute()
        # self.insmask_head.recompute()

        # ops init
        self.cast = ops.Cast()
        self.cumsum = ops.CumSum()
        self.reshape = ops.Reshape()
        self.stack = ops.Stack(axis=5)
        self.concat4 = ops.Concat(axis=4)
        self.transpose = ops.Transpose()
        self.zeros = ops.Zeros()
        self.sin = ops.Sin()
        self.cos = ops.Cos()
        self.fill = ops.Fill()
        self.concat0 = ops.Concat(axis=0)
        self.concat1 = ops.Concat(axis=1)
        self.squeeze = ops.Squeeze(0)
        self.concat_1 = ops.Concat(axis=-1)
        self.tile = ops.Tile()
        self.expand_dim = ops.ExpandDims()
        self.zeros_like = ops.ZerosLike()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        """embed construct"""
        x = x[0]
        mask = self.zeros((x.shape[0], x.shape[2], x.shape[3]), msp.float32)
        src_list = []
        pos_list = []
        features = []
        src = self.embed1(x)
        src_list.append(src)
        src = self.embed2(src)
        src_list.append(src)
        src = self.embed3(src)
        src_list.append(src)
        src = self.embed4(src)
        src_list.append(src)
        for src in src_list:
            interpolate = P.ResizeNearestNeighbor(src.shape[-2:])
            ms = interpolate(mask[None])
            ms = self.cast(ms, msp.bool_)[0]
            features.append((src, ms))
            features_pos = self.PositionEmbeddingSine(ms)
            pos_list.append(features_pos)

        src, ms = features[-1]
        src_proj = self.input_proj(src)
        src_copy = src_proj.copy()
        n, c, h, w = src_proj.shape
        src_proj = self.reshape(src_proj, (n//self.num_frames, self.num_frames, c, h, w))
        src_proj = self.transpose(src_proj, (0, 2, 1, 3, 4))
        src_proj = self.reshape(src_proj,
                                (src_proj.shape[0],
                                 src_proj.shape[1],
                                 src_proj.shape[2],
                                 src_proj.shape[3]*src_proj.shape[4]))
        ms = self.reshape(ms, (n//self.num_frames, self.num_frames, h*w))
        pos_embed = self.transpose(pos_list[-1], (0, 2, 1, 3, 4))
        pos_embed = self.reshape(pos_embed,
                                 (pos_embed.shape[0],
                                  pos_embed.shape[1],
                                  pos_embed.shape[2],
                                  pos_embed.shape[3]*pos_embed.shape[4]))
        query_embed = self.query_embed.embedding_table

        # backbone construct
        bs, c, h, w = src_proj.shape
        src_proj = self.transpose(self.reshape(src_proj, (bs, c, h * w)), (2, 0, 1))
        pos_embed = self.transpose(self.reshape(pos_embed, (bs, c, h * w)), (2, 0, 1))
        query_embed = self.tile(self.expand_dim(query_embed, 1), (1, bs, 1))
        mask = self.reshape(ms, (bs, h * w))

        tgt = self.zeros_like(query_embed)
        memory = self.encoder(src_proj,
                              src_key_padding_mask=mask,
                              pos=pos_embed)
        hs = self.decoder(tgt,
                          memory,
                          memory_key_padding_mask=mask,
                          pos=pos_embed,
                          query_pos=query_embed)
        hs_t = self.transpose(hs, (0, 2, 1, 3))
        memory = self.reshape(self.transpose(memory, (1, 2, 0)), (bs, c, h, w))

        outputs_class = self.class_embed(hs_t)
        outputs_coord = self.sigmoid(self.bbox_embed(hs_t))

        if self.aux_loss:
            output = self.concat_1([outputs_class, outputs_coord])
        else:
            output = self.concat_1([outputs_class[-1, ...], outputs_coord[-1, ...]])

        # head construct
        _, c, s_h, s_w = src_copy.shape
        src = []
        bs_f = features[-1][0].shape[0]//self.num_frames
        for i in range(3):
            _, c_f, h, w = features[i][0].shape
            feature = self.reshape(features[i][0], (bs_f, self.num_frames, c_f, h, w))
            src.append(feature)
        n_f = self.num_queries//self.num_frames
        outputs_seg_masks = []

        # image level processing using box attention
        for i in range(self.num_frames):
            hs_f = hs_t[-1][:, i*n_f:(i+1)*n_f, :]
            memory_f = self.reshape(memory[:, :, i, :], (bs_f, c, s_h, s_w))
            mask_f = self.reshape(ms[:, i, :], (bs_f, s_h, s_w))
            bbox_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f)
            seg_masks_f = self.mask_head(memory_f,
                                         bbox_mask_f,
                                         [src[2][:, i], src[1][:, i], src[0][:, i]])
            outputs_seg_masks_f = self.reshape(seg_masks_f,
                                               (bs_f,
                                                n_f,
                                                24,
                                                seg_masks_f.shape[-2],
                                                seg_masks_f.shape[-1]))
            outputs_seg_masks.append(outputs_seg_masks_f)
        frame_masks = self.concat0(outputs_seg_masks)
        outputs_seg_masks = []

        # instance level processing using 3D convolution
        for i in range(frame_masks.shape[1]):
            mask_ins = self.expand_dim(frame_masks[:, i], 0)
            mask_ins = self.transpose(mask_ins, (0, 2, 1, 3, 4))
            outputs_seg_masks.append(self.insmask_head(mask_ins))
        outputs_seg_masks = self.transpose(self.squeeze(self.concat1(outputs_seg_masks)),
                                           (1, 0, 2, 3))
        outputs_seg_masks = self.reshape(outputs_seg_masks,
                                         (1, self.num_queries,
                                          outputs_seg_masks.shape[-2],
                                          outputs_seg_masks.shape[-1]))
        return output, outputs_seg_masks

    def PositionEmbeddingSine(self, mask):
        """Sine encoding
        """
        n, h, w = mask.shape
        mask = self.reshape(mask, (n//self.num_frames, self.num_frames, h, w))
        not_mask = ~mask
        not_mask = self.cast(not_mask, msp.float32)
        z_embed = self.cumsum(not_mask, 1)
        y_embed = self.cumsum(not_mask, 2)
        x_embed = self.cumsum(not_mask, 3)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale
        pos_x = x_embed[:, :, :, :, None] / self.dim_t
        pos_y = y_embed[:, :, :, :, None] / self.dim_t
        pos_z = z_embed[:, :, :, :, None] / self.dim_t
        pos_x = self.stack([self.sin(pos_x[:, :, :, :, 0::2]),
                            self.cos(pos_x[:, :, :, :, 1::2])])
        pos_x = self.reshape(pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2],
                                     pos_x.shape[3], pos_x.shape[4]*pos_x.shape[5]))
        pos_y = self.stack([self.sin(pos_y[:, :, :, :, 0::2]),
                            self.cos(pos_y[:, :, :, :, 1::2])])
        pos_y = self.reshape(pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2],
                                     pos_y.shape[3], pos_y.shape[4]*pos_y.shape[5]))
        pos_z = self.stack([self.sin(pos_z[:, :, :, :, 0::2]),
                            self.cos(pos_z[:, :, :, :, 1::2])])
        pos_z = self.reshape(pos_z, (pos_z.shape[0], pos_z.shape[1], pos_z.shape[2],
                                     pos_z.shape[3], pos_z.shape[4]*pos_z.shape[5]))
        pos = self.concat4((pos_z, pos_y, pos_x))
        pos = self.transpose(pos, (0, 1, 4, 2, 3))
        return pos
