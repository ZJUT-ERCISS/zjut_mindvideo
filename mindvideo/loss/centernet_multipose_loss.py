# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
centernet multipose loss
"""
import math
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype


from mindvideo.utils.class_factory import ClassFactory, ModuleType
from mindvideo.loss import FocalLoss

__all__ = ['RegLoss', 'CenterNetMultiPoseLoss']
class GatherFeature(nn.Cell):
    """
    Gather feature at specified position

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self):
        super(GatherFeature, self).__init__()
        self.tile = ops.Tile()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.gather_nd = ops.GatherNd()

    def construct(self, feat, ind):
        """gather by specified index"""
        # (b, N)->(b*N, 1)
        b, n = self.shape(ind)
        ind = self.reshape(ind, (-1, 1))
        ind_b = nn.Range(0, b, 1)()
        ind_b = self.reshape(ind_b, (-1, 1))
        ind_b = self.tile(ind_b, (1, n))
        ind_b = self.reshape(ind_b, (-1, 1))
        index = self.concat((ind_b, ind))
        # (b, N, 2)
        index = self.reshape(index, (b, n, -1))
        # (b, N, c)
        feat = self.gather_nd(feat, index)
        return feat


class TransposeGatherFeature(nn.Cell):
    """
    Transpose and gather feature at specified position

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self):
        super(TransposeGatherFeature, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 3, 1)
        self.gather_feat = GatherFeature()

    def construct(self, feat, ind):
        """(b, c, h, w)->(b, h*w, c)"""
        feat = self.transpose(feat, self.perm_list)
        b, _, _, c = self.shape(feat)
        feat = self.reshape(feat, (b, -1, c))
        # (b, N, c)
        feat = self.gather_feat(feat, ind)
        return feat


class RegLoss(nn.Cell):
    """
    Warpper for regression loss.

    Args:
        mode(str): L1 or Smoothed L1 loss. Default: "l1"

    Returns:
        Tensor, regression loss.
    """

    def __init__(self, mode='l1'):
        super(RegLoss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.gather_feature = TransposeGatherFeature()
        if mode == 'l1':
            self.loss = nn.L1Loss(reduction='sum')
        elif mode == 'sl1':
            self.loss = nn.SmoothL1Loss()
        else:
            self.loss = None

    def construct(self, output, mask, ind, target):
        """Warpper for regression loss."""
        pred = self.gather_feature(output, ind)
        mask = self.cast(mask, mstype.float32)
        num = self.reduce_sum(mask, ())
        mask = self.expand_dims(mask, 2)
        target = target * mask
        pred = pred * mask
        regr_loss = self.loss(pred, target)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss


@ClassFactory.register(ModuleType.LOSS)
class CenterNetMultiPoseLoss(nn.Cell):
    """
    Provide pose estimation network losses.

    Args:
        reg_loss (str): Regression loss, it can be L1 loss or Smooth L1 loss: (['l1', 'sl1']).
            Default='l1'.
        hm_weight (int): Loss weight for keypoint heatmaps. Default=1.
        wh_weight (int): Loss weight for bounding box size. Default=0.1.
        off_weight (int): Loss weight for keypoint local offsets. Default=1.
        reg_offset (bool): Whether to use regress local offset. Default=True.
        reid_dim (int): Feature embed dim. Default=128.
        nID (int): Totoal number of identities in dataset. Default=14455.
        batch_size (int): Number of imgs.


    Returns:
        Tensor, total loss.
    """

    def __init__(self, reg_loss, hm_weight, wh_weight, off_weight, reg_offset, reid_dim, nid, batch_size):
        super(CenterNetMultiPoseLoss, self).__init__()
        self.crit = FocalLoss()
        # self.crit_wh = RegWeightedL1Loss() if not config.net.dense_hp else nn.L1Loss(reduction='sum')
        self.crit_wh = RegLoss(reg_loss)
        # wh
        self.crit_reg = RegLoss(reg_loss)  # reg_loss = 'l1'
        self.hm_weight = hm_weight  # hm_weight = 1 :loss weight for keypoint heatmaps
        self.wh_weight = wh_weight  # wh_weight = 0.1 : loss weight for bounding box size
        self.off_weight = off_weight  # off_weight = 1 : loss weight for keypoint local offsets
        self.reg_offset = reg_offset  # reg_offset = True : regress local offset

        # self.reg_ind = self.hm_hp_ind + 1 if self.reg_offset else self.hm_hp_ind
        self.reg_ind = "reg" if self.reg_offset else "wh"

        # define id
        self.emb_dim = reid_dim  # dataset.reid_dim = 128
        self.nid = nid  # nId = 14455
        self.classifier = nn.Dense(self.emb_dim, self.nid).to_float(mstype.float16)
        self.id_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.emb_scale = math.sqrt(2) * math.log(self.nid - 1)  # fix np
        self.s_det = Parameter(Tensor(-1.85 * np.ones(1), mstype.float32))
        self.s_id = Parameter(Tensor(-1.05 * np.ones(1), mstype.float32))
        # self.s_id = Tensor(-1.05 * self.ones(1, mindspore.float32))

        self.normalize = ops.L2Normalize(axis=1)
        self.greater = ops.Greater()
        self.expand_dims = ops.ExpandDims()
        self.tile = ops.Tile()
        self.multiples_1 = (1, 1, 128)
        # self.multiples_2 = (1, 1, 14455)
        self.select = ops.Select()
        self.zeros = ops.Zeros()
        self.exp = ops.Exp()
        self.squeeze = ops.Squeeze(0)
        self.transpose_gather = TransposeGatherFeature()
        self.reshape = ops.Reshape()
        self.reshape_mul = batch_size * 500
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.sigmoid = nn.Sigmoid()
        self.clip_by_value = ops.clip_by_value

    def construct(self, feature, hm, reg_mask, ind, wh, reg, ids):
        """Defines the computation performed."""
        output_hm = feature["hm"]  # FocalLoss()

        # sigmoid and then clip by value
        output_hm = self.sigmoid(output_hm)
        dt = self.dtype(output_hm)
        output_hm = self.clip_by_value(output_hm,
                                       self.cast(ops.tuple_to_array((1e-4,)), dt),
                                       self.cast(ops.tuple_to_array((1-1e-4,)), dt))

        hm_loss = self.crit(output_hm, hm)

        output_id = feature["feature_id"]  # SoftmaxCrossEntropyWithLogits()
        id_head = self.transpose_gather(output_id, ind)  # id_head=[1,500,128]

        # id_head = id_head[reg_mask > 0]
        reg_mask = self.cast(reg_mask, ms.int32)
        cond = self.greater(reg_mask, 0)  # cond=[1,500]
        cond_cast = self.cast(cond, ms.int32)
        expand_output = self.expand_dims(cond_cast, 2)
        tile_out = self.tile(expand_output, self.multiples_1)
        tile_cast = self.cast(tile_out, ms.bool_)
        fill_zero = self.zeros(id_head.shape, mstype.float32)  # fill_zero=[1,500,128]
        id_head = self.select(tile_cast, id_head, fill_zero)  # id_head=[1,500,128]

        id_head = self.emb_scale * self.normalize(id_head)  # id_head=[1,500,128]

        zero_input = self.zeros(ids.shape, mstype.int32)
        id_target = self.select(cond, ids, zero_input)  # id_target=[1,500]
        id_target_out = self.reshape(id_target, (self.reshape_mul,))
        # expand_output = self.expand_dims(id_target, 2)
        # tile_out = self.tile(expand_output, self.multiples_2)

        c_out = self.reshape(id_head, (self.reshape_mul, 128))
        c_out = self.cast(c_out, mstype.float16)
        id_output = self.classifier(c_out)         # id_output=[1,500,14455]
        id_output = self.cast(id_output, ms.float32)
        # id_output = self.squeeze(id_output)      # id_output=[500,14455]
        # id_target = self.squeeze(tile_out)       # id_target=[500,14455]
        id_loss = self.id_loss(id_output, id_target_out)

        output_wh = feature["wh"]  # Regl1Loss
        wh_loss = self.crit_reg(output_wh, reg_mask, ind, wh)

        off_loss = 0
        if self.reg_offset and self.off_weight > 0:  # Regl1Loss
            output_reg = feature[self.reg_ind]
            off_loss = self.crit_reg(output_reg, reg_mask, ind, reg)

        det_loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss
        loss = self.exp(-1.0*self.s_det) * det_loss + self.exp(-1.0*self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        return loss