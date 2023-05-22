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
"""box ops"""

import mindspore
from mindspore import Tensor, ops, nn
from mindspore import numpy as np
from mindspore.ops import operations as P


class MultiIou(nn.Cell):
    """
    multi iou calculating Iou between pred boxes and gt boxes.

    Args:
        pred_bbox(tensor):predicted bbox.
        gt_bbox(tensor):Ground Truth bbox.

    Returns:
        Tensor, iou of predicted box and ground truth box.
    """

    def __init__(self):
        super().__init__()
        self.max = P.Maximum()
        self.min = P.Minimum()
        self.max_value = Tensor(8388608, mindspore.float32)
        self.min_value = Tensor(0, mindspore.float32)

    def construct(self, pred_bbox, gt_bbox):
        """construct calculating iou"""
        lt = self.max(pred_bbox[..., :2], gt_bbox[..., :2])
        rb = self.min(pred_bbox[..., 2:], gt_bbox[..., 2:])
        wh = ops.clip_by_value((lt - rb), self.min_value, self.max_value)
        wh_1 = pred_bbox[..., 2:] - pred_bbox[..., :2]
        wh_2 = gt_bbox[..., 2:] - gt_bbox[..., :2]
        inter = wh[..., 0] * wh[..., 1]
        union = wh_1[..., 0] * wh_1[..., 1] + wh_2[..., 0] * wh_2[..., 1]
        union = union - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou


class BoxIou(nn.Cell):
    """calculate box iou

    Args:
        boxes1(Tensor):[x0, y0, x1, y1] format
        boxes2(Tensor):[x0, y0, x1, y1] format

    Returns:
    """

    def __init__(self):
        super().__init__()
        self.gather = ops.Gather()
        self.squeeze = ops.Squeeze()
        self.max = P.Maximum()
        self.min = P.Minimum()
        self.max_value = Tensor(8388608, mindspore.float32)
        self.min_value = Tensor(0, mindspore.float32)
        self.indice1 = Tensor(np.array([0, 1]), mindspore.int32)
        self.indice2 = Tensor(np.array([2, 3]), mindspore.int32)

    def construct(self, boxes1, boxes2):
        """construct boxiou"""
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)

        lt = self.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = self.min(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = ops.clip_by_value((rb-lt), self.min_value, self.max_value)

        inter = wh[:, :, 0] * wh[:, :, 1]

        # union = area1.expand_dims(0) + area2 - inter
        union = area1[:, None] + area2 - inter
        iou = (inter+1e-6) / (union+1e-6)
        return iou, union, inter, wh

    def _box_area(self, boxes):
        """box area"""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class GeneralizedBoxIou(nn.Cell):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    Args:
        boxes1(Tensor):[x0, y0, x1, y1] format
        boxes2(Tensor):[x0, y0, x1, y1] format

    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """

    def __init__(self):
        super().__init__()
        self.gather = ops.Gather()
        self.indice1 = Tensor(np.array([0, 1]), mindspore.int32)
        self.indice2 = Tensor(np.array([2, 3]), mindspore.int32)
        self.max = P.Maximum()
        self.min = P.Minimum()
        self.max_value = Tensor(8388608, mindspore.float32)
        self.min_value = Tensor(0, mindspore.float32)
        self.box_iou = BoxIou()

    def construct(self, boxes_1, boxes_2):
        """construct GeneralizedBoxIou"""
        iou, union, _, _ = self.box_iou(boxes_1, boxes_2)

        rb = self.max(boxes_1[:, None, 2:], boxes_2[:, 2:])
        lt = self.min(boxes_1[:, None, :2], boxes_2[:, :2])

        wh = ops.clip_by_value((rb-lt), self.min_value, self.max_value)

        area = wh[:, :, 0] * wh[:, :, 1]
        return iou - ((area - union)+1e-6) / (area+1e-6)
