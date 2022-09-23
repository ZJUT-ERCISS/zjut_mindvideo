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

import numpy
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
        super(MultiIou, self).__init__()
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


class Box_Iou(nn.Cell):
    """calculate box iou

    Args:
        boxes1(Tensor):[x0, y0, x1, y1] format
        boxes2(Tensor):[x0, y0, x1, y1] format

    Returns:
    """

    def __init__(self):
        super(Box_Iou, self).__init__()
        self.gather = ops.Gather()
        self.squeeze = ops.Squeeze()
        self.max = P.Maximum()
        self.min = P.Minimum()
        self.max_value = Tensor(8388608, mindspore.float32)
        self.min_value = Tensor(0, mindspore.float32)
        self.indice1 = Tensor(np.array([0, 1]), mindspore.int32)
        self.indice2 = Tensor(np.array([2, 3]), mindspore.int32)

    def construct(self, boxes1, boxes2):
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)

        # lt_1 = self.gather(boxes1, self.indice1, 1).expand_dims(0)
        # lt_2 = self.gather(boxes2, self.indice1, 1)
        # lt = self.max(lt_1, lt_2)
        # rb_1 = self.gather(boxes1, self.indice2, 1).expand_dims(0)
        # rb_2 = self.gather(boxes2, self.indice2, 1)
        # rb = self.min(rb_1, rb_2)
        lt = self.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = self.min(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = ops.clip_by_value((rb - lt), self.min_value, self.max_value)

        # inter1 = self.gather(wh, self.indice1[0], 2)
        # inter2 = self.gather(wh, self.indice1[1], 2)
        # inter = inter1 * inter2
        inter = wh[:, :, 0] * wh[:, :, 1]

        # union = area1.expand_dims(0) + area2 - inter
        union = area1[:, None] + area2 - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou, union, inter, wh

    def _box_area(self, boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (x1, y1, x2, y2) coordinates.

        Arg:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format

        Returns:
            area (Tensor[N]): area for each box
        """
        # boxes1 = self.squeeze(self.gather(boxes, self.indice2[0], 1))
        # boxes2 = self.squeeze(self.gather(boxes, self.indice1[0], 1))
        # boxes3 = self.squeeze(self.gather(boxes, self.indice2[1], 1))
        # boxes4 = self.squeeze(self.gather(boxes, self.indice1[1], 1))
        # out = (boxes1 - boxes2) * (boxes3 - boxes4)
        # return out
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class Generalized_Box_Iou(nn.Cell):
    """Generalized Box IoU

    Args:
        boxes1(Tensor):[x0, y0, x1, y1] format
        boxes2(Tensor):[x0, y0, x1, y1] format

    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)

    """

    def __init__(self):
        super(Generalized_Box_Iou, self).__init__()
        self.gather = ops.Gather()
        self.indice1 = Tensor(np.array([0, 1]), mindspore.int32)
        self.indice2 = Tensor(np.array([2, 3]), mindspore.int32)
        self.max = P.Maximum()
        self.min = P.Minimum()
        self.max_value = Tensor(8388608, mindspore.float32)
        self.min_value = Tensor(0, mindspore.float32)
        self.box_iou = Box_Iou()

    def construct(self, boxes1, boxes2):
        # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union, inter, wh1 = self.box_iou(boxes1, boxes2)

        # lt_1 = self.gather(boxes1, self.indice1, 1).expand_dims(0)
        # lt_2 = self.gather(boxes2, self.indice1, 1)
        # lt = self.max(lt_1, lt_2)
        # rb_1 = self.gather(boxes1, self.indice2, 1).expand_dims(0)
        # rb_2 = self.gather(boxes2, self.indice2, 1)
        # rb = self.min(rb_1, rb_2)
        lt = self.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = self.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = ops.clip_by_value((rb - lt), self.min_value, self.max_value)

        # inter1 = self.gather(wh, self.indice1[0], 2)
        # inter2 = self.gather(wh, self.indice1[1], 2)
        # area = inter1 * inter2
        area = wh[:, :, 0] * wh[:, :, 1]
        giou = iou - ((area - union) + 1e-6) / (area + 1e-6)
        return giou
