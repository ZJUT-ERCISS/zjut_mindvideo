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
"""VisTR neck"""
import mindspore
from mindspore import nn, Tensor, ops
from mindspore.ops import operations as P
from mindvideo.models.layers.bbox import MultiIou
from mindvideo.models.layers.hungarian import Hungarian


class HungarianMatcher(nn.Cell):
    r"""This class computes an assignment between the targets
    and the predictions of the network

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general,there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_frames: int = 36, cost_class: float = 1,
                 cost_bbox: float = 1, cost_giou: float = 1):
        r"""Creates the matcher

        Args:
            cost_class: This is the relative weight of the classification
                        error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the
                        bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the
                        bounding box in the matching cost
        """
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.softmax = nn.Softmax(axis=-1)
        self.unsqueeze = P.ExpandDims()
        self.mean = P.ReduceMean(keep_dims=False)
        self.concat = P.Concat(axis=0)
        self.multiiou = MultiIou()
        self.frame_index = Tensor(list(range(self.num_frames)), dtype=mindspore.int32)
        self.blank = Tensor([], mindspore.dtype.int32)
        self.topk = P.TopK()
        self.hungarian = Hungarian(10)
        self.ones = P.Ones()
        self.stack = P.Stack()
        self.unstack = P.Unstack(axis=-1)
        self.stack_1 = P.Stack(axis=-1)
        self.stack1 = P.Stack(axis=1)
        self.zero = ops.Zeros()
        self.equal = ops.Equal()
        self.select = ops.Select()
        self.unique = ops.Unique()
        self.cast = ops.Cast()
        self.contrast_matrix = Tensor([1 for _ in range(self.num_frames)], mindspore.int32)
        self.subscript = ops.arange(start=0, stop=self.num_frames, step=1)
        self.subscript2 = Tensor([0 for _ in range(self.num_frames)], mindspore.int32)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()
        self.tile = ops.Tile()
        self.abs = ops.Abs()
        self.gather = ops.GatherNd()
        self.expand_dims = ops.ExpandDims()

    def _CxcywhToXyxy(self, x):
        """CxCyWH_to_XYXY

        Args:
            x(tensor):last dimension is four

        Returns:
            Tensor, last dimension is four
        """
        x_c, y_c, w, h = self.unstack(x)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        out = self.stack_1(b)
        return out

    def construct(self, pred_logits, pred_boxes, tgt_labels, tgt_boxes, tgt_valid):
        """ Performs the sequence level matching
        """
        bs = pred_logits.shape[0]
        indices_src = []
        indices_tgt = []
        row_ind = self.ones((10,), mindspore.int32)
        col_ind = self.ones((10,), mindspore.int32)
        for i in range(bs):
            out_prob = self.softmax(pred_logits[i])
            out_bbox = pred_boxes[i]
            tgt_ids = tgt_labels[i]
            tgt_bbox = tgt_boxes[i]
            tgt_val = tgt_valid[i]
            num_out = 10
            num_tgt = len(tgt_ids)//self.num_frames
            out_prob_split = out_prob.reshape(self.num_frames, num_out, out_prob.shape[-1]).transpose(1, 0, 2)
            out_bbox = out_bbox.reshape(self.num_frames, num_out, out_bbox.shape[-1]).transpose(1, 0, 2)
            out_bbox_split = self.unsqueeze(out_bbox, 1)

            tgt_bbox_split = self.unsqueeze(tgt_bbox.reshape(num_tgt, self.num_frames, 4), 0)

            tgt_val_split = tgt_val.reshape(num_tgt, self.num_frames)

            out_bbox_split = self.cast(out_bbox_split, mindspore.float32)
            tgt_bbox_split = self.cast(tgt_bbox_split, mindspore.float32)

            frame_index = self.tile(self.frame_index, (num_tgt,))

            cost_index = self.stack1((frame_index, tgt_ids))
            class_cost = self.transpose(out_prob_split, (1, 2, 0))
            class_cost = self.gather(class_cost, cost_index)
            class_cost = self.transpose(class_cost, (1, 0))
            class_cost = self.reshape(class_cost, (num_out, num_tgt, self.num_frames))
            class_cost = -1 * self.mean(class_cost, -1)
            bbox_cost = self.mean(self.abs(out_bbox_split-tgt_bbox_split), (-1, -2))

            bbox_cost = self.mean((out_bbox_split-tgt_bbox_split).abs(), (-1, -2))

            iou_cost = -1 * self.multiiou(self._CxcywhToXyxy(out_bbox_split),
                                          self._CxcywhToXyxy(tgt_bbox_split)).mean(-1)
            # TODO: only deal with box and mask with empty target
            cost_matrix = self.cost_class*class_cost + self.cost_bbox * \
                bbox_cost + self.cost_giou*iou_cost
            cost_matrix = self.transpose(cost_matrix, (1, 0))
            _, col_ind, row_ind = self.hungarian(cost_matrix)

            index_i, index_j = [], []
            for j, n in enumerate(row_ind):
                tgt_valid_ind_j, mask = self._nonzero(tgt_val_split[j])
                index_i.append((tgt_valid_ind_j*num_out + n) * mask)
                index_j.append((tgt_valid_ind_j + col_ind[j] * self.num_frames))
            if not index_i or not index_j:
                indices_src.append(self.blank)
                indices_tgt.append(self.blank)
            else:
                index_i = self.concat(index_i).expand_dims(0)
                index_j = self.concat(index_j).expand_dims(0)
                indices_src.append(index_i)
                indices_tgt.append(index_j)
        indices_src = self.concat(indices_src)
        indices_tgt = self.concat(indices_tgt)
        indices_src = ops.stop_gradient(indices_src)
        indices_tgt = ops.stop_gradient(indices_tgt)
        return indices_src, indices_tgt

    def _nonzero(self, x):
        mask = self.equal(x, self.contrast_matrix)
        out = self.select(mask, self.subscript, self.subscript2)
        mask = self.cast(mask, mindspore.int32)
        return out, mask
