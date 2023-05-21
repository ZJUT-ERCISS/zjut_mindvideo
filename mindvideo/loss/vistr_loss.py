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
"""
VisTR loss
"""
import numpy as np
import mindspore as ms
from mindvideo.models.layers import bbox
from mindspore import ops, Tensor, nn, ms_function
from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = ['DiceLoss', 'SigmoidFocalLoss', 'SetCriterion']


@ms_function
def full_like(x, num, dtype):
    y = ops.ones_like(x)
    y = ops.cast(y, dtype)
    y = y*num
    return y


@ClassFactory.register(ModuleType.LOSS)
class DiceLoss(nn.Cell):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.reducesum = ops.ReduceSum()
        self.flatten = ops.Flatten()

    def construct(self, inputs, targets, num_boxes):
        """
        construct diceloss
        """
        inputs = self.sigmoid(inputs)
        # inputs = self.reshape(inputs, (-1, 40))
        inputs = self.flatten(inputs)
        numerator = self.reducesum((2 * (inputs * targets)), 1)
        denominator1 = self.reducesum(inputs, 1)
        denominator2 = self.reducesum(targets, 1)
        denominator = denominator1 + denominator2
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_boxes


@ClassFactory.register(ModuleType.LOSS)
class SigmoidFocalLoss(nn.Cell):
    """
    Args:
        alpha(float):Default: 0.25.
        gamma(float):Default: 2.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.flatten = ops.Flatten()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha
        self.gamma = gamma

    def construct(self, inputs, targets, num_boxes):
        prob = self.sigmoid(inputs)
        ce_loss = self.BCEWithLogitsLoss(inputs, targets)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_boxes


@ClassFactory.register(ModuleType.LOSS)
class SetCriterion(nn.LossBase):
    r"""vistr loss contains loss_labels, loss_masks and loss_boxes.
    Args:
        num_classes(int): Types of segmented objects.
        matcher(cell): Match predictions to GT.
        weight_dict(dict): Weights for different losses.
        eos_coef(float): Background class weights.
        aux_loss(bool): wether or not to computer aux loss.
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, aux_loss):
        super(SetCriterion, self).__init__()
        # param
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.aux_loss = aux_loss

        # ops
        self.fill = ops.Fill()
        self.stack2 = ops.Stack(2)
        self.unstack = ops.Unstack(axis=-1)
        self.stack_1 = ops.Stack(axis=-1)
        self.scatter_update = ops.TensorScatterUpdate()
        self.gathernd = ops.GatherNd()
        self.ones_like = ops.OnesLike()
        self.reduce_sum = ops.ReduceSum()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.Generalized_Box_Iou = bbox.GeneralizedBoxIou()
        self.abs = ops.Abs()
        self.cast = ops.Cast()
        # losses
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.l1_loss_none = nn.L1Loss(reduction='none')
        self.sigmoid_focal_loss = SigmoidFocalLoss()
        self.dice_loss = DiceLoss()

    def loss_labels(self, pred_logits, labels, indices):

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_labels = self.gathernd(labels, tgt_idx)
        target_classes = self.fill(ms.int32, pred_logits.shape[:2], self.num_classes)
        target_classes = self.scatter_update(target_classes,
                                             self.reshape(src_idx, (-1, 2)),
                                             self.reshape(target_labels, (-1,)))

        loss_ce = self.cross_entropy(self.reshape(pred_logits, (-1, pred_logits.shape[2])),
                                     self.reshape(target_classes, (-1,)))

        return loss_ce

    def loss_boxes(self, pred_boxes, tgt_boxes, indices, num_boxes):

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_boxes = self.gathernd(pred_boxes, src_idx)
        tgt_boxes = self.gathernd(tgt_boxes, tgt_idx)
        src_boxes = self.reshape(src_boxes, (-1, src_boxes.shape[-1]))
        tgt_boxes = self.reshape(tgt_boxes, (-1, tgt_boxes.shape[-1]))
        # calculate bbox loss
        loss_bbox = self.l1_loss_none(src_boxes, tgt_boxes)

        # calculate giou loss
        src_boxes = self._CxcywhToXyxy(src_boxes)
        tgt_boxes = self._CxcywhToXyxy(tgt_boxes)
        loss_giou = self.Generalized_Box_Iou(src_boxes, tgt_boxes).diagonal()
        loss_giou = self.ones_like(loss_giou) - loss_giou

        # average these loss
        loss_bbox = self.reduce_sum(loss_bbox) / num_boxes
        loss_giou = self.reduce_sum(loss_giou) / num_boxes
        return loss_bbox, loss_giou

    def loss_masks(self, src_masks, masks, indices, num_boxes):

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # pad target_masks to maximum shape
        # target_masks, valid = misc.nested_tensor_from_tensor_list(masks, split=False)
        target_masks = masks  # TODO now only support batch_size=1
        src_masks = self.gathernd(src_masks, src_idx)  # (1, 72, 75, 104)
        src_masks = ops.interpolate(src_masks,  # (1, 72, 75, 104)
                                    sizes=target_masks.shape[-2:],  # (x, x)
                                    mode='bilinear',
                                    coordinate_transformation_mode="asymmetric")
        src_masks = self.reshape(src_masks, (num_boxes, -1))

        target_masks = self.gathernd(target_masks, tgt_idx)
        target_masks = self.reshape(target_masks, (num_boxes, -1))

        loss_mask = self.sigmoid_focal_loss(src_masks, target_masks, num_boxes)
        loss_dice = self.dice_loss(src_masks, target_masks, num_boxes)

        return loss_dice, loss_mask

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pass

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # indices: (batch_size, 2, n)
        bs = indices.shape[0]
        src_idx = indices[:, :, 0]
        batch_idx = nn.Range(0, bs, 1)()
        batch_idx = self.expand_dims(batch_idx, 1)
        batch_idx = ops.repeat_elements(batch_idx, indices.shape[1], 1)
        batch_src_idx = self.stack2((batch_idx, src_idx))
        return batch_src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute predictions following indices
        # indices: (batch_size, 2, n)
        bs = indices.shape[0]
        tgt_idx = indices[:, :, 1]
        batch_idx = nn.Range(0, bs, 1)()
        batch_idx = self.expand_dims(batch_idx, 1)
        batch_idx = ops.repeat_elements(batch_idx, indices.shape[1], 1)
        batch_tgt_idx = self.stack2((batch_idx, tgt_idx))
        return batch_tgt_idx

    def _get_outputs(self, outputs):
        # 转成字典结构
        if self.aux_loss:
            aux_pred_logits = outputs[:-1, :, :, :42]
            aux_pred_boxes = outputs[:-1, :, :, 42:]
            pred_logits = outputs[-1, :, :, :42]
            pred_boxes = outputs[-1, :, :, 42:]
        else:
            aux_pred_logits = 0
            aux_pred_boxes = 0
            pred_logits = outputs[:, :, :42]
            pred_boxes = outputs[:, :, 42:]
        return pred_logits, pred_boxes, aux_pred_logits, aux_pred_boxes

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

    def resize_annotation(self, labels, boxes, valid, masks, resize_shape):
        """resize masks
        """
        resize_shape = resize_shape[0]
        h1 = self.cast(resize_shape[0], ms.int32)
        w1 = self.cast(resize_shape[1], ms.int32)
        h2 = self.cast(resize_shape[2], ms.int32)
        w2 = self.cast(resize_shape[3], ms.int32)
        ##
        i = int(self.cast(resize_shape[4], ms.int32))
        h_crop = int(self.cast(resize_shape[5], ms.int32))
        j = int(self.cast(resize_shape[6], ms.int32))
        w_crop = int(self.cast(resize_shape[7], ms.int32))
        ##
        h3 = self.cast(resize_shape[8], ms.int32)
        w3 = self.cast(resize_shape[9], ms.int32)
        # 去除之前补零的
        ins_num = int(self.cast(resize_shape[10], ms.int32))
        valid = valid[:, :ins_num]
        labels = labels[:, :ins_num]
        boxes = boxes[:, :ins_num]
        masks = masks[:, :ins_num]
        #
        masks = masks[0]
        if masks.shape[0] > 0:
            # first resize
            out = masks[:, None]
            ops_nearest1 = ops.ResizeNearestNeighbor((int(h1), int(w1)))
            out = ops_nearest1(out) > 0.5
            out = self.cast(out, ms.float32)
            # second resize
            ops_nearest2 = ops.ResizeNearestNeighbor((int(h2), int(w2)))
            out = ops_nearest2(out) > 0.5
            out = self.cast(out, ms.float32)
            # crop
            out = out[:, :, i:i + h_crop, j:j + w_crop]
            # third resize
            ops_nearest3 = ops.ResizeNearestNeighbor((int(h3), int(w3)))
            out = ops_nearest3(out) > 0.5
            out = self.cast(out[:, 0], ms.float32)
        else:
            out = np.zeros((masks.shape[0], h3, w3))
        return labels, boxes, valid, out[None, ...]

    def construct(self, outputs, pred_masks, labels, boxes, valid, masks, resize_shape):
        r""" This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        # labels, boxes, valid, masks = targets
        labels, boxes, valid, masks = self.resize_annotation(labels, boxes, valid, masks, resize_shape)

        pred_logits, pred_boxes, aux_pred_logits, aux_pred_boxes = self._get_outputs(outputs)

        num_boxes = labels.shape[0]*labels.shape[1]

        losses = Tensor(0, dtype=ms.float32)
        indices = self.matcher(pred_logits, pred_boxes, labels, boxes, valid)  # debug
        indices = self.stack2(indices)
        losses = losses + self.weight_dict['loss_ce']*self.loss_labels(pred_logits, labels, indices)
        loss_bbox, loss_giou = self.loss_boxes(pred_boxes, boxes, indices, num_boxes)
        losses = losses + self.weight_dict['loss_bbox']*loss_bbox+self.weight_dict['loss_giou']*loss_giou

        loss_masks, loss_dice = self.loss_masks(pred_masks, masks, indices, num_boxes)
        losses = losses + self.weight_dict["loss_mask"]*loss_masks + self.weight_dict['loss_dice']*loss_dice

        if self.aux_loss:
            for i, aux_output in enumerate(aux_pred_logits):
                indices = self.matcher(aux_output, aux_pred_boxes[i], labels, boxes, valid)  # debug
                indices = self.stack2(indices)
                losses = losses + self.weight_dict['loss_ce']*self.loss_labels(aux_output, labels, indices)
                loss_bbox, loss_giou = self.loss_boxes(pred_boxes, aux_pred_boxes[i], indices, num_boxes)
                losses = losses + self.weight_dict['loss_bbox']*loss_bbox+self.weight_dict['loss_giou']*loss_giou
        return losses
