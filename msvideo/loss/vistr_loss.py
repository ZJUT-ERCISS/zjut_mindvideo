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
import mindspore
from mindspore import numpy as np
# import numpy as np
from mindspore import ops, Tensor, nn
from mindspore import dtype as mstype
from mindvision.msvideo.utils import misc
from mindvision.msvideo.engine.ops import bbox
from mindvision.msvideo.example.vistr.grad_ops import grad_l1
from mindvision.msvideo.models.neck.instance_sequence_match_v2 import HungarianMatcher


def _get_src_permutation_idx(indices, bs):
    # permute predictions following indices
    input_indices = nn.Range(0, bs, 1)()
    oneslike = ops.OnesLike()
    concat = ops.Concat()
    stack = ops.Stack()
    transpose = ops.Transpose()
    unstack = ops.Unstack()
    gather = ops.Gather()
    input_perm = (1, 0)
    batch_idx = []
    cast = ops.Cast()
    indices = cast(indices, mindspore.float32)
    for i, ind in enumerate(input_indices):
        src = gather(indices, ind, 0)
        batch_idx_ones = oneslike(src)*i
        batch_idx.append(batch_idx_ones)
    batch_idxes = concat(batch_idx)
    src_idx = concat(unstack(indices))
    idx = stack([batch_idxes, src_idx])
    idx = transpose(idx, input_perm)
    return idx


def _get_tgt_permutation_idx(indices, bs):
    # permute targets following indices
    input_indices = nn.Range(0, bs, 1)()
    gather = ops.Gather()
    oneslike = ops.OnesLike()
    concat = ops.Concat()
    stack = ops.Stack()
    transpose = ops.Transpose()
    unstack = ops.Unstack()
    input_perm = (1, 0)
    batch_idx = []
    cast = ops.Cast()
    indices = cast(indices, mindspore.float32)
    for i, ind in enumerate(input_indices):
        tgt = gather(indices, ind, 0)
        batch_idx_ones = oneslike(tgt)*i
        batch_idx.append(batch_idx_ones)
    batch_idxes = concat(batch_idx)
    tgt_idx = concat(unstack(indices))
    idx = stack([batch_idxes, tgt_idx])
    idx = transpose(idx, input_perm)
    return idx

# TODO np全改成tensor
def numpy_one_hot(targets, num_classes):
    """one hot"""
    one_hots = []
    bs, l = targets.shape
    for target in targets:
        one_hot = np.zeros((len(target), num_classes))
        index = np.arange(0, target.shape[0], 1)
        one_hot[index, target] = 1
        one_hots.append(one_hot)
    return np.concatenate(one_hots).reshape((bs, l, num_classes))


class SetCriterion:
    r"""
    Args:
        weight_dict('loss_ce', 'loss_bbox', 'loss_giou',
                    'loss_mask', 'loss_dice')
        loss(labels, boxes, masks)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, aux_loss):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.aux_loss = aux_loss
        self.oneslike = ops.OnesLike()
        self.concat = ops.Concat()
        self.stack = ops.Stack()
        self.transpose = ops.Transpose()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mul = ops.Mul()
        self.div = ops.RealDiv()
        self.fill = ops.Fill()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.zeros = ops.Zeros()
        self.equal = ops.Equal()
        self.select = ops.Select()
        self.one_hot = ops.OneHot(axis=-1)
        self.update = ops.TensorScatterUpdate()
        self.reducesum = ops.ReduceSum(False)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.eos_coef = eos_coef
        self.empty_weight = list(1 for i in range(1, 43))
        self.empty_weight[-1] = self.eos_coef
        self.num_classes = num_classes
        self.gather = ops.Gather()
        self.resize_bilinear = ops.ResizeBilinear((10, 4))
        self.gathernd = ops.GatherNd()
        self.flatten = ops.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = 0.25
        self.gamma = 2
        self.ones = ops.Ones()
        self.l1loss = nn.L1Loss(reduction='none')
        self.unstack = ops.Unstack(axis=-1)
        self.stack_1 = ops.Stack(axis=-1)
        self.Generalized_Box_Iou = bbox.Generalized_Box_Iou()
        self.log = ops.Log()

    def softmax_ce_withlogit_withweight(self, logits, target_classes, weight):
        """softmax ce with logits with weight"""
        logits = logits.asnumpy()
        target_classes = target_classes.asnumpy()
        weight = np.array(weight, np.float32)
        target_classes_oh = numpy_one_hot(target_classes,
                                          self.num_classes + 1)
        softmax = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
        weighted_target = weight * target_classes_oh
        ce = -(weighted_target * np.log(softmax)).sum(axis=2)
        sum_weight_target = weight[target_classes.astype(np.int32)].sum()
        ce = ce.sum() / sum_weight_target

        ce_grad_src = ((softmax - target_classes_oh) *
                       weighted_target.sum(axis=2, keepdims=True) / sum_weight_target)
        return ce_grad_src

    def sigmoid_focal_loss(self, inputs, targets, num_boxes):
        prob = self.sigmoid(inputs)
        ce_loss = self.BCEWithLogitsLoss(inputs, targets)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
            sigmoid_flocal_grad_loss = alpha_t * (((self.gamma*(1 - p_t)) ** (self.gamma - 1)) * self.log(p_t) - ((1-p_t) ** self.gamma) / p_t)



        return loss.mean(1).sum() / num_boxes, sigmoid_flocal_grad_loss


    def dice_loss(self, inputs, targets, num_boxes):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = self.sigmoid(inputs)
        # inputs = self.reshape(inputs, (-1, 40))
        inputs = self.flatten(inputs)
        numerator = self.reducesum((2 * (inputs * targets)), 1)
        denominator1 = self.reducesum(inputs, 1)
        denominator2 = self.reducesum(targets, 1)
        denominator = denominator1 + denominator2
        loss = 1 - (numerator + 1) / (denominator + 1)
        # loss grad
        loss_grad = -((2*targets*(targets + inputs + 1) - 2*targets*inputs -1) / (targets + inputs +1) ** 2) * inputs * (1 - inputs)

        return loss.sum() / num_boxes, loss_grad
    
    def loss_labels(self, outputs, targets, indices):
        bs = outputs["pred_logits"].shape[0]
        input_indices = nn.Range(0, bs, 1)()

        indices_src, indices_tgt = indices
        src_logits = outputs['pred_logits']

        idx = _get_src_permutation_idx(indices_src, bs)
        target_class_o = []
        indices_tgt = self.cast(indices_tgt, mindspore.float32)
        # for i, ind in enumerate(input_indices):
        #     tgt = self.gather(indices_tgt, ind, 0)
        #     tgt = self.cast(tgt, mindspore.int32)
        #     target_class = targets[i]['labels'][tgt]
        #     target_class_o.append(target_class)
        for i, ind in enumerate(input_indices):
            target = targets[i]
            mark = target['valid'][0][0]
            tgt = self.gather(indices_tgt, ind, 0)
            tgt = self.cast(tgt, mindspore.int32)
            if mark == 1:
                target_class = target['labels'][tgt]
                target_class_o.append(target_class)
            else:
                target['labels'][0] = self.num_classes
                target_class = target['labels'][tgt]
                target_class_o.append(target_class)

        target_classes_o = self.concat(target_class_o)

        target_classes = self.fill(mindspore.dtype.int32,
                                   src_logits.shape[:2], self.num_classes)
        idx = self.cast(idx, mstype.int32)
        target_classes_o = self.cast(target_classes_o, mstype.int32)
        target_classes = self.update(target_classes, idx, target_classes_o)

        labels_int = self.cast(target_classes, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = src_logits.transpose(0, 2, 1)
        logits_ = self.reshape(logits_, (-1, self.num_classes+1))
        labels_float = self.cast(labels_int, mstype.float32)
        weights = self.zeros(labels_float.shape, mstype.float32)
        for i in range(self.num_classes+1):
            fill_weight = self.fill(
                mstype.float32, labels_float.shape, self.empty_weight[i])
            # 找对应的class
            equal_ = self.equal(labels_float, i)
            weights = self.select(equal_, fill_weight, weights)
        one_hot_labels = self.one_hot(
            labels_int, self.num_classes+1, self.on_value, self.off_value)
        logits_ = logits_.astype(mstype.float32)

        loss_ce = self.ce(logits_, one_hot_labels)
        loss_ce = self.mul(weights, loss_ce)
        loss_ce = self.div(self.reducesum(loss_ce), self.reducesum(weights))

        ce_grad_src = self.softmax_ce_withlogit_withweight(src_logits, target_classes, self.empty_weight)

        losses = {'loss_ce': loss_ce,
                  'loss_ce_grad_src': ce_grad_src}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        losses = {}
        mark = targets[0]['valid'][0][0]
        bs = outputs["pred_logits"].shape[0]
        input_indices = nn.Range(0, bs, 1)()
        num_boxes = 0
        for target in targets:
            num_boxes = num_boxes + len(target['labels'])
        indices_src, indices_tgt = indices
        idx = _get_src_permutation_idx(indices_src, bs)
        idx = self.cast(idx, mindspore.int32)
        temp_src = outputs['pred_boxes'][0][0]
        outputs['pred_boxes'][0][0] = mindspore.numpy.zeros((4), mindspore.float32)
        src_boxes = self.gathernd(outputs['pred_boxes'], idx)
        if mark == 1:
            src_boxes[0] = temp_src

        target_box = []
        indices_tgt = self.cast(indices_tgt, mindspore.float32)
        # for i, ind in enumerate(input_indices):
        #     tgt = self.gather(indices_tgt, ind, 0)
        #     tgt = self.cast(tgt, mindspore.int32)
        #     target = targets[i]
        #     box = target['boxes'][tgt]
        #     target_box.append(box)
        for i, ind in enumerate(input_indices):
            tgt = self.gather(indices_tgt, ind, 0)
            tgt = self.cast(tgt, mindspore.int32)
            target = targets[i]
            temp_tgt = target['boxes'][0]
            target['boxes'][0] = mindspore.numpy.zeros((4), mindspore.float32)
            box = target['boxes'][tgt]
            if mark == 1:
                box[0] = temp_tgt
            target_box.append(box)

        target_boxes = self.concat(target_box)

        loss_bbox = self.l1loss(src_boxes, target_boxes)
        l1_grad_src = grad_l1(src_boxes.asnumpy(), target_boxes.asnumpy())
        l1_grad_src = Tensor(l1_grad_src, mstype.float32)
        l1_grad_src_full = mindspore.numpy.zeros_like(outputs['pred_boxes'])
        l1_grad_src_full = self.update(l1_grad_src_full, idx, l1_grad_src)
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # l1_grad_src_full tensor
        losses['loss_bbox_grad_src'] = l1_grad_src_full.asnumpy()


        src_boxes = self.cast(src_boxes, mindspore.float32)
        target_boxes = self.cast(target_boxes, mindspore.float32)

        src_boxes = self._CxcywhToXyxy(src_boxes)
        target_boxes = self._CxcywhToXyxy(target_boxes)

        # giou_grad_src array
        giou, giou_grad_src = self.Generalized_Box_Iou(src_boxes, target_boxes)
        giou_grad_src = Tensor(giou_grad_src, mstype.float32)
        giou_grad_src_full = mindspore.numpy.zeros_like(outputs['pred_boxes'])
        giou_grad_src_full = self.update(giou_grad_src_full, idx, giou_grad_src)

        loss_giou = 1 - giou.diagonal()  
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['loss_giou_grad_src'] = giou_grad_src_full.asnumpy()
        return losses

    def loss_masks(self, outputs, targets, indices):
        mark = targets[0]['valid'][0][0]
        indices_src, indices_tgt = indices
        bs = outputs["pred_logits"].shape[0]
        num_boxes = 0
        for target in targets:
            num_boxes = num_boxes + len(target['labels'])

        src_idx = _get_src_permutation_idx(indices_src, bs)
        tgt_idx = _get_tgt_permutation_idx(indices_tgt, bs)

        src_masks = outputs["pred_masks"]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = misc.nested_tensor_from_tensor_list(
            [t["masks"] for t in targets], split=False)
        # src_masks = src_masks[src_idx]
        src_idx = self.cast(src_idx, mindspore.int32)
        temp_src = src_masks[0][0]
        src_masks[0][0] = mindspore.numpy.ones((src_masks.shape[-2:]), mindspore.float32) * -100
        src_masks = self.gathernd(src_masks, src_idx)
        if mark == 1:
            src_masks[0] = temp_src

        resize_bilinear = ops.ResizeBilinear((target_masks.shape[-2:]))
        src_masks = src_masks.expand_dims(1)
        src_masks = resize_bilinear(src_masks)
        src_masks = src_masks.squeeze(1)
        src_masks = self.flatten(src_masks)
        # target_masks = self.flatten(target_masks[tgt_idx])
        tgt_idx = self.cast(tgt_idx, mindspore.int32)
        temp_tgt = target_masks[0][0]
        target_masks[0][0] = mindspore.numpy.zeros((target_masks.shape[-2:]), mindspore.float32)
        # target_masks = self.reshape(self.gathernd(target_masks, tgt_idx), (-1, 40))
        h = target_masks.shape[2]
        w = target_masks.shape[3]
        target_masks = self.gathernd(target_masks, tgt_idx)
        if mark == 1:
            target_masks[0] = temp_tgt
        target_masks = self.flatten(target_masks)


        loss_mask, loss_grad_mask = self.sigmoid_focal_loss(src_masks, target_masks, num_boxes)
        loss_grad_mask = loss_grad_mask.reshape(num_boxes, h, w)
        loss_grad_mask_full = mindspore.numpy.zeros((1, 360, h, w))
        # loss_grad_mask_full = resize_bilinear(loss_grad_mask_full.expand_dims(1)).squeeze(1)
        loss_grad_mask_full = self.update(loss_grad_mask_full, src_idx, loss_grad_mask)
        dif = 500 - w
        dif_fill = mindspore.numpy.zeros((bs, 360, 300, dif), mstype.float32)
        concat = ops.Concat(axis=-1)
        loss_grad_mask_full = concat((loss_grad_mask_full, dif_fill))

        loss_dice, loss_grad_dice = self.dice_loss(src_masks, target_masks, num_boxes)
        loss_grad_dice = loss_grad_dice.reshape(num_boxes, h, w)
        loss_grad_dice_full = mindspore.numpy.zeros((1, 360, h, w))
        # loss_grad_dice_full = resize_bilinear(loss_grad_dice_full)
        loss_grad_dice_full = self.update(loss_grad_dice_full, src_idx, loss_grad_dice)
        dif = 500 - w
        dif_fill = mindspore.numpy.zeros((bs, 360, 300, dif), mstype.float32)
        loss_grad_mask_full = concat((loss_grad_dice_full, dif_fill))

        losses = {
            "loss_mask": loss_mask,
            "loss_mask_grad_src": loss_grad_mask_full.asnumpy(),
            "loss_dice": loss_dice,
            "loss_dice_grad_src": loss_grad_dice_full.asnumpy()
        }
        return losses

    def _get_outputs(self, outputs):
        # 转成字典结构
        out, outputs_seg_masks = outputs
        if self.aux_loss:
            n_aux_outs = out.shape[0] - 1
            aux_outs = []
            for i in range(n_aux_outs):
                aux_out = {'pred_logits': out[i, ..., :42],
                           'pred_boxes': out[i, ..., 42:]}
                aux_outs.append(aux_out)
            outputs = {
                'pred_logits': out[-1, ..., :42],
                'pred_boxes': out[-1, ..., 42:],
                'pred_masks': outputs_seg_masks,
                'aux_outputs': aux_outs
            }
        else:
            outputs = {'pred_logits': out[:, :, :42],
                       'pred_boxes': out[:, :, 42:],
                       'pred_masks': outputs_seg_masks
                       }
        return outputs

    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        # assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices)

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

    def __call__(self, outputs, targets):
        r""" This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        labels, boxes, valid, masks = targets
        target = {}
        target['labels'] = labels.squeeze(0)
        target['boxes'] = boxes.squeeze(0)
        target['valid'] = valid
        target['masks'] = masks.squeeze(0)
        targets = [target]

        outputs = self._get_outputs(outputs)
        indices = self.matcher(outputs, targets)
        num_boxes = 0
        for target in targets:
            num_boxes = num_boxes + len(target['labels'])

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        if self.aux_loss:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                # 按顺序往losses里输入
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

def build_criterion():
    matcher = HungarianMatcher()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    weight_dict["loss_mask"] = 1
    weight_dict["loss_dice"] = 1

    aux_weight_dict = {}
    for i in range(6 - 1):
        aux_weight_dict.update(
            {k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(41, matcher, weight_dict, 0.1, ['labels', 'boxes', 'masks'], True)
    return criterion
