from mindspore import nn
from mindspore import numpy as np
from mindvision.msvideo.dataset import ytvos2
from mindvision.msvideo.models.vistr import vistr_r50
from mindspore import context, load_checkpoint, load_param_into_net, ops
from mindvision.msvideo.models.blocks.vistr_loss import SetCriterion
from mindvision.msvideo.models.neck.instance_sequence_match import HungarianMatcher
from mindvision.msvideo.models.loss.vistr_loss import build_criterion
from pycocotools.cocoeval import COCOeval

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_id=1)

network = vistr_r50(pretrained=False)
param_dict = load_checkpoint(
    "/home/zgz/VisTR/vistr_r50.ckpt")
load_param_into_net(network, param_dict)


weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
weight_dict['loss_giou'] = 2
weight_dict["loss_mask"] = 1
weight_dict["loss_dice"] = 1

aux_weight_dict = {}
for i in range(6 - 1):
    aux_weight_dict.update(
        {k + f'_{i}': v for k, v in weight_dict.items()})
weight_dict.update(aux_weight_dict)


eos_coef = 0.1
num_classes = 41
losses = ['labels', 'boxes', 'masks']
matcher = HungarianMatcher(num_frames=36, cost_class=1, cost_bbox=1, cost_giou=1)
criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict,
                              eos_coef=eos_coef, losses=losses, aux_loss=False)

dataset_train = ytvos2.ytvos(path="/usr/dataset/VOS/",
                            split='train',
                            seq=36,
                            batch_size=1,
                            repeat_num=1,
                            shuffle=False)
            
# criterion = build_criterion()
dataset_train = dataset_train.run()
# out = np.ones((6, 1, 360,46))
# outputs_seg_masks = np.ones((1, 360, 75, 85))

class LossCell(nn.Cell):
    """Cell with loss function.."""

    def __init__(self, net, loss):
        super(LossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, data):
        """Cell with loss function."""
        image = data[0]
        feature = self._net(image)
        return self._loss(feature, (data[1], data[2], data[3], data[4]))

    @property
    def backbone_network(self):
        """Return net."""
        return self._net

network = LossCell(network, criterion)

for data in dataset_train.create_tuple_iterator():
    # labels = np.ones((360))
    # boxes = np.ones((360, 4))
    # valid = np.ones((1, 360))
    # masks = np.ones((360 ,10, 4))
    loss = network(data)
    print(loss)
