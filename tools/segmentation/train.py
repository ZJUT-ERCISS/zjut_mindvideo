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
"""VisTR train"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import torch
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import nn
import mindspore.common.dtype as mstype
import mindspore

import mindspore.dataset as ds
from mindspore.common import set_seed
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindvideo.utils.callbacks import LossMonitor

from mindvideo.models.layers.instance_sequence_match import HungarianMatcher
from mindvideo.utils.config import parse_args, Config
from mindvideo.loss.builder import build_loss

from mindvideo.schedule.builder import get_lr
from mindvideo.data.builder import build_dataset, build_transforms
from mindvideo.models import build_model

set_seed(10)
ds.config.set_prefetch_size(1)


class LossCell(nn.Cell):
    """Cell with loss function.."""

    def __init__(self, net, loss):
        super().__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, video, labels, boxes, valids, masks, resize_shape):
        """Cell with loss function."""
        outputs, pred_masks = self._net(video)
        return self._loss(outputs, pred_masks, labels, boxes, valids, masks, resize_shape)


def main(pargs):
    """
    vistr resnet50 train
    """
    config = Config(pargs.config)
    context.set_context(**config.context)

    # perpare dataset
    transforms = build_transforms(config.data_loader.train.map.operations)
    data_set = build_dataset(config.data_loader.train.dataset)
    data_set.transform = transforms
    dataset_train = data_set.run()

    ckpt_save_dir = config.train.ckpt_path

    # set network
    network = build_model(config.model)

    # lr
    lr_cfg = config.learning_rate
    lr_embed_cfg = config.learning_rate

    lr = get_lr(lr_cfg)
    lr_embed = get_lr(lr_embed_cfg)

    #  Define optimizer.
    param_dicts = [
        {
            'params': [par for par in network.trainable_params() if 'embed' not in par.name],
            'lr': lr,
            'weight_decay': config.weight_decay
        },
        {
            'params': [par for par in network.trainable_params() if 'embed' in par.name],
            'lr': lr_embed,
            'weight_decay': config.weight_decay
        }
    ]
    config.optimizer.params = param_dicts
    network_opt = nn.AdamWeightDecay(config.optimizer.params)

    if config.train.pre_trained:
        # load pretrain model
        param_dict = load_checkpoint(config.train.pretrained_model)
        load_param_into_net(network, param_dict)

    # Define losgs function.
    matcher = HungarianMatcher(num_frames=config.matcher.num_frames,
                               cost_class=config.matcher.cost_class,
                               cost_bbox=config.matcher.cost_bbox,
                               cost_giou=config.matcher.cost_giou)
    config.loss.matcher = matcher
    # get weight_dict
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    weight_dict["loss_mask"] = 1
    weight_dict["loss_dice"] = 1
    aux_weight_dict = {}
    for i in range(6 - 1):
        aux_weight_dict.update(
            {k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    config.loss.weight_dict = weight_dict

    # set loss
    network_loss = build_loss(config.loss)

    # set checkpoint for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=config.train.save_checkpoint_steps,
        keep_checkpoint_max=config.train.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=config.model_name,
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # init the whole Model
    net_with_loss = LossCell(network, network_loss)

    loss_scale_manager = mindspore.FixedLossScaleManager(config.loss_scale)
    model = Model(net_with_loss,
                  optimizer=network_opt,
                  loss_scale_manager=loss_scale_manager)

    # Begin to train.
    print('[Start training `{}`]'.format(config.model_name))
    print("=" * 80)
    model.train(config.train.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=config.dataset_sink_mode)
    print('[End of training `{}`]'.format(config.model_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
