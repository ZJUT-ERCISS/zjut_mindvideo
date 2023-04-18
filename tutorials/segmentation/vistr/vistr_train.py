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
import torch
import argparse
import os
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import nn
import mindspore.common.dtype as mstype
import mindspore

import mindspore.dataset as ds
from mindspore.common import set_seed
from mindspore.communication import init, get_rank, get_group_size
from mindspore.profiler import Profiler
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from msvideo.utils.callbacks import LossMonitor

from msvideo.data import ytvos
from msvideo.models.vistr import VistrCom
from msvideo.loss.vistr_loss import SetCriterion
from msvideo.models.layers.instance_sequence_match import HungarianMatcher
from msvideo.schedule.lr_schedule import piecewise_constant_lr

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


def vistr_r50_train(args_opt):
    """
    vistr resnet50 train
    """
    if args_opt.run_distribute:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'
        context.set_context(mode=context.GRAPH_MODE,
                            device_target='GPU')
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.AUTO_PARALLEL,
                                            search_mode="dynamic_programming")

        if args_opt.profiler:
            profiler = Profiler(output_path='./profiler_data')

        dataset_train = ytvos.Ytvos(path=args_opt.dataset_path,
                                    split='train',
                                    seq=args_opt.num_frames,
                                    batch_size=args_opt.batch_size,
                                    repeat_num=args_opt.repeat_num,
                                    shuffle=args_opt.shuffle,
                                    shard_id=rank_id,
                                    num_shards=device_num)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target='GPU')

        if args_opt.profiler:
            profiler = Profiler(output_path='./profiler_data')

        dataset_train = ytvos.Ytvos(path=args_opt.dataset_path,
                                    split='train',
                                    seq=args_opt.num_frames,
                                    batch_size=args_opt.batch_size,
                                    repeat_num=args_opt.repeat_num,
                                    shuffle=args_opt.shuffle,
                                    shard_id=None,
                                    num_shards=None)
    dataset_train = dataset_train.run()
    step_size = dataset_train.get_dataset_size()
    ckpt_save_dir = args_opt.ckpt_save_dir

    network = VistrCom(name=args_opt.name,
                       num_frames=args_opt.num_frames,
                       num_queries=args_opt.num_queries,
                       dropout=args_opt.dropout,
                       aux_loss=args_opt.aux_loss,
                       num_class=args_opt.num_classes)
    
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_path)
        load_param_into_net(network, param_dict)

    # Set learning rate scheduler.
    lr = piecewise_constant_lr(
        [step_size * args_opt.lr_drop, step_size * args_opt.epochs],
        [args_opt.lr, args_opt.lr * 0.1]
    )
    lr_embed = piecewise_constant_lr(
        [step_size * args_opt.lr_drop, step_size * args_opt.epochs],
        [args_opt.lr_embed, args_opt.lr_embed * 0.1]
    )

    #  Define optimizer.
    param_dicts = [
        {
            'params': [par for par in network.trainable_params() if 'embed' not in par.name],
            'lr': lr,
            'weight_decay': args_opt.weight_decay
        },
        {
            'params': [par for par in network.trainable_params() if 'embed' in par.name],
            'lr': lr_embed,
            'weight_decay': args_opt.weight_decay
        }
    ]

    # Define losgs function.
    matcher = HungarianMatcher(num_frames=args_opt.num_frames,
                               cost_class=args_opt.cost_class,
                               cost_bbox=args_opt.cost_bbox,
                               cost_giou=args_opt.cost_giou)
    network_loss = SetCriterion(num_classes=args_opt.num_classes,
                                matcher=matcher,
                                weight_dict=args_opt.weight_dict,
                                eos_coef=args_opt.eos_coef,
                                aux_loss=args_opt.aux_loss)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=args_opt.ckpt_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model_name, directory=ckpt_save_dir, config=ckpt_config)
    # Init the model.
    # network.to_float(mstype.float16)
    # network_loss.to_float(mstype.float32)
    net_with_loss = LossCell(network, network_loss)

    network_opt = nn.AdamWeightDecay(param_dicts)
    loss_scale_manager = mindspore.FixedLossScaleManager(args_opt.loss_scale)
    model = Model(net_with_loss,
                  optimizer=network_opt,
                  loss_scale_manager=loss_scale_manager)

    # Begin to train.
    print('[Start training `{}`]'.format(args_opt.model_name))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=False)
    print('[End of training `{}`]'.format(args_opt.model_name))

    if args_opt.profiler:
        profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisTR_r50 Train.")
    parser.add_argument("--dataset_path", type=str,
                        default="/usr/dataset/VOS/")
    parser.add_argument("--model_name", type=str, default="vistr_r50")
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--epoch_size", type=int,
                        default=18, help="Train epoch size.")
    parser.add_argument('--epochs', default=18, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_embed', default=0.00001, type=float)
    parser.add_argument('--lr_drop', default=12, type=int)
    parser.add_argument("--pretrained", type=bool,
                        default=False, help="Load pretrained model.")
    # model
    parser.add_argument("--name", type=str, default="ResNet50")
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=41)
    parser.add_argument("--num_queries", type=int, default=360)
    # dataset
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of batch size.")
    parser.add_argument("--repeat_num", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--pretrained_path", type=str,
                        default="./pretrained_path",
                        help="Location of Pretrained Model.")
    parser.add_argument("--ckpt_save_dir", type=str,
                        default="./ckpt_save_dir", help="Location of training outputs.")
    parser.add_argument("--ckpt_max", type=int, default=10,
                        help="Max number of checkpoint files.")
    parser.add_argument("--dataset_sink_mode", default=False,
                        help="The dataset sink mode.")
    parser.add_argument("--run_distribute", type=bool,
                        default=False, help="Run distribute.")
    parser.add_argument("--num_parallel_workers", type=int,
                        default=8, help="Number of parallel workers.")
    parser.add_argument("--profiler", type=bool,
                        default=False, help="Use Profiler.")
    # matcher
    parser.add_argument("--cost_class", type=int, default=1)
    parser.add_argument("--cost_bbox", type=int, default=1)
    parser.add_argument("--cost_giou", type=int, default=1)
    # loss
    parser.add_argument("--weight_dict", type=dict)
    parser.add_argument("--eos_coef", type=float, default=0.1)
    parser.add_argument("--losses", type=list,
                        default=['labels', 'boxes', 'masks'])
    parser.add_argument("--aux_loss", type=bool, default=True)
    parser.add_argument("--loss_scale", type=float, default=1024.0)

    args = parser.parse_known_args()[0]

    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    weight_dict["loss_mask"] = 1
    weight_dict["loss_dice"] = 1

    aux_weight_dict = {}
    for i in range(6 - 1):
        aux_weight_dict.update(
            {k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    args.weight_dict = weight_dict
    vistr_r50_train(args)
