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
"""MindSpore Vision Video training script."""

import argparse
from mindspore import nn
from mindspore import context
from mindspore import nn, load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from msvideo.data.transforms import VideoRandomCrop, VideoRescale, VideoResize, VideoReOrder, VideoRandomHorizontalFlip, VideoCenterCrop
from msvideo.models.c3d import C3D
from msvideo.data import UCF101
from msvideo.utils.callbacks import ValAccMonitor
from msvideo.utils.check_param import Validator, Rel
from msvideo.schedule import warmup_step_lr


def c3d_ucf101_train(args_opt):
    """C3D train."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            init()
        else:
            init("nccl")

        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          parameter_broadcast=True)
        ckpt_save_dir = args_opt.ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
        dataset = UCF101(args_opt.data_url,
                         split="train",
                         seq=16,
                         seq_mode="average",
                         num_parallel_workers=args_opt.num_parallel_workers,
                         shuffle=True,
                         num_shards=device_num,
                         shard_id=rank_id,
                         batch_size=args_opt.batch_size,
                         repeat_num=args_opt.repeat_num)
    else:
        dataset = UCF101(args_opt.data_url,
                         split="train",
                         seq=16,
                         seq_mode="average",
                         num_parallel_workers=args_opt.num_parallel_workers,
                         shuffle=True,
                         batch_size=args_opt.batch_size,
                         repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir

    # perpare dataset
    transforms = [VideoResize([128, 171]),
                  VideoRescale(shift="msvideo/example/c3d/resized_mean_sports1m.npy"),
                  VideoRandomCrop([112, 112]),
                  VideoRandomHorizontalFlip(0.5),
                  VideoReOrder([3, 0, 1, 2])]
    dataset.transform = transforms
    dataset_train = dataset.run()
    Validator.check_int(dataset_train.get_dataset_size(), 0, Rel.GT)
    step_size = dataset_train.get_dataset_size()

    # eval dataset
    dataset_e = UCF101(args_opt.data_url,
                       split="test",
                       seq=16,
                       seq_mode="average",
                       num_parallel_workers=args_opt.num_parallel_workers,
                       shuffle=False,
                       batch_size=args_opt.batch_size,
                       repeat_num=args_opt.repeat_num)

    # perpare dataset
    transforms = [VideoResize([128, 171]),
                  VideoRescale(shift="msvideo/example/c3d/resized_mean_sports1m.npy"),
                  VideoCenterCrop([112, 112]),
                  VideoReOrder([3, 0, 1, 2])]
    dataset_e.transform = transforms
    dataset_eval = dataset_e.run()

    # set network
    network = C3D(num_classes=args_opt.num_classes)
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_path)
        load_param_into_net(network, param_dict)
    # Set lr scheduler
    if args_opt.lr_decay_mode == 'exponential':
        lr = warmup_step_lr(lr=args_opt.learning_rate,
                            lr_epochs=args_opt.milestone,
                            steps_per_epoch=step_size,
                            warmup_epochs=args_opt.warmup_epochs,
                            max_epoch=args_opt.epoch_size,
                            gamma=args_opt.gamma)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define metrics.
    metrics = {'Top1_Acc': nn.Top1CategoricalAccuracy()}

    # Define optimizer.
    network_opt = nn.SGD(network.trainable_params(),
                         lr,
                         momentum=args_opt.momentum,
                         weight_decay=args_opt.weight_decay)

    # set checkpoint for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='c3d_ucf101',
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # Init the model.
    model = Model(network,
                  network_loss,
                  network_opt,
                  metrics=metrics)

    # begin to train0.
    print('[Start training `{}`]'.format('c3d_ucf101'))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback,
                           LossMonitor(),
                           ValAccMonitor(model, 
                                         dataset_eval, 
                                         num_epochs=args_opt.epoch_size,
                                         metric_name='Top1_Acc')],
                dataset_sink_mode=args_opt.dataset_sink_mode)
    print('[End of training `{}`]'.format('c3d_ucf101'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C3D train.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=150, help='Train epoch size.')
    parser.add_argument('--pretrained', type=bool, default=False,help='Load pretrained model.')
    parser.add_argument('--pretrained_path', default=None,help='Path to pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=20, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./c3d", help='Location of training outputs.')
    parser.add_argument('--num_parallel_workers', type=int, default=2, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=101, help='Number of classification.')
    parser.add_argument('--lr_decay_mode', type=str, default="exponential", help='Learning rate decay mode.')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Warmup epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Decay rate of learning rate.')
    parser.add_argument('--learning_rate', type=int, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--milestone', type=list, default=[10, 20, 30, 40, 75], help='A list of milestone.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the moving average.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay for the optimizer.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool, default=False, help='Distributed parallel training.')

    args = parser.parse_known_args()[0]
    c3d_ucf101_train(args)
