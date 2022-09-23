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
""" Video Swin Transformer training script. """

import argparse

from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import nn

from mindspore.common import set_seed
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import Accuracy
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.profiler import Profiler
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindvision.engine.callback import LossMonitor
from mindvision.engine.lr_schedule.lr_schedule import warmup_cosine_annealing_lr_v1
from mindvision.msvideo.dataset import Kinetic400
from mindvision.msvideo.dataset import transforms
from mindvision.msvideo.models import swin3d_t, swin3d_s, swin3d_b, swin3d_l


set_seed(42)


def swin_tiny_train(args_opt):
    """Swin3d train."""
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    if args_opt.profiler:
        profiler = Profiler(output_path='./profiler_data')
    # Data Pipeline.
    if args_opt.run_distribute:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset_train = Kinetic400(path=args_opt.dataset_path,
                                   split='train',
                                   seq=32,
                                   seq_mode='interval',
                                   batch_size=args_opt.batch_size,
                                   shuffle=False,
                                   num_parallel_workers=args_opt.num_parallel_workers,
                                   frame_interval=2,
                                   num_clips=1,
                                   num_shards=device_num,
                                   shard_id=rank_id)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset_train = Kinetic400(path=args_opt.dataset_path,
                                   split='train',
                                   seq=32,
                                   seq_mode='interval',
                                   batch_size=args_opt.batch_size,
                                   shuffle=False,
                                   num_parallel_workers=args_opt.num_parallel_workers,
                                   frame_interval=2,
                                   num_clips=1
                                   )
        ckpt_save_dir = args_opt.ckpt_save_dir
    dataset_train.transform = [transforms.VideoShortEdgeResize(size=256, interpolation='linear'),
                               transforms.VideoRandomCrop(size=(224, 224)),
                               transforms.VideoRescale(shift=0),
                               transforms.VideoReOrder(order=(3, 0, 1, 2)),
                               transforms.VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    dataset_train = dataset_train.run()
    step_size = dataset_train.get_dataset_size()

    # Create model.
    if args_opt.model_name == "swin3d_t":
        network = swin3d_t()
    elif args_opt.model_name == "swin3d_s":
        network = swin3d_s()
    elif args_opt.model_name == "swin3d_b":
        network = swin3d_b()
    elif args_opt.model_name == "swin3d_l":
        network = swin3d_l()
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model_dir)
        load_param_into_net(network, param_dict)

    # Set learning rate scheduler.
    lr = warmup_cosine_annealing_lr_v1(lr=0.001, steps_per_epoch=step_size,
                                       warmup_epochs=2.5, max_epoch=30, t_max=30, eta_min=0)

    #  Define optimizer.
    network_opt = nn.AdamWeightDecay(network.trainable_params(), lr, beta1=0.9, beta2=0.999, weight_decay=0.02)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=args_opt.ckpt_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model_name, directory=ckpt_save_dir, config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc": Accuracy()})

    # Begin to train.
    print('[Start training `{}`]'.format(args_opt.model_name))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr.tolist())],
                dataset_sink_mode=False)
    print('[End of training `{}`]'.format(args_opt.model_name))
    if args_opt.profiler:
        profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin3d_t Train.")
    parser.add_argument("--dataset_path", type=str, default="/usr/publicfile/kinetics-400")
    parser.add_argument("--model_name", type=str, default="swin3d_t")
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=3)
    parser.add_argument("--epoch_size", type=int, default=30, help="Train epoch size.")
    parser.add_argument("--pretrained", type=bool, default=True, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of batch size.")
    parser.add_argument("--ckpt_save_dir", type=str, default="./swin3d_t", help="Location of training outputs.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default="/home/yutw/vision/mindvision/msvideo/example/vist/pretrained/ms_swin_tiny_patch244_window877_kinetics400_1k.ckpt",
                        help="Location of Pretrained Model.")
    parser.add_argument("--ckpt_max", type=int, default=100, help="Max number of checkpoint files.")
    parser.add_argument("--dataset_sink_mode", default=False, help="The dataset sink mode.")
    parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute.")
    parser.add_argument("--num_parallel_workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--profiler", type=bool, default=False, help="Use Profiler.")
    args = parser.parse_known_args()[0]

    swin_tiny_train(args)
