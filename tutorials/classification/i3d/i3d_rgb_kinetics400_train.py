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
""" I3D training script. """

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net, ParallelMode

from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Accuracy
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor


from msvideo.data.kinetics400 import Kinetic400
from msvideo.models.i3d import I3D
from msvideo.data.transforms import VideoToTensor, VideoRandomCrop, VideoRandomHorizontalFlip, VideoResize
from msvideo.schedule.lr_schedule import warmup_step_lr




def i3d_rgb_train(args_opt):
    """I3D train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)

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
        dataset_train = Kinetic400(path=args_opt.dataset_path,
                                   batch_size=args_opt.batch_size,
                                   split='train',
                                   shuffle=True,
                                   seq=64,
                                   num_parallel_workers=args_opt.num_parallel_workers,
                                   num_shards=device_num,
                                   shard_id=rank_id)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset_train = Kinetic400(path=args_opt.dataset_path,
                                   batch_size=args_opt.batch_size,
                                   split='train',
                                   shuffle=True,
                                   seq=64,
                                   num_parallel_workers=args_opt.num_parallel_workers)
        ckpt_save_dir = args_opt.ckpt_save_dir

    transforms = [VideoResize([256, 256]),
                  VideoRandomCrop([224, 224]),
                  VideoRandomHorizontalFlip(0.5),
                  VideoToTensor()]
    dataset_train.transform = transforms
    dataset_train = dataset_train.run()
    step_size = dataset_train.get_dataset_size()

    # Create model.
    network = I3D()

    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model_dir)
        load_param_into_net(network, param_dict)

    # Set learning rate scheduler.
    lr = warmup_step_lr(lr=args_opt.learning_rate,
                        lr_epochs=args_opt.milestone,
                        steps_per_epoch=step_size,
                        warmup_epochs=args_opt.warmup_epochs,
                        max_epoch=args_opt.epoch_size,
                        gamma=args_opt.gamma)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define optimizer.
    network_opt = nn.SGD(network.trainable_params(),
                         lr,
                         momentum=args_opt.momentum,
                         weight_decay=args_opt.weight_decay,
                         loss_scale=args_opt.loss_scale)

    # set checkpoint for the network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model_name, directory=ckpt_save_dir, config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"Accuracy": Accuracy()})

    # Begin to train.
    print('[Start training `{}`]'.format(args_opt.model_name))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr.tolist())],
                dataset_sink_mode=False)
    print('[End of training `{}`]'.format(args_opt.model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I3d_rgb_train Train.")
    parser.add_argument("--dataset_path", type=str, default="/home/publicfile/kinetics-dataset")
    parser.add_argument("--model_name", type=str, default="i3d_rgb")
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=3)
    parser.add_argument("--epoch_size", type=int, default=100, help="Train epoch size.")
    parser.add_argument("--pretrained", type=bool, default=True, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of batch size.")
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument("--ckpt_save_dir", type=str, default="./i3d_rgb", help="Location of training outputs.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default=".vscode/ms_ckpt/tf2ms_i3d_rgb_imagenet.ckpt",
                        help="Location of Pretrained Model.")
    parser.add_argument('--warmup_epochs', type=int, default=4, help='Warmup epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Decay rate of learning rate.')
    parser.add_argument('--learning_rate', type=int, default=0.0012, help='Initial learning rate.')
    parser.add_argument('--milestone', type=list, default=[60, 70], help='A list of milestone.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the moving average.')
    parser.add_argument('--weight_decay', type=float, default=0.0004, help='Weight decay for the optimizer.')
    parser.add_argument('--loss_scale', type=float, default=1024, help='Loss scale for the optimizer.')
    parser.add_argument("--dataset_sink_mode", default=False, help="The dataset sink mode.")
    parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute.")
    parser.add_argument("--num_parallel_workers", type=int, default=8, help="Number of parallel workers.")

    args = parser.parse_known_args()[0]

    i3d_rgb_train(args)
