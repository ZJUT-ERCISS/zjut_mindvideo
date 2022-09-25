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
""" ARN training script. """

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net, ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.nn import MSELoss
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

from msvideo.models.arn import ARN
from msvideo.data.ucf101 import UCF101
from msvideo.data.transforms.video_to_tensor import VideoToTensor
from msvideo.data.transforms import VideoResize, VideoReshape, VideoNormalize

from msvideo.utils.callbacks import SaveCallback
from msvideo.utils.task_acc import TaskAccuracy


def arn_rgb_train(args_opt):
    """ARN train."""
    context.set_context(mode=context.PYNATIVE_MODE,
                        # mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target, device_id=args_opt.device_id)

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
        dataset_train = UCF101(path=args_opt.dataset_path,
                               batch_size=args_opt.batch_size,
                               split='train',
                               shuffle=True,
                               seq=16,
                               num_parallel_workers=args_opt.num_parallel_workers,
                               num_shards=device_num,
                               shard_id=rank_id)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset_train = UCF101(path=args_opt.dataset_path,
                               batch_size=args_opt.batch_size,
                               split='train',
                               shuffle=False,
                               seq=16,
                               num_parallel_workers=args_opt.num_parallel_workers,
                               suffix="task",
                               task_num=args_opt.step_size,
                               task_n=5,
                               #    task_k=5,
                               task_k=1,
                               task_q=1)
        dataset_valid = UCF101(path=args_opt.dataset_path,
                               batch_size=args_opt.batch_size,
                               split='test',
                               shuffle=False,
                               seq=16,
                               num_parallel_workers=args_opt.num_parallel_workers,
                               suffix="task",
                               task_num=100,
                               task_n=5,
                               #    task_k=5,
                               task_k=1,
                               task_q=1
                               )
        ckpt_save_dir = args_opt.ckpt_save_dir

    transforms = [
        VideoReshape((-1, 240, 320, 3)),
        VideoResize((128, 128)),
        VideoToTensor(),
        VideoNormalize((0.3474, 0.3474, 0.3474), (0.2100, 0.2100, 0.2100)),
        VideoReshape((3, -1, 16, 128, 128))

    ]

    dataset_train.transform = transforms
    dataset_train = dataset_train.run()

    dataset_valid.transform = transforms
    dataset_valid = dataset_valid.run()

    step_size = dataset_train.get_dataset_size()

    # network = arn(support_num_per_class=5)
    network = ARN()
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model_dir)
        load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = MSELoss()

    # Define optimizer.
    network_opt = nn.Adam(network.trainable_params(),
                          learning_rate=args_opt.learning_rate,
                          )

    # set checkpoint for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size, keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(
        prefix=args_opt.model_name, directory=ckpt_save_dir, config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss,
                  optimizer=network_opt, metrics={"Accuracy": TaskAccuracy()})

    # Begin to train.
    print(f"[Start training {args_opt.model_name}]")
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(),
                           SaveCallback(model, dataset_valid)
                           ],
                dataset_sink_mode=False)
    print(f"[End of training {args_opt.model_name}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arn_train Train.")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/publicfile/UCF101")
    parser.add_argument("--model_name", type=str, default="arn")
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--step_size", type=int,
                        default=100000, help="Train epoch size.")
    parser.add_argument("--pretrained", type=bool,
                        default=False, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of batch size.")
    parser.add_argument('--keep_checkpoint_max', type=int,
                        default=10, help='Max number of checkpoint files.')
    parser.add_argument("--ckpt_save_dir", type=str,
                        default="./arn", help="Location of training outputs.")
    parser.add_argument("--epoch_size", type=int,
                        default=1, help="Train epoch size.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default=None,
                        help="Location of Pretrained Model.")
    parser.add_argument('--learning_rate', type=int,
                        default=0.001, help='Initial learning rate.')
    parser.add_argument('--milestone', type=list,
                        default=[60, 70], help='A list of milestone.')
    parser.add_argument("--dataset_sink_mode", default=False,
                        help="The dataset sink mode.")
    parser.add_argument("--run_distribute", type=bool,
                        default=False, help="Run distribute.")
    parser.add_argument("--num_parallel_workers", type=int,
                        default=1, help="Number of parallel workers.")

    args = parser.parse_known_args()[0]
    arn_rgb_train(args)
