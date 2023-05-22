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
"""MindSpore Vision Video training script."""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../..")))

import numpy as np
import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.communication.management import init, get_rank, get_group_size

from mindvideo.utils.check_param import Validator, Rel
from mindvideo.utils.config import parse_args, Config
from mindvideo.loss.builder import build_loss
from mindvideo.schedule.builder import get_lr
from mindvideo.optim.builder import build_optimizer
from mindvideo.data.builder import build_dataset, build_transforms
from mindvideo.models import build_model


def train_classification(pargs):
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)
    # run distribute
    if config.train.run_distribute:
        if config.device_target == "Ascend":
            init()
        else:
            init("nccl")
        context.set_auto_parallel_context(device_num=get_group_size(),
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        ckpt_save_dir = config.train.ckpt_path + "ckpt_" + str(get_rank()) + "/"
    else:
        ckpt_save_dir = config.train.ckpt_path

    # perpare dataset
    transforms = build_transforms(config.data_loader.train.map.operations)
    data_set = build_dataset(config.data_loader.train.dataset)
    data_set.transform = transforms
    dataset_train = data_set.run()
    Validator.check_int(dataset_train.get_dataset_size(), 0, Rel.GT)
    batches_per_epoch = dataset_train.get_dataset_size()

    # set network
    network = build_model(config.model)

    # set loss
    network_loss = build_loss(config.loss)
    # set lr
    lr_cfg = config.learning_rate
    lr_cfg.steps_per_epoch = int(batches_per_epoch / config.data_loader.group_size)
    lr = get_lr(lr_cfg)

    # set optimizer
    config.optimizer.params = network.trainable_params()
    config.optimizer.learning_rate = lr
    network_opt = build_optimizer(config.optimizer)

    if config.train.pre_trained:
        # load pretrain model
        param_dict = load_checkpoint(config.train.pretrained_model)
        load_param_into_net(network, param_dict)

    # set checkpoint for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=config.train.save_checkpoint_steps,
        keep_checkpoint_max=config.train.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=config.model_name,
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # init the whole Model
    model = Model(network,
                  network_loss,
                  network_opt,
                  metrics={"Accuracy": Accuracy()})

    # begin to train
    print('[Start training `{}`]'.format(config.model_name))
    print("=" * 80)
    model.train(config.train.epochs,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor()],
                dataset_sink_mode=config.dataset_sink_mode)
    print('[End of training `{}`]'.format(config.model_name))

    if config.export.do_export:
        input_tensor = ms.Tensor(np.ones(config.export.input_shape), ms.float32)
        ms.export(network,
                  input_tensor,
                  file_name=config.export.file_name,
                  file_format=config.export.file_format)

if __name__ == '__main__':
    args = parse_args()
    train_classification(args)
