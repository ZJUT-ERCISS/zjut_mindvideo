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
""" MindSpore Vision Video evaluation script. """

import argparse
from mindspore import nn, context, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindvideo.utils.check_param import Validator, Rel
from mindvideo.data.transforms import VideoCenterCrop, VideoRescale, VideoResize, VideoReOrder
from mindvideo.models.c3d import C3D
from mindvideo.data import UCF101
from mindvideo.utils.callbacks import EvalLossMonitor


def c3d_ucf101_eval(args_opt):
    """C3D eval."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)

    dataset = UCF101(args_opt.data_url,
                     split="test",
                     seq=16,
                     seq_mode="average",
                     num_parallel_workers=args_opt.num_parallel_workers,
                     shuffle=False,
                     batch_size=args_opt.batch_size,
                     repeat_num=args_opt.repeat_num)

    # perpare dataset
    transforms = [VideoResize([128, 171]),
                  VideoRescale(shift="tutorials/classification/c3d/resized_mean_sports1m.npy"),
                  VideoCenterCrop([112, 112]),
                  VideoReOrder([3, 0, 1, 2])]
    dataset.transform = transforms
    dataset_eval = dataset.run()
    Validator.check_int(dataset_eval.get_dataset_size(), 0, Rel.GT)

    # set network
    network = C3D(num_classes=args_opt.num_classes, keep_prob=(1.0, 1.0, 1.0))

    # load pretrained weights
    param_dict = load_checkpoint(args_opt.pretrained_path)
    load_param_into_net(network, param_dict)

    # loss
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # metrics
    eval_metrics = {'Top1_Acc': nn.Top1CategoricalAccuracy(),
                    'Top5_Acc': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network,
                  network_loss,
                  metrics=eval_metrics)
    print_cb = EvalLossMonitor(model)

    # Begin to eval.
    result = model.eval(dataset_eval,
                        callbacks=[print_cb],
                        dataset_sink_mode=args_opt.dataset_sink_mode)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C3D eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', default="/home/publicfile/UCF101", help='Location of data.')
    parser.add_argument('--pretrained_path', type=str, default=".vscode/ms_ckpts/c3d_20220912.ckpt",
                        help='Location of pretrained ckpt.')
    parser.add_argument('--num_parallel_workers', type=int, default=1, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=101, help='Number of classification.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool, default=False, help='Distributed parallel testing.')

    args = parser.parse_known_args()[0]
    c3d_ucf101_eval(args)
