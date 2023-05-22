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
""" ARN eval script. """

import argparse

from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.nn import MSELoss
from mindspore.train import Model

from mindvideo.models.arn import ARN
from mindvideo.data.ucf101 import UCF101
from mindvideo.data.transforms.video_to_tensor import VideoToTensor
from mindvideo.data.transforms import VideoResize, VideoReshape, VideoNormalize

from mindvideo.utils.task_acc import TaskAccuracy

def arn_rgb_eval(args_opt):
    """ARN eval."""
    context.set_context(mode=context.PYNATIVE_MODE,
                        # mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target, device_id=args_opt.device_id)

    # Data Pipeline.
    dataset_eval = UCF101(path=args_opt.dataset_path,
                          batch_size=args_opt.batch_size,
                          split='test',
                          shuffle=False,
                          seq=16,
                          num_parallel_workers=args_opt.num_parallel_workers,
                          suffix="task",
                          task_num=100,
                          task_n=5,
                          # task_k=5,
                          task_k=1,
                          task_q=1
                          )
    transforms = [
        VideoReshape((-1, 240, 320, 3)),
        VideoResize((128, 128)),
        VideoToTensor(),
        VideoNormalize((0.3474, 0.3474, 0.3474), (0.2100, 0.2100, 0.2100)),
        VideoReshape((3, -1, 16, 128, 128))
    ]
    dataset_eval.transform = transforms
    dataset_eval = dataset_eval.run()

    # Create model.
    # network = ARN(support_num_per_class=5)
    network = ARN()

    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model_dir)
        load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = MSELoss()

    # Init the model.
    model = Model(network, loss_fn=network_loss,
                  metrics={"Accuracy": TaskAccuracy()})

    # Begin to eval.
    print('[Start eval `{}`]'.format(args_opt.model_name))
    result = model.eval(dataset_eval, dataset_sink_mode=False)
    print("result:", result["Accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arn_eval Eval.")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/publicfile/UCF101")
    parser.add_argument("--model_name", type=str, default="arn")
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=2)
    parser.add_argument("--pretrained", type=bool,
                        default=True, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of batch size.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default="/home/huyt/807_ARN_ucf_CROSS0.7446666666666667.ckpt",
                        help="Location of Pretrained Model.")
    parser.add_argument("--dataset_sink_mode", default=False,
                        help="The dataset sink mode.")
    parser.add_argument("--num_parallel_workers", type=int,
                        default=1, help="Number of parallel workers.")

    args = parser.parse_known_args()[0]
    arn_rgb_eval(args)
