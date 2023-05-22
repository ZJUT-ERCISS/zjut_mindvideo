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
"""MindSpore Vision Video infer script."""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import numpy as np
import decord

import mindspore as ms
from mindspore import context, ops, Tensor

from mindvideo.utils.config import parse_args, Config
from mindvideo.utils.load import load_model
from mindvideo.data.builder import build_dataset, build_transforms
from mindvideo.models import build_model


def infer_classification(pargs):
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)
    
    # perpare dataset
    transforms = build_transforms(config.data_loader.eval.map.operations)
    data_set = build_dataset(config.data_loader.eval.dataset)

    # set network and load pretrain model
    ckpt_path = config.infer.pretrained_model
    network = None
    if os.path.splitext(ckpt_path)[-1] == '.ckpt':
        network = build_model(config.model)
    
    network = load_model(ckpt_path, network)

    expand_dims = ops.ExpandDims()

    # 随机生成一个指定视频
    vis_num = len(data_set.video_path)
    vid_idx = np.random.randint(vis_num)
    video_path = data_set.video_path[vid_idx]

    if isinstance(video_path, list):
        video_path = video_path[np.random.randint(len(video_path))]
    print(video_path)

    video_reader = decord.VideoReader(video_path, num_threads=1)
    img_set = []

    for k in range(16):
        im = video_reader[k].asnumpy()
        img_set.append(im)
    video = np.stack(img_set, axis=0)
    for t in transforms:
        video = t(video)
    video = Tensor(video, ms.float32)
    video = expand_dims(video, 0)
    # Begin to eval.
    result = network(video)
    result = result.asnumpy()
    print("This is {}-th category".format(result.argmax()))

    return result


if __name__ == '__main__':
    args = parse_args()
    result = infer_classification(args)
    # print(result)
