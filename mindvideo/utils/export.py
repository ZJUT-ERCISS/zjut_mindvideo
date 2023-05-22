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
import numpy as np
import mindspore as ms
from mindspore import context, nn, load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindvideo.utils.check_param import Validator, Rel
from mindvideo.utils.config import parse_args, Config
from mindvideo.utils.load import load_model
from mindvideo.data.builder import build_dataset, build_transforms
from mindvideo.models import build_model


def export_classification(pargs):
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)

    # perpare dataset
    if config.export.include_dataset:
        transforms = build_transforms(config.data_loader.eval.map.operations)
        dataset = build_dataset(config.data_loader.eval.dataset)
        dataset.transform = transforms
        dataset = dataset.run()
        Validator.check_int(dataset.get_dataset_size(), 0, Rel.GT)

    # set network and load pretrain model
    ckpt_path = config.export.pretrained_model
    network = None
    if os.path.splitext(ckpt_path)[-1] == '.ckpt':
        network = build_model(config.model)
    
    network = load_model(ckpt_path, network)


    input_tensor = ms.Tensor(np.ones(config.export.input_shape), ms.float32)
    ms.export(network,
              input_tensor,
              file_name=config.export.file_name,
              file_format=config.export.file_format)


if __name__ == '__main__':
    args = parse_args()
    export_classification(args)
