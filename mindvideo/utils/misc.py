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
"""
misc for VisTR
"""
import mindspore
from mindspore import ops


def get_mask(tensor):
    """get img masks"""
    shape = (tensor.shape[0], tensor.shape[2], tensor.shape[3])
    zeros = ops.Zeros()
    mask = zeros(shape, mindspore.float32)
    return mask


def _max_by_axis(the_list):
    """type: (List[List[int]]) -> List[int]"""
    maxes = the_list[0]
    maxs = []
    for i in maxes:
        maxs.append(i)
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            if maxes[index] >= item:
                maxs[index] = maxes[index]
            else:
                maxs[index] = item
    return maxs


def nested_tensor_from_tensor_list(tensor_list, split=True):
    """Normalize the input image data"""
    # TODO make this more general
    zeros = ops.Zeros()
    ones = ops.Ones()
    if split:
        tensor_list = [tensor for tensor in tensor_list]
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([img.shape for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        tensor = zeros((b, c, h, w), dtype)
        mask = ones((b, h, w), mindspore.dtype.bool_)
        for i, img in enumerate(tensor_list):
            tensor[i][: img.shape[0], : img.shape[1], : img.shape[2]] = img
            mask[i][: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return tensor, mask
