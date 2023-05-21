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
"""Mask Utils."""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from .windows import window_partition

def compute_mask(depth, height, width, window_size, shift_size):
    r"""
    Calculate attention mask for SW-MSA.

    Args:
        depth, height, width (int): Numbers of depth, height, width
            dimensions.
        window_size (Tuple(int)): Input window size.
        shift_size (Tuple(int)): Input shift_size.

    Returns:
        Tensor, attention mask.
    """
    t_padded = int(np.ceil(depth / window_size[0])) * window_size[0]
    h_padded = int(np.ceil(height / window_size[1])) * window_size[1]
    w_padded = int(np.ceil(width / window_size[2])) * window_size[2]
    img_mask = np.zeros((1, t_padded, h_padded, w_padded, 1))
    cnt = 0
    d_slices = (slice(-window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None))
    h_slices = (slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None))
    w_slices = (slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None))

    for i in d_slices:
        for j in h_slices:
            for k in w_slices:
                img_mask[:, i, j, k, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows[:, np.newaxis, :] - mask_windows[:, :, np.newaxis]
    attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.),
                       dtype=mstype.float32)
    return attn_mask
