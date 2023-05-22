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
"""Image frame split to windows utils."""


def limit_window_size(input_size, window_size, shift_size):
    r"""
    Limit the window size and shift size for window W-MSA and SW-MSA.
    If window size is larger than input size, we don't partition or shift
    windows.

    Args:
        input_size (tuple[int]): Input size of features. E.g. (16, 56, 56).
        window_size (tuple[int]): Target window size. E.g. (8, 7, 7).
        shift_size (int): depth of video. E.g. (4, 3, 3).

    Returns:
        Tuple[int], limited window size and shift size.
    """

    use_window_size = list(window_size)
    use_shift_size = [0, 0, 0]
    if shift_size:
        use_shift_size = list(shift_size)

    for i in range(len(input_size)):
        if input_size[i] <= window_size[i]:
            use_window_size[i] = input_size[i]
            if shift_size:
                use_shift_size[i] = 0
    window_size = tuple(use_window_size)
    shift_size = tuple(use_shift_size)

    return window_size, shift_size


def window_partition(features, window_size):
    r"""
    Window partition function for Swin Transformer.

    Args:
        features: Original features of shape (B, D, H, W, C).
        window_size (tuple[int]): Window size.

    Returns:
        Tensor of shape (B * num_windows, window_size * window_size, C).
    """

    batch_size, depth, height, width, channel_num = features.shape
    windows = features.reshape(
        batch_size,
        depth // window_size[0], window_size[0],
        height // window_size[1], window_size[1],
        width // window_size[2], window_size[2],
        channel_num
    )
    windows = windows.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    windows = windows.reshape(-1, window_size[0] *
                              window_size[1] * window_size[2], channel_num)
    return windows


def window_reverse(windows, window_size, batch_size, depth,
                   height, width):
    r"""
    Window reverse function for Swin Transformer.

    Args:
        windows: Partitioned features of shape (B*num_windows, window_size,
            window_size, C).
        window_size (tuple[int]): Window size.
        batch_size (int): Batch size of video.
        depth (int): depth of video.
        height (int): Height of video.
        width (int): Width of video.

    Returns:
        Tensor of shape (B, D, H, W, C).
    """

    windows = windows.view(
        batch_size,
        depth // window_size[0],
        height // window_size[1],
        width // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1
    )
    windows = windows.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    windows = windows.view(batch_size, depth, height, width, -1)
    return windows
