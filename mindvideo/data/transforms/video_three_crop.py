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
"""Video transforms functions."""

import numpy as np

import mindspore.dataset.transforms.py_transforms as trans

from mindvideo.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class VideoThreeCrop(trans.PyTensorOperation):
    """
    Crop each frame of the input video into three crops with equal intervals along the shorter side.

    Args:
        crop_size (tuple[int]): The output size of the cropped image.
    """

    def __init__(self, size=(224, 224)):
        self.crop_size = size

    def __call__(self, video):
        """
        Args:
            Video(list): Video to be three-cropped.

        Returns:
            seq video: Cropped seq video.
        """
        _, h, w, _ = video.shape
        crop_h, crop_w = self.crop_size
        assert crop_h == h or crop_w == w
        if crop_h == h:
            w_step = (w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == w:
            h_step = (h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]
        cropped = []

        for x_offset, y_offset in offsets:
            crop = video[:, y_offset:y_offset + crop_h, x_offset:x_offset + crop_w, :]
            cropped.append(crop)
        cropped = np.concatenate(cropped, axis=0)

        return cropped
