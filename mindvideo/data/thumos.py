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
""" THUMOS dataset. TODO: finish these dataset API. """

import os
import scipy.io
from mindvision.dataset.meta import ParseDataset
from mindvision.mindvideo.dataset.ucf101 import ParseUCF101
from mindvision.mindvideo.dataset.video_dataset import VideoDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["Thumos", "ParseThumos"]


@ClassFactory.register(ModuleType.DATASET)
class Thumos(VideoDataset):
    """
     Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test", "val" or "background". Default: "infer".
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
        align(boolean): The video contains multiple actions. Default: False.
        batch_size (int): Batch size of dataset. Default:32.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.

    Examples:
        >>> from mindvision.mindvideo.dataset.thumos import Thumos
        >>> dataset = Thumos("./data","train")
        >>> dataset = dataset.run()

    The thumos structure of THUMOS dataset looks like:

        .
        |-thumos
          |-- train                      // contains 101 classes from UCF101.
          |   |-- ApplyEyeMakeup
          |     |-- v_ApplyEyeMakeup_g08_c01.avi     // video file
          |     |-- v_ApplyEyeMakeup_g01_c02.avi     // video file
          |     ...
          |   |-- ApplyLipstick
          |     |-- v_ApplyLipstick_g01_c01.avi      // video file
          |     |-- v_ApplyLipstick_g01_c02.avi      // video file
          |    ...
          |-- TH14_test_set_mp4                  // contains 1574 test video files.
          |   |-- video_test_0000001.mp4      // video file
          |   |-- video_test_0000002.mp4      // video file
          |   |-- video_test_0000003.mp4      // video file
          |   ...
          |-- validation                  // contains 1010 validation video files.
          |   |-- video_validation_0000001.mp4      // video file
          |   |-- video_validation_0000002.mp4      // video file
          |   |-- video_validation_0000003.mp4      // video file
          |   ...
          |-- videos                  // contains 2500 background video files.
          |   |-- video_background_0000001.mp4      // video file
          |   |-- video_background_0000002.mp4      // video file
          |   |-- video_background_0000003.mp4      // video file
          |   ...
          |-- test_set_meta.mat                  // annotation file
          |-- validation_set.mat                 // annotation file
          |-- background_set.mat                 // annotation file
          ...
    """

    def __init__(self,
                 path,
                 split="train",
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=False,
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False
                 ):
        load_data = ParseUCF101(os.path.join(path, split)).parse_dataset
        super().__init__(path=path,
                         split=split,
                         load_data=load_data,
                         transform=transform,
                         target_transform=target_transform,
                         seq=seq,
                         seq_mode=seq_mode,
                         align=align,
                         batch_size=batch_size,
                         repeat_num=repeat_num,
                         shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         download=download)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        mapping = os.listdir(os.path.join(self.path, "train"))
        mapping.sort()
        return mapping

    def download_dataset(self):
        """Download the Thumos data if it doesn't exist already."""
        raise ValueError("Thumos dataset download is not supported.")



class ParseThumos(ParseDataset):
    """
    Parse Thumos dataset.
    """

    def parse_dataset(self):
        """Traverse the Thumos dataset file to get the path and label."""
        parse_ucf101 = ParseUCF101(self.path)
        split = os.path.split(parse_ucf101.path)[-1]
        base_path = os.path.dirname(parse_ucf101.path)
        if split == "train":
            return ParseUCF101(self.path).parse_dataset()
        if split == "test":
            desc = "test"
            video_path = "TH14_test_set_mp4"
            mat_file = "test_set_meta.mat"
        if split == "val":
            desc = "validation"
            video_path = "validation"
            mat_file = "validation_set.mat"
        if split == "background":
            desc = "background"
            video_path = "video"
            mat_file = "background_set.mat"
        mat_file = os.path.join(base_path, mat_file)
        base_path = os.path.join(base_path, video_path)
        info = scipy.io.loadmat(mat_file)
        video_path, video_label = [], []
        for row in info[f"{desc}_videos"][0]:
            path = os.path.join(base_path, row[0])
            video_path.append(path)
            video_label.append(int(row[3]) - 1)
        return video_path, video_label
