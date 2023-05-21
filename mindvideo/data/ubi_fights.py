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
""" UBI FIGHTS dataset. TODO: finish these dataset API. """

import os
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["UbiFights", "ParseUbiFights"]


@ClassFactory.register(ModuleType.DATASET)
class UbiFights(VideoDataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test" or "infer". Default: None.
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part"), "discrete", or "align" fetch. Default: "align".
        align(boolean): The video contains multiple actions.Default: True.
        batch_size (int): Batch size of dataset. Default:32.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.

    Examples:
        >>> from mindvision.mindvideo.dataset.ubi_fights import UbiFights
        >>> dataset = UbiFights("./data", "test")
        >>> dataset = dataset.run()

    The ubi_fights structure of UBI-fights dataset looks like:

        .
        |-ubi_fights
            |-- annotation
            |   |-- F_0_1_0_0_0.csv         // annotation file
            |   |-- F_100_1_2_0_0.csv       // annotation file
            |   |-- F_101_1_2_0_0.csv       //annotation file
            |    ...
            |-- videos
            |   |-- fight
            |       |-- F_0_1_0_0_0.mp4     // video file
            |       |-- F_100_1_2_0_0.mp4   // video file
            |   |-- normal
            |       |-- N_0_0_0_1_0.mp4     // video file
            |       |-- N_100_1_0_1_0.mp4   // video file
            |   |-- test_videos.csv   // split file
            |   |-- train_videos.csv  // split file
            ...
    """

    def __init__(self,
                 path,
                 split="test",
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=True,
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False
                 ):
        load_data = ParseUbiFights(os.path.join(path, split)).parse_dataset
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
        split = os.path.split(self.path)[-1]
        base_path = os.path.dirname(self.path)
        split_file = os.path.join(base_path, f"{split}_videos.csv")
        mapping = []
        with open(split_file, "rb")as f:
            rows = f.readlines()
            for row in rows:
                row = str(row)
                if row[0] == 'N':
                    type_id = 0
                if row[0] == 'F':
                    type_id = 1
                mapping.append(type_id)
        return mapping

    def download_dataset(self):
        """Download the UBI-fights data if it doesn't exist already."""
        raise ValueError("UBI-fights dataset download is not supported.")


class ParseUbiFights(ParseDataset):
    """
    Parse Ubi-fights dataset.
    """

    def parse_dataset(self):
        """Traverse the Ubi-Fights dataset file to get the path and label."""
        parse_ubifights = ParseUbiFights(self.path)
        split = os.path.split(parse_ubifights.path)[-1]
        base_path = os.path.dirname(parse_ubifights.path)
        video_label, video_path = [], []
        split_file = os.path.join(base_path, f"{split}_videos.csv")
        with open(split_file, "r")as f:
            rows = f.readlines()
            file_dir = os.path.join(base_path, "video")
            for row in rows:
                row = row.strip()
                if row[0] == 'N':
                    label_type = "normal"
                if row[0] == 'F':
                    label_type = "fight"
                video_path.append(os.path.join(file_dir, label_type, row))
                annotation_path = os.path.join(base_path, "annotation", f"{row}.csv")
                with open(annotation_path, "r") as an:
                    info = an.readlines()
                    label = list(map(int, info))
                video_label.append(label)
        return video_path, video_label
