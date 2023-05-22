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
""" charades dataset. TODO: finish these dataset API. """

import os
import csv
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["Charades", "ParseCharades"]


@ClassFactory.register(ModuleType.DATASET)
class Charades(VideoDataset):
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
        >>> from mindvision.mindvideo.dataset.charades import Charades
        >>> dataset = Charades("./data","train")
        >>> dataset = dataset.run()

    The charades structure of Charades dataset looks like:

        .
        |-charades
            |-- Charades
            |   |-- ___Charades_v1_test.csv      // csv file
            |   |-- ___Charades_v1_test.csv      // csv file
            |   |--Charades_v1_classes.txt       //class label file
            |    ...
            |-- Charades_v1_480
            |   |-- 001YG.mp4.mp4       // video file
            |   |-- 003WS.mp4   // video file
            ...
    """

    def __init__(self,
                 path,
                 split="train",
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
        load_data = ParseCharades(os.path.join(path, split)).parse_dataset
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
        cls_file = os.path.join(self.path, "Charades", "Charades_v1_classes.txt ")
        mapping = []
        with open(cls_file, "rb")as f:
            content = f.readlines()
            for row in content:
                info = row[5:]
                mapping.append(info)
        return mapping

    def download_dataset(self):
        """Download the Charades data if it doesn't exist already."""
        raise ValueError("Charades dataset download is not supported.")


class ParseCharades(ParseDataset):
    """
    Parse Charades dataset.
    """
    urlpath = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"

    def parse_dataset(self):
        """Traverse the Charades dataset file to get the path and label."""
        parse_charades = ParseCharades(self.path)
        split = os.path.split(parse_charades.path)[-1]
        video_label, video_path = [], []
        base_path = os.path.dirname(parse_charades.path)
        csv_file = os.path.join(base_path, f"Charades/Charades_v1_{split}.csv")
        video_base_path = os.path.join(base_path, "Charades_v1_480")
        margin = 0.2
        with open(csv_file, "r")as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                file_path = os.path.join(video_base_path, f"{row['id']}.mp4")
                action_list = row['actions'].split(';')
                video_len = float(row['length'])
                for action in action_list:
                    info = action.split(' ')
                    label = []
                    cls = int(info[0][1:])
                    start = min(max(float(info[1]), 0), video_len)
                    end = min(max(float(info[2]), 0), video_len)
                    if end - start < margin:
                        continue
                    label.append([cls, start / video_len, end / video_len])
                    video_path.append(file_path)
                    video_label.append(label)
        return video_path, video_label
