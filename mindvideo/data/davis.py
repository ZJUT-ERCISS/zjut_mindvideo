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
""" DAVIS dataset. TODO: finish these dataset API. """

import os
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["Davis", "ParseDavis"]


@ClassFactory.register(ModuleType.DATASET)
class Davis(VideoDataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "val" or "trainval". Default: None.
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part"), "discrete", or "align" fetch. Default: "align".
        align(boolean): The video contains multiple actions.Default: True.
        quality(str):The Picture quality,"1080p" or "480p".Default:"480p".
        batch_size (int): Batch size of dataset. Default:32.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.

    Examples:
        >>> from mindvision.mindvideo.dataset.davis import Davis
        >>> dataset = Davis("./data", "train")
        >>> dataset = dataset.run()

    The davis structure of Davis dataset looks like:

        .
        |-davis
            |-- Annotation
            |   |-- 1080p
            |       |-- bear
            |           |-- 00000.png   // annotation file
            |           |-- 00000.png   // annotation file
            |   ...
            |       |-- blackswan
            |           |-- 00000.png   // annotation file
            |           |-- 00000.png   // annotation file
            |   ...
            |   |-- 480p
            |       |-- bear
            |           |-- 00000.png   // annotation file
            |           |-- 00001.png   // annotation file
            |   ...
            |       |-- blackswan
            |           |-- 00000.png   // annotation file
            |           |-- 00001.png   // annotation file
            |   ...
            |-- ImageSets
            |   |--1080p
            |       |--train.txt    //split file
            |       |--trainval.txt //split file
            |       |--val.txt  //split file
            |   |--480p
            |       |--train.txt    //split file
            |       |--trainval.txt //split file
            |       |--val.txt  //split file
            |-- JPEGImages
            |   |-- 1080p
            |       |-- bear
            |           |-- 00000.png   //a frame of video file
            |           |-- 00000.png   //a frame of video file
            |   ...
            |       |-- blackswan
            |           |-- 00000.png   //a frame of video file
            |           |-- 00000.png   //a frame of video file
            |   ...
            |   |-- 480p
            |       |-- bear
            |           |-- 00000.png   //a frame of video file
            |           |-- 00001.png   //a frame of video file
            |   ...
            |       |-- blackswan
            |           |-- 00000.png   //a frame of video file
            |           |-- 00001.png   //a frame of video file
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
                 quality="480p",
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False
                 ):
        load_data = ParseDavis(os.path.join(path, f"{split}_{quality}")).parse_dataset
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
        quality = os.path.split(self.path)[-1]
        split = os.path.split(self.path)[-2]
        base_path = os.path
        split_file = os.path.join(base_path, "ImageSets", quality, f"{split}.txt")
        mapping = []
        with open(split_file, "r")as f:
            rows = f.readlines()
            cls_dic = {}
            for row in rows:
                jpg = row.split(' ')[0]
                cls = jpg.split('/')[-2]
                if cls not in cls_dic:
                    mapping.append(cls)
        return mapping

    def download_dataset(self):
        """Download the UBI-fights data if it doesn't exist already."""
        raise ValueError("UBI-fights dataset download is not supported.")


class ParseDavis(ParseDataset):
    """
    Parse Davis dataset.
    """

    def parse_dataset(self):
        """Traverse the Davis dataset file to get the path and label."""
        parse_davis = ParseDavis(self.path)
        info = os.path.split(self.path)[-1]
        quality = info.split('_')[-1]
        split = info.split('_')[-2]
        base_path = os.path.dirname(parse_davis.path)
        video_label, video_path = [], []
        split_file = os.path.join(base_path, "ImageSets", quality, f"{split}.txt")
        with open(split_file, "r")as f:
            rows = f.readlines()
            cls_dic = {}
            video = []
            for row in rows:
                if not row:
                    continue
                jpg = row.split(' ')[0]
                cls = jpg.split('/')[-2]
                if cls not in cls_dic:
                    video_label.append(len(cls_dic))
                    cls_dic.setdefault(cls, len(cls_dic))
                    if video:
                        video_path.append(video)
                        video.clear()
                video.append(jpg)
            return video_path, video_label
