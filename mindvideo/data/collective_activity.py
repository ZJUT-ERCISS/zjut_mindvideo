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
""" Collective Activity dataset. TODO: finish these dataset API. """

import os
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["CollectiveActivity", "ParseCollectiveActivity"]


@ClassFactory.register(ModuleType.DATASET)
class CollectiveActivity(VideoDataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part"), "discrete", or "align" fetch. Default: "align".
        align(boolean): The video contains multiple actions.Default: False.
        batch_size (int): Batch size of dataset. Default:1.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.
        suffix(str): Storage format of video. Default: "picture".

    Examples:
        >>> from mindvision.mindvideo.dataset.collective_activity import CollectiveActivity
        >>> dataset = CollectiveActivity("./data")
        >>> dataset = dataset.run()

    The collective_activity structure of CollectiveActivity dataset looks like:
        .
        |-collective_activity  //contains 44 video sequences.
            |-- seq01
            |   |-- annotations.txt
            |   |-- frame0001.jpg
            |   |-- frame0002.jpg
            |   |-- frame0003.jpg
            |   |-- frame0004.jpg
            |   ...
            |-- seq02
            |   |-- annotations.txt
            |   |-- frame0001.jpg
            |   |-- frame0002.jpg
            |   |-- frame0003.jpg
            |   |-- frame0004.jpg
            |   ...
            |...

    """

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=False,
                 batch_size=1,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False,
                 suffix="picture"
                 ):
        load_data = ParseCollectiveActivity(path).parse_dataset
        super().__init__(path=path,
                         split="",
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
                         download=download,
                         suffix=suffix)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        raise ValueError("Collective_activity dataset has no label.")

    def download_dataset(self):
        """Download the Collective_activity data if it doesn't exist already."""
        raise ValueError("Collective_activity dataset download is not supported.")


class ParseCollectiveActivity(ParseDataset):
    """
    Parse Collective_activity dataset.
    """

    def parse_dataset(self):
        """Traverse the Mot16 dataset file to get the path and label."""
        parse_mot16 = ParseCollectiveActivity(self.path)
        video_list = os.listdir(parse_mot16.path)
        video_label, video_path = [], []
        for video in video_list:
            video_info = []
            txt_file = os.path.join(parse_mot16.path, video, "annotations.txt")
            img_list_path = os.path.join(parse_mot16.path, video)
            seq_len = len(os.listdir(img_list_path)) - 1
            img_list = [os.path.join(img_list_path,
                                     "frame" + str(i + 1).zfill(4)) + str(".jpg") for i in range(seq_len - 1)]
            video_path.append(img_list)
            with open(txt_file, "r") as f:
                rows = f.readlines()
                for row in rows:
                    row = row.split('\t')
                    row[-1] = row[-1].strip()
                    info = list(map(int, row))
                    video_info.append(info)
                video_label.append(video_info)
        return video_path, video_label
