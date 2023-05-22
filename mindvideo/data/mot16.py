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
""" MOT16 dataset. TODO: finish these dataset API. """

import os
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["Mot16", "ParseMot16"]


@ClassFactory.register(ModuleType.DATASET)
class Mot16(VideoDataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", or "test". Default: "train".
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
        >>> from mindvision.mindvideo.dataset.mot16 import Mot16
        >>> dataset = Mot16("./data", "train")
        >>> dataset = dataset.run()

    The davis structure of mot16 dataset looks like:
        .
        |-mot16
            |-- test        //contains 7 videos.
            |   |-- MOT16-01
            |       |-- det
            |           |-- det.txt   // annotation file
            |       |-- img1
            |           |-- 000001.jpg   // a frame of image in a video.
            |           |-- 000002.jpg   // a frame of image in a video.
            |           |...
            |       |-- seqinfo.ini     // video information file.
            |   ...
            |-- train       //contains 7 videos.
            |   |-- MOT16-02
            |       |-- det
            |           |-- det.txt   // annotation file
            |       |-- gt
            |           |-- gt.txt   // annotation file
            |       |-- img1
            |           |-- 000001.jpg   // a frame of image in a video.
            |           |-- 000002.jpg   // a frame of image in a video.
            |           |...
            |       |-- seqinfo.ini     // video information file.

    """

    def __init__(self,
                 path,
                 split="train",
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
        load_data = ParseMot16(os.path.join(path, split)).parse_dataset
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
                         download=download,
                         suffix=suffix)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        raise ValueError("Mot16 dataset has no label.")

    def download_dataset(self):
        """Download the Mot16 data if it doesn't exist already."""
        raise ValueError("Mot16 dataset download is not supported.")


class ParseMot16(ParseDataset):
    """
    Parse Mot16 dataset.
    """
    urlpath = "https://motchallenge.net/data/MOT16.zip"

    def parse_dataset(self, *args):
        """Traverse the Mot16 dataset file to get the path and label."""
        parse_mot16 = ParseMot16(self.path)
        split = os.path.split(self.path)[-1]
        video_list = os.listdir(parse_mot16.path)
        video_label, video_path = [], []
        for video in video_list:
            if split == "test":
                txt_file = os.path.join(parse_mot16.path, video, "det", "det.txt")
            if split == "train":
                txt_file = os.path.join(parse_mot16.path, video, "gt", "gt.txt")
            video_det = []
            img_list_path = os.path.join(parse_mot16.path, video, "img1")
            with open(txt_file, "r") as f:
                rows = f.readlines()
                seq_len = int(rows[-1].split(',')[0])
                img_list = [os.path.join(img_list_path, str(i + 1).zfill(6)) + str(".jpg") for i in range(seq_len)]
                video_path.append(img_list)
                for row in rows:
                    row = row.strip()
                    info = list(map(float, row.split(',')))
                    video_det.append(info)
                video_label.append(video_det)
        return video_path, video_label
