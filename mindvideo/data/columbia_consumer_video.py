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
""" columbia consumer video dataset. TODO: finish these dataset API. """

import os
import re
from typing import Optional, Callable

from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["CCV", "ParseCCV"]


@ClassFactory.register(ModuleType.DATASET)
class CCV(VideoDataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports, options["train", "test", "val", "infer"]. Default: "train".
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
        align(bool): The video contains multiple actions. Default: False.
        batch_size (int): Batch size of dataset. Default:16.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.

    Examples:
        >>> from mindvision.mindvideo.dataset.collective_activity import CCV
        >>> dataset = CCV("./data","train")
        >>> dataset = dataset.run()

    The directory structure of columbia consumer video dataset looks like:

        .
        |-columbia consumer video
            |-- train
            |   |-- 1MdlncaU5JE.mp4      // video file
            |   |-- 3dKrnfAgTmk.mp4      // video file
            |    ...
            |-- test
            |   |-- -0n50a7seNI.mp4       // video file
            |   |-- -IByByzFDyQ           // video file
            |    ...
            |-- categoryName.txt          //category file.
            |-- trainVidID.txt      //train dataset split file.
            |-- trainLabel.txt        //train dataset label file.
            |-- testVidID.txt          //test dataset split file.
            |-- testLabel.txt           //test dataset label file.
            ...
    """

    def __init__(self,
                 path: str,
                 split: str = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 seq: int = 16,
                 seq_mode: str = "part",
                 align: bool = False,
                 batch_size: int = 16,
                 repeat_num: int = 1,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[bool] = None,
                 shard_id: Optional[bool] = None,
                 download: bool = False
                 ):
        load_data = ParseCCV(os.path.join(path, split)).parse_dataset
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
        parse_ccv = ParseCCV(os.path.join(self.path, self.split))
        mapping, _ = parse_ccv.load_cls_file()
        return mapping

    def download_dataset(self):
        """Download the columbia consumer video data if it doesn't exist already."""
        raise ValueError("Columbia consumer video dataset download is not supported.")


class ParseCCV(ParseDataset):
    """
    Parse columbia consumer video dataset.
    """

    def load_cls_file(self):
        """Parse the category file."""
        cls_txt = os.path.join(self.path, "categoryName.txt")
        mapping = []
        mapping.append("index from 1.")
        with open(cls_txt, "r")as f:
            rows = f.readlines()
            for row in rows:
                mapping.append(row.strip())
        return mapping

    def parse_dataset(self, *args):
        """Traverse the columbia consumer video dataset file to get the path and label."""
        parse_ccv = ParseCCV(self.path)
        split = os.path.split(parse_ccv.path)[-1]
        video_label, video_path = [], []
        label_txt_file = os.path.join(os.path.dirname(parse_ccv.path), f"{split}Label.txt")
        path_txt_file = os.path.join(os.path.dirname(parse_ccv.path), f"{split}VidId.txt")
        with open(label_txt_file, "r")as f:
            rows = f.readlines()
            for row in rows:
                row = row.strip()
                info = list(map(int, row.split(' ')))
                info = [i * info[i] for i in range(1, 21)]
                label = sum(info)
                video_label.append(label)
        with open(path_txt_file, "r")as f:
            rows = f.readlines()
            for row in rows:
                row = row.strip()
                path = os.path.join(parse_ccv.path, row)
                video_path.append(path)
        return video_path, video_label

    def modify_file_name(self, path):
        video_list = os.listdir(path)
        for video_file in video_list:
            video_id = re.findall(r"\[(.+?)]", video_file)
            if video_id:
                os.rename(os.path.join(path, video_file), os.path.join(path, f"{video_id}.mp4"))
