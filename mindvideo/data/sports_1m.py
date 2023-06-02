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
""" sports-1m dataset. TODO: finish these dataset API. """

import os
import csv
import json

from mindvideo.data import transforms
from mindvideo.data.meta import ParseDataset
from mindvideo.data.video_dataset import VideoDataset

from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = ["Sport1M", "ParseSport1M"]


@ClassFactory.register(ModuleType.DATASET)
class Sport1M(VideoDataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test" or "infer". Default: None.
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label.
                Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
        align(boolean): The video contains multiple actions. Default: False.
        batch_size (int): Batch size of dataset. Default:32.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.
                Default: 1.
        num_shards (int, optional): Number of shards that the dataset will be divided into.
                Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool): Whether to download the dataset. Default: False.
        frame_interval (int): Frame interval of the sample strategy. Default: 1.
        num_clips (int): Number of clips sampled in one video. Default: 1.
    Examples:
        >>> from mindvision.mindvideo.dataset.sports_1m import Sport1M
        >>> dataset = Sport1M("./data/","train")
        >>> dataset = dataset.run()

    """

    def __init__(self,
                 path,
                 split=None,
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
                 download=False,
                 frame_interval=1,
                 num_clips=1
                 ):
        load_data = ParseSport1M(os.path.join(path, split)).parse_dataset
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
                         frame_interval=frame_interval,
                         num_clips=num_clips
                         )

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        csv_file = os.path.join(self.path, f"sport-1m_{self.split}.csv")
        mapping = []
        with open(csv_file, "r")as f:
            f_csv = csv.DictReader(f)
            c = 0
            for row in f_csv:
                if not cls:
                    cls = row['label']
                    mapping.append(cls)
                if row['label'] != cls:
                    c += 1
                    cls = row['label']
                    mapping.append(cls)
        return mapping

    def download_dataset(self):
        """Download the Sport1M data if it doesn't exist already."""
        raise ValueError("Sport1M dataset download is not supported.")

    def default_transform(self):
        """Set the default transform for dataset."""
        size = 256
        order = (3, 0, 1, 2)
        trans = [
            transforms.VideoShortEdgeResize(size=size, interpolation='linear'),
            transforms.VideoCenterCrop(size=(224, 224)),
            transforms.VideoRescale(shift=0),
            transforms.VideoReOrder(order=order),
            transforms.VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
        ]

        return trans


class ParseSport1M(ParseDataset):
    """
    Parse sport_1m dataset.
    """
    def load_cls_file(self):
        """Parse the category file."""
        base_path = os.path.dirname(self.path)
        csv_file = os.path.join(base_path, f"sport-1m_train.csv")
        cls2id = {}
        id2cls = []
        cls_file = os.path.join(base_path, "cls2index.json")
        print(cls_file)
        if os.path.isfile(cls_file):
            with open(cls_file, "r")as f:
                cls2id = json.load(f)
            id2cls = [*cls2id]
            return id2cls, cls2id
        with open(csv_file, "r")as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                if row['label'] not in cls2id:
                    cls2id.setdefault(row['label'], len(cls2id))
                    id2cls.append(row['label'])
        f.close()
        os.mknod(cls_file)
        with open(cls_file, "w")as f:
            f.write(json.dumps(cls2id))
        return id2cls, cls2id

    def parse_dataset(self, *args):
        """Traverse the Sport1M dataset file to get the path and label."""
        parse_Sport1M = ParseSport1M(self.path)
        split = os.path.split(parse_Sport1M.path)[-1]
        video_label, video_path = [], []
        _, cls2id = self.load_cls_file()
        with open(os.path.join(os.path.dirname(parse_Sport1M.path),
                               f"sport-1m_{split}.csv"), "rt")as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                start = row['time_start'].zfill(6)
                end = row['time_end'].zfill(6)
                file_name = f"{row['youtube_id']}_{start}_{end}.mp4"
                video_path.append(os.path.join(self.path, file_name))
                video_label.append(cls2id[row['label']])
        return video_path, video_label
