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
""" ACTIVITYNET dataset. TODO: finish these dataset API. """

import json
import os
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["Activitynet", "ParseActivitynet"]


@ClassFactory.register(ModuleType.DATASET)
class Activitynet(VideoDataset):
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
        >>> from mindvision.mindvideo.dataset.activitynet import Activitynet
        >>> dataset = Activitynet("./data", "train")
        >>> dataset = dataset.run()

    The activitynet structure of Activitynet dataset looks like:

        .
        |-activitynet
            |-- train_val
            |   |-- v_--mFXNrRZ5E.mp4      // video file
            |   |-- v_4WrU5OdkvY0.mp4      // video file
            |    ...
            |-- test
            |   |-- v_--tFD65KaK4.mp4       // video file
            |   |-- v_3j-CWo_hYBo.mp4   // video file
            |-- activity_net.v1-3.min.json  // annotation file
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
        load_data = ParseActivitynet(os.path.join(path, split)).parse_dataset
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
        _, _, mapping = ParseActivitynet(os.path.join(self.path, self.split)).load_json()
        return mapping

    def download_dataset(self):
        """Download the Activitynet dataset if it doesn't exist already."""
        raise ValueError("Activitynet dataset download is not supported.")


class ParseActivitynet(ParseDataset):
    """
    Parse Activitynet dataset.
    """

    def load_json(self):
        """Parse json file."""
        base_path = os.path.dirname(self.path)
        json_file = os.path.join(base_path, "activity_net.v1-3.min.json")
        cls2index = {}
        index2cls = []
        with open(json_file, "rb")as f:
            row_data = json.load(f)
            database = row_data["database"]
            for row in database:
                d = database[row]
                annotation_list = d['annotations']
                if not annotation_list:
                    continue
                for annotation in annotation_list:
                    label_name = annotation['label']
                    if not label_name in cls2index:
                        cls2index.setdefault(label_name, cls2index.__len__())
                        index2cls.append(label_name)
        return database, cls2index, index2cls

    def parse_dataset(self):
        """Traverse the Charades dataset file to get the path and label."""
        parse_charades = ParseActivitynet(self.path)
        video_label, video_path = [], []
        base_path = os.path.dirname(parse_charades.path)
        os.rename(os.path.join(base_path, "train_val"), os.path.join(base_path, "train"))
        database, cls2index, _ = self.load_json()
        video_list = os.listdir(parse_charades.path)
        for video in video_list:
            video_id = video.split('.')[0]
            info = database.get(video_id)
            video_len = info["duration"]
            annotation_list = info['annotations']
            if not annotation_list:
                continue
            for annotation in annotation_list:
                label = []
                segment = annotation['segment']
                start = segment[0]
                end = segment[1]
                label_name = annotation['label']
                label.append([cls2index.get(label_name), start / video_len, end / video_len])
                video_path.append(os.path.join(parse_charades.path, video))
                video_label.append(label)
        return video_path, video_label
