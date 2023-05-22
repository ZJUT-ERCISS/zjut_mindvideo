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
""" MSVD dataset. TODO: finish these dataset API. """

import os
from mindvision.dataset.meta import ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.video_dataset import VideoDataset

__all__ = ["MSVD", "ParseMSVD"]


@ClassFactory.register(ModuleType.DATASET)
class MSVD(VideoDataset):
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
        >>> from mindvision.mindvideo.dataset.msvd import MSVD
        >>> dataset = MSVD("./data")
        >>> dataset = dataset.run()

    The msvd structure of MSVD dataset looks like:

        .
        |-msvd
            |-- YouTubeClips
            |   |-- -4wsuPCjDBc_5_15.avi       // video file
            |   |-- -7KMZQEsJW4_205_208.avi    // video file
            |   |-- -8y1Q0rA3n8_108_115.avi    //video file
            |    ...
            |-- AllVideoDescriptions.txt       //annotation file
            ...
    """

    def __init__(self,
                 path,
                 split="",
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
        load_data = ParseMSVD(os.path.join(path, split)).parse_dataset
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
        parse_msvd = ParseMSVD(self.path)
        mapping, _, _ = parse_msvd.load_label()
        return mapping

    def download_dataset(self):
        """Download the MSVD data if it doesn't exist already."""
        raise ValueError("MSVD dataset download is not supported.")


class ParseMSVD(ParseDataset):
    """
    Parse MSVD dataset.
    """

    def load_label(self):
        """Parse annotation file."""
        label_file = os.path.join(self.path, "AllVideoDescriptions.txt")
        label2index = {}
        index2label = []
        id2cls = {}
        with open(label_file, "rb")as f:
            row_data = f.readlines()
            info_list = row_data[8:]
            for info in info_list:
                info = str(info)
                video_id = info.split(' ', maxsplit=1)[0]
                cls = info[len(video_id) + 1:]
                if cls not in label2index:
                    label2index.setdefault(cls, len(label2index))
                    index2label.append(cls)
                id2cls.setdefault(id, label2index.get(cls))

        return index2label, label2index, id2cls

    def parse_dataset(self):
        """Traverse the MSVD dataset file to get the path and label."""
        parse_msvd = ParseMSVD(self.path)
        video_label, video_path = [], []
        video_file_path = os.path.join(parse_msvd.path, "YouTubeClips")
        video_list = os.listdir(video_file_path)
        _, _, id2cls = self.load_label()
        for video in video_list:
            video_id = video.split('.')[0]
            label = id2cls[video_id]
            video_path.append(os.path.join(video_file_path, video))
            video_label.append(label)
        return video_path, video_label
