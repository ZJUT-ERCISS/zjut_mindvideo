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
""" HMDB dataset. TODO: finish these dataset API. """

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.mindvideo.dataset.ucf101 import ParseUCF101
from mindvision.mindvideo.dataset import UCF101

__all__ = ["HMDB51", "ParseHMDB51"]


@ClassFactory.register(ModuleType.DATASET)
class HMDB51(UCF101):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test" or "infer". Default: None.
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
        >>> from mindvision.mindvideo.dataset.hmdb51 import HMDB51
        >>> dataset = HMDB51("./data/")
        >>> dataset = dataset.run()

     About HMDB51 dataset:

        The HMDB51 dataset consists of 6766 videos in 51 classes.

        Here is the original HMDB51 dataset structure.
        You can unzip the dataset files into the following directory structure and read them by MindSpore Vision's API.

        .
        |-hmdb-51                                // contains 51 file folder
            |-- brush_hair                        // contains 107 videos
            |   |-- April_09_brush_hair_u_nm_np1_ba_goo_0.avi      // video file
            |   |-- April_09_brush_hair_u_nm_np1_ba_goo_1.avi      // video file
            |    ...
            |-- cartwheel                         // contains 107 image files
            |   |-- (Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi       // video file
            |   |-- Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8.avi   // video file
            |    ...
            ...

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
                 download=False
                 ):
        super(HMDB51, self).__init__(path=path,
                                     split=split,
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


class ParseHMDB51(ParseUCF101):
    """
    Parse HMDB51 dataset and generated the UCF dataset look l
    """
    urlpath = "https://gitee.com/link?target=https%3A%2F%2Fserre-lab.clps.brown.edu%2Fresource%2Fhmdb-a-large-human-motion-database%2F%23Downloads"
