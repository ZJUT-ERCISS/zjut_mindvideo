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
""" YTvos dataset. """
import os
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Optional, Callable
from mindvideo.data.pycocotools.ytvos import YTVOS
from mindvideo.data.meta import Dataset
from mindvideo.data.meta import ParseDataset
from mindvideo.data.transforms import ytvos_transform
from mindvideo.data.transforms.cocopolytomask import ConvertCocoPolysToMask
from mindvideo.utils.class_factory import ClassFactory, ModuleType
from mindspore import ops
import mindspore.dataset.transforms.py_transforms as trans


__all__ = ["YTVOSDataset", "Ytvos"]


class YTVOSDataset:
    """
    Generate YTVOSDataset
    Args:
        img_folfer(string):path of image file
        ann_file(string):path of annotation file
        num_frames(int):number of frames in one video.Default: 36
    Expamle:
    >>> img_folfer = "./data/img"
    >>> ann_file = "./data/ann"
    >>> dataset = YTVosDataset(img_folder, ann_file)
    """

    def __init__(self, img_folder, ann_file, num_frames=36, data_len='all'):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.num_frames = num_frames
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        self.concat = ops.Concat(axis=0)
        #######
        # if data_len != 'all':
        #     self.vid_ids = self.vid_ids[:1]
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))
        if data_len != 'all':
            self.vid_ids = self.vid_ids[:1]
            self.vid_infos = self.vid_infos[:1]
            self.img_ids = self.img_ids[:1]


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid,  frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        img_path_list = []
        vid_len = len(self.vid_infos[vid]['file_names'])
        inds = list(range(self.num_frames))
        inds = [i % vid_len for i in inds][::-1]
        for j in range(self.num_frames):
            img_path = os.path.join(
                str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
            img_path_list.append(img_path)
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'vid_len': vid_len,
                  'frame_id': frame_id, 'annotations': target}
        return img_path_list, target


@ClassFactory.register(ModuleType.DATASET)
class Ytvos(Dataset):
    """
    Args:
        path (string): Root directory of the Mnist dataset or inference image.
        split (str): The dataset split supports "train", "test" or "infer". Default: None.
        transform (callable, optional): A function transform that takes in a video. Default:None.
        target_transform (callable, optional): A function transform that takes in a label.
                Default: None.
        seq(int): The number of frames of captured video. Default: 16.
        seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
        batch_size (int): Batch size of dataset. Default:32.
        repeat_num (int): The repeat num of dataset. Default:1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
        columns_list(list):show the type and number of dataset'output
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        num_shards (int, optional): Number of shards that the dataset will be divided into.
                Default: None.
    Example:
        >>> from mindvideo.data.ytvos import ytvos
        >>> path = "./dataset/vos"
        >>> split = 'train'
        >>> seq = 36
        >>> batch_size = 1
        >>> repeat_num = 1
        >>> shuffle = False
        >>> dataset = ytvos(path=path, split=split, seq=seq,
                            batch_sizez=batch_size, shuffle=shuffle)

    The directory structure of Kinetic-400 dataset looks like:

        .
        |-VOS
            |-- train
            |   |-- JPEGImages
            |       |-- 00a23ccf53
            |       ... |-- 00000.jpg
            |           |-- 00001.jpg
            |           |-- 00ad5016a4
            |           ...
            |-- test
            |   |-- JPEGImages
            |       |-- 00a23ccf53
            |       ... |-- 00000.jpg
            |           |-- 00001.jpg
            |           |-- 00ad5016a4
            |           ...
            |-- val
            |   |-- JPEGImages
            |       |-- 00a23ccf53
            |       ... |-- 00000.jpg
            |           |-- 00001.jpg
            |           |-- 00ad5016a4
            |           ...
            |-- annotations
            |   |-- instances_train_sub.json
            |   |-- instances_val_sub.json
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 seq: int = 36,
                 batch_size: int = 1,
                 repeat_num: int = 1,
                 shuffle: bool = False,
                 num_parallel_workers: int = 1,
                 shard_id: int = None,
                 num_shards: int = None,
                 data_len: str = 'all',
                 transforms: Optional[Callable] = None
                 ):
        parseytvos = ParseYtvos(path)
        self.transforms = transforms
        parseytvos.init(image_set=split, num_frames=seq)
        parseytvos.build(data_len=data_len)
        load_data = parseytvos.parse_dataset
        self.columns_list = ['video', 'labels', 'boxes', 'valid', 'masks', 'resize_shape']
        super().__init__(path=path,
                         split=split,
                         load_data=load_data,
                         batch_size=batch_size,
                         repeat_num=repeat_num,
                         shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         resize=300,
                         transform=None,
                         target_transform=None,
                         mode=None,
                         columns_list=self.columns_list,
                         schema_json=None,
                         trans_record=None)

    @property
    def index2label(self):
        mapping = []
        root = Path(self.path)
        img_folder = root / "train/JPEGImages"
        ann_file = root / "annotations" / 'instances_train_sub.json'
        dataset = YTVOSDataset(img_folder, ann_file, num_frames=36)
        num = len(dataset.img_ids)
        for i in range(num):
            _, target = dataset[i]
            mapping.append(target)
        return mapping

    def download_dataset(self):
        """Download the ytvos data if it doesn't exist already."""
        raise ValueError("ytvos dataset download is not supported.")

    def default_transform(self):
        default_trans = [DeFaultTrans()]
        return default_trans

    def pipelines(self):
        if self.transforms:
            pipelines_trans = self.transforms
        else:
            pipelines_trans = self.default_transform()
        self.dataset = self.dataset.map(operations=pipelines_trans,
                                        input_columns=self.columns_list,
                                        num_parallel_workers=self.num_parallel_workers)

    def run(self):
        """dataset pipeline"""
        self.pipelines()
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.repeat(self.repeat_num)
        return self.dataset


class ParseYtvos(ParseDataset):
    """
    Parse ytvos dataset.
    """

    def init(self, image_set, num_frames):
        self.image_set = image_set
        self.num_frames = num_frames
        self.prepare = ConvertCocoPolysToMask(True)
        self.inds = list(range(self.num_frames))

    def build(self, data_len):
        """
        build ytvos dataset
        """
        root = Path(self.path)
        path = {
            "train": (root / "train/JPEGImages", root / "annotations" / 'instances_train_sub.json'),
            "val": (root / "val/JPEGImages", root / "annotations" / 'instances_val_sub.json'),
        }
        img_folder, ann_file = path[self.image_set]
        dataset = YTVOSDataset(img_folder, ann_file, num_frames=self.num_frames, data_len=data_len)
        self.data = dataset

    def parse_dataset(self, *args):
        """
        Traverse ytvos dataset file to get the label ,boxes ,valid, masks and resize shape
        """
        if not args:
            path_list = []
            labels = []
            num = len(self.data.img_ids)
            # if data_len != "all":
            #     num = num - 61844
            for i in range(num):
                path_list.append(i)
            return path_list, labels
        resize_shape = []
        # if data_len == "all":
        path, target = self.data[(args[0])]
        # else:
        #     path, target = self.data[(args[0] + 61844)]
        img = Image.open(path[0]).convert('RGB')
        vid_len = target['vid_len']
        inds = [i % vid_len for i in self.inds][::-1]
        target = self.prepare(img, target, inds, self.num_frames)
        return path, target["labels"], target["boxes"], target["valid"], target["masks"], resize_shape
        


@ClassFactory.register(ModuleType.PIPELINE)
class DeFaultTrans(trans.PyTensorOperation):
    """
    ytvos default transform
    """

    def __init__(self):
        self.cast = ops.Cast()
        self.trans = ytvos_transform.make_coco_transforms()

    def __call__(self, path, label, box, valid, mask, resize_shape):
        video = []
        for im in path:
            im_path = bytes.decode(im.tobytes())
            im = Image.open(im_path).convert('RGB')
            video.append(im)
        video, box, mask, resize_shape, label, valid = self.trans(video, box, mask, resize_shape, label, valid)
        video = video.astype(np.float32)
        labels = label.astype(np.int32)
        boxes = box.astype(np.float32)
        valids = valid.astype(np.int32)
        masks = mask.astype(np.float32)
        resize_shape = resize_shape.astype(np.float32)
        return video, labels, boxes, valids, masks, resize_shape

