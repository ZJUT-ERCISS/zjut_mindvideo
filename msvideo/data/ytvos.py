import os
from pathlib import Path
from mindspore import ops
from pycocotools.ytvos import YTVOS
from mindvision.dataset.meta import ParseDataset
from mindvision.msvideo.dataset.transforms import ytvos_transform
from mindvision.msvideo.dataset.video_dataset import VideoDataset
from mindvision.msvideo.dataset.transforms.ytvos_transform import default_trans


class YTVOSDataset:
    def __init__(self, img_folder, ann_file, return_masks: True, num_frames: 36):
        self.img_folder = img_folder
        self.ann_file = ann_file
        # self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        # self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        self.concat = ops.Concat(axis=0)
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid,  frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        # img = []
        img_path_list = []
        vid_len = len(self.vid_infos[vid]['file_names'])
        inds = list(range(self.num_frames))
        inds = [i % vid_len for i in inds][::-1]
        # if random
        # random.shuffle(inds)
        for j in range(self.num_frames):
            img_path = os.path.join(
                str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
            # img.append(Image.open(img_path).convert('RGB'))
            # 放入图片路径
            img_path_list.append(img_path)
        # img = Image.open(img_path_list[0]).convert('RGB')
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'vid_len': vid_len,
                  'frame_id': frame_id, 'annotations': target}
        # target = self.prepare(img, target, inds, self.num_frames)
        return img_path_list, target


class ytvos(VideoDataset):
    """
    Args:
        path = "/usr/dataset/VOS/"
        split = "train"

    The directory structure of Kinetic-400 dataset looks like:

        .
        |-VOS
            |-- train
            |       |-- JPEGImages    
            |                    |-- 00a23ccf53 
            |                    |            |-- 00000.jpg
            |                    |            |-- 00001.jpg
            |                    |-- 00ad5016a4
            |-- test
            |       |-- JPEGImages    
            |                    |-- 00a23ccf53 
            |                    |            |-- 00000.jpg
            |                    |            |-- 00001.jpg
            |                    |-- 00ad5016a4
            |-- val
            |       |-- JPEGImages    
            |                    |-- 00a23ccf53 
            |                    |            |-- 00000.jpg
            |                    |            |-- 00001.jpg
            |                    |-- 00ad5016a4
            |-- annotations
            |       |-- instances_train_sub.json 
            |       |-- instances_val_sub.json
    """

    def __init__(self,
                 path,
                 split=None,
                 transform=default_trans(),
                 target_transform=None,
                 seq=36,
                 seq_mode=None,
                 align=False,
                 batch_size=1,
                 repeat_num=1,
                 shuffle=None,
                 suffix='ytvos',
                 columns_list=['video', 'labels'],
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False
                 ):
        parseytvos = ParseYtvos(path)
        parseytvos.init(image_set=split, num_frames=seq)
        load_data = parseytvos.parse_dataset
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
                         suffix=suffix,
                         columns_list=columns_list,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         download=download)

    @property
    def index2label(self):
        mapping = []
        img_folder = "/data0/VOS/train/JPEGImages"
        ann_file = "/data0/VOS/annotations/instances_train_sub.json"
        dataset = YTVOSDataset(img_folder, ann_file,
                               return_masks=True, num_frames=36)
        num = len(dataset.img_ids)
        for i in range(num):
            _, target = dataset[i]
            mapping.append(target)
        return mapping

    def download_dataset(self):
        """Download the HMDB51 data if it doesn't exist already."""
        raise ValueError("HMDB51 dataset download is not supported.")

    def default_transform(self):
        return ytvos_transform.default_trans()

    def pipelines(self):
        trans = self.default_transform()
        self.dataset = self.dataset.map(operations=[trans],
                                        input_columns=[self.columns_list[0]],
                                        num_parallel_workers=self.num_parallel_workers)

    def run(self):
        """dataset pipeline"""
        # self.pipelines()
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.repeat(self.repeat_num)
        return self.dataset


class ParseYtvos(ParseDataset):
    """
    Parse kinetic-400 dataset.
    """

    def init(self, image_set, num_frames):
        self.image_set = image_set
        self.num_frames = num_frames

    def build(self):
        # root = Path(ytvos_path)
        # assert root.exists(), f'provided YTVOS path {root} does not exist'
        root = Path(self.path)
        PATHS = {
            "train": (root / "train/JPEGImages", root / "annotations" / 'instances_train_sub.json'),
            "val": (root / "valid/JPEGImages", root / "annotations" / 'instances_val_sub.json'),
        }
        masks = True
        img_folder, ann_file = PATHS[self.image_set]
        dataset = YTVOSDataset(img_folder, ann_file,
                               return_masks=masks, num_frames=self.num_frames)
        return dataset

    def parse_dataset(self):
        path_list = []
        target_lsit = []
        dataset = self.build()
        num = len(dataset.img_ids)
        for i in range(num):
            path, target = dataset[i]
            path_list.append(path)
            target_lsit.append(target)
        return path_list, target_lsit