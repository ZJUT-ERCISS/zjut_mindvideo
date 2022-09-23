import os
import numpy as np
from PIL import Image
from pathlib import Path
import mindspore as ms
from mindspore import ops, Tensor
from pycocotools.ytvos import YTVOS
from mindvision.dataset.meta import ParseDataset
from mindvision.msvideo.dataset.transforms import ytvos_transform
from mindvision.msvideo.dataset.video_dataset import VideoDataset
from mindvision.dataset.meta import Dataset
from mindvision.msvideo.dataset.transforms.cocopolytomask import ConvertCocoPolysToMask
from mindspore.dataset.vision import c_transforms as T
from mindspore.dataset.vision import py_transforms as T_p
import mindspore.dataset.transforms.py_transforms as trans


class YTVOSDataset:
    def __init__(self, img_folder, ann_file, num_frames: 36):
        self.img_folder = img_folder
        self.ann_file = ann_file
        # self._transforms = transforms
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
        img = Image.open(img_path_list[0]).convert('RGB')
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'vid_len': vid_len,
                  'frame_id': frame_id, 'annotations': target}
        # target = self.prepare(img, target, inds, self.num_frames)
        return img_path_list, target


class ytvos(Dataset):
    def __init__(self,
                 path,
                 split=None,
                 transform=None,
                 target_transform=None,
                 seq=36,
                 seq_mode=None,
                 batch_size=1,
                 repeat_num=1,
                 shuffle=None,
                 columns_list=['video', 'labels', 'boxes', 'valid', 'masks'],
                 num_parallel_workers=1,
                 num_shards=None
                 ):
        parseytvos = ParseYtvos(path)
        parseytvos.init(image_set=split, num_frames=seq)
        parseytvos.build()
        load_data = parseytvos.parse_dataset
        super().__init__(path=path,
                         split=split,
                         load_data=load_data,
                         batch_size=batch_size,
                         repeat_num=repeat_num,
                         shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=None,
                         resize=300,
                         transform=transform,
                         target_transform=target_transform,
                         mode=seq_mode,
                         columns_list=columns_list,
                         schema_json=None,
                         trans_record=None)

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
        trans = [default_trans()]
        return trans

    def pipelines(self):
        trans = self.default_transform()
        self.dataset = self.dataset.map(operations=trans,
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
    Parse kinetic-400 dataset.
    """

    def init(self, image_set, num_frames):
        self.image_set = image_set
        self.num_frames = num_frames
        self.prepare = ConvertCocoPolysToMask(True)
        self.inds = list(range(self.num_frames))

    def build(self):
        # root = Path(ytvos_path)
        # assert root.exists(), f'provided YTVOS path {root} does not exist'
        root = Path(self.path)
        PATHS = {
            "train": (root / "train/JPEGImages", root / "annotations" / 'instances_train_sub.json'),
            "val": (root / "valid/JPEGImages", root / "annotations" / 'instances_val_sub.json'),
        }
        img_folder, ann_file = PATHS[self.image_set]
        dataset = YTVOSDataset(img_folder, ann_file, num_frames=self.num_frames)
        self.data = dataset

    def parse_dataset(self, *args):
        if not args:
            path_list = []
            labels = []
            num = len(self.data.img_ids)
            for i in range(num):
                path_list.append(i)
            return path_list, labels

        path, target = self.data[args[0]]
        img = Image.open(path[0]).convert('RGB')
        vid_len = target['vid_len']
        inds = [i % vid_len for i in self.inds][::-1]
        target = self.prepare(img, target, inds, self.num_frames)
        return path, target["labels"], target["boxes"], target["valid"], target["masks"]


class default_trans(trans.PyTensorOperation):
    def __init__(self):
        self.cast = ops.Cast()
        self.transform = T_p.ToTensor()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.trans = ytvos_transform.make_coco_transforms()

    def __call__(self, path, label, box, valid, mask):
        video = []
        for im in path:
            im_path = bytes.decode(im.tostring())
            im = Image.open(im_path).convert('RGB')
            # width = int((im.size[0]*300) / im.size[1])
            # height = 300
            # im = im.resize((width, height), resample=Image.Resampling.BILINEAR)
            # im = np.array(im, dtype=np.float32)
            # im = im.transpose((2, 0, 1))
            # im = (im - self.mean[:, None, None]) / self.std[:, None, None]
            # im = np.expand_dims(im, axis=0)
            video.append(im)
        video, box, mask = self.trans(video, box, mask)
        labels = np.array(label, dtype=np.int32)
        boxes = np.array(box, dtype=np.float32)
        valids = np.array(valid, dtype=np.int32)
        masks = np.array(mask, dtype=np.float32)
        return video, labels, boxes, valids, masks
