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
"""VisTR infer"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import cv2
import torch
import json
import math
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from mindvideo.utils import misc
import mindspore
from mindspore import nn
from mindspore.dataset.vision import py_transforms as T_p
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, ops
from mindvideo.utils.config import parse_args, Config
from mindvideo.models import build_model

palette_data1 = [0, 0, 0, 0, 0, 255]
palette_data2 = [0, 0, 0, 255, 0, 0]
palette_data3 = [0, 0, 0, 255, 0, 255]
palette_data4 = [0, 0, 0, 255, 255, 255]
palette_data5 = [0, 0, 0, 0, 255, 255, 0]
palette_data6 = [0, 0, 0, 0, 255, 0]
palette_data7 = [0, 0, 0, 0, 255, 255]
palette_data8 = [0, 0, 0, 0, 0, 100]
palette_data9 = [0, 0, 0, 0, 100, 255]
palette_data10 = [0, 0, 0, 100, 100, 255]

palette_data = [palette_data1, palette_data2, palette_data3, palette_data4, palette_data5,
                palette_data6, palette_data7, palette_data8, palette_data9, palette_data10]

def main(pargs):
    """
    vistr resnet50 infer
    """
    config = Config(pargs.config)
    context.set_context(**config.context)

    cast = ops.Cast()

    transform = T_p.ToTensor()

    mean = np.array(config.infer.mean)
    std = np.array(config.infer.std)

    num_frames = config.infer.num_frames
    num_ins = config.infer.num_ins

    ann_path = os.path.join(config.data_loader.train.dataset.path, "annotations/val.json")
    folder = os.path.join(config.data_loader.train.dataset.path, "train/JPEGImages")
    videos = json.load(open(ann_path, 'rb'))['videos']

    # set network
    network = build_model(config.model)
    param_dict = load_checkpoint(config.infer.pretrained_model)
    load_param_into_net(network, param_dict)
    weight = np.load(config.infer.weights)
    weight = mindspore.Tensor(weight, mindspore.float32)
    network.mask_head.dcn.conv_weight = weight

    vis_num = len(videos)
    if config.infer.video_num != 'all':
        vis_num = config.infer.video_num
    result = []

    ms_sigmoid = ops.Sigmoid()
    concat = ops.Concat(axis=0)
    softmax = nn.Softmax(axis=-1)
    expand_dims = ops.ExpandDims()

    result = []
    for i in range(vis_num):
        print("Process video: ", i)
        id_ = videos[i]['id']
        length = videos[i]['length']
        file_names = videos[i]['file_names']

        v_name, _ = os.path.split(file_names[0])
        path = os.path.join("./output", v_name)
        if not os.path.exists(path):
            os.makedirs(path)

        img_set = []
        if length < num_frames:
            clip_names = file_names*(math.ceil(num_frames/length))
            clip_names = clip_names[:num_frames]
        else:
            clip_names = file_names[:num_frames]
        if clip_names == []:
            continue
        if len(clip_names) < num_frames:
            clip_names.extend(file_names[:num_frames-len(clip_names)])
        path_im = []
        for k in range(num_frames):
            im = Image.open(os.path.join(folder, clip_names[k]))
            _, im_name = os.path.split(clip_names[k])
            path_im.append(os.path.join(path, im_name))
            h = im.size[1]
            w = im.size[0]
            width = int((im.size[0]*300) / im.size[1])
            height = 300
            im = im.resize((width, height), resample=Image.Resampling.BILINEAR)
            im = transform(im)
            im = (im - mean[:, None, None]) / std[:, None, None]
            im = Tensor(im, mindspore.float32)
            im = expand_dims(im, 0)
            img_set.append(im)
        img = concat(img_set)
        images = Tensor(img, mindspore.float32)
        images = images.expand_dims(axis=0)
        if images.shape[-1] <= 700:
            pred, pred_mask = network(images)

            pred_logits = pred[-1, ..., :42]
            pred_boxes = pred[-1, ..., 42:]
            pred_logits = softmax(pred_logits)[0, :, :-1]
            pred_boxes = pred_boxes[0]
            pred_masks = pred_mask[0]

            pred_masks = pred_masks.reshape(36, 10, pred_masks.shape[-2], pred_masks.shape[-1])
            resize_bilinear = ops.ResizeBilinear((h, w))
            pred_masks = resize_bilinear(pred_masks)
            pred_masks = ms_sigmoid(pred_masks).asnumpy() > 0.5

            pred_masks = pred_masks[:length]
            pred_logits = pred_logits.reshape(num_frames, num_ins, pred_logits.shape[-1]).asnumpy()
            pred_logits = pred_logits[:length]
            pred_scores = np.max(pred_logits, axis=-1)
            pred_logits = np.argmax(pred_logits, axis=-1)
            for m in range(num_ins):
                segmentation = []
                if pred_masks[:, m].max() == 0:
                    continue
                out_score = pred_scores[:, m].mean()
                category_id = np.argmax(np.bincount(pred_logits[:, m]))
                out_instance = {'video_id': id_, 'score': float(out_score), 'category_id': int(category_id)}
                for n in range(length):
                    if pred_scores[n, m] < 0.001:
                        segmentation.append(None)
                    else:
                        if os.path.exists(path_im[n]):
                            img = cv2.imread(path_im[n])
                        else:
                            img = cv2.imread(os.path.join(folder, clip_names[n]))
                        mask = (pred_masks[n, m]).astype('uint8')
                        get_mask_out(img, mask, path_im[n], m)
                        rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
                        rle["counts"] = rle["counts"].decode("utf-8")
                        segmentation.append(rle)
                out_instance['segmentations'] = segmentation
                result.append(out_instance)
    # out_file = open(config.infer.save_path, 'w', encoding='utf-8')
    # json.dump(result, out_file)
    # out_file.close()
    with open(config.infer.save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)

def get_mask_out(img, mask, path_im, m):
    mask_p = Image.fromarray(mask, "P")

    mask_p.putpalette(palette_data[m])
    mask_p.save("./output/1.png")
    mask_p = cv2.imread("./output/1.png")
    masked_img = cv2.addWeighted(img, 0.5, mask_p, 0.5, 0)
    cv2.imwrite(path_im, masked_img)


if __name__ == "__main__":
    args = parse_args()
    main(args)
