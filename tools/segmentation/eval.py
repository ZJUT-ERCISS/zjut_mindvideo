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
"""VisTR eval"""
import torch
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import json
import math
import pylab
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
from mindvideo.data.pycocotools.ytvos import YTVOS
from mindvideo.data.pycocotools.ytvoseval import YTVOSeval

def main(pargs):
    """
    vistr resnet50 eval
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
    folder = os.path.join(config.data_loader.train.dataset.path, "train/JPEGImages/")
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
        for k in range(num_frames):
            im = Image.open(os.path.join(folder, clip_names[k]))
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
                        mask = (pred_masks[n, m]).astype('uint8')
                        out_mask = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
                        out_mask["counts"] = out_mask["counts"].decode("utf-8")
                        segmentation.append(out_mask)
                out_instance['segmentations'] = segmentation
                result.append(out_instance)
    # out_file = open(config.infer.save_path, 'w', encoding='utf-8')
    # json.dump(result, out_file)
    # out_file.close()
    with open(config.infer.save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)

    evaluate(config)

def evaluate(cfg):
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    annType = ['segm','bbox','keypoints']
    annType = annType[0]      #specify type here

    annFile = os.path.join(cfg.data_loader.train.dataset.path, "annotations/val.json")
    cocoGt=YTVOS(annFile)
    #initialize COCO detections api
    resFile='./result.json'

    cocoDt=cocoGt.loadRes(resFile)
    imgIds=sorted(cocoGt.getVidIds())
    imgIds=imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]
    # running evaluation
    cocoEval = YTVOSeval(cocoGt,cocoDt,annType)
    cocoEval.params.vidIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    args = parse_args()
    main(args)
