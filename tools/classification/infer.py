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
"""MindSpore Vision Video infer script."""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import numpy as np
import moviepy.editor as mpy
import decord
import cv2
import json

import mindspore as ms
from mindspore import context, ops, Tensor

from mindvideo.utils.config import parse_args, Config
from mindvideo.utils.load import load_model
from mindvideo.data.builder import build_dataset, build_transforms
from mindvideo.models import build_model


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.6
FONTCOLOR = (255, 255, 255)
BGBLUE = (0, 119, 182)
THICKNESS = 1
LINETYPE = 1


def infer_classification(pargs):
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)
    
    # perpare dataset
    transforms = build_transforms(config.data_loader.eval.map.operations)
    data_set = build_dataset(config.data_loader.eval.dataset)

    # set network and load pretrain model
    ckpt_path = config.infer.pretrained_model
    network = None
    if os.path.splitext(ckpt_path)[-1] == '.ckpt':
        network = build_model(config.model)
    
    network = load_model(ckpt_path, network)

    expand_dims = ops.ExpandDims()

    # Randomly generates a specified video
    vis_num = len(data_set.video_path)
    vid_idx = np.random.randint(vis_num)
    if config.model_name == 'arn':
        video_path = data_set.video_path[vid_idx]
        className = []
        img_set = []
        for v in video_path:
            # get label
            className.append(v.split('/')[-2])

            # get video
            video_reader = decord.VideoReader(v, num_threads=1)
            for k in range(16):
                im = video_reader[k].asnumpy()
                img_set.append(im)
        video = np.stack(img_set, axis=0)
        for t in transforms:
            video = t(video)
        video = Tensor(video, ms.float32)
        video = expand_dims(video, 0)
        # Begin to eval.
        result = network(video)
        result = result.asnumpy()
        video_label = []
        for i in range(5):
            print("This is {}-th category".format(result[0][i].argmax()))
            video_label.append(className[int(result[0][i].argmax())])
    else:
        video_path = data_set.video_path[vid_idx]
        if isinstance(video_path, list):
            video_path = video_path[np.random.randint(len(video_path))]
        video_reader = decord.VideoReader(video_path, num_threads=1)
        img_set = []

        if config.model_name in ['Swin3D-T', 'Swin3D-S', 'Swin3D-B']:
            for k in range(32):
                im = video_reader[k].asnumpy()
                img_set.append(im)
        else:
            for k in range(16):
                im = video_reader[k].asnumpy()
                img_set.append(im)
        video = np.stack(img_set, axis=0)
        for t in transforms:
            video = t(video)
        video = Tensor(video, ms.float32)
        video = expand_dims(video, 0)
        # Begin to eval.
        result = network(video)
        result = result.asnumpy()
        print("This is {}-th category".format(result.argmax()))
    if config.data_loader.eval.dataset.type == 'Kinetic400':
        cls_file = os.path.join(data_set.path, "cls2index.json")
        with open(cls_file, "r")as f:
            cls2id = json.load(f)
        className = {v:k for k, v in cls2id.items()}
        video_label = className[int(result.argmax())]
    elif config.model_name == 'c3d' and config.data_loader.eval.dataset.type == 'UCF101':
        cls_file = os.path.join(data_set.path, "ucfTrainTestlist", "classInd.txt")
        cls2id = {}
        with open(cls_file, "r")as f:
            rows = f.readlines()
            for row in rows:
                index = int(row.split(' ')[0])
                cls = row.split(' ')[-1].strip()
                cls2id.setdefault(cls, index - 1)
        className = {v:k for k, v in cls2id.items()}
        video_label = className[int(result.argmax())]

    return result, video_path, video_label

def add_label(frame, label, BGCOLOR=BGBLUE):
    threshold = 30

    def split_label(label):
        label = label.split()
        lines, cline = [], ''
        for word in label:
            if len(cline) + len(word) < threshold:
                cline = cline + ' ' + word
            else:
                lines.append(cline)
                cline = word
        if cline != '':
            lines += [cline]
        return lines
    
    if len(label) > 30:
        label = split_label(label)
    else:
        label = [label]
    label = ['Action: '] + label
    
    sizes = []
    for line in label:
        sizes.append(cv2.getTextSize(line, FONTFACE, FONTSCALE, THICKNESS)[0])
    box_width = max([x[0] for x in sizes]) + 10
    text_height = sizes[0][1]
    box_height = len(sizes) * (text_height + 6)
    
    cv2.rectangle(frame, (0, 0), (box_width, box_height), BGCOLOR, -1)
    for i, line in enumerate(label):
        location = (5, (text_height + 6) * i + text_height + 3)
        cv2.putText(frame, line, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    return frame

if __name__ == '__main__':
    args = parse_args()
    result, video_path, video_label = infer_classification(args)

    if isinstance(video_path, list):
        for j in range(-5, 0):
            label = video_label[j]
            video = decord.VideoReader(video_path[j])
            frames = [x.asnumpy() for x in video]
            vid_frames = []
            for i in range(1, 50):
                vis_frame = add_label(frames[i], label)
                vid_frames.append(vis_frame)

            vid = mpy.ImageSequenceClip(vid_frames, fps=24)
            vid.write_gif(f'./vis_result{j}.gif')
    else:
        video = decord.VideoReader(video_path)
        frames = [x.asnumpy() for x in video]
        vid_frames = []
        for i in range(1, 50):
            vis_frame = add_label(frames[i], video_label)
            vid_frames.append(vis_frame)

        vid = mpy.ImageSequenceClip(vid_frames, fps=24)
        vid.write_gif('./vis_result.gif')
