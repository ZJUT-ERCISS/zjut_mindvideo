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
"""Optical Flow implementation. TODO: the optical flow should replace to outside utils."""

import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

from mindvideo.utils.check_param import Validator


def cal_for_frames(video_path):
    """Calculate optical flow using a list of frames."""
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for frame_curr in tqdm(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_tvl1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def cal_for_video(video_path):
    """Calculate optical flow of a video."""
    flow = []
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, im = cap.read()
        if ret:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            frames.append(im)
        else:
            break

    prev = frames[0]
    for frame_curr in tqdm(frames[1:]):
        curr = frame_curr
        tmp_flow = compute_tvl1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_tvl1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow."""
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(prev, curr, None)

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    return np.clip(flow, 0, 255)


def save_flow(video_flows, flow_path, save_format='jpg'):
    """Save video flows in specified format.

    TODO: explain the meaning of flow path. what is u an v?
    """
    support_format = ['npy', 'jpg']
    Validator.check_string(save_format, support_format)
    if not os.path.exists(flow_path):
        os.mkdir(flow_path)
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    if save_format == 'npy':
        for i, flow in enumerate(video_flows):
            np.save(os.path.join(flow_path, 'u', "{:06d}.{}".format(
                i, save_format)), flow[:, :, 0])
            np.save(os.path.join(flow_path, 'v', "{:06d}.{}".format(
                i, save_format)), flow[:, :, 1])
    elif save_format == 'jpg':
        for i, flow in enumerate(video_flows):
            cv2.imwrite(os.path.join(flow_path, 'u',
                                     "{:06d}.{}".format(i, save_format)), flow[:, :, 0])
            cv2.imwrite(os.path.join(flow_path, 'v',
                                     "{:06d}.{}".format(i, save_format)), flow[:, :, 1])


def extract_flow(video_path, flow_path, save_format='jpg'):
    """Extract flow from video frames.

    Args:
        video_path (str): The path of video. If `video_path` is a file directory,
            the function will extract optical flow from jpeg images in the directory.
            Else if `video_path` is a video, then extract optical flow frame by frame.
        flow_path (str): The path where saves the optical flow.
        save_format (str): Optical flow save format, can be 'npy' or 'jpg'. Default: 'jpg'.
    Returns:
        None

    Examples:
        >>> vpath = "./path_to_video"
        >>> save_path = "./path_to_saved_flow"
        >>> extract_flow(vpath, save_path)
    """
    if os.path.isdir(video_path):
        flow = cal_for_frames(video_path)
    else:
        flow = cal_for_video(video_path)
    save_flow(flow, flow_path, save_format)
