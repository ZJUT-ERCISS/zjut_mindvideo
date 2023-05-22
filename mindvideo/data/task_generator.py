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
# ==============================================================================
"""The module is used to generate few shot learning tasks."""

import random
import imageio.v3 as iio


class TaskGenerator:
    """
    N-way K-Shot Tasks generator for getting video path and its corresponding label.
    There are N categories in each task, including K labeled samples in each category.
    """

    def __init__(self, path, cls, n, k, q):
        """
        Init Task Generator.

        Args:
            path(str): video file path.
            cls(list): the ending index of the video for each category, the index is start from 1.
            n(int): the number of categories per task.
            k(int): the number of label samples in each category
            q(int): the number of unlabeled samples in each category.
        """

        self.cls = cls
        self.path = path
        self.n = n
        self.k = k
        self.q = q
        self.task_list = [i for i in range(len(self.cls))]
        random.shuffle(self.task_list)

    def __getitem__(self, item):
        """Get the video and label of the task for each item."""
        file_list = []
        data = []
        for c in self.task_list[self.n * item:self.n * (item + 1)]:
            if c == 0:
                files = random.sample(
                    self.path[0:self.cls[c]], self.k + self.q)
            else:
                files = random.sample(
                    self.path[self.cls[c - 1]:self.cls[c]], self.k + self.q)
            file_list.append(files)
        for f in file_list:
            with open(f, 'rb')as rf:
                content = rf.read()
                video = iio.imread(content, index=None, format_hint=".avi")
                sub_video = video[0:16]
                data.append(sub_video)
        return data

    def __len__(self):
        """Get the the size of tasks."""
        return len(self.TaskList) / self.N
