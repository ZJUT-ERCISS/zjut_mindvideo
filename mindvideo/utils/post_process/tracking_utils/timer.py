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
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""A simple timer."""
import time


class Timer:
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.start_time = 0.
        self.avg_time = 0.

        self.diff = 0.
        self.calls = 0
        self.duration = 0.

    def tic(self):
        """ using time.time instead of time.clock because time time.clock
            does not normalize for multithreading"""
        self.start_time = time.time()

    def toc(self, average=True):
        """toc"""
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg_time = self.total_time / self.calls
        if average:
            self.duration = self.avg_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        """clear"""
        self.total_time = 0.
        self.start_time = 0.
        self.avg_time = 0.

        self.diff = 0.
        self.calls = 0
        self.duration = 0.
