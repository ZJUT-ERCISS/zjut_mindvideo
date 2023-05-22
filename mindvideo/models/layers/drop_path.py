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
"""Drop Path block."""
from mindspore import nn
from mindspore import ops

__all__ = ['ProbDropPath3D']


class ProbDropPath3D(nn.Cell):
    """
    Drop path per sample using a fixed probability.
    Use keep_prob param as the probability for keeping network units.

    Args:
        keep_prob (int): Network unit keeping probability.
        ndim (int): Number of dropout features' dimension.

    Inputs:
        Tensor of ndim dimension.

    Outputs:
        A path-dropped tensor.
    """

    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.rand = ops.UniformReal(seed=0)
        self.floor = ops.Floor()
        self.shape = ops.Shape()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            random_tensor = self.rand((x_shape[0], 1, 1, 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor

        return x
