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
"""Hungarian algorithm."""
import mindspore as ms
from mindspore import Tensor
from mindspore import nn, ops


__all__ = ['Hungarian']


class Hungarian(nn.Cell):
    """Given a cost matrix, calculate the best assignment that cost the least.
    This ops now only support square matrix.

    Args:
        dim (int): The size of the input square matrix.

    Inputs:
        x(Tensor): The input cost matrix.

    Returns:
        Tensor[bool]: The best assignment, there can be multiple solutions.
        Tensor[int32]: The indices of row assignment.
        Tensor[int32]: The indices of column assignment.

    Examples:
    >>> cost_matrix = Tensor([[7, 5, 9, 8, 11],
                              [9, 12, 7, 11, 9],
                              [8, 5, 4, 6, 9],
                              [7, 3, 6, 9, 6],
                              [4, 6, 7, 5, 11]], dtype=ms.float32)
    >>> hung = Hungarian(dim=5)
    >>> print(hung(cost_matrix))

    """

    def __init__(self, dim):
        super().__init__()
        self.n = dim
        self.on_value, self.off_value = Tensor(1.0, ms.float32), Tensor(0.0, ms.float32)
        self.row_ind = Tensor(list(range(self.n)), dtype=ms.int32)
        self.tensor_zero = self.off_value
        self.inf = float("inf")
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()
        self.zeroslike = ops.ZerosLike()
        self.reduce_min = ops.ReduceMin(keep_dims=True)
        self.tile = ops.Tile()
        self.cast = ops.Cast()
        self.equal = ops.Equal()
        self.reduce_sum = ops.ReduceSum()
        self.trans = ops.Transpose()
        self.topk = ops.TopK()
        self.scalar2tensor = ops.ScalarToTensor()
        self.expand_dim = ops.ExpandDims()
        self.onehot = ops.OneHot()
        self.squeeze = ops.Squeeze()
        self.logical_or = ops.LogicalOr()
        self.logical_and = ops.LogicalAnd()
        self.reduce_any = ops.ReduceAny()
        self.reduce_all = ops.ReduceAll()
        self.reduce_max = ops.ReduceMax(keep_dims=True)
        self.concat0 = ops.Concat(0)
        self.concat1 = ops.Concat(1)
        self.slice = ops.Slice()

    def create_onehot(self, idx):
        """Calculate one hot vector according to input indice."""
        return self.onehot(idx, self.n, self.on_value, self.off_value)

    def get_assign(self, assign_matrix):
        """Make every row of assign matrix has at most one assignment."""
        temp_matrix = assign_matrix.copy()
        for _ in range(self.n):
            row_cnt = self.reduce_sum(temp_matrix, 1)
            zero_pos = self.equal(row_cnt, 0.0)
            row_cnt = row_cnt.masked_fill(zero_pos, self.inf)
            minn = self.reduce_min(row_cnt)
            row_idx = -1
            for idx, cnt in enumerate(row_cnt):
                if cnt == minn and cnt != self.inf:
                    row_idx = idx
                    break
            if row_idx == -1:
                break
            col_idx = self.squeeze(self.topk(temp_matrix[row_idx], 1)[1])
            row_idx = self.scalar2tensor(row_idx, ms.int32)
            # assign_matrix
            assign_matrix[row_idx] = self.create_onehot(col_idx)
            assign_matrix = self.trans(assign_matrix, (1, 0))
            assign_matrix[col_idx] = self.create_onehot(row_idx)
            assign_matrix = self.trans(assign_matrix, (1, 0))
            # temp_matrix
            temp_matrix[row_idx] = self.zeroslike(temp_matrix[col_idx])
            temp_matrix = self.trans(temp_matrix, (1, 0))
            temp_matrix[col_idx] = self.zeroslike(temp_matrix[row_idx])
            temp_matrix = self.trans(temp_matrix, (1, 0))

        return assign_matrix

    def try_assign(self, x):
        """Try assignment, if succeed return the result."""
        zeros = self.zeroslike(x)
        assign_matrix = self.equal(x, zeros)
        assign_matrix = self.cast(assign_matrix, ms.float32)
        try_matrix = self.get_assign(assign_matrix)
        abandoned_matrix = assign_matrix - try_matrix
        try_matrix = self.cast(try_matrix, ms.bool_)
        abandoned_matrix = self.cast(abandoned_matrix, ms.bool_)
        if self.reduce_all(self.reduce_any(try_matrix, 1)):
            return x, try_matrix
        temp_x = x.copy()
        col_mask = self.cast(self.zeroslike(x), ms.bool_)
        row_mask = self.cast(self.zeroslike(x), ms.bool_)
        for _ in range(self.n):
            try_cnt = self.reduce_any(try_matrix.masked_fill(col_mask, 0.0), 1)
            abandon_cnt = self.reduce_any(abandoned_matrix, 1)
            flag = True
            for idx, cnt in enumerate(try_cnt):
                # 找到未指派过且有0元素的行
                if not cnt and abandon_cnt[idx]:
                    # 该行打勾
                    row_mask[idx] = self.ones((self.n,), ms.bool_)
                    # 此行中0元素对应列打勾
                    col_mask = self.logical_or(col_mask, self.tile(abandoned_matrix[idx], (self.n, 1)))
                    abandoned_matrix[idx] = self.zeros((self.n,), ms.bool_)
                    # 对打勾的列中所含的独立0元素的行打勾
                    cross_mask = self.logical_and(col_mask, try_matrix)
                    row_mask = self.logical_or(
                        row_mask,
                        self.tile(self.expand_dim(self.reduce_any(cross_mask, 1), 1), (1, self.n)))
                    flag = False
                    break
            if flag:
                break
        cover_row_mask = self.ones((self.n, self.n), ms.bool_).masked_fill(row_mask, 0.0)
        temp_x = temp_x.masked_fill(cover_row_mask, self.inf)
        temp_x = temp_x.masked_fill(col_mask, self.inf)
        minn = self.reduce_min(temp_x)
        x = x - self.cast(row_mask, ms.float32)*minn
        x = x + self.cast(col_mask, ms.float32)*minn
        return x, try_matrix

    def construct(self, x):
        """Hungarian construct."""
        r, c = x.shape
        mark = min(r, c)
        if r < self.n:
            padding = self.ones((self.n-r, c), ms.float32)*self.reduce_max(x)
            x = self.concat0((x, padding))
        if c < self.n:
            padding = self.ones((self.n, self.n-c), ms.float32)*self.reduce_max(x)
            x = self.concat1((x, padding))

        x = x - self.tile(self.reduce_min(x, 1), (1, self.n))
        x = self.trans(x, (1, 0))
        x = x - self.tile(self.reduce_min(x, 1), (1, self.n))
        x = self.trans(x, (1, 0))

        try_matrix = self.zeros((self.n, self.n), ms.bool_)
        while True:
            x, try_matrix = self.try_assign(x)
            if self.reduce_all(self.reduce_any(try_matrix, 1)):
                break
        best_assign = self.cast(try_matrix, ms.float32)
        row_ind = self.row_ind
        col_ind = self.squeeze(self.topk(best_assign, 1)[1])
        row_ind = self.slice(row_ind, (0,), (mark,))
        col_ind = self.slice(col_ind, (0,), (mark,))
        return try_matrix, row_ind, col_ind
