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
""" ARN training script. """

import numpy as np
from mindspore.nn.metrics.metric import rearrange_inputs
from mindspore.nn.metrics import Accuracy


class TaskAccuracy(Accuracy):
    r"""
    Calculates the accuracy for classification and multilabel data.

    The accuracy class has two local variables, the correct number and the total number of samples, that are used to
    compute the frequency with which `y_pred` matches `y`. This frequency is ultimately returned as the accuracy: an
    idempotent operation that simply divides the correct number by the total number.

    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): The metric to calculate the accuracy over a dataset. Supports 'classification' and
          'multilabel'. 'classification' means the dataset label is single. 'multilabel' means the dataset has multiple
          labels. Default: 'classification'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import nn, Tensor
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> y = Tensor(np.array([[1, 0], [1, 0], [0, 1]]), mindspore.float32)
        >>> metric = nn.Accuracy('one_hot')
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> accuracy = metric.eval()
    """

    def __init__(self, label_format='one_hot'):
        super(TaskAccuracy, self).__init__()
        self.label_format = label_format
        self.clear()

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the local variables. For 'classification', if the index of the maximum of the predict value
        matches the label, the predict result is correct. For 'multilabel', the predict value match the label,
        the predict result is correct.

        Args:
            inputs: Logits and labels. `y_pred` stands for logits, `y` stands for labels. `y_pred` and `y`
            must be a `Tensor`, a list or an array.
            For the 'one_hot' evaluation type, `y_pred` is a list of floating numbers in range :math:`[0, 1]`
            and the shape is :math:`(1, N, C)` in most cases (not strictly), where :math:`N` is the number of
            cases and :math:`C` is the number of categories. `y` must be in one-hot format that shape
            is :math:`(1, N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.
            For 'single' evaluation type, `y` is not one-hot format :match:'(1, N)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError(
                f"For TaskAccuracy.update, it needs 2 inputs (predicted value, true value), but got {len(inputs)}")
        y_pred = self._convert_data(inputs[0])
        y_pred = y_pred.argmax(axis=2)
        y = self._convert_data(inputs[1])

        if self.label_format == 'one_hot':
            y = y.argmax(axis=2)
        result = (np.equal(y_pred, y) * 1).reshape(-1)
        self._correct_num += result.sum()
        self._total_num += result.shape[0]
