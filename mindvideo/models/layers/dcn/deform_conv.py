
from functools import lru_cache
import math
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn
from mindspore.common.initializer import initializer

import collections
from itertools import repeat

__all__ = ['TorchDeformConv']

####### utils ########


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

######################


def _output_size(input, weight, padding, dilation, stride):
    channels = weight.shape[0]
    output_size = (input.shape[0], channels)
    for d in range(len(input.shape) - 2):
        in_size = input.shape[d + 2]
        pad = padding[d]
        kernel = dilation[d] * (weight.shape[d + 2] - 1) + 1
        stride_ = stride[d]
        output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
    if not all(map(lambda s: s > 0, output_size)):
        raise ValueError(
            "convolution input is too small (output would be {})".format(
                "x".join(map(str, output_size))
            )
        )
    return output_size


@lru_cache(maxsize=128)
def _cal_im2col_step(input_size, default_size):
    """
    Calculate proper im2col step size, which should be divisible by input_size and not larger
    than prefer_size. Meanwhile the step size should be as large as possible to be more
    efficient. So we choose the largest one among all divisors of input_size which are smaller
    than prefer_size.
    :param input_size: input batch size .
    :param default_size: default preferred im2col step size.
    :return: the largest proper step size.
    """
    if input_size <= default_size:
        return input_size
    best_step = 1
    for step in range(2, min(int(math.sqrt(input_size)) + 1, default_size)):
        if input_size % step == 0:
            if input_size // step <= default_size:
                return input_size // step
            best_step = step

    return best_step


class TorchDeformConv(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        deformable_groups=1,
        has_bias=False,
        norm=None,
        activation=None,
    ):
        """
        docstring.
        """
        super(TorchDeformConv, self).__init__()

        assert not has_bias
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = 64

        self.norm = norm
        self.activation = activation

        # weight_value = initializer('he_uniform',
        #                            (out_channels, in_channels // self.groups, *self.kernel_size),
        #                            ms.float32)
        # self.weight = ms.Parameter(weight_value)
        self.conv_weight = ops.ones((out_channels, in_channels // self.groups, *self.kernel_size), ms.float32)

        self.bias = None

        # self.backward_op =
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()

        self.bufs_ = [self.zeros((1,), ms.float32),
                      self.zeros((1,), ms.float32)]

        self.forward_op = ops.Custom(
            "/home/hcx/zjut_mindvideo_v2/zjut_mindvideo-master/mindvideo/models/layers/dcn/ms_dcn.so:ms_deformable_conv_forward", out_shape=lambda x0, x1, x2, x3, x4, x5, x6,
            x7, x8, x9, x10, x11, x12, x13, x14, x15, x16: x3, out_dtype=lambda x0, x1, x2, x3, x4, x5, x6,
            x7, x8, x9, x10, x11, x12, x13, x14, x15, x16: x3, func_type="aot")
        self.bprop_input = ops.Custom(
            "/home/hcx/zjut_mindvideo_v2/zjut_mindvideo-master/mindvideo/models/layers/dcn/ms_dcn_bprop.so:msDCN_backward_input", out_shape=lambda x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15, x16, x17: x2, out_dtype=lambda x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15, x16, x17: x2, func_type="aot")
        self.bprop_filter = ops.Custom(
            "/home/hcx/zjut_mindvideo_v2/zjut_mindvideo-master/mindvideo/models/layers/dcn/ms_dcn_bprop.so:msDCN_backward_filter", out_shape=lambda x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15, x16, x17: x2, out_dtype=lambda x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15, x16, x17: x2, func_type="aot")

    def bprop(self, x, offset, out, dout):
        input = x
        grad_output = dout
        grad_input = ops.zeros_like(input)
        grad_offset = ops.zeros_like(offset)
        cur_im2col_step = _cal_im2col_step(x.shape[0], self.im2col_step)
        grad_weight = ops.zeros_like(self.conv_weight)

        self.bprop_input(input,
                         offset,
                         grad_output,
                         grad_input,
                         grad_offset,
                         self.conv_weight,
                         self.bufs_[0],
                         self.conv_weight.shape[3],
                         self.conv_weight.shape[2],
                         self.stride[1],
                         self.stride[0],
                         self.padding[1],
                         self.padding[0],
                         self.dilation[1],
                         self.dilation[0],
                         self.groups,
                         self.deformable_groups,
                         cur_im2col_step,
                         )

        self.bprop_filter(input,
                          offset,
                          dout,
                          grad_weight,
                          self.bufs_[0],
                          self.bufs_[1],
                          self.conv_weight.shape[3],
                          self.conv_weight.shape[2],
                          self.stride[1],
                          self.stride[0],
                          self.padding[1],
                          self.padding[0],
                          self.dilation[1],
                          self.dilation[0],
                          self.groups,
                          self.deformable_groups,
                          1,
                          cur_im2col_step)
        return (grad_input, grad_offset, grad_weight, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def construct(self, x, offset):

        if x is not None and len(x.shape) != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(len(x.shape))
            )
        output_size = _output_size(x, self.conv_weight, self.padding, self.dilation, self.stride)
        output_buffer = self.zeros(output_size, ms.float32)
        # bufs_ = [self.zeros((1,), ms.float32),
        #          self.zeros((1,), ms.float32)]  # columns, ones

        cur_im2col_step = _cal_im2col_step(x.shape[0], self.im2col_step)
        assert (x.shape[0] % cur_im2col_step) == 0, "im2col step must divide batchsize"

        output = self.forward_op(x,
                                 self.conv_weight,
                                 offset,
                                 output_buffer,
                                 self.bufs_[0],
                                 self.bufs_[1],
                                 self.conv_weight.shape[3],
                                 self.conv_weight.shape[2],
                                 self.stride[1],
                                 self.stride[0],
                                 self.padding[1],
                                 self.padding[0],
                                 self.dilation[1],
                                 self.dilation[0],
                                 self.groups,
                                 self.deformable_groups,
                                 cur_im2col_step,)

        # print("bufs_[0]:", bufs_[0].sum())
        # print("output:", output.sum())

        if self.norm is not None:
            output = self.norm(output)
        if self.activation is not None:
            output = self.activation(output)
        return output
