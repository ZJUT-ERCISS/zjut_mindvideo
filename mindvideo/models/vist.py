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
"""Video Swin Transformer."""

from typing import Optional
import ml_collections as collections
import numpy as np

from mindspore import dtype
from mindspore import nn
from mindspore import ops
from mindspore import Parameter
from mindspore import Tensor

from mindvideo.utils.check_param import Rel
from mindvideo.utils.check_param import Validator
from mindvideo.utils.class_factory import ClassFactory, ModuleType

from mindvideo.models.layers import Roll3D
from mindvideo.models.layers import GlobalAvgPooling3D
from mindvideo.models.layers import Identity
from mindvideo.models.layers import ProbDropPath3D
from mindvideo.models.layers import FeedForward
from mindvideo.models.layers import DropoutDense

__all__ = [
    'Swin3D',
    'swin3d_t',
    'swin3d_s',
    'swin3d_b',
    'swin3d_l',
]


def limit_window_size(input_size, window_size, shift_size):
    r"""
    Limit the window size and shift size for window W-MSA and SW-MSA.
    If window size is larger than input size, we don't partition or shift
    windows.

    Args:
        input_size (tuple[int]): Input size of features. E.g. (16, 56, 56).
        window_size (tuple[int]): Target window size. E.g. (8, 7, 7).
        shift_size (int): depth of video. E.g. (4, 3, 3).

    Returns:
        Tuple[int], limited window size and shift size.
    """

    use_window_size = list(window_size)
    use_shift_size = [0, 0, 0]
    if shift_size:
        use_shift_size = list(shift_size)

    for i in range(len(input_size)):
        if input_size[i] <= window_size[i]:
            use_window_size[i] = input_size[i]
            if shift_size:
                use_shift_size[i] = 0
    window_size = tuple(use_window_size)
    shift_size = tuple(use_shift_size)

    return window_size, shift_size


def window_partition(features, window_size):
    r"""
    Window partition function for Swin Transformer.

    Args:
        features: Original features of shape (B, D, H, W, C).
        window_size (tuple[int]): Window size.

    Returns:
        Tensor of shape (B * num_windows, window_size * window_size, C).
    """

    batch_size, depth, height, width, channel_num = features.shape
    windows = features.reshape(
        batch_size,
        depth // window_size[0], window_size[0],
        height // window_size[1], window_size[1],
        width // window_size[2], window_size[2],
        channel_num
    )
    windows = windows.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    windows = windows.reshape(-1, window_size[0] *
                              window_size[1] * window_size[2], channel_num)
    return windows


def window_reverse(windows, window_size, batch_size, depth,
                   height, width):
    r"""
    Window reverse function for Swin Transformer.

    Args:
        windows: Partitioned features of shape (B*num_windows, window_size,
            window_size, C).
        window_size (tuple[int]): Window size.
        batch_size (int): Batch size of video.
        depth (int): depth of video.
        height (int): Height of video.
        width (int): Width of video.

    Returns:
        Tensor of shape (B, D, H, W, C).
    """

    windows = windows.view(
        batch_size,
        depth // window_size[0],
        height // window_size[1],
        width // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1
    )
    windows = windows.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    windows = windows.view(batch_size, depth, height, width, -1)
    return windows


def compute_mask(depth, height, width, window_size, shift_size):
    r"""
    Calculate attention mask for SW-MSA.

    Args:
        depth, height, width (int): Numbers of depth, height, width
            dimensions.
        window_size (Tuple(int)): Input window size.
        shift_size (Tuple(int)): Input shift_size.

    Returns:
        Tensor, attention mask.
    """
    t_padded = int(np.ceil(depth / window_size[0])) * window_size[0]
    h_padded = int(np.ceil(height / window_size[1])) * window_size[1]
    w_padded = int(np.ceil(width / window_size[2])) * window_size[2]
    img_mask = np.zeros((1, t_padded, h_padded, w_padded, 1))
    cnt = 0
    d_slices = (slice(-window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None))
    h_slices = (slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None))
    w_slices = (slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None))

    for i in d_slices:
        for j in h_slices:
            for k in w_slices:
                img_mask[:, i, j, k, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows[:, np.newaxis, :] - mask_windows[:, :, np.newaxis]
    attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.),
                       dtype=dtype.float32)
    return attn_mask


class WindowAttention3D(nn.Cell):
    r"""
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        in_channels (int): Number of input channels.
        window_size (tuple[int]): The depth length, height and width of the window. Default: (8, 7, 7).
        num_head (int): Number of attention heads. Default: 3.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        attn_keep_prob (float, optional): Dropout keep ratio of attention weight. Default: 1.0.
        proj_keep_prob (float, optional): Dropout keep ratio of output. Deault: 1.0.

    Inputs:
        - `x` (Tensor) - Tensor of shape (B, N, C).
        - `mask` (Tensor) - (0 / - inf) mask with shape of (num_windows, N, N) or None.

    Outputs:
        Tensor of shape (B, N, C), which is equal to the input **x**.

    Examples:
        >>> input = ops.Zeros()((1024, 392, 96), mindspore.float32)
        >>> net = WindowAttention3D(96, (8, 7, 7), 3, True, None, 0., 0.)
        >>> output = net(input)
        >>> print(output)
        (1024, 392, 96)
    """

    def __init__(self,
                 in_channels: int = 96,
                 window_size: int = (8, 7, 7),
                 num_head: int = 3,
                 qkv_bias: Optional[bool] = True,
                 qk_scale: Optional[float] = None,
                 attn_kepp_prob: Optional[float] = 1.0,
                 proj_keep_prob: Optional[float] = 1.0
                 ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_head = num_head
        head_dim = in_channels // num_head
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias
        init_tensor = np.random.randn(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            num_head
        )
        init_tensor = Tensor(init_tensor, dtype=dtype.float32)
        # self.relative_position_bias_table: [2*Wt-1 * 2*Wh-1 * 2*Ww-1, nH]
        self.relative_position_bias_table = Parameter(
            init_tensor
        )
        # get pair-wise relative position index for each token in a window
        coords_d = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[1])
        coords_w = np.arange(self.window_size[2])
        # coords: [3, Wd, Wh, Ww]
        coords = np.stack(
            np.meshgrid(
                coords_d,
                coords_h,
                coords_w,
                indexing='ij'
            )
        )
        coords_flatten = np.reshape(coords, (coords.shape[0], -1))
        # relative_coords: [3, Wd*Wh*Ww, Wd*Wh*Ww]
        relative_coords = coords_flatten[:, :, np.newaxis] - \
            coords_flatten[:, np.newaxis, :]
        # relative_coords: [Wh*Ww, Wh*Ww, 2]
        relative_coords = relative_coords.transpose(1, 2, 0)
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * \
            (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        # self.relative_position_index: [Wd*Wh*Ww, Wd*Wh*Ww]
        self.relative_position_index = Parameter(Tensor(relative_coords.sum(-1)), requires_grad=False)
        # QKV Linear layer
        self.qkv = nn.Dense(in_channels, in_channels * 3, has_bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_kepp_prob)
        self.proj = nn.Dense(in_channels, in_channels)
        self.proj_dropout = nn.Dropout(proj_keep_prob)
        self.softmax = nn.Softmax(axis=-1)
        # ops definition
        self.batch_matmul = ops.BatchMatMul()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()

    def construct(self, x, mask=None):
        """Construct WindowAttention3D."""

        batch_size, window_num, channel_num = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, window_num, 3,
                          self.num_head, channel_num // self.num_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        # q, k, v: [B, nH, N, C]
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale
        attn = self.batch_matmul(query, key.transpose(0, 1, 3, 2))
        # relative_position_bias: [Wd*Wh*Ww, Wd*Wh*Ww, nH]
        relative_position_bias = self.relative_position_bias_table[
            self.reshape(
                self.relative_position_index[:window_num, :window_num], (-1,)
            )
        ]
        relative_position_bias = self.reshape(
            relative_position_bias, (window_num, window_num, -1)
        )
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)
        relative_position_bias = self.expand_dims(relative_position_bias, 0)
        # attn: [B, nH, N, N]
        attn = attn + relative_position_bias
        # masked attention
        if mask is not None:
            n_w = mask.shape[0]
            mask = self.expand_dims(mask, 1)
            mask = self.expand_dims(mask, 0)
            attn = attn.view(batch_size // n_w, n_w, self.num_head,
                             window_num, window_num) + mask
            attn = attn.view(-1, self.num_head, window_num, window_num)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        x = self.batch_matmul(attn, value).transpose(
            0, 2, 1, 3).reshape(batch_size, window_num, channel_num)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class SwinTransformerBlock3D(nn.Cell):
    """
    A Video Swin Transformer Block. The implementation of this block follows
    the paper "Video Swin Transformer".

    Args:
        embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
        input_size (int | tuple(int)): input feature size. Default: (16, 56, 56).
        num_head (int): number of attention head of the current Swin3d block. Default: 3.
        window_size (int): window size of window attention. Default: (8, 7, 7).
        shift_size (tuple[int]): shift size for shifted window attention. Default: (4, 3, 3).
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): if True, add a learnable bias to query, key,value. Default: True.
        qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set True. Default: None.
        keep_prob (float): dropout keep probability. Default: 1.0.
        attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.0.
        droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 1.0.
        act_layer (nn.Cell): activation layer. Default: nn.GELU.
        norm_layer (nn.Cell): normalization layer. Default: 'layer_norm'.

    Inputs:
        - **x** (Tensor) - Input feature of shape (B, D, H, W, C).
        - **mask_matrix** (Tensor) - Attention mask for cyclic shift.

    Outputs:
        Tensor of shape (B, D, H, W, C)

    Examples:
        >>> net1 = SwinTransformerBlock3D()
        >>> input = ops.Zeros()((8,16,56,56,96), mindspore.float32)
        >>> output = net1(input, None)
        >>> print(output.shape)
    """

    def __init__(self,
                 embed_dim: int = 96,
                 input_size: int = (16, 56, 56),
                 num_head: int = 3,
                 window_size: int = (8, 7, 7),
                 shift_size: int = (4, 3, 3),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 keep_prob: float = 1.,
                 attn_keep_prob: float = 1.,
                 droppath_keep_prob: float = 1.,
                 act_layer: nn.Cell = nn.GELU,
                 norm_layer: str = 'layer_norm'
                 ):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        # get window size and shift size
        self.window_size, self.shift_size = limit_window_size(
            input_size, window_size, shift_size)
        # check self.shift_size whether is smaller than self.window_size and
        # larger than 0
        Validator.check_int_range(
            self.shift_size[0], 0, self.window_size[0],
            Rel.INC_LEFT,
            arg_name="shift size", prim_name="SwinTransformerBlock3D")
        Validator.check_int_range(
            self.shift_size[1], 0, self.window_size[1],
            Rel.INC_LEFT, arg_name="shift size",
            prim_name="SwinTransformerBlock3D")
        Validator.check_int_range(
            self.shift_size[2], 0, self.window_size[2],
            Rel.INC_LEFT, arg_name="shift size",
            prim_name="SwinTransformerBlock3D")

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.mlp_ratio = mlp_ratio
        if isinstance(self.embed_dim, int):
            self.embed_dim = (self.embed_dim,)

        # the first layer norm
        if norm_layer == 'layer_norm':
            self.norm1 = nn.LayerNorm(self.embed_dim, epsilon=1e-5)
        else:
            self.norm1 = Identity()
        # the second layer norm
        if norm_layer == 'layer_norm':
            self.norm2 = nn.LayerNorm(self.embed_dim, epsilon=1e-5)
        else:
            self.norm2 = Identity()

        # window attention 3D block
        self.attn = WindowAttention3D(self.embed_dim[0],
                                      window_size=self.window_size,
                                      num_head=num_head,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      attn_kepp_prob=attn_keep_prob,
                                      proj_keep_prob=keep_prob
                                      )

        self.drop_path = ProbDropPath3D(
            droppath_keep_prob) if droppath_keep_prob < 1. else Identity()

        mlp_hidden_dim = int(self.embed_dim[0] * mlp_ratio)

        # reuse classification.models.block.feed_forward as MLP here
        self.mlp = FeedForward(in_features=self.embed_dim[0],
                               hidden_features=mlp_hidden_dim,
                               activation=act_layer,
                               keep_prob=keep_prob
                               )

    def _construc_part1(self, x, mask_matrix):
        """"Construct W-MSA and SW-MSA."""
        batch_size, depth, height, width, channel_num = x.shape
        window_size = self.window_size
        shift_size = self.shift_size
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - depth % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - height % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - width % window_size[2]) % window_size[2]
        x_padded = []
        pad = nn.Pad(paddings=(
            (pad_d0, pad_d1),
            (pad_t, pad_b),
            (pad_l, pad_r),
            (0, 0)))
        for i in range(x.shape[0]):
            x_b = x[i]
            x_b = pad(x_b)
            x_padded.append(x_b)
        x = ops.Stack(axis=0)(x_padded)
        # cyclic shift
        _, t_padded, h_padded, w_padded, _ = x.shape
        if [i for i in shift_size if i > 0]:
            shifted_x = Roll3D(shift=[-i for i in shift_size])(x)
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows: (B*nW, Wd*Wh*Ww, C)
        x_windows = window_partition(shifted_x, window_size)
        # W-MSA/SW-MSA
        # attn_windows: (B*nW, Wd*Wh*Ww, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (channel_num,)))
        shifted_x = window_reverse(attn_windows, window_size, batch_size,
                                   t_padded, h_padded, w_padded)
        # reverse cyclic shift
        if [i for i in shift_size if i > 0]:
            x = Roll3D(shift=shift_size)(shifted_x)
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :depth, :height, :width, :]
        return x

    def _construct_part2(self, x):
        """Construct MLP."""
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        return x

    def construct(self, x, mask_matrix=None):
        """Construct 3D Swin Transformer Block."""
        shortcut = x
        x = self._construc_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self._construct_part2(x)
        return x


class PatchMerging(nn.Cell):
    """
    Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Cell): Normalization layer. Default: nn.LayerNorm。

    Inputs:
        - **x** (Tensor) - Input feature of shape (B, D, H, W, C).

    Outputs:
        Tensor of shape (B, D, H/2, W/2, 2*C)
    """

    def __init__(self,
                 dim: int = 96,
                 norm_layer: str = 'layer_norm'):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Dense(4 * self.dim, 2 * self.dim, has_bias=False)
        if norm_layer == 'layer_norm':
            self.norm = nn.LayerNorm((4 * self.dim,))
        else:
            self.norm = Identity()

    def construct(self, x):
        """Construct Patch Merging Layer."""

        batch_size, _, height, width, _ = x.shape

        # padding
        pad_input = (height % 2 == 1) or (width % 2 == 1)
        if pad_input:
            x_padded = []
            pad = nn.Pad(
                paddings=(
                    (0, 0), (0, height %
                             2), (0, width %
                                  2), (0, 0)))
            for i in range(batch_size):
                x_b = x[i]
                x_b = pad(x_b)
                x_padded.append(x_b)
            x_padded = ops.Stack(axis=0)(x_padded)
            x = x_padded
        # x_0, x_1, x_2, x_3: (B, D, H/2, W/2, C)
        x_0 = x[:, :, 0::2, 0::2, :]
        x_1 = x[:, :, 1::2, 0::2, :]
        x_2 = x[:, :, 0::2, 1::2, :]
        x_3 = x[:, :, 1::2, 1::2, :]
        # x: (B, D, H/2, W/2, 4*C)
        x = ops.Concat(axis=-1)([x_0, x_1, x_2, x_3])

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformerStage3D(nn.Cell):
    r"""
    A basic Swin Transformer layer for one stage.

    Args:
        embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
        input_size (tuple[int]): input feature size. Default. (16, 56, 56).
        depth (int): depth of the current Swin3d stage. Default: 2.
        num_head (int): number of attention head of the current Swin3d stage. Default: 3.
        window_size (int): window size of window attention. Default: (8, 7, 7).
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): if qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: Truee
        qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set. Default: None.
        keep_prob (float): dropout keep probability. Default: 1.0.
        attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.
        droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 0.8.
        norm_layer(string): normalization layer. Default: 'layer_norm'.
        downsample (nn.Cell | None, optional): downsample layer at the end of swin3d stage. Default: PatchMerging.

    Inputs:
        A video feature of shape (N, D, H, W, C)
    Returns:
        Tensor of shape (N, D, H / 2, W / 2, 2 * C)
    """

    def __init__(self,
                 embed_dim=96,
                 input_size=(16, 56, 56),
                 depth=2,
                 num_head=3,
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 keep_prob=1.,
                 attn_keep_prob=1.,
                 droppath_keep_prob=0.8,
                 norm_layer='layer_norm',
                 downsample=PatchMerging
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock3D(
                embed_dim=embed_dim,
                num_head=num_head,
                input_size=input_size,
                window_size=self.window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                keep_prob=keep_prob,
                attn_keep_prob=attn_keep_prob,
                droppath_keep_prob=droppath_keep_prob[i] if isinstance(
                    droppath_keep_prob, list) else droppath_keep_prob,
                norm_layer=norm_layer
            )
            for i in range(depth)])
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=embed_dim, norm_layer=norm_layer)
        self.window_size, self.shift_size = limit_window_size(
            input_size, self.window_size, self.shift_size)
        self.attn_mask = compute_mask(
            input_size[0], input_size[1], input_size[2],
            self.window_size, self.shift_size)

    def construct(self, x):
        """Construct a basic stage layer for VideoSwinTransformer."""
        for blk in self.blocks:
            x = blk(x, self.attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed3D(nn.Cell):
    """
    Video to Patch Embedding.

    Args:
        input_size (tuple[int]): Input feature size.
        patch_size (int): Patch token size. Default: (2,4,4).
        in_channels (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        patch_norm (bool): if True, add normalization after patch embedding. Default: True.
    Inputs:
        An original Video tensor in data format of 'NCDHW'.

    Returns:
        An embedded tensor in data format of 'NDHWC'.
    """

    def __init__(self, input_size=(16, 224, 224), patch_size=(2, 4, 4),
                 in_channels=3, embed_dim=96, norm_layer='layer_norm', patch_norm=True):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.project = nn.Conv3d(in_channels, embed_dim, has_bias=True,
                                 kernel_size=patch_size, stride=patch_size)

        if norm_layer == 'layer_norm' and patch_norm:
            if isinstance(self.embed_dim, int):
                self.embed_dim = (self.embed_dim,)
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.norm = Identity()

        self.output_size = [input_size[0] // patch_size[0],
                            input_size[1] // patch_size[1],
                            input_size[2] // patch_size[2]]

    def construct(self, x):
        """Construct Patch Embedding for 3D features."""
        # padding
        _, _, depth, height, width = x.shape
        pad_d = 0
        pad_b = 0
        pad_r = 0
        x_padded = []
        if width % self.patch_size[2] != 0:
            pad_r = self.patch_size[2] - width % self.patch_size[2]
        if height % self.patch_size[1] != 0:
            pad_b = self.patch_size[1] - height % self.patch_size[1]
        if depth % self.patch_size[0] != 0:
            pad_d = self.patch_size[0] - depth % self.patch_size[0]
        pad = nn.Pad(paddings=(
            (0, pad_d),
            (0, pad_b),
            (0, pad_r),
            (0, 0)
        ))
        for i in range(x.shape[0]):
            x_b = x[i]
            x_b = pad(x_b)
            x_padded.append(x_b)
        x = ops.Stack(axis=0)(x_padded)
        x = self.project(x)  # B C D Wh Ww
        batch_size, channel_num, depth_w, height_w, width_w = x.shape
        x = x.reshape(batch_size, channel_num, -1).transpose(0, 2, 1)
        x = self.norm(x)
        x = x.view(-1, depth_w, height_w, width_w, channel_num)
        return x


class SwinTransformer3D(nn.Cell):
    """
    Video Swin Transformer backbone.
    A mindspore implementation of : `Video Swin Transformer` http://arxiv.org/abs/2106.13230

    Args:
        input_size (int | tuple(int)): input feature size. Default: (16, 56, 56).
        embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
        depths (tuple[int]): depths of each Swin3d stage. Default: (2, 2, 6, 2).
        num_heads (tuple[int]): number of attention head of each Swin3d stage. Default: (3, 6, 12, 24).
        window_size (int): window size of window attention. Default: (8, 7, 7).
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): if qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: True.
        qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set. Default: None.
        keep_prob (float): dropout keep probability. Default: 1.0.
        attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.
        droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 0.8.
        norm_layer (string): normalization layer. Default: 'layer_norm'.

    Inputs:
        - **x** (Tensor) - Tensor of shape 'NDHWC'.

    Outputs:
        Tensor of shape 'NCDHW'.
    """

    def __init__(self,
                 input_size=(16, 56, 56),
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 keep_prob=1.,
                 attn_keep_prob=1.,
                 droppath_keep_prob=0.8,
                 norm_layer='layer_norm',
                 ):
        super(SwinTransformer3D, self).__init__()
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.input_feature_size = input_size
        self.patch_embedding_drop = nn.Dropout(keep_prob=keep_prob)
        # stochastic depth decay rule
        depth_decay_rate = list(np.linspace(1, droppath_keep_prob, sum(depths)))
        # build swin3d stages
        self.stages = nn.CellList()
        for i_stage in range(self.num_stages):
            stage = SwinTransformerStage3D(
                embed_dim=int(embed_dim * 2 ** i_stage),
                input_size=(self.input_feature_size[0],
                            self.input_feature_size[1] // (2 ** i_stage),
                            self.input_feature_size[2] // (2 ** i_stage)),
                depth=depths[i_stage],
                num_head=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                keep_prob=keep_prob,
                attn_keep_prob=attn_keep_prob,
                droppath_keep_prob=depth_decay_rate[sum(
                    depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                # use PatchMerging depending on i_layer
                downsample=PatchMerging if i_stage < self.num_stages - 1 else None
            )
            self.stages.append(stage)
        self.feature_dim = int(embed_dim * 2 ** (self.num_stages - 1))
        # if self.norm is 'layer_norm', apply nn.LayerNorm
        if norm_layer == 'layer_norm':
            self.norm = nn.LayerNorm((self.feature_dim,))
        # otherwise, use Identity()
        else:
            self.norm = Identity()

    def construct(self, x):
        x = self.patch_embedding_drop(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = x.transpose(0, 4, 1, 2, 3)
        return x


class Swin3D(nn.Cell):
    """
    Constructs a swin3d architecture corresponding to
    `Video Swin Transformer <http://arxiv.org/abs/2106.13230>`.

    Args:
        num_classes (int): The number of classification. Default: 400.
        patch_size (int): Patch size used by window attention. Default: (2, 4, 4).
        window_size (int): Window size used by window attention. Default: (8, 7, 7).
        embed_dim (int): Embedding dimension of the featrue generated from patch embedding layer. Default: 96.
        depths (int): Depths of each stage in Swin3d Tiny module. Default: (2, 2, 6, 2).
        num_heads (int): Numbers of heads of each stage in Swin3d Tiny module. Default: (3, 6, 12, 24).
        representation_size (int): Feature dimension of the last layer in backbone. Default: 768.
        droppath_keep_prob (float): The drop path keep probability. Default: 0.9.
        input_size (int | tuple(int)): Input feature size. Default: (32, 224, 224).
        in_channels (int): Input channels. Default: 3.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): If qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        keep_prob (float): Dropout keep probability. Default: 1.0.
        attn_keep_prob (float): Keeping probability for attention dropout. Default: 1.0.
        norm_layer (string): Normalization layer. Default: 'layer_norm'.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        pooling_keep_dim (bool): Specifies whether to keep dimension shape the same as input feature. Default: False.
        head_bias (bool): Specifies whether the head uses a bias vector. Default: True.
        head_activation (Union[str, Cell, Primitive]): Activate function applied in the head. Default: None.
        head_keep_prob (float): Head's dropout keeping rate, between [0, 1]. Default: 0.5.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    About Swin:

    TODO: Swin3d introduction.

    Citation:

    .. code-block::

        @article{liu2021video,
            title={Video Swin Transformer},
            author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
            journal={arXiv preprint arXiv:2106.13230},
            year={2021}
        }
    """

    def __init__(self,
                 num_classes: int,
                 patch_size: int,
                 window_size: int,
                 embed_dim: int,
                 depths: int,
                 num_heads: int,
                 representation_size: int,
                 droppath_keep_prob: float,
                 input_size: int = (32, 224, 224),
                 in_channels: int = 3,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 keep_prob: float = 1.,
                 attn_keep_prob: float = 1.,
                 norm_layer: str = 'layer_norm',
                 patch_norm: bool = True,
                 pooling_keep_dim: bool = False,
                 head_bias: bool = True,
                 head_activation: Optional[str] = None,
                 head_keep_prob: float = 0.5,
                 ):
        super(Swin3D, self).__init__()
        self.embed = PatchEmbed3D(input_size=input_size,
                                  patch_size=patch_size,
                                  in_channels=in_channels,
                                  embed_dim=embed_dim,
                                  norm_layer=norm_layer,
                                  patch_norm=patch_norm
                                  )
        self.backbone = SwinTransformer3D(input_size=self.embed.output_size,
                                          embed_dim=embed_dim,
                                          depths=depths,
                                          num_heads=num_heads,
                                          window_size=window_size,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          keep_prob=keep_prob,
                                          attn_keep_prob=attn_keep_prob,
                                          droppath_keep_prob=droppath_keep_prob,
                                          norm_layer=norm_layer
                                          )
        self.neck = GlobalAvgPooling3D(keep_dims=pooling_keep_dim)
        self.head = DropoutDense(input_channel=representation_size,
                                 out_channel=num_classes,
                                 has_bias=head_bias,
                                 activation=head_activation,
                                 keep_prob=head_keep_prob
                                 )
        self.softmax = nn.Softmax()

    def construct(self, x):
        """construct for Swin3d."""
        # print(x.shape)
        b, c, nl, h, w = x.shape
        # x.shape : (B, C, D x Crop x Clip, H, W)
        if nl > 32:
            x = x.reshape(b, c, -1, 32, h, w)
            x = x.transpose(2, 0, 1, 3, 4, 5)
            x = x.reshape(-1, c, 32, h, w)
        x = self.embed(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        if nl > 32:
            x = self.softmax(x)
            x = x.reshape(-1, b, 400)
            x = x.mean(axis=0, keep_dims=False)
        return x


@ClassFactory.register(ModuleType.MODEL)
def swin3d_t(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 96,
             depths: int = (2, 2, 6, 2),
             num_heads: int = (3, 6, 12, 24),
             representation_size: int = 768,
             droppath_keep_prob: float = 0.9,
             ) -> nn.Cell:
    """
    Video Swin Transformer Tiny (swin3d-T) model.
    """
    config = collections.ConfigDict()
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    return Swin3D(**config)


@ClassFactory.register(ModuleType.MODEL)
def swin3d_s(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 96,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (3, 6, 12, 24),
             representation_size: int = 768,
             droppath_keep_prob: float = 0.9,
             ) -> nn.Cell:
    """
    Video Swin Transformer Small (swin3d-S) model.
    """
    config = collections.ConfigDict()
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    return Swin3D(**config)


@ClassFactory.register(ModuleType.MODEL)
def swin3d_b(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 128,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (4, 8, 16, 32),
             representation_size: int = 1024,
             droppath_keep_prob: float = 0.7,
             ) -> nn.Cell:
    """
    Video Swin Transformer Base (swin3d-B) model.
    """
    config = collections.ConfigDict()
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    return Swin3D(**config)


@ClassFactory.register(ModuleType.MODEL)
def swin3d_l(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 192,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (6, 12, 24, 48),
             representation_size: int = 1536,
             droppath_keep_prob: float = 0.9,
             ) -> nn.Cell:
    """
    Video Swin Transformer Large (swin3d-L) model.
    """
    config = collections.ConfigDict()
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    return Swin3D(**config)
