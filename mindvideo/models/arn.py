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
""" ARN network."""

from mindspore import context
from typing import Optional

from mindspore import nn
from mindspore import ops

from mindvideo.models.layers.conv_norm_activation import ConvNormActivation
from mindvideo.models.layers.unit3d import Unit3D
from mindvideo.models.layers.c3d_backbone import C3DBackbone
from mindvideo.models.layers.maxpool3d import MaxPool3D
from mindvideo.utils.class_factory import ClassFactory, ModuleType

__all__ = [
    'SpatialAttention',
    'SimilarityNetwork',
    'ARNEmbedding',
    'ARNBackbone',
    'ARNNeck',
    'ARNHead',
    'ARN'
]


class SpatialAttention(nn.Cell):
    """
    Initialize spatial attention unit which refine the aggregation step
    by re-weighting block contributions.

    Args:
        in_channels: The number of channels of the input feature.
        out_channels: The number of channels of the output of hidden layers.

    Returns:
        Tensor of shape (1, 1, H, W).
    """

    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 16
                 ):
        super(SpatialAttention, self).__init__()
        self.layer1 = Unit3D(in_channels, out_channels)
        self.layer2 = Unit3D(out_channels, out_channels)
        self.max_pool = ops.MaxPool3D(kernel_size=(2, 1, 1), strides=(2, 1, 1))
        self.conv3d = nn.Conv3d(out_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out = self.layer1(x)
        out = self.max_pool(out)
        out = self.layer2(out)
        out = self.max_pool(out)
        out = self.sigmoid(self.conv3d(out))
        return out


class SimilarityNetwork(nn.Cell):
    """Similarity learning between query and support clips as paired
    relation descriptors for RelationNetwork.

    Args:
        in_channels (int): Number of channels of the input feature. Default: 2.
        out_channels (int): Number of channels of the output feature. Default: 64.
        input_size (int): Size of input features. Default: 64.
        hidden_size (int): Number of channels in the hidden fc layers. Default: 8.
    Returns:
        Tensor, output tensor.
    """

    def __init__(self, in_channels=2, out_channels=64, input_size=64, hidden_size=8):
        super(SimilarityNetwork, self).__init__()

        self.layer1 = ConvNormActivation(in_channels, out_channels)
        self.layer2 = ConvNormActivation(out_channels, out_channels)
        self.layer3 = ConvNormActivation(out_channels, out_channels)
        self.layer4 = ConvNormActivation(out_channels, out_channels)

        self.fc1 = nn.Dense(out_channels * (input_size // 16)
                            * (input_size // 16), hidden_size)
        self.fc2 = nn.Dense(hidden_size, 1)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out = self.layer1(x)
        out = self.maxpool2d(out)  # 2,2
        out = self.maxpool2d(out + self.layer2(out))
        out = self.maxpool2d(out + self.layer3(out))
        out = self.maxpool2d(out + self.layer4(out))
        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


class ARNEmbedding(nn.Cell):
    """
    Embedding for ARN based on Unit3d-built 4-layer Conv or C3d.

    Args:
        support_num_per_class (int): Number of samples in support set per class. Default: 1.
        query_num_per_class (int): Number of samples in query set per class. Default: 1.
        class_num (int): Number of classes. Default: 5.
        is_c3d (bool): Specifies whether the network uses C3D as embedding for ARN. Default: False.
        in_channels: The number of channels of the input feature. Default: 3.
        out_channels: The number of channels of the output of hidden layers (only used when is_c3d is set to False).
            Default: 64.

    Returns:
        Tensor, output 2 tensors.
    """

    def __init__(self,
                 support_num_per_class: int = 1,
                 query_num_per_class: int = 1,
                 class_num: int = 5,
                 is_c3d: bool = True,
                 in_channels: Optional[int] = 3,
                 out_channels: Optional[int] = 64,
                 ) -> None:
        super(ARNEmbedding, self).__init__()
        self.support_num_per_class = support_num_per_class
        self.query_num_per_class = query_num_per_class
        self.class_num = class_num
        self.concat = ops.Concat(0)
        self.squeeze = ops.Squeeze(0)
        if is_c3d:
            # reusing c3d backbone as embedding
            self.embedding = C3DBackbone(in_channels)
        else:
            # reusing unit3d block for building Conv-4 architecture as embedding
            self.embedding = nn.SequentialCell(
                Unit3D(in_channels, out_channels),
                MaxPool3D(kernel_size=2, strides=2),
                Unit3D(out_channels, out_channels),
                MaxPool3D(kernel_size=2, strides=2),
                Unit3D(out_channels, out_channels),
                Unit3D(out_channels, out_channels)
            )

    def construct(self, data):
        """Construct embedding for ARN."""
        data = self.squeeze(data)
        data = data.transpose((1, 0, 2, 3, 4))
        support = data[:self.support_num_per_class *
                       self.class_num, :, :, :, :]
        query = data[self.support_num_per_class * self.class_num:self.support_num_per_class * self.class_num +
                     self.query_num_per_class * self.class_num, :, :, :, :]
        support_features = self.embedding(support)
        query_features = self.embedding(query)
        features = self.concat((support_features, query_features))

        return features


class ARNBackbone(nn.Cell):
    """ARN architecture. TODO: these architecture is slight complex. we will discuses later.

    Args:
        jigsaw (int): Number of the output dimension for spacial-temporal jigsaw discriminator. Default: 10.
        support_num_per_class (int): Number of samples in support set per class. Default: 1.
        query_num_per_class (int): Number of samples in query set per class. Default: 1.
        class_num (int): Number of classes. Default: 5.

    Returns:
        Tensor, output 2 tensors.

    Examples:
        >>> ARNBackbone(10, 5, 3, 5)
    """

    def __init__(self,
                 jigsaw: int = 10,
                 support_num_per_class: int = 1,
                 query_num_per_class: int = 1,
                 class_num: int = 5,
                 seq: int = 16
                 ):
        super(ARNBackbone, self).__init__()
        self.jigsaw = jigsaw
        self.support_num_per_class = support_num_per_class
        self.query_num_per_class = query_num_per_class
        self.class_num = class_num
        self.concat = ops.Concat(0)
        self.seq = seq
        self.spatial_detector = SpatialAttention(64, self.seq)

    def construct(self, features):
        """test construct of arn backbone"""
        support_features = features[:self.support_num_per_class *
                                    self.class_num, :, :, :, :]
        query_features = features[self.support_num_per_class *
                                  self.class_num:, :, :, :, :]
        channel = support_features.shape[1]
        temporal_dim = support_features.shape[2]
        width = support_features.shape[3]
        height = support_features.shape[4]
        support_ta = 1 + self.spatial_detector(support_features)
        query_ta = 1 + self.spatial_detector(query_features)
        support_features = (support_features * support_ta).reshape(self.support_num_per_class * self.class_num, channel,
                                                                   temporal_dim * width * height)  # C * N
        query_features = (query_features * query_ta)
        query_features = query_features.reshape(
            self.query_num_per_class * self.class_num,
            channel, temporal_dim * width * height)  # C * N
        features = self.concat((support_features, query_features))

        return features


class ARNNeck(nn.Cell):
    """
    ARN neck architecture.

    Args:
        class_num (int): Number of classes. Default: 5.
        support_num_per_class (int): Number of samples in support set per class. Default: 1.
        sigma: Controls the slope of PN. Default: 100.

    Returns:
        Tensor, output 2 tensors.
    """

    def __init__(self,
                 class_num: int = 5,
                 support_num_per_class: int = 1,
                 sigma: int = 100
                 ):
        super(ARNNeck, self).__init__()
        self.class_num = class_num
        self.support_num_per_class = support_num_per_class
        self.sigma = sigma
        self.mm = ops.MatMul(transpose_b=True)
        self.sigmoid = ops.Sigmoid()
        self.expand = ops.ExpandDims()
        self.stack_feature = ops.Stack(0)
        self.concat = ops.Concat(0)

    def power_norm(self, x):
        """
        Define the operation of Power Normalization.

        Args:
            x (Tensor): Tensor of shape :math:`(C_{in}, C_{in})`.

        Returns:
            Tensor of shape: math:`(C_{out}, C_{out})`.
        """
        out = 2.0 * self.sigmoid(self.sigma * x) - 1.0
        return out

    def construct(self, features):
        """test construct of arn neck"""
        support_features = features[:self.support_num_per_class * self.class_num]
        query_features = features[self.support_num_per_class * self.class_num:]
        channel = support_features.shape[1]
        so_support_features = []
        so_query_features = []

        for dd in range(support_features.shape[0]):
            s = support_features[dd, :, :].reshape(channel, -1)
            s = (1.0 / s.shape[1]) * self.mm(s, s)
            so_support_features.append(self.power_norm(s / s.trace()))
        so_support_features = self.stack_feature(so_support_features)

        for dd in range(query_features.shape[0]):
            t = query_features[dd, :, :].view(channel, -1)
            t = (1.0 / t.shape[1]) * self.mm(t, t)
            so_query_features.append(self.power_norm(t / t.trace()))
        so_query_features = self.stack_feature(so_query_features)

        so_support_features = so_support_features.reshape(
            self.class_num, self.support_num_per_class, 1, channel, channel).mean(1)  # z-shot, average
        so_query_features = self.expand(so_query_features, 1)  # 1 * C * C
        features = self.concat((so_support_features, so_query_features))

        return features


class ARNHead(nn.Cell):
    """
    ARN head architecture.

    Args:
        class_num (int): Number of classes. Default: 5.
        query_num_per_class (int): Number of query samples per class. Default: 1.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 class_num: int = 5,
                 query_num_per_class: int = 1
                 ):
        super(ARNHead, self).__init__()
        self.class_num = class_num
        self.query_num_per_class = query_num_per_class
        self.expand = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.cat_relation = ops.Concat(1)
        self.repeat = ops.Tile()
        self.relation_network = SimilarityNetwork(
            out_channels=64, input_size=64)
        self.concat = ops.Concat(0)

    def construct(self, features):
        """test construct of arn head"""
        so_support_features = features[:-
                                       self.query_num_per_class * self.class_num]
        so_query_features = features[-self.query_num_per_class *
                                     self.class_num:]
        channel = so_support_features.shape[-1]
        support_feature_ex = self.expand(so_support_features, 0)
        support_feature_ex = self.concat(
            (support_feature_ex, support_feature_ex, support_feature_ex, support_feature_ex, support_feature_ex))
        query_feature_ex = self.expand(so_query_features, 0)
        query_feature_ex = self.concat(
            (query_feature_ex, query_feature_ex, query_feature_ex, query_feature_ex, query_feature_ex))
        query_feature_ex = self.transpose(query_feature_ex, (1, 0, 2, 3, 4))
        query_feature_ex = query_feature_ex.view(-1, 1, channel, channel)
        support_feature_ex = support_feature_ex.view(-1, 1, channel, channel)
        relation_pairs = self.cat_relation(
            (support_feature_ex, query_feature_ex))
        relations = self.relation_network(relation_pairs)
        relations = relations.reshape(-1, self.class_num)
        relations = self.expand(relations, 0)

        return relations


@ClassFactory.register(ModuleType.MODEL)
class ARN(nn.Cell):
    """
    Constructs a ARN architecture from
    `Few-shot Action Recognition via Permutation-invariant Attention <https://arxiv.org/pdf/2001.03905.pdf>`.

    Args:
        support_num_per_class (int): Number of samples in support set per class. Default: 1.
        query_num_per_class (int): Number of samples in query set per class. Default: 1.
        class_num (int): Number of classes. Default: 5.
        is_c3d (bool): Specifies whether the network uses C3D as embendding for ARN. Default: False.
        in_channels: The number of channels of the input feature. Default: 3.
        out_channels: The number of channels of the output of hidden layers (only used when is_c3d is set to False).
            Default: 64.
        jigsaw (int): Number of the output dimension for spacial-temporal jigsaw discriminator. Default: 10.
        sigma: Controls the slope of PN. Default: 100.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(E, N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(CLASSES_NUM, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.mindvideo.models import arn
        >>>
        >>> net = arn(5, 3, 5, False, 3, 64, ops.MaxPool3D(kernel_size=2, strides=2), 10, 100)
        >>> x = ms.Tensor(np.random.randn(1, 10, 3, 16, 128, 128), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (5, 5)

    About ARN:

    TODO: ARN introduction.

    Citation:

    .. code-block::

        @article{zhang2020few,
            title={Few-shot Action Recognition with Permutation-invariant Attention},
            author={Zhang, Hongguang and Zhang, Li and Qi, Xiaojuan and Li, Hongdong and Torr, Philip HS
                and Koniusz, Piotr},
            journal={arXiv preprint arXiv:2001.03905},
            year={2020}
        }
    """

    def __init__(self,
                 support_num_per_class: int = 1,
                 query_num_per_class: int = 1,
                 class_num: int = 5,
                 is_c3d: bool = False,
                 in_channels: Optional[int] = 3,
                 out_channels: Optional[int] = 64,
                 jigsaw: int = 10,
                 sigma: int = 100):
        super(ARN, self).__init__()

        self.embedding = ARNEmbedding(support_num_per_class=support_num_per_class,
                                      query_num_per_class=query_num_per_class,
                                      class_num=class_num,
                                      is_c3d=is_c3d,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      )
        self.backbone = ARNBackbone(jigsaw=jigsaw,
                                    support_num_per_class=support_num_per_class,
                                    query_num_per_class=query_num_per_class,
                                    class_num=class_num
                                    )
        self.neck = ARNNeck(class_num=class_num,
                            support_num_per_class=support_num_per_class,
                            sigma=sigma)
        self.head = ARNHead(class_num=class_num,
                            query_num_per_class=query_num_per_class
                            )

    def construct(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
