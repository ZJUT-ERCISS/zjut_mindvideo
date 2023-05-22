## mindvideo.models

### SpatialAttention

> class mindvideo.models.SpatialAttention(in_channels: int = 64,
                 out_channels: int = 16)

Initialize spatial attention unit which refine the aggregation step by re-weighting block contributions.

- base: nn.Cell

**Parameters:**

- in_channels: The number of channels of the input feature.
- out_channels: The number of channels of the output of hidden layers.

**Return:**

Tensor of shape (1, 1, H, W).




### SimilarityNetwork

> class mindvideo.models.SimilarityNetwork(in_channels=2, out_channels=64, input_size=64, hidden_size=8)

Similarity learning between query and support clips as paired relation descriptors for RelationNetwork.

- base: nn.Cell

**Parameters:**

- in_channels (int): Number of channels of the input feature. Default: 2.
- out_channels (int): Number of channels of the output feature. Default: 64.
- input_size (int): Size of input features. Default: 64.
- hidden_size (int): Number of channels in the hidden fc layers. Default: 8.

**Return:**

Tensor, output tensor.


### ARNEmbedding

> class mindvideo.models.ARNEmbedding(support_num_per_class: int = 1,
                 query_num_per_class: int = 1,
                 class_num: int = 5,
                 is_c3d: bool = True,
                 in_channels: Optional[int] = 3,
                 out_channels: Optional[int] = 64)

Embedding for ARN based on Unit3d-built 4-layer Conv or C3d.

- base: nn.Cell

**Parameters:**

- support_num_per_class (int): Number of samples in support set per class. Default: 1.
- query_num_per_class (int): Number of samples in query set per class. Default: 1.
- class_num (int): Number of classes. Default: 5.
- is_c3d (bool): Specifies whether the network uses C3D as embedding for ARN. Default: False.
- in_channels: The number of channels of the input feature. Default: 3.
- out_channels: The number of channels of the output of hidden layers (only used when is_c3d is set to False). Default: 64.

**Return:**

Tensor, output 2 tensors.


### ARNBackbone

> class mindvideo.models.ARNBackbone(jigsaw: int = 10,
                 support_num_per_class: int = 1,
                 query_num_per_class: int = 1,
                 class_num: int = 5,
                 seq: int = 16)

ARN architecture. 

- base: nn.Cell

**Parameters:**

- jigsaw (int): Number of the output dimension for spacial-temporal jigsaw discriminator. Default: 10.
- support_num_per_class (int): Number of samples in support set per class. Default: 1.
- query_num_per_class (int): Number of samples in query set per class. Default: 1.
- class_num (int): Number of classes. Default: 5.

**Return:**

Tensor, output 2 tensors.


### ARNNeck

> class mindvideo.models.ARNNeck(class_num: int = 5,
                 support_num_per_class: int = 1,
                 sigma: int = 100)

ARN neck architecture.

- base: nn.Cell

**Parameters:**

- class_num (int): Number of classes. Default: 5.
- support_num_per_class (int): Number of samples in support set per class. Default: 1.
- sigma: Controls the slope of PN. Default: 100.

**Return:**

Tensor, output 2 tensors.

> def mindvideo.models.ARNNeck.power_norm(x)

Define the operation of Power Normalization.

**Parameters:**

x (Tensor): Tensor of shape :math:`(C_{in}, C_{in})`.

**Return:**

Tensor of shape: math:`(C_{out}, C_{out})`.


### ARNHead

> class mindvideo.models.ARNHead(class_num: int = 5,
                 query_num_per_class: int = 1)

ARN head architecture.

- base: nn.Cell

**Parameters:**

- class_num (int): Number of classes. Default: 5.
- query_num_per_class (int): Number of query samples per class. Default: 1.

**Return:**

Tensor, output tensors.


### ARN

> class mindvideo.models.ARN(support_num_per_class: int = 1,
                 query_num_per_class: int = 1,
                 class_num: int = 5,
                 is_c3d: bool = False,
                 in_channels: Optional[int] = 3,
                 out_channels: Optional[int] = 64,
                 jigsaw: int = 10,
                 sigma: int = 100)

Constructs a ARN architecture from `Few-shot Action Recognition via Permutation-invariant Attention <https://arxiv.org/pdf/2001.03905.pdf>`.

- base: nn.Cell

**Parameters:**

- support_num_per_class (int): Number of samples in support set per class. Default: 1.
- query_num_per_class (int): Number of samples in query set per class. Default: 1.
- class_num (int): Number of classes. Default: 5.
- is_c3d (bool): Specifies whether the network uses C3D as embendding for ARN. Default: False.
- in_channels: The number of channels of the input feature. Default: 3.
- out_channels: The number of channels of the output of hidden layers (only used when is_c3d is set to False). Default: 64.
- jigsaw (int): Number of the output dimension for spacial-temporal jigsaw discriminator. Default: 10.
- sigma: Controls the slope of PN. Default: 100.

**Inputs:**

- x(Tensor): Tensor of shape :math:`(E, N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(CLASSES_NUM, CLASSES_{out})`


### C3D

> class mindvideo.models.C3D(in_d: int = 16,
                 in_h: int = 112,
                 in_w: int = 112,
                 in_channel: int = 3,
                 kernel_size: Union[int, Tuple[int]] = (3, 3, 3),
                 head_channel: Union[int, Tuple[int]] = (4096, 4096),
                 num_classes: int = 400,
                 keep_prob: Union[float, Tuple[float]] = (0.5, 0.5, 1.0))

Constructs a C3D architecture.

- base: nn.Cell

**Parameters:**

- in_d: Depth of input data, it can be considered as frame number of a video. Default: 16.
- in_h: Height of input frames. Default: 112.
- in_w: Width of input frames. Default: 112.
- in_channel(int): Number of channel of input data. Default: 3.
- kernel_size(Union[int, Tuple[int]]): Kernel size for every conv3d layer in C3D. Default: (3, 3, 3).
- head_channel(Tuple[int]): Hidden size of multi-dense-layer head. Default: [4096, 4096].
- num_classes(int): Number of classes, it is the size of classfication score for every sample, i.e. :math:`CLASSES_{out}`. Default: 400.
- keep_prob(Tuple[int]): Probability of dropout for multi-dense-layer head, the number of probabilities equals the number of dense layers.
- pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded from network. If `False`, it will create a c3d model with uniform initialization for weight and bias.

**Inputs:**

- x(Tensor): Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`.



### BasicBlock

> class mindvideo.models.BasicBlock(cin, cout, stride=1, dilation=1)

Basic residual block for dla.

- base: nn.Cell

**Parameters:**

- cin(int): Input channel.
- cout(int): Output channel.
- stride(int): Covolution stride. Default: 1.
- dilation(int): The dilation rate to be used for dilated convolution. Default: 1.

**Return:**

Tensor, the feature after covolution.


### Root

> class mindvideo.models.Root(in_channels, out_channels, kernel_size, residual)

Get HDA node which play as the root of tree in each stage.

- base: nn.Cell

**Parameters:**

- cin(int): Input channel.
- cout(int):Output channel.
- kernel_size(int): Covolution kernel size.
- residual(bool): Add residual or not.

**Return:**

Tensor, HDA node after aggregation.


### Tree

> class mindvideo.models.Tree(levels, block, in_channels, out_channels, stride=1, level_root=False,
                 root_dim=0, root_kernel_size=1, dilation=1, root_residual=False)

Construct the deep aggregation network through recurrent. Each stage can be seen as a tree with multiple children.

- base: nn.Cell

**Parameters:**

- levels(list int): Tree height of each stage.
- block(Cell): Basic block of the tree.
- in_channels(list int): Input channel of each stage.
- out_channels(list int): Output channel of each stage.
- stride(int): Covolution stride. Default: 1.
- level_root(bool): Whether is the root of tree or not. Default: False.
- root_dim(int): Input channel of the root node. Default: 0.
- root_kernel_size(int): Covolution kernel size at the root. Default: 1.
- dilation(int): The dilation rate to be used for dilated convolution. Default: 1.
- root_residual(bool): Add residual or not. Default: False.

**Return:**

Tensor, the root ida node.


### DLA34

> class mindvideo.models.DLA34(levels, channels, block=None, residual_root=False)

Construct the downsampling deep aggregation network.

- base: nn.Cell

**Parameters:**

- levels(list int): Tree height of each stage.
- channels(list int): Input channel of each stage
- block(Cell): Initial basic block. Default: BasicBlock.
- residual_root(bool): Add residual or not. Default: False

**Return:**

tuple of Tensor, the root node of each stage.


### DlaDeformConv

> class mindvideo.models.DlaDeformConv(cin, cout)

Deformable convolution v2 with bn and relu.

- base: nn.Cell

**Parameters:**

- cin(int): Input channel
- cout(int): Output_channel

**Return:**

Tensor, results after deformable convolution and activation


### IDAUp

> class mindvideo.models.IDAUp(out, channels, up_f)

IDAUp sample.

- base: nn.Cell

**Return:**

List.


### DLAUp

> class mindvideo.models.DLAUp(startp, channels, scales, in_channels=None)

DLAUp sample.

- base: nn.Cell

**Return:**

List.


### DLASegConv

> class mindvideo.models.DLASegConv(down_ratio: int,
                 last_level: int,
                 out_channel: int = 0,
                 stage_levels: Tuple[int] = (1, 1, 1, 2, 2, 1),
                 stage_channels: Tuple[int] = (16, 32, 64, 128, 256, 512))

The DLA backbone network.

- base: nn.Cell

**Parameters:**

- down_ratio(int): The ratio of input and output resolution
- last_level(int): The ending stage of the final upsampling
- stage_levels(tuple[int]): The tree height of each stage block
- stage_channels(tuple[int]): The feature channel of each stage

**Return:**

Tensor, the feature map extracted by dla network


### FairmotDla34

> class mindvideo.models.FairmotDla34(down_ratio: int,
                 last_level: int,
                 out_channel: int = 0,
                 stage_levels: Tuple[int] = (1, 1, 1, 2, 2, 1),
                 stage_channels: Tuple[int] = (16, 32, 64, 128, 256, 512))

Constructs a Fairmot architecture.

- base: nn.Cell

**Parameters:**

- down_ratio(int): Output stride. Currently only supports 4. Default: 4.
- last_level(int): Last level of dla layers used for deep layer aggregation(DLA) module. Default: 5.
- head_channel(int): Channel of input of second conv2d layer in heads. Default: 256.
- head_conv2_ksize(Union[int, Tuple]): Kernel size of second conv2d layer. Default: 1.
- hm(int): Number of heatmap channels. Default: 1.
- wh(int): Dimension of offset and size output, i.e. position of bbox, it equals 4 if regress left, top, right, bottom of bbox, else 2. Default: 4.
- feature_id(int): Dimension of identity embedding. Default: 128.
- reg(int): Dimension of local offset. Default: 2.
- pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded from network. If `False`, it will create a fairmot model with default initialization. Default: False.

**Inputs:**

- x(Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`.


### Inception3dModule

> class mindvideo.models.Inception3dModule(in_channels, out_channels)

Inception3dModule definition.

- base: nn.Cell

**Parameters:**

- in_channels (int):  The number of channels of input frame images.
- out_channels (int): The number of channels of output frame images.

**Return:**

Tensor, output tensor.


### InceptionI3d

> class mindvideo.models.InceptionI3d(in_channels=3)

InceptionI3d architecture. 

- base: nn.Cell

**Parameters:**

- in_channels (int): The number of channels of input frame images(default 3).

**Return:**

Tensor, output tensor.


### I3dHead

> class mindvideo.models.I3dHead(in_channels, num_classes=400, dropout_keep_prob=0.5)

I3dHead definition.

- base: nn.Cell

**Parameters:**

- in_channels: Input channel.
- num_classes (int): The number of classes .
- dropout_keep_prob (float): A float value of prob.

**Return:**

Tensor, output tensor.


### I3D

> class mindvideo.models.I3D(in_channel: int = 3,
                 num_classes: int = 400,
                 keep_prob: float = 0.5,
                 pooling_keep_dim: bool = True,
                 backbone_output_channel=1024)

Constructs a I3D architecture.

- base: nn.Cell

**Parameters:**

- in_channel(int): Number of channel of input data. Default: 3.
- num_classes(int): Number of classes, it is the size of classfication score for every sample, i.e. :math:`CLASSES_{out}`. Default: 400.
- keep_prob(float): Probability of dropout for multi-dense-layer head, the number of probabilities equals the number of dense layers. Default: 0.5.
- pooling_keep_dim: whether to keep dim when pooling. Default: True.
- pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded from network. If `False`, it will create a i3d model with uniform initialization for weight and bias. Default: False.

**Inputs:**

- x(Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`.


### NonLocalBlockND

> class mindvideo.models.NonLocalBlockND(in_channels,
            inter_channels=None,
            mode='embedded',
            sub_sample=True,
            bn_layer=True)

Classification backbone for nonlocal. Implementation of Non-Local Block with 4 different pairwise functions.

- base: nn.Cell

**Parameters:**

- in_channels (int): original channel size.
- inter_channels (int): channel size inside the block if not specified reduced to half.
- mode: 4 mode to choose (gaussian, embedded, dot, and concatenation).
- bn_layer: whether to add batch norm.

**Inputs:**

- x(Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.


### NLInflateBlockBase3D

> class mindvideo.models.NLInflateBlockBase3D(in_channels,
            inter_channels=None,
            mode='embedded',
            sub_sample=True,
            bn_layer=True)

ResNet residual block base definition.

- base: ResidualBlockBase3D

**Parameters:**

- in_channel (int): Input channel.
- out_channel (int): Output channel.
- stride (int): Stride size for the first convolutional layer. Default: 1.
- group (int): Group convolutions. Default: 1.
- base_width (int): Width of per group. Default: 64.
- norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
- down_sample (nn.Cell, optional): Downsample structure. Default: None.

**Return:**

Tensor, output tensor.


### NLInflateBlock3D

> class mindvideo.models.NLInflateBlockBase3D(in_channel: int,
                 out_channel: int,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 non_local: bool = False,
                 non_local_mode: str = 'dot',
                 **kwargs)

ResNet3D residual block definition.

- base: ResidualBlock3D

**Parameters:**

- in_channel (int): Input channel.
- out_channel (int): Output channel.
- stride (int): Stride size for the second convolutional layer. Default: 1.
- group (int): Group convolutions. Default: 1.
- base_width (int): Width of per group. Default: 64.
- norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
- down_sample (nn.Cell, optional): Downsample structure. Default: None.

**Return:**

Tensor, output tensor.


### NLInflateResNet3D


> class mindvideo.models.NLInflateResNet3D(block: Optional[nn.Cell],
                                layer_nums: Tuple[int],
                                stage_channels: Tuple[int] = (64, 128, 256, 512),
                                stage_strides: Tuple[int] = ((1, 1, 1),
                                                            (1, 2, 2),
                                                            (1, 2, 2),
                                                            (1, 2, 2)),
                                down_sample: Optional[nn.Cell] = Unit3D,
                                inflate: Tuple[Tuple[int]] = ((1, 1, 1),
                                                            (1, 0, 1, 0),
                                                            (1, 0, 1, 0, 1, 0),
                                                            (0, 1, 0)),
                                non_local: Tuple[Tuple[int]] = ((0, 0, 0),
                                                                (0, 1, 0, 1),
                                                                (0, 1, 0, 1, 0, 1),
                                                                (0, 0, 0)),
                                **kwargs)

Inflate3D with ResNet3D backbone and non local block.

- base: ResNet3D

**Parameters:**

- block (Optional[nn.Cell]): THe block for network.
- layer_nums (list): The numbers of block in different layers.
- norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.
- stage_strides: Stride size for ResNet3D convolutional layer.
- non_local: Determine whether to apply nonlocal block in this block.

**Inputs:**

- x(Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor, output tensor.


### nonlocal3d


> class mindvideo.models.nonlocal3d(in_d: int = 32,
                        in_h: int = 224,
                        in_w: int = 224,
                        num_classes: int = 400,
                        keep_prob: float = 0.5,
                        backbone: Optional[nn.Cell] = NLResInflate3D50,
                        avg_pool: Optional[nn.Cell] = AdaptiveAvgPool3D,
                        flatten: Optional[nn.Cell] = nn.Flatten,
                        head: Optional[nn.Cell] = DropoutDense)

nonlocal3d model from Xiaolong Wang. "Non-local Neural Networks." https://arxiv.org/pdf/1711.07971v3

- base: nn.Cell

**Parameters:**

- in_d: Depth of input data, it can be considered as frame number of a video. Default: 32.
- in_h: Height of input frames. Default: 224.
- in_w: Width of input frames. Default: 224.
- num_classes(int): Number of classes, it is the size of classfication score for every sample, i.e. :math:`CLASSES_{out}`. Default: 400.
- pooling_keep_dim: whether to keep dim when pooling. Default: True.
- keep_prob(float): Probability of dropout for multi-dense-layer head, the number of probabilities equals the number of dense layers.
- pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded from network. If `False`, it will create a nonlocal3d model with uniform initialization for weight and bias.
- backbone: Bcxkbone of nonlocal3d.
- avg_pool: Avgpooling and flatten.
- head: LinearClsHead architecture.

**Inputs:**

- x(Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`..

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`.


### Conv2Plus1d


> class mindvideo.models.Conv2Plus1d(in_channel,
                 mid_channel,
                 out_channel,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 norm=nn.BatchNorm3d,
                 activation=nn.ReLU)

R(2+1)d conv12 block. It implements spatial-temporal feature extraction in a sperated way.

- base: nn.Cell

**Parameters:**

- in_channels (int):  The number of channels of input frame images.
- out_channels (int):  The number of channels of output frame images.
- kernel_size (tuple): The size of the spatial-temporal convolutional layer kernels.
- stride (Union[int, Tuple[int]]): Stride size for the convolutional layer. Default: 1.
- group (int): Splits filter into groups, in_channels and out_channels must be divisible by the number of groups. Default: 1.
- norm (Optional[nn.Cell]): Norm layer that will be stacked on top of the convolution layer. Default: nn.BatchNorm3d.
- activation (Optional[nn.Cell]): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

**Return:**

Tensor, its channel size is calculated from in_channel, out_channel and kernel_size.


### R2Plus1dNet


> class mindvideo.models.R2Plus1dNet(block: Optional[nn.Cell],
                 layer_nums: Tuple[int],
                 stage_channels: Tuple[int] = (64, 128, 256, 512),
                 stage_strides: Tuple[Tuple[int]] = ((1, 1, 1),
                                                     (2, 2, 2),
                                                     (2, 2, 2),
                                                     (2, 2, 2)),
                 num_classes: int = 400,
                 **kwargs)

Generic R(2+1)d generator.

- base: ResNet3D

**Parameters:**

- block (Optional[nn.Cell]): THe block for network.
- layer_nums (Tuple[int]): The numbers of block in different layers.
- stage_channels (Tuple[int]): Output channel for every res stage. Default: (64, 128, 256, 512).
- stage_strides (Tuple[Tuple[int]]): Strides for every res stage.Default:((1, 1, 1),  (2, 2, 2), (2, 2, 2), (2, 2, 2)).
- conv12 (nn.Cell, optional): Conv1 and conv2 config in resblock. Default: Conv2Plus1D.
- base_width (int): The width of per group. Default: 64.
- norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.
- num_classes(int): Number of categories in the action recognition dataset.
- keep_prob(float): Dropout probability in classification stage.
- kwargs (dict, optional): Key arguments for "make_res_layer" and resblocks.

**Return:**

Tensor, output tensor.


### WindowAttention3D

> class mindvideo.models.WindowAttention3D(in_channels: int = 96,
                 window_size: int = (8, 7, 7),
                 num_head: int = 3,
                 qkv_bias: Optional[bool] = True,
                 qk_scale: Optional[float] = None,
                 attn_kepp_prob: Optional[float] = 1.0,
                 proj_keep_prob: Optional[float] = 1.0)

Window based multi-head self attention (W-MSA) module with relative position bias. It supports both of shifted and non-shifted window.

- base: nn.Cell

**Parameters:**

- in_channels (int): Number of input channels.
- window_size (tuple[int]): The depth length, height and width of the window. Default: (8, 7, 7).
- num_head (int): Number of attention heads. Default: 3.
- qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True.
- qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
- attn_keep_prob (float, optional): Dropout keep ratio of attention weight. Default: 1.0.
- proj_keep_prob (float, optional): Dropout keep ratio of output. Deault: 1.0.

**Inputs:**

- `x` (Tensor) - Tensor of shape (B, N, C).
- `mask` (Tensor) - (0 / - inf) mask with shape of (num_windows, N, N) or None.



**Return:**

Tensor of shape (B, N, C), which is equal to the input **x**.


### SwinTransformerBlock3D

> class mindvideo.models.SwinTransformerBlock3D(embed_dim: int = 96,
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
                 norm_layer: str = 'layer_norm')

A Video Swin Transformer Block. The implementation of this block follows the paper "Video Swin Transformer".

- base: nn.Cell

**Parameters:**

- embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
- input_size (int | tuple(int)): input feature size. Default: (16, 56, 56).
- num_head (int): number of attention head of the current Swin3d block. Default: 3.
- window_size (int): window size of window attention. Default: (8, 7, 7).
- shift_size (tuple[int]): shift size for shifted window attention. Default: (4, 3, 3).
- mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
- qkv_bias (bool): if True, add a learnable bias to query, key,value. Default: True.
- qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set True. Default: None.
- keep_prob (float): dropout keep probability. Default: 1.0.
- attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.0.
- droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 1.0.
- act_layer (nn.Cell): activation layer. Default: nn.GELU.
- norm_layer (nn.Cell): normalization layer. Default: 'layer_norm'.


**Inputs:**

- **x** (Tensor) - Input feature of shape (B, D, H, W, C).
- **mask_matrix** (Tensor) - Attention mask for cyclic shift.


**Return:**

Tensor of shape (B, D, H, W, C)


### PatchMerging

> class mindvideo.models.PatchMerging(dim: int = 96,
                 norm_layer: str = 'layer_norm')

Patch Merging Layer.

- base: nn.Cell

**Parameters:**

- dim (int): Number of input channels.
- norm_layer (nn.Cell): Normalization layer. Default: nn.LayerNorm


**Inputs:**

- **x** (Tensor) - Input feature of shape (B, D, H, W, C).


**Return:**

Tensor of shape (B, D, H/2, W/2, 2*C)


### SwinTransformerStage3D

> class mindvideo.models.SwinTransformerStage3D(embed_dim=96,
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
                 downsample=PatchMerging)

A basic Swin Transformer layer for one stage.

- base: nn.Cell

**Parameters:**

- embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
- input_size (tuple[int]): input feature size. Default. (16, 56, 56).
- depth (int): depth of the current Swin3d stage. Default: 2.
- num_head (int): number of attention head of the current Swin3d stage. Default: 3.
- window_size (int): window size of window attention. Default: (8, 7, 7).
- mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
- qkv_bias (bool): if qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: Truee
- qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set. Default: None.
- keep_prob (float): dropout keep probability. Default: 1.0.
- attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.
- droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 0.8.
- norm_layer(string): normalization layer. Default: 'layer_norm'.
- downsample (nn.Cell | None, optional): downsample layer at the end of swin3d stage. Default: PatchMerging.


**Inputs:**

A video feature of shape (N, D, H, W, C)

**Return:**

Tensor of shape (N, D, H / 2, W / 2, 2 * C)


### PatchEmbed3D

> class mindvideo.models.PatchEmbed3D(input_size=(16, 224, 224), patch_size=(2, 4, 4),
                 in_channels=3, embed_dim=96, norm_layer='layer_norm', patch_norm=True)

Video to Patch Embedding.

- base: nn.Cell

**Parameters:**

- input_size (tuple[int]): Input feature size.
- patch_size (int): Patch token size. Default: (2,4,4).
- in_channels (int): Number of input video channels. Default: 3.
- embed_dim (int): Number of linear projection output channels. Default: 96.
- norm_layer (nn.Module, optional): Normalization layer. Default: None.
- patch_norm (bool): if True, add normalization after patch embedding. Default: True.


**Inputs:**

An original Video tensor in data format of 'NCDHW'.

**Return:**

An embedded tensor in data format of 'NDHWC'.


### SwinTransformer3D

> class mindvideo.models.SwinTransformer3D(input_size=(16, 56, 56),
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
                 norm_layer='layer_norm')

Video Swin Transformer backbone. A mindspore implementation of : `Video Swin Transformer` http://arxiv.org/abs/2106.13230

- base: nn.Cell

**Parameters:**

- input_size (int | tuple(int)): input feature size. Default: (16, 56, 56).
- embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
- depths (tuple[int]): depths of each Swin3d stage. Default: (2, 2, 6, 2).
- num_heads (tuple[int]): number of attention head of each Swin3d stage. Default: (3, 6, 12, 24).
- window_size (int): window size of window attention. Default: (8, 7, 7).
- mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
- qkv_bias (bool): if qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: True.
- qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set. Default: None.
- keep_prob (float): dropout keep probability. Default: 1.0.
- attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.
- droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 0.8.
- norm_layer (string): normalization layer. Default: 'layer_norm'.

**Inputs:**

- **x** (Tensor) - Tensor of shape 'NDHWC'.

**Return:**

Tensor of shape 'NCDHW'.


### Swin3D

> class mindvideo.models.Swin3D(input_size=(16, 56, 56),
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
                 norm_layer='layer_norm')

Constructs a swin3d architecture corresponding to `Video Swin Transformer <http://arxiv.org/abs/2106.13230>`.

- base: nn.Cell

**Parameters:**

- num_classes (int): The number of classification. Default: 400.
- patch_size (int): Patch size used by window attention. Default: (2, 4, 4).
- window_size (int): Window size used by window attention. Default: (8, 7, 7).
- embed_dim (int): Embedding dimension of the featrue generated from patch embedding layer. Default: 96.
- depths (int): Depths of each stage in Swin3d Tiny module. Default: (2, 2, 6, 2).
- num_heads (int): Numbers of heads of each stage in Swin3d Tiny module. Default: (3, 6, 12, 24).
- representation_size (int): Feature dimension of the last layer in backbone. Default: 768.
- droppath_keep_prob (float): The drop path keep probability. Default: 0.9.
- input_size (int | tuple(int)): Input feature size. Default: (32, 224, 224).
- in_channels (int): Input channels. Default: 3.
- mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
- qkv_bias (bool): If qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: True.
- qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
- keep_prob (float): Dropout keep probability. Default: 1.0.
- attn_keep_prob (float): Keeping probability for attention dropout. Default: 1.0.
- norm_layer (string): Normalization layer. Default: 'layer_norm'.
- patch_norm (bool): If True, add normalization after patch embedding. Default: True.
- pooling_keep_dim (bool): Specifies whether to keep dimension shape the same as input feature. Default: False.
- head_bias (bool): Specifies whether the head uses a bias vector. Default: True.
- head_activation (Union[str, Cell, Primitive]): Activate function applied in the head. Default: None.
- head_keep_prob (float): Head's dropout keeping rate, between [0, 1]. Default: 0.5.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`


### swin3d_t

> def mindvideo.models.swin3d_t(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 96,
             depths: int = (2, 2, 6, 2),
             num_heads: int = (3, 6, 12, 24),
             representation_size: int = 768,
             droppath_keep_prob: float = 0.9)

Video Swin Transformer Tiny (swin3d-T) model.

**Parameters:**

num_classes (int): Number of categories.
patch_size (int): Size of swin3d patch segmentation.
window_size (int): Size of swin3d window.
embed_dim (int): Dimension output by the patch embedding.
depths (int): Depth of each stage.
num_heads (int): Number of heads in window attention.
representation_size (int): Size of features output at the last layer of backbone.
droppath_keep_prob (float): Retetion probability of drop path.

**Returns:**

swin3d_t: nn.Cell


### swin3d_s

> def mindvideo.models.swin3d_s(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 96,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (3, 6, 12, 24),
             representation_size: int = 768,
             droppath_keep_prob: float = 0.9)

Video Swin Transformer Small (swin3d-S) model.

**Parameters:**

num_classes (int): Number of categories.
patch_size (int): Size of swin3d patch segmentation.
window_size (int): Size of swin3d window.
embed_dim (int): Dimension output by the patch embedding.
depths (int): Depth of each stage.
num_heads (int): Number of heads in window attention.
representation_size (int): Size of features output at the last layer of backbone.
droppath_keep_prob (float): Retetion probability of drop path.

**Returns:**

swin3d_s: nn.Cell

### swin3d_b

> def mindvideo.models.swin3d_b(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 128,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (4, 8, 16, 32),
             representation_size: int = 1024,
             droppath_keep_prob: float = 0.7)

Video Swin Transformer Base (swin3d-B) model.

**Parameters:**

num_classes (int): Number of categories.
patch_size (int): Size of swin3d patch segmentation.
window_size (int): Size of swin3d window.
embed_dim (int): Dimension output by the patch embedding.
depths (int): Depth of each stage.
num_heads (int): Number of heads in window attention.
representation_size (int): Size of features output at the last layer of backbone.
droppath_keep_prob (float): Retetion probability of drop path.

**Returns:**

swin3d_b: nn.Cell

### swin3d_l

> def mindvideo.models.swin3d_l(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 192,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (6, 12, 24, 48),
             representation_size: int = 1536,
             droppath_keep_prob: float = 0.9)

Video Swin Transformer Large (swin3d-L) model.

**Parameters:**

num_classes (int): Number of categories.
patch_size (int): Size of swin3d patch segmentation.
window_size (int): Size of swin3d window.
embed_dim (int): Dimension output by the patch embedding.
depths (int): Depth of each stage.
num_heads (int): Number of heads in window attention.
representation_size (int): Size of features output at the last layer of backbone.
droppath_keep_prob (float): Retetion probability of drop path.

**Returns:**

swin3d_l: nn.Cell


### GroupNorm3d

> class mindvideo.models.GroupNorm3d(num_groups, num_channels, eps=1e-05, affine=True, gamma_init='ones', beta_init='zeros')

modify from mindspore.nn.GroupNorm, add depth

- base: nn.Cell

**Parameters:**

num_groups (int): Number of groups to be divided along the channel dimension.
num_channels (int): Number of channels.
eps(float): The value added to the denominator.
affine (bool): When set to True, a learnable affine transformation parameter is added to the layer.
gamma_init (str): Method of initializing the gamma parameter.
beta_init (str): Method of initializing the beta parameter.

**Return:**

Tensor, output tensor.


### VistrCom

> class mindvideo.models.VistrCom(name: str = 'ResNet50',
                 train_embeding: bool = True,
                 num_queries: int = 360,
                 num_pos_feats: int = 64,
                 num_frames: int = 36,
                 temperature: int = 10000,
                 normalize: bool = True,
                 scale: float = None,
                 hidden_dim: int = 384,
                 d_model: int = 384,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: int = 0.1,
                 activation: str = "relu",
                 normalize_before: bool = False,
                 return_intermediate_dec: bool = True,
                 aux_loss: bool = True,
                 num_class: int = 41)

Vistr Architecture.

- base: nn.Cell

**Parameters:**

name (str): The type of ResNet.
train_embeding (bool): Whether to train embeding or not.
num_queries （int）: Number of instances.
num_pos_feats (int): The encoding length of each dimension.
num_frames (int)： Number of frames.
temperature (int): Coefficient.
normalize (bool): Whether to normalize. If True, normalize.
scale (float): Coefficient.
hidden_dim (int): Dimensions required by the input vector in the encoder.
d_model (int): Number of expected features entered by the backbone
nhead (int): Number of heads in multi head attention.
num_encoder_layers (int): Layer number of encoders.
num_decoder_layers (int): Layer number of decoders.
dim_feedforward (int): Dimensions of the feedforward network model in backbone
dropout (int): Value of dropout.
activation(str): Activation function.
normalize_before (bool): Whether is normalized or not before.
return_intermediate_dec (bool): Whether to return intermediate output
aux_loss (bool): Whether to calculate the loss of the middle layer.
num_class (int): Number of categories.

**Return:**

Tensor, output tensor.


### BlockX3D

> class mindvideo.models.BlockX3D(in_channel,
                 out_channel,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 inflate: int = 2,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 block_idx: int = 0,
                 se_ratio: float = 0.0625,
                 use_swish: bool = True,
                 drop_connect_rate: float = 0.0,
                 bottleneck_factor: float = 2.25,
                 **kwargs)

BlockX3D 3d building block for X3D.

- base: ResidualBlock3D

**Parameters:**

- in_channel (int): Input channel.
- out_channel (int): Output channel.
- conv12(nn.Cell, optional): Block that constructs first two conv layers. It can be `Inflate3D`, `Conv2Plus1D` or other custom blocks, this block should construct a layer where the name of output feature channel size is `mid_channel` for the third conv layers. Default: Inflate3D.
- inflate (int): Whether to inflate kernel.
- spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
- down_sample (nn.Module | None): DownSample layer. Default: None.
- block_idx (int): the id of the block.
- se_ratio (float | None): The reduction ratio of squeeze and excitation unit. If set as None, it means not using SE unit. Default: None.
- use_swish (bool): Whether to use swish as the activation function before and after the 3x3x3 conv. Default: True.
- drop_connect_rate (float): dropout rate. If equal to 0.0, perform no dropout.
- bottleneck_factor (float): Bottleneck expansion factor for the 3x3x3 conv.

**Return:**

Tensor, output tensor.


### ResNetX3D

> class mindvideo.models.ResNetX3D(block: Optional[nn.Cell],
                 layer_nums: Tuple[int],
                 stage_channels: Tuple[int],
                 stage_strides: Tuple[Tuple[int]],
                 drop_rates: Tuple[float],
                 down_sample: Optional[nn.Cell] = Unit3D,
                 bottleneck_factor: float = 2.25)

X3D backbone definition.

- base: ResNet3D

**Parameters:**

- block (Optional[nn.Cell]): THe block for network.
- layer_nums (list): The numbers of block in different layers.
- stage_channels (Tuple[int]): Output channel for every res stage.
- stage_strides (Tuple[Tuple[int]]): Stride size for ResNet3D convolutional layer.
- drop_rates (list): list of the drop rate in different blocks. The basic rate at which blocks are dropped, linearly increases from input to output blocks.
- down_sample (Optional[nn.Cell]): Residual block in every resblock, it can transfer the input feature into the same channel of output. Default: Unit3D.
- bottleneck_factor (float): Bottleneck expansion factor for the 3x3x3 conv.
- fc_init_std (float): The std to initialize the fc layer(s).

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor, output tensor.


### X3DHead

> class mindvideo.models.X3DHead(pool_size,
                 input_channel,
                 out_channel=2048,
                 num_classes=400,
                 dropout_rate=0.5)

x3d head architecture.

- base: nn.Cell

**Parameters:**

- input_channel (int): The number of input channel.
- out_channel (int): The number of inner channel. Default: 2048.
- num_classes (int): Number of classes. Default: 400.
- dropout_rate (float): Dropout keeping rate, between [0, 1]. Default: 0.5.

**Return:**

Tensor


### x3d

> class mindvideo.models.x3d(block: Type[BlockX3D],
                 depth_factor: float,
                 num_frames: int,
                 train_crop_size: int,
                 num_classes: int,
                 dropout_rate: float,
                 bottleneck_factor: float = 2.25,
                 eval_with_clips: bool = False)

x3d architecture. Christoph Feichtenhofer. "X3D: Expanding Architectures for Efficient Video Recognition." https://arxiv.org/abs/2004.04730

- base: nn.Cell

**Parameters:**

- block (Type[BlockX3D]): The block of X3D.
- depth_factor (float): Depth expansion factor.
- num_frames (int): The number of frames of the input clip.
- train_crop_size (int): The spatial crop size for training.
- num_classes (int): the channel dimensions of the output.
- dropout_rate (float): dropout rate. If equal to 0.0, perform no dropout.
- bottleneck_factor (float): Factor of bottleneck.
- eval_with_clips (bool): If evalidate with clips, eval_with_clips is True.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`


### x3d_m

> def mindvideo.models.x3d_m(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 2.2,
          num_frames: int = 16,
          train_crop_size: int = 224,
          eval_with_clips: bool = False)

X3D middle model.

**Parameters:**

- num_classes (int): the channel dimensions of the output.
- dropout_rate (float): dropout rate. If equal to 0.0, perform no dropout.
- depth_factor (float): Depth expansion factor.
- num_frames (int): The number of frames of the input clip.
- train_crop_size (int): The spatial crop size for training.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`


### x3d_s

> def mindvideo.models.x3d_s(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 2.2,
          num_frames: int = 13,
          train_crop_size: int = 160,
          eval_with_clips: bool = False)

X3D small model.

**Parameters:**

- num_classes (int): the channel dimensions of the output.
- dropout_rate (float): dropout rate. If equal to 0.0, perform no dropout.
- depth_factor (float): Depth expansion factor.
- num_frames (int): The number of frames of the input clip.
- train_crop_size (int): The spatial crop size for training.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`


### x3d_xs

> def mindvideo.models.x3d_xs(num_classes: int = 400,
           dropout_rate: float = 0.5,
           depth_factor: float = 2.2,
           num_frames: int = 4,
           train_crop_size: int = 160,
           eval_with_clips: bool = False)

X3D x-small model.

**Parameters:**

- num_classes (int): the channel dimensions of the output.
- dropout_rate (float): dropout rate. If equal to 0.0, perform no dropout.
- depth_factor (float): Depth expansion factor.
- num_frames (int): The number of frames of the input clip.
- train_crop_size (int): The spatial crop size for training.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`


### x3d_l

> def mindvideo.models.x3d_l(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 5.0,
          num_frames: int = 16,
          train_crop_size: int = 312,
          eval_with_clips: bool = False)

X3D large model.

**Parameters:**

- num_classes (int): the channel dimensions of the output.
- dropout_rate (float): dropout rate. If equal to 0.0, perform no dropout.
- depth_factor (float): Depth expansion factor.
- num_frames (int): The number of frames of the input clip.
- train_crop_size (int): The spatial crop size for training.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, CLASSES_{out})`