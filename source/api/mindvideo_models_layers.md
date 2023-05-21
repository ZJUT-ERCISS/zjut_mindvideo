## mindvideo.model.layers

### AdaptiveAvgPool3D

> class mindvideo.model.layers.AdaptiveAvgPool3D(output_size)

Applies a 3D adaptive average pooling over an input tensor which is typically of shape`(N, C, D_{in}, H_{in}, W_{in})` and output shape`(N, C, D_{out}, H_{out}, W_{out})`. where `N` is batch size. `C` is channel number.


- base: nn.Cell

**Parameters:**

- output_size(Union[int, tuple[int]]): The target output size of the form D x H x W. Can be a tuple (D, H, W) or a single number D for a cube D x D x D.

**Inputs:**

- x(Tensor): The input Tensor in the form of :math:`(N, C, D_{in}, H_{in}, W_{in})`.

**Return:**

Tensor, the pooled Tensor in the form of :math:`(N, C, D_{out}, H_{out}, W_{out})`.


### AvgPool3D

> class mindvideo.model.layers.AvgPool3D(kernel_size=(1, 1, 1), strides=(1, 1, 1))

Average pooling for 3d feature.


- base: nn.Cell

**Parameters:**

- kernel_size(Union[int, tuple[int]]): The size of kernel window used to take the average value, Default: (1, 1, 1).
- strides(Union[int, tuple[int]]): The distance of kernel moving. Default: (1, 1, 1).

**Inputs:**

- x(Tensor): The input Tensor.

**Return:**

Tensor, the pooled Tensor.


### GlobalAvgPooling3D

> class mindvideo.model.layers.GlobalAvgPooling3D(keep_dims: bool = True)

A module of Global average pooling for 3D video features.


- base: nn.Cell

**Parameters:**

- keep_dims (bool): Specifies whether to keep dimension shape the same as input feature. E.g. `True`. Default: False

**Return:**

Tensor, output tensor.


### MultiIou

> class mindvideo.model.layers.MultiIou()

Multi iou calculating Iou between pred boxes and gt boxes.


- base: nn.Cell

**Parameters:**

None

**Inputs:**

- pred_bbox(tensor):predicted bbox.
- gt_bbox(tensor):Ground Truth bbox.

**Return:**

Tensor, iou of predicted box and ground truth box.


### BoxIou

> class mindvideo.model.layers.BoxIou()

calculate box iou

- base: nn.Cell

**Parameters:**

None

**Inputs:**

- boxes1(Tensor):[x0, y0, x1, y1] format
- boxes2(Tensor):[x0, y0, x1, y1] format

**Return:**

Tensor


### BoxIou

> class mindvideo.model.layers.BoxIou()

Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] format. Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2).

- base: nn.Cell

**Parameters:**

None

**Inputs:**

- boxes1(Tensor):[x0, y0, x1, y1] format
- boxes2(Tensor):[x0, y0, x1, y1] format

**Return:**

a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)


### ConvNormActivation

> class mindvideo.model.layers.ConvNormActivation(in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 has_bias: bool = False)

Convolution/Depthwise fused with normalization and activation blocks definition.

- base: nn.Cell

**Parameters:**

- in_planes (int): Input channel.
- out_planes (int): Output channel.
- kernel_size (int): Input kernel size.
- stride (int): Stride size for the first convolutional layer. Default: 1.
- groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
- norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution layer. Default: nn.BatchNorm2d.
- activation (nn.Cell, optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

**Return:**

Tensor, output tensor.


### Conv2dNormResAct

> class mindvideo.model.layers.Conv2dNormResAct(in_channels, out_channels, kernel_size, stride, padding, residual=False)

Convolution/Depthwise fused with normalization and activation blocks definition.

- base: nn.Cell

**Parameters:**

- in_channels (int): The channel number of the input tensor of the Conv2d layer.
- out_channels (int): The channel number of the output tensor of the Conv2d layer.
- kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
- stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
- padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the input.
- residual (bool): Whether the input value needs to be added.


**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.


### Conv2dTransPadBN

> class mindvideo.model.layers.Conv2dTransPadBN(in_channels, out_channels, kernel_size, stride, padding, output_padding=0)

Convolution/Depthwise fused with normalization and activation blocks definition.

- base: nn.Cell

**Parameters:**

- in_channels (int): The channel number of the input tensor of the Conv2d layer.
- out_channels (int): The channel number of the output tensor of the Conv2d layer.
- kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
- stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
- padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the input.
- output_padding (int): The number of padding of the output.


**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.


### C3DBackbone

> class mindvideo.model.layers.C3DBackbone(in_channel=3, kernel_size=(3, 3, 3))

C3D backbone. It works when the of input data is in the shape of :math:`(B, C, T, H, W)`.

- base: nn.Cell

**Parameters:**

- in_channel(int): Number of input data. Default: 3.
- kernel_size(Union[int, Tuple[int]]): Kernel size for every conv3d layer in C3D. Default: (3, 3, 3).

**Return:**

Tensor, infer output tensor.


### DeformConv2d

> class mindvideo.model.layers.DeformConv2d(inc, outc, kernel_size=3, stride=1, pad_mode='same', padding=0, has_bias=False, modulation=True)

Deformable convolution opertor.

- base: nn.Cell

**Parameters:**

- inc(int): Input channel.
- outc(int): Output channel.
- kernel_size (int): Convolution window. Default: 3.
- stride (int): The distance of kernel moving. Default: 1.
- padding (int): Implicit paddings size on both sides of the input. Default: 1.
- has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
- modulation (bool): If True, modulated defomable convolution (Deformable ConvNets v2). Default: True.

**Return:**

Tensor, detection of images(bboxes, score, keypoints and category id of each objects)

### _get_offset_base

> def mindvideo.model.layers._get_offset_base(offset_shape, stride)

Get base position index from deformable shift of each kernel element.

### _get_feature_by_index

> def mindvideo.model.layers._get_feature_by_index(x, p_h, p_w)

Gather feature by specified index.

### _regenerate_feature_map

> def mindvideo.model.layers._regenerate_feature_map(x_offset)

Get rescaled feature map which was enlarged by ks**2 times.


### ProbDropPath3D

> class mindvideo.model.layers.ProbDropPath3D(keep_prob)

Drop path per sample using a fixed probability. Use keep_prob param as the probability for keeping network units.

- base: nn.Cell

**Parameters:**

- keep_prob (int): Network unit keeping probability.
- ndim (int): Number of dropout features' dimension.

**Inputs:**

Tensor of ndim dimension.

**Return:**

A path-dropped tensor.


### DropoutDense

> class mindvideo.model.layers.DropoutDense(input_channel: int,
                 out_channel: int,
                 has_bias: bool = True,
                 activation: Optional[Union[str, nn.Cell]] = None,
                 keep_prob: float = 1.0)

Dropout + Dense architecture.

- base: nn.Cell

**Parameters:**

- input_channel (int): The number of input channel.
- out_channel (int): The number of output channel.
- has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
- activation (Union[str, Cell, Primitive]): activate function applied to the output. Eg. `ReLU`. Default: None.
- keep_prob (float): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of input. Default: 1.0.

**Return:**

Tensor, output tensor.


### FairMOTSingleHead

> class mindvideo.model.layers.FairMOTSingleHead(in_channel, head_conv=0, classes=100, kernel_size=3, bias_init=Zero())

Simple convolutional head, two conv2d layers will be created if head_conv > 0, else there is only one conv2d layer.

- base: nn.Cell

**Parameters:**

- in_channel(int): Channel size of input feature.
- head_conv(int): Channel size between two conv2d layers, there will be only one conv2d layer if head_conv equals 0. Default: 0.
- classes(int): Number of classes, channel size of output tensor.
- kernel_size(Union[int, tuple]): The kernel size of first conv2d layer.
- bias_init(Union[Tensor, str, Initializer, numbers.Number]): Bias initialization of last conv2d layer. The input value is the same as `mindspore.common.initializer.initializer`.

**Return:**

Tensor, the classification result.


### FairMOTMultiHead

> class mindvideo.model.layers.FairMOTMultiHead(heads, in_channel, head_conv=0, kernel_size=3)

Fairmot net multi-conv head, the combination of single heads.

- base: nn.Cell

**Parameters:**

- heads(dict): A dict contains name and output dimension of heads, the name is the key, and output dimension is the value. For fairmot, it must have 'hm', 'wh', 'id', 'reg' heads.
- in_channel(int): Channel size of input feature.
- head_conv(int): Channel size between two conv2d layers, there will be only one conv2d layer if head_conv equals 0. Default: 0.
- kernel_size(Union[int, tuple]): The kernel size of first conv2d layer.
- bias_init(Union[Tensor, str, Initializer, numbers.Number]): Bias initialization of last conv2d layer. The input value is the same as `mindspore.common.initializer.initializer`.

**Return:**

Tensor, the multi-head classification results.


### FeedForward

> class mindvideo.model.layers.FeedForward(in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0)

Feed Forward layer implementation.

- base: nn.Cell

**Parameters:**

- in_features (int): The dimension of input features.
- hidden_features (int): The dimension of hidden features. Default: None.
- out_features (int): The dimension of output features. Default: None
- activation (nn.Cell): Activation function which will be stacked on top of the
- normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
- keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

**Return:**

Tensor, output tensor.


### Hungarian

> class mindvideo.model.layers.Hungarian(dim)

Given a cost matrix, calculate the best assignment that cost the least. This ops now only support square matrix.

- base: nn.Cell

**Parameters:**

- dim (int): The size of the input square matrix.

**Inputs:**

x(Tensor): The input cost matrix.

**Returns:**

- Tensor[bool]: The best assignment, there can be multiple solutions.
- Tensor[int32]: The indices of row assignment.
- Tensor[int32]: The indices of column assignment.


> def mindvideo.model.layers.Hungarian.create_onehot(idx)

Calculate one hot vector according to input indice.

**Return:**

Tensor: One hot vector.


> def mindvideo.model.layers.Hungarian.get_assign(assign_matrix)

Make every row of assign matrix has at most one assignment.

**Return:**

Tensor: assign matrix.


> def mindvideo.model.layers.Hungarian.try_assign(x)

Try assignment, if succeed return the result.

**Return:**

Tensor: The best assignment, there can be multiple solutions.


### Inflate3D

> class mindvideo.model.layers.Inflate3D(in_channel: int,
                 out_channel: int,
                 mid_channel: int = 0,
                 stride: tuple = (1, 1, 1),
                 kernel_size: tuple = (3, 3, 3),
                 conv2_group: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm3d,
                 activation: List[Optional[Union[nn.Cell, str]]] = (nn.ReLU, None),
                 inflate: int = 1)

Inflate3D block definition.

- base: nn.Cell

**Parameters:**

- in_channel (int):  The number of channels of input frame images.
- out_channel (int):  The number of channels of output frame images.
- mid_channel (int): The number of channels of inner frame images.
- kernel_size (tuple): The size of the spatial-temporal convolutional layer kernels.
- stride (Union[int, Tuple[int]]): Stride size for the second convolutional layer. Default: 1.
- conv2_group (int): Splits filter into groups for the second conv layer, in_channels and out_channels must be divisible by the number of groups. Default: 1.
- norm (Optional[nn.Cell]): Norm layer that will be stacked on top of the convolution layer. Default: nn.BatchNorm3d.
- activation (List[Optional[Union[nn.Cell, str]]]): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer.
Default: nn.ReLU, None.
- inflate (int): Whether to inflate two conv3d layers and with different kernel size.

**Return:**

Tensor, output tensor.


### HungarianMatcher

> class mindvideo.model.layers.HungarianMatcher(num_frames: int = 36, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1)

This class computes an assignment between the targets and the predictions of the network.
For efficiency reasons, the targets don't include the no_object. Because of this, in general,there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are un-matched (and thus treated as non-objects).

- base: nn.Cell

**Parameters:**

- num_frames: The number of frames.
- cost_class: This is the relative weight of the classification error in the matching cost.
- cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost.
- cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost.

**Return:**

Tensor, output tensor.


> def mindvideo.model.layers.HungarianMatcher._CxcywhToXyxy(x)

CxCyWH_to_XYXY

**Parameters:**

x(tensor):last dimension is four

**Return:**

Tensor, last dimension is four


### MaskHeadSmallConv

> class mindvideo.model.layers.MaskHeadSmallConv(dim, fpn_dims, context_dim)

MaskHeadSmallConv:Simple convolutional head, using group norm. Upsampling is done using a FPN approach.

- base: nn.Cell

**Parameters:**

- dim(int):Size of the embeddings (dimension of the transformer) + Number of attention heads inside the transformer's attentions.
- fpn_dims(dict):three dims for FPN.
- context_dim(int):Size of the embeddings (dimension of the transformer).

**Inputs:**

- x(Tensor):sequence of encoded features
- bbox_mask(Tensor): the attention softmax of bbox
- fpns(list[Tensor]):images features without positional encoding

**Return:**

Tensor.


### MaxPool3D

> class mindvideo.model.layers.MaxPool3D(kernel_size=1,
                                        strides=1,
                                        pad_mode="VALID",
                                        pad_list=0,
                                        ceil_mode=None,
                                        data_format="NCDHW")

3D max pooling operation. Applies a 3D max pooling over an input Tensor which can be regarded as a composition of 3D planes.

- base: nn.Cell

**Parameters:**

-  kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value, is an int number that represents depth, height and width of the kernel, or a tuple of three int numbers that represent depth, height and width respectively. Default: 1.
- strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents the depth, height and width of movement are both strides, or a tuple of three int numbers that represent depth, height and width of movement respectively. Default: 1.
- pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive. Default: "valid".
- pad_list (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2], pad[3], pad[4] and pad[5] correspondingly.
- ceil_mode (bool): Whether to use ceil instead of floor to calculate output shape. Only effective in "pad" mode. When "pad_mode" is "pad" and "ceil_mode" is "None", "ceil_mode" will be set as "False". Default: None.
- data_format (str) : The optional value for data format. Currently only support 'NCDHW'. Default: 'NCDHW'.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`. Data type must be float16 or float32.

**Return:**

Tensor, with shape :math:`(N, C, D_{out}, H_{out}, W_{out})`. Has the data type with `x`.


### Maxpool3DwithPad

> class mindvideo.model.layers.Maxpool3DwithPad(kernel_size,
                                            padding,
                                            strides=1,
                                            pad_mode='SYMMETRIC')

3D max pooling with padding operation.

- base: nn.Cell

**Parameters:**

- kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value, is an int number that represents depth, height and width of the kernel, or a tuple of three int numbers that represent depth, height and width respectively. Default: 1.
- padding (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2], pad[3], pad[4] and pad[5] correspondingly.
- strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents not only the depth, height of movement but also the width of movement,, or a tuple of three int numbers that represent depth, height and width of movement respectively. Default: 1.
- pad_mode (str): The optional value of pad mode is "same" or "valid" or "SYMMETRIC". Default: "SYMMETRIC".

**Return:**

Tensor, output tensor.


### MHAttentionMsp

> class mindvideo.model.layers.MHAttentionMsp(query_dim, hidden_dim, num_heads, dropout=0.0, bias=True)

This is a 2D attention module, which only returns the attention softmax (no multiplication by value).

- base: nn.Cell

**Parameters:**

- query_dim(int): The number of channels in input sequence.
- hidden_dim(int): The number of channels in output sequence.
- num_heads(int): parallel attention heads.
- dropout(float):The dropout rate.Default: 0.0.
- bias(bool): Whether the Conv layer has a bias parameter. Default: True.

**Return:**

Tensor, output tensor.


### MLP

> class mindvideo.model.layers.MLP(input_dim, hidden_dim, output_dim, num_layers)

Very simple multi-layer perceptron (also called FFN).

- base: nn.Cell

**Parameters:**

- input_dim(int): The number of channels in the input space.
- hidden_dim(int): The number of extra channels
- output_dim(int): The number of channels in the output space.
- num_layers(int): The number of layers in the mlp

**Return:**

tensor, one tensor


### linear

> def mindvideo.model.layers.linear(input_arr, weight, bias=None)

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

**Parameters:**

- Input: :math:`(N, *, in_features)` N is the batch size, `*` means any number of additional dimensions
- Weight: :math:`(out_features, in_features)`
- Bias: :math:`(out_features)`
- Output: :math:`(N, *, out_features)`

**Return:**

Tensor.


### MultiheadAttention

> class mindvideo.model.layers.MultiheadAttention(embed_dim, num_heads, dropout=0.)

multi head attention

- base: nn.Cell

**Parameters:**

- embed_dim(int): total dimension of the model
- num_heads(int): parallel attention heads
- dropout(float): a Dropout layer on attn_output_weights.Default:0.

**Return:**

tensor


### ResidualBlockBase

> class mindvideo.model.layers.ResidualBlockBase(in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None)

ResNet residual block base definition.

- base: nn.Cell

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


### ResidualBlock

> class mindvideo.model.layers.ResidualBlock(in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None)

ResNet residual block definition.

- base: nn.Cell

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


### ResNet

> class mindvideo.model.layers.ResNet(block: Type[Union[ResidualBlockBase, ResidualBlock]],
                                    layer_nums: List[int],
                                    group: int = 1,
                                    base_width: int = 64,
                                    norm: Optional[nn.Cell] = None)

ResNet architecture.

- base: nn.Cell

**Parameters:**

- block (Type[Union[ResidualBlockBase, ResidualBlock]]): THe block for network.
- layer_nums (list): The numbers of block in different layers.
- group (int): The number of Group convolutions. Default: 1.
- base_width (int): The width of per group. Default: 64.
- norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

**Return:**

Tensor of shape :math:`(N, 2048, 7, 7)`


### ResidualBlockBase3D

> class mindvideo.model.layers.ResidualBlockBase3D(in_channel: int,
                                                out_channel: int,
                                                mid_channel: int = 0,
                                                conv12: Optional[nn.Cell] = Inflate3D,
                                                group: int = 1,
                                                base_width: int = 64,
                                                norm: Optional[nn.Cell] = None,
                                                down_sample: Optional[nn.Cell] = None,
                                                **kwargs)

ResNet3D residual block base definition.

- base: nn.Cell

**Parameters:**

- in_channel (int): Input channel.
- out_channel (int): Output channel.
- conv12(nn.Cell, optional): Block that constructs first two conv layers. It can be `Inflate3D`, `Conv2Plus1D` or other custom blocks, this block should construct a layer where the name of output feature channel size is `mid_channel` for the third conv layers. Default: Inflate3D.
- group (int): Group convolutions. Default: 1.
- base_width (int): Width of per group. Default: 64.
- norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
- down_sample (nn.Cell, optional): Downsample structure. Default: None.
- **kwargs(dict, optional): Key arguments for "conv12", it can contain "stride", "inflate", etc.

**Return:**

Tensor, output tensor.


### ResidualBlock3D

> class mindvideo.model.layers.ResidualBlock3D(in_channel: int,
                 out_channel: int,
                 mid_channel: int = 0,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 activation: List[Optional[Union[nn.Cell, str]]] = (nn.ReLU, None),
                 down_sample: Optional[nn.Cell] = None,
                 **kwargs)

ResNet3D residual block definition.

- base: nn.Cell

**Parameters:**

- in_channel (int): Input channel.
- out_channel (int): Output channel.
- mid_channel (int): Inner channel.
- conv12(nn.Cell, optional): Block that constructs first two conv layers. It can be `Inflate3D`, `Conv2Plus1D` or other custom blocks, this block should construct a layer where the name of output feature channel size is `mid_channel` for the third conv layers. Default: Inflate3D.
- group (int): Group convolutions. Default: 1.
- base_width (int): Width of per group. Default: 64.
- norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
- activation (List[Optional[Union[nn.Cell, str]]]): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU, None.
- down_sample (nn.Cell, optional): Downsample structure. Default: None.
- **kwargs(dict, optional): Key arguments for "conv12", it can contain "stride", "inflate", etc.

**Return:**

Tensor, output tensor.


### ResNet3D

> class mindvideo.model.layers.ResNet3D(block: Optional[nn.Cell],
                 layer_nums: Tuple[int],
                 stage_channels: Tuple[int] = (64, 128, 256, 512),
                 stage_strides: Tuple[Tuple[int]] = ((1, 1, 1),
                                                     (1, 2, 2),
                                                     (1, 2, 2),
                                                     (1, 2, 2)),
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = Unit3D,
                 **kwargs)

ResNet3D architecture.

- base: nn.Cell

**Parameters:**

- block (Optional[nn.Cell]): THe block for network.
- layer_nums (Tuple[int]): The numbers of block in different layers.
- stage_channels (Tuple[int]): Output channel for every res stage. Default: [64, 128, 256, 512].
- stage_strides (Tuple[Tuple[int]]): Strides for every res stage. Default:[[1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2]].
- group (int): The number of Group convolutions. Default: 1.
- base_width (int): The width of per group. Default: 64.
- norm (nn.Cell, optional): The module specifying the normalization layer to use. Default: None.
- down_sample(nn.Cell, optional): Residual block in every resblock, it can transfer the input feature into the same channel of output. Default: Unit3D.
- kwargs (dict, optional): Key arguments for "make_res_layer" and resblocks.

**Inputs:**

- **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, T_{in}, H_{in}, W_{in})`.


**Return:**

Tensor of shape :math:`(N, 2048, 7, 7, 7)`


### Roll3D

> class mindvideo.model.layers.Roll3D(shift)

Roll Tensors of shape (B, D, H, W, C).

- base: nn.Cell

**Parameters:**

- shift (tuple[int]): shift size for target rolling.

**Inputs:**

Tensor of shape (B, D, H, W, C).


**Return:**

Rolled Tensor.


### make_divisible

> def mindvideo.model.layers.make_divisible(v: float,
                   divisor: int,
                   min_value: Optional[int] = None)

It ensures that all layers have a channel number that is divisible by 8.

**Parameters:**

- v (int): original channel of kernel.
- divisor (int): Divisor of the original channel.
- min_value (int, optional): Minimum number of channels.

**Return:**

Number of channel.


### SqueezeExcite3D

> class mindvideo.model.layers.SqueezeExcite3D(dim_in, ratio, act_fn: Union[str, nn.Cell] = Swish)

Squeeze-and-Excitation (SE) block implementation.

- base: nn.Cell

**Parameters:**

- dim_in (int): the channel dimensions of the input.
- ratio (float): the channel reduction ratio for squeeze.
- act_fn (Union[str, nn.Cell]): the activation of conv_expand: Default: Swish.

**Return:**

Tensor.


### Swish

> class mindvideo.model.layers.Swish()

Swish activation function: x * sigmoid(x).

- base: nn.Cell

**Parameters:**

None

**Return:**

Tensor.


### Unit3D

> class mindvideo.model.layers.Unit3D(in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 pad_mode: str = 'pad',
                 padding: Union[int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 group: int = 1,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 norm: Optional[nn.Cell] = nn.BatchNorm3d,
                 pooling: Optional[nn.Cell] = None,
                 has_bias: bool = False)

Conv3d fused with normalization and activation blocks definition.

- base: nn.Cell

**Parameters:**

- in_channels (int):  The number of channels of input frame images.
- out_channels (int):  The number of channels of output frame images.
- kernel_size (tuple): The size of the conv3d kernel.
- stride (Union[int, Tuple[int]]): Stride size for the first convolutional layer. Default: 1.
- pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "pad".
- padding (Union[int, Tuple[int]]): Implicit paddings on both sides of the input x. If `pad_mode` is "pad" and `padding` is not specified by user, then the padding size will be  `(kernel_size - 1) // 2` for C, H, W channel.
- dilation (Union[int, Tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1
- group (int): Splits filter into groups, in_channels and out_channels must be divisible by the number of groups. Default: 1.
- activation (Optional[nn.Cell]): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.
- norm (Optional[nn.Cell]): Norm layer that will be stacked on top of the convolution layer. Default: nn.BatchNorm3d.
- pooling (Optional[nn.Cell]): Pooling layer (if not None) will be stacked on top of all the former layers. Default: None.
- has_bias (bool): Whether to use Bias.

**Return:**

Tensor, output tensor.


### TransformerDecoder

> class mindvideo.model.layers.TransformerDecoder(decoder_layers, norm=None, return_intermediate=False)

Transformer decoder is a stack of N decoder layers.

- base: nn.Cell

**Parameters:**

- decoder_layers(nn.cell):an instance of the TransformerDecoderLayer() class
- norm(nn.cell):the layer normalization component (optional).Default=None
- return_intermediate(bool):return intermediate result.Default=False

**Inputs:**

- tgt(tensor): the sequence to the decoder
- memory(tensor): the sequence from the last layer of the encoder
- tgt_key_padding_mask(tensor): the mask for the tgt keys per batch
- memory_key_padding_mask(tensor): he mask for the memory keys per batch
- pos(tensor): memory's encoded position
- query_pos(tensor): tgt's encoded position

**Return:**

Tensor.


### TransformerDecoderLayer

> class mindvideo.model.layers.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False)

Transformer decoder layer is made up of self-attn and feedforward network.

- base: nn.Cell

**Parameters:**

- d_model(int): the number of expected features in the input
- nhead(int): the number of heads in the multiheadattention models
- dim_feedfroward(int): the dimension of the feedforward network model.Default=2048
- dropout(float): the dropout value.Default=0.1
- activation(str): the activation function of the intermediate layer, can be a string ("relu" or "gelu") or a unary callable. Default="relu"
- normalize_before(bool): done normalize before decoderlayer. Default:False

**Inputs:**

- tgt(tensor): the sequence to the decoder
- memory(tensor): the sequence from the last layer of the encoder
- tgt_key_padding_mask(tensor): the mask for the tgt keys per batch
- memory_key_padding_mask(tensor): he mask for the memory keys per batch
- pos(tensor): memory's encoded position
- query_pos(tensor): tgt's encoded position

**Return:**

Tensor.


### TransformerEncoder

> class mindvideo.model.layers.TransformerEncoder(encoder_layers, norm=None)

Transformer encoder is a stack of N encoder layers.

- base: nn.Cell

**Parameters:**

- encoder_layers: an list of TransformerEncoderlayer class's instance
- norm: the layer normalization component

**Inputs:**

- src: the sequence to encoder
- src_key_padding_mask: the mask for the src key per batch
- pos: the sequence's encoder position

**Return:**

Tensor.


### TransformerEncoderLayer

> class mindvideo.model.layers.TransformerEncoder(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False)

Transformer encoder layer is made up of self-attn and feedforward network.

- base: nn.Cell

**Parameters:**

- d_model(int): the number of expected features in the input
- nhead(int): the number of heads in the multiheadattention models
- dim_feedfroward(int): the dimension of the feedforward network model.Default=2048
- dropout(float): the dropout value.Default=0.1
- activation(str): the activation function of the intermediate layer, can be a string ("relu" or "gelu") or a unary callable. Default="relu"
- normalize_before(bool): done normalize before decoderlayer.Default:False

**Inputs:**

- src: the sequence to encoder
- src_key_padding_mask: the mask for the src key per batch
- pos: the sequence's encoder position

**Return:**

Tensor.