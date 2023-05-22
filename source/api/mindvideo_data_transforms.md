## mindvideo.data.transforms

### VideoCenterCrop


> class mindvideo.data.transforms.VideoCenterCrop(size=(224, 224)):

Crop each frame of the input video at the center to the given size.
If input frame of video size is smaller than output size, input video will be padded with 0 before cropping.

- base: trans.PyTensorOperation

**Parameters:**

- size (Union[int, sequence]): The output size of the cropped image. If size is an integer, a square crop of size (size, size) is returned. If size is a sequence of length 2, it should be (height, width).Default:(224,224)

**Return:**

None

### VideoNormalize


> class mindvideo.data.transforms.VideoNormalize(mean, std):

VideoNormalize the input numpy.ndarray video of shape (C, T, H, W) with the specified mean and standard deviation.

- base: trans.PyTensorOperation

**Note:**

The values of the input image need to be in the range [0.0, 1.0]. If not so, call `VideoReOrder` and `VideoRescale` first.

**Parameters:**

- mean (Union[float, sequence]): list or tuple of mean values for each channel, arranged in channel order. The values must be in the range [0.0, 1.0]. If a single float is provided, it will be filled to the same length as the channel.
- std (Union[float, sequence]): list or tuple of standard deviation values for each channel, arranged in channel order. The values must be in the range (0.0, 1.0]. If a single float is provided, it will be filled to the same length as the channel.

**Return:**

None

### VideoRandomCrop

> class mindvideo.data.transforms.VideoRandomCrop(size):
    
Crop the given video sequences (t x h x w x c) at a random location.

**Parameters:**

- size (sequence or int): Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.

**Return:**

None

> def mindvideo.data.transforms.VideoNormalize.get_params(img, output_size)

Get parameters for ``crop`` for a random crop.

**Parameters:**

- img (PIL Image): Image to be cropped.
- output_size (tuple): Expected output size of the crop.

**Return:**

- tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.


### VideoRandomHorizontalFlip


> class mindvideo.data.transforms.VideoRandomHorizontalFlip(prob=0.5):

Flip every frame of the video with a given probability.

- base: trans.PyTensorOperation

**Parameters:**

- prob (float): probability of the image being flipped. Default: 0.5.

**Return:**

None


### VideoReOrder

> class mindvideo.data.transforms.VideoReOrder(new_order):

Rearrange the order of dims of data.

- base: trans.PyTensorOperation

**Parameters:**

- new_order(tuple), new_order of output.

**Return:**

None


### VideoRescale

> class mindvideo.data.transforms.VideoRescale(rescale=1 / 255.0, shift=0.0):

Rescale the input video frames with the given rescale and shift. This operator will rescale the input video with: output = image * rescale + shift.

- base: trans.PyTensorOperation

**Parameters:**

- rescale (float): Rescale factor.
- shift (float, str): Shift factor, if `shift` is a string, it should be the path to a `.npy` file with shift data in it.

**Return:**

None


### VideoReshape

> class mindvideo.data.transforms.VideoReshape(shape):

Reshape data.

- base: trans.PyTensorOperation

**Parameters:**

- shape(tuple), shape of output.

**Return:**

None


### VideoResize


> class mindvideo.data.transforms.VideoResize(size, interpolation="bilinear"):

Resize the given video sequences (t, h, w, c) at the given size.

- base: trans.PyTensorOperation

**Parameters:**

- size(Union[tuple[int], int]): Desired output size after resize.
- interpolation (str): TO DO. Default: "bilinear".

**Return:**

None



### VideoShortEdgeResize


> class mindvideo.data.transforms.VideoShortEdgeResize(new_order=(3, 0, 1, 2))):

Resize the given video sequences (t, h, w, x, c) at the given size. And make sure the smallest dimension in (h, w) is 256 pixels.

- base: trans.PyTensorOperation

**Parameters:**

- new_order(tuple), new_order of output.

**Return:**

None


### VideoToTensor


> class mindvideo.data.transforms.VideoToTensor(order=(3, 0, 1, 2)):

Convert the input video frames in type numpy.ndarray of shape (T, H, W, C) in the range [0, 255] to numpy.ndarray of shape (C, T, H, W) in the range [-1.0, 1.0] with the desired dtype.

- base: trans.PyTensorOperation

**Parameters:**

- new_order(tuple), new_order of output.

**Return:**

None


### RandomHorizontalFlip


> class mindvideo.data.transforms.RandomHorizontalFlip(p=0.5):

Flip every frame of the video ,bboxes and masks with a given probability.

- base: trans.PyTensorOperation

**Parameters:**

- size(int): Desired output size after resize.
- interpolation (str): TO DO. Default: "bilinear".

**Return:**

None

> def mindvideo.data.transforms.RandomHorizontalFlip.hflip(clip, boxes, masks, resize_shape, label, valid):

flip img, boxes and masks


### ResizeShape


> class mindvideo.data.transforms.ResizeShape(size):

Resize img and boxes with at the given size.

- base: trans.PyTensorOperation

**Parameters:**

- size(Union[tuple[int]): Desired output size after resize.

**Return:**

None


### RandomResize


> class mindvideo.data.transforms.RandomResize(sizes, max_size=None):

Resize img and boxes with at the given size.

- base: trans.PyTensorOperation

**Parameters:**

- size(Union[tuple[int]): Desired output size after resize.
- max_size(int): Limit the length after resize. Default:None

**Return:**

None

> def mindvideo.data.transforms.RandomResize.resize(clip, boxes, masks, resize_shape, size, label, valid, max_size=None):

resize img and boxes, then save the resize_shape.size can be min_size (scalar) or (w, h) tuple


### PhotometricDistort


> class mindvideo.data.transforms.PhotometricDistort():

photometric distortion

- base: trans.PyTensorOperation

**Parameters:**

None

**Return:**

None


### RandomContrast


> class mindvideo.data.transforms.RandomContrast(lower=0.5, upper=1.5):

random contrast on img

- base: trans.PyTensorOperation

**Parameters:**

- lower(float): smallest random value.Default: 0.5
- upper(float): largest random value. Default: 1.5

**Return:**

None


### ConvertColor


> class mindvideo.data.transforms.ConvertColor(current='BGR', transform='HSV'):

Change image color space.

- base: trans.PyTensorOperation

**Parameters:**

- current(str): current color space.Default: 'BGR'
- transform(str): largest random value. Default: 'HSV'

**Return:**

None


### RandomSaturation


> class mindvideo.data.transforms.RandomSaturation(lower=0.5, upper=1.5)

Random saturation on the second channel.

- base: trans.PyTensorOperation

**Parameters:**

- lower(float): smallest random value.Default: 0.5
- upper(float): largest random value. Default: 1.5

**Return:**

None


### RandomHue


> class mindvideo.data.transforms.RandomHue(delta=18.0)

Adjust the hue of RGB images by a random factor.

- base: trans.PyTensorOperation

**Parameters:**

- delta(float):The value to randomly increase or decrease the Hue of image.Default:18.0

**Return:**

None


### RandomBrightness


> class mindvideo.data.transforms.RandomBrightness(delta=32)

Adjust the brightness of images by a random factor.

- base: trans.PyTensorOperation

**Parameters:**

- delta(float):the value to random adjust_brightness.Default:32

**Return:**

None


### RandomLightingNoise


> class mindvideo.data.transforms.RandomLightingNoise()

Randomly transform channels.6 transformation modes are set, one is randomly selected, and the order of the three BGR channels is changed

- base: trans.PyTensorOperation

**Parameters:**

None

**Return:**

None


### SwapChannels


> class mindvideo.data.transforms.SwapChannels(swaps)

swap channels.

- base: trans.PyTensorOperation

**Parameters:**

- swap: int

**Return:**

None


### Compose


> class mindvideo.data.transforms.Compose(transforms)

compose transform

- base: trans.PyTensorOperation

**Parameters:**

- transforms(list): Data transforms.

**Return:**

None


### RandomSizeCrop


> class mindvideo.data.transforms.RandomSizeCrop(min_size: int, max_size: int)

random crop img and boxes. save region data.

- base: trans.PyTensorOperation

**Parameters:**

- min_size(int): the min size of cropped image.
- max_size(int): the max size of cropped image.

**Return:**

None


### Normalize


> class mindvideo.data.transforms.Normalize(mean, std)

normalize on img and boxes.

- base: trans.PyTensorOperation

**Parameters:**

- tmean(list):mean for normalize.
- std(list):standard deviation for narmalize.

**Return:**

None


### RescaleShape


> class mindvideo.data.transforms.RescaleShape(h, w)

resize img,boxes and masks.save resize_shape.

- base: trans.PyTensorOperation

**Parameters:**

- h(int): height of resize shape.
- w(int): width of resize shape.

**Return:**

None