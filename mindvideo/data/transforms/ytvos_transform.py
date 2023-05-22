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
"""ytvos dataset transforms functions."""
import random
import PIL
from numpy import random as rand
import numpy as np
from PIL import Image
import cv2
import mindspore.dataset.transforms.py_transforms as trans
from mindvideo.utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class RandomHorizontalFlip(trans.PyTensorOperation):
    """
    Flip every frame of the video ,bboxes and masks with a given probability.
    Args:
        prob (float): probability of the image being flipped. Default: 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, masks, resize_shape, label, valid):
        if random.random() < self.p:
            return hflip(img, boxes, masks, resize_shape, label, valid)
        return img, boxes, masks, resize_shape, label, valid


def hflip(clip, boxes, masks, resize_shape, label, valid):
    """
    flip img, boxes and masks
    """
    flipped_image = []
    for image in clip:
        # h x w x c
        image = np.array(image)
        image = np.flip(image, 1)
        image = Image.fromarray(np.uint8(image))
        flipped_image.append(image)

    w = clip[0].size[0]

    boxes = boxes.copy()
    boxes = boxes[:, [2, 1, 0, 3]] * \
        np.array([-1, 1, -1, 1]) + np.array([w, 0, w, 0])

    masks = masks[..., ::-1]

    return flipped_image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class ResizeShape(trans.PyTensorOperation):
    """
    resize img and boxes with at the given size
    Args:
       size(Union[tuple[int]): Desired output size after resize.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes, masks, resize_shape, label, valid):
        return resize(img, boxes, masks, resize_shape, self.size, label, valid, max_size=None)


@ClassFactory.register(ModuleType.PIPELINE)
class RandomResize(trans.PyTensorOperation):
    """
    resize img and boxes with at the given size
    Args:
       size(Union[tuple[int]): Desired output size after resize.
       max_size(int): Limit the length after resize. Default:None
    """

    def __init__(self, sizes, max_size=None):
        # assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, boxes, masks, resize_shape, label, valid):
        size = random.choice(self.sizes)
        # print(size)
        return resize(img, boxes, masks, resize_shape, size, label, valid, self.max_size)


def resize(clip, boxes, masks, resize_shape, size, label, valid, max_size=None):
    """
    resize img and boxes, then save the resize_shape.size can be min_size (scalar) or (w, h) tuple
    """
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(clip[0].size, size, max_size)
    rescaled_image = []
    # T_resize = Resize(size)
    for image in clip:
        image = image.resize(size, resample=PIL.Image.Resampling.BILINEAR)
        rescaled_image.append(image)

    ratios = tuple(float(s) / float(s_orig)
                   for s, s_orig in zip(rescaled_image[0].size, clip[0].size))
    ratio_width, ratio_height = ratios

    # boxes = target["boxes"]
    scaled_boxes = boxes * \
        np.array([ratio_width, ratio_height,
                  ratio_width, ratio_height])

    w, h = size
    resize_shape = np.append(resize_shape, (h, w))
    return rescaled_image, scaled_boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class PhotometricDistort(trans.PyTensorOperation):
    """
    photometric distortion
    """

    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, clip, boxes, masks, resize_shape, label, valid):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, boxes, masks, resize_shape, label, valid = self.rand_brightness(img, boxes, masks, resize_shape, label, valid)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, boxes, masks, resize_shape, label, valid = distort(img, boxes, masks, resize_shape, label, valid)
            img, boxes, masks, resize_shape, label, valid = self.rand_light_noise(img, boxes, masks, resize_shape, label, valid)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class RandomContrast(trans.PyTensorOperation):
    """
    random contrast on img
    Args:
       lower(float): smallest random value.Default: 0.5
       upper(float): largest random value. Default: 1.5
    """

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, masks, resize_shape, label, valid):

        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class ConvertColor(trans.PyTensorOperation):
    """
    Change image color space.
    Args:
       current(str): current color space.Default: 'BGR'
       transform(str): largest random value. Default: 'HSV'
    """

    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes, masks, resize_shape, label, valid):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class RandomSaturation(trans.PyTensorOperation):
    """
    random saturation on the second channel
    Args:
        lower(float): smallest random value.Default: 0.5
        upper(float): largest random value. Default: 1.5
    """

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, masks, resize_shape, label, valid):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class RandomHue(trans.PyTensorOperation):
    """
    Adjust the hue of RGB images by a random factor.
    Args:
        delta(float):The value to randomly increase or decrease the Hue of image.Default:18.0
    """
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes, masks, resize_shape, label, valid):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class RandomBrightness(trans.PyTensorOperation):
    """
    Adjust the brightness of images by a random factor.
    Args:
        delta(float):the value to random adjust_brightness.Default:32
    """
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes, masks, resize_shape, label, valid):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class RandomLightingNoise(trans.PyTensorOperation):
    """
    Randomly transform channels.6 transformation modes are set, one is randomly selected,
    and the order of the three BGR channels is changed
    """
    def __init__(self):
        self.perms = [(0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0)]

    def __call__(self, image, boxes, masks, resize_shape, label, valid):
        if rand.randint(2):
            num = rand.randint(0, 5)
            swap = self.perms[int(num)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class SwapChannels(trans.PyTensorOperation):
    """
    swap channels.
    """
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


@ClassFactory.register(ModuleType.PIPELINE)
class Compose(trans.PyTensorOperation):
    """
    compose transform
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, masks, resize_shape, label, valid):
        for t in self.transforms:
            image, boxes, masks, resize_shape, label, valid = t(image, boxes, masks, resize_shape, label, valid)
        return image, boxes, masks, resize_shape, label, valid

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {0}".format(t)
        format_string += "\n)"
        return format_string


@ClassFactory.register(ModuleType.PIPELINE)
class RandomSizeCrop(trans.PyTensorOperation):
    """
    random crop img and boxes.save region data.
    Args:
        min_size(int):
        max_size(int):
    """
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def get_params(self, img, output_size):
        """
        get region
        """
        w, h = img.size[0], img.size[1]
        th, tw = output_size

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, img: PIL.Image.Image, boxes, masks, resize_shape, label, valid):
        w = random.randint(self.min_size, min(img[0].width, self.max_size))
        h = random.randint(self.min_size, min(img[0].height, self.max_size))
        region = self.get_params(img[0], [h, w])
        return crop(img, boxes, masks, region, resize_shape, label, valid)


def crop(clip, boxes, masks, region, resize_shape, label, valid):
    """
    crop on img and boxes.save region data.
    """
    min_value = 0
    max_value = float('inf')
    cropped_image = []
    i, j, h, w = region
    for image in clip:
        image = np.array(image)
        image = image[i:i + h, j:j + w, :]
        image = Image.fromarray(np.uint8(image))
        cropped_image.append(image)

    max_size = np.array([w, h], dtype='float32')
    cropped_boxes = boxes - np.array([j, i, j, i])
    cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
    a, b, _ = cropped_boxes.shape
    max_size = np.tile(max_size, (a, b, 1))

    cropped_boxes = np.minimum(cropped_boxes, max_size)
    cropped_boxes = np.clip(cropped_boxes, min_value, max_value)

    boxes = cropped_boxes.reshape(-1, 4)

    # masks = masks[:, i:i + h, j:j + w]
    resize_shape = np.append(resize_shape, (i, h, j, w))

    return cropped_image, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class YtvosToTensor(trans.PyTensorOperation):
    """
    convert Image to numpy.array.
    """
    def __call__(self, clip, boxes, masks, resize_shape, label, valid):
        img = []
        for im in clip:
            # c h w array
            if isinstance(im, Image.Image):
                img.append(np.array(im).transpose(2, 0, 1)/255.0)
            elif isinstance(im, np.ndarray):
                img.append(im.transpose(2, 0, 1)/255.0)
            else:
                raise ValueError(f"images should be in type `PIL` or `np.ndarray`, but got {type(im)}")
        return img, boxes, masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class Normalize(trans.PyTensorOperation):
    """
    normalize on img and boxes.
    Args:
        mean(list):mean for normalize.
        std(list):standard deviation for narmalize.
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, clip, boxes, masks, resize_shape, label, valid):
        image = []
        for im in clip:
            im = (im - self.mean[:, None, None]) / self.std[:, None, None]
            im = np.expand_dims(im, axis=0)
            image.append(im)

        h, w = image[0].shape[-2:]
        # image = self.concat(image)
        image = np.concatenate(image, axis=0)

        boxes = self.box_xyxy_to_cxcywh(boxes)
        boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        return image, boxes, masks, resize_shape, label, valid

    def box_xyxy_to_cxcywh(self, x):
        x0 = np.expand_dims(x[..., 0], axis=-1)
        y0 = np.expand_dims(x[..., 1], axis=-1)
        x1 = np.expand_dims(x[..., 2], axis=-1)
        y1 = np.expand_dims(x[..., 3], axis=-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return np.concatenate(b, axis=-1)


@ClassFactory.register(ModuleType.PIPELINE)
class RescaleShape(trans.PyTensorOperation):
    """
    resize img,boxes and masks.save resize_shape.
    Args:
        h(int): height of resize shape.
        w(int): width of resize shape.
    """
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, clip, boxes, masks, resize_shape, label, valid):
        rescaled_image = []
        for im in clip:
            im = im.resize((self.w, self.h), resample=PIL.Image.Resampling.BILINEAR)
            rescaled_image.append(im)
        resize_shape = np.append(resize_shape, (self.h, self.w))
        # 记录instance的数量
        resize_shape = np.append(resize_shape, masks.shape[0])

        ratios = tuple(float(s) / float(s_orig)
                       for s, s_orig in zip(rescaled_image[0].size, clip[0].size))
        ratio_width, ratio_height = ratios

        # boxes = target["boxes"]
        scaled_boxes = boxes * \
            np.array([ratio_width, ratio_height,
                      ratio_width, ratio_height])
        # 补boxes
        shape1 = 360 - scaled_boxes.shape[0]
        blank_boxes = np.zeros([shape1, 4], dtype=np.float32)
        scaled_boxes = np.concatenate((scaled_boxes, blank_boxes), axis=0)

        # 补masks
        shape2 = 360 - masks.shape[0]
        blank_masks = np.zeros([shape2, masks.shape[1], masks.shape[2]], dtype=np.bool_)
        scaled_masks = np.concatenate((masks, blank_masks), axis=0)

        return rescaled_image, scaled_boxes, scaled_masks, resize_shape, label, valid


@ClassFactory.register(ModuleType.PIPELINE)
class ConcatBlank(trans.PyTensorOperation):
    """
    zero padding on label and valid.
    """
    def __call__(self, clip, boxes, masks, resize_shape, label, valid):
        shape1 = 360 - label.shape[0]
        blank_label = np.zeros((shape1), dtype=np.int32)
        scaled_label = np.concatenate((label, blank_label), axis=0)

        shape2 = 360 - valid.shape[0]
        blank_valid = np.zeros((shape2), dtype=np.int32)
        scaled_valid = np.concatenate((valid, blank_valid), axis=0)

        return clip, boxes, masks, resize_shape, scaled_label, scaled_valid


def make_coco_transforms():
    # default train transforms
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    return Compose([
        ConcatBlank(),
        RandomHorizontalFlip(),
        RandomResize(scales, max_size=800),
        PhotometricDistort(),
        Compose([
            RandomResize([400]),
            RandomSizeCrop(384, 600),
            # RandomResize([300], max_size=540)  # for r50
            # T.RandomResize([280], max_size=504),#for r101
        ]),
        RescaleShape(168, 300),
        YtvosToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
