import PIL
import random
from numpy import random as rand
import numpy as np
from PIL import Image
from mindspore import Tensor, ops
import cv2
from mindspore.dataset.vision import py_transforms as T
import mindspore.dataset.transforms.py_transforms as trans
from msvideo.data.transforms.video_normalize import VideoNormalize


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, masks):
        if random.random() < self.p:
            return hflip(img, boxes, masks)
        return img, boxes, masks


def hflip(clip, boxes, masks):
    flipped_image = []
    for image in clip:
        # h x w x c
        image = np.array(image)
        image = np.flip(image, 1)
        image = Image.fromarray(np.uint8(image))
        flipped_image.append(image)

    w = clip[0].size[0]
    # h = clip[0].shape[1]

    boxes = boxes.copy()
    # boxes = boxes[:, [2, 1, 0, 3]] * \
    #     Tensor([-1, 1, -1, 1]) + Tensor([w, 0, w, 0])
    boxes = boxes[:, [2, 1, 0, 3]] * \
        np.array([-1, 1, -1, 1]) + np.array([w, 0, w, 0])

    # revierse = ops.ReverseV2(axis=[-1])
    masks = masks[..., ::-1]

    return flipped_image, boxes, masks


class ResizeShape(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes, masks):
        return resize(img, boxes, masks, self.size, max_size=None)


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        # assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, boxes, masks):
        size = random.choice(self.sizes)
        # print(size)
        return resize(img, boxes, masks, size, self.max_size)


def resize(clip, boxes, masks, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

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
        else:
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
    mask_list = []
    nearest = ops.ResizeNearestNeighbor((h, w))
    if masks.shape[0] > 0:
        # data = Tensor(masks, mindspore.float32)
        # data3 = data.expand_dims(axis=1)
        # out = nearest(data3)[:, 0]
        # out = out.asnumpy()
        # out = out > 0.5
        for mask in masks:
            im = Image.fromarray(mask)
            im = im.resize((w, h), resample=Image.Resampling.NEAREST)
            im = np.array(im)
            im = np.expand_dims(im, axis=0)
            mask_list.append(im)
        out = np.concatenate(mask_list, axis=0)
    else:
        out = np.zeros((masks.shape[0], h, w))
    return rescaled_image, scaled_boxes, out


class PhotometricDistort(object):
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

    def __call__(self, clip, boxes, masks):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, boxes, masks = self.rand_brightness(img, boxes, masks)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, boxes, masks = distort(img, boxes, masks)
            img, boxes, masks = self.rand_light_noise(img, boxes, masks)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, boxes, masks


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, masks):

        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, masks


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes, masks):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, masks


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, masks):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, boxes, masks


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes, masks):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, masks


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes, masks):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, masks


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes, masks):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, masks


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, masks):
        for t in self.transforms:
            image, boxes, masks = t(image, boxes, masks)
        return image, boxes, masks

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def get_params(self, img, output_size):
        w, h = img.size[0], img.size[1]
        th, tw = output_size

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h != th else 0
        j = random.randint(0, w - tw) if w != tw else 0
        return i, j, th, tw

    def __call__(self, img: PIL.Image.Image, boxes, masks):
        w = random.randint(self.min_size, min(img[0].width, self.max_size))
        h = random.randint(self.min_size, min(img[0].height, self.max_size))
        region = self.get_params(img[0], [h, w])
        return crop(img, boxes, masks, region)


def crop(clip, boxes, masks, region):
    min_value = 0
    max_value = float('inf')
    cropped_image = []
    i, j, h, w = region
    for image in clip:
        image = np.array(image)
        image = image[i:i + h, j:j + w, :]
        image = Image.fromarray(np.uint8(image))
        cropped_image.append(image)

    # i, j, h, w = region

    # should we do something wrt the original size?
    # target["size"] = np.array([h, w])

    # boxes = target["boxes"]
    max_size = np.array([w, h], dtype='float32')
    cropped_boxes = boxes - np.array([j, i, j, i])
    cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
    a, b, c = cropped_boxes.shape
    max_size = np.tile(max_size, (a, b, 1))
    # np.min(cropped_boxes, max_size)
    # cropped_boxes = torch.min(cropped_boxes, max_size)

    # cropped_boxes = Tensor(cropped_boxes, mindspore.float32)
    # max_size = Tensor(max_size, mindspore.float32)
    cropped_boxes = np.minimum(cropped_boxes, max_size)
    cropped_boxes = np.clip(cropped_boxes, min_value, max_value)

    boxes = cropped_boxes.reshape(-1, 4)

    masks = masks[:, i:i + h, j:j + w]

    return cropped_image, boxes, masks


class ToTensor(object):
    def __call__(self, clip, boxes, masks):
        img = []
        for im in clip:
            # c h w array
            img.append(T.ToTensor()(im))
        return img, boxes, masks


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, clip, boxes, masks):
        image = []
        for im in clip:
            im = (im - self.mean[:, None, None]) / self.std[:, None, None]
            im = np.expand_dims(im, axis=0)
            image.append(im)

        h, w = image[0].shape[-2:]
        # image = self.concat(image)
        image = np.concatenate(image, axis=0)

        boxes = self.box_xyxy_to_cxcywh(boxes)
        boxes = boxes / np.array([w, h, w, h], dtype='float32')
        masks = np.array(masks, dtype="float32")
        return image, boxes, masks

    def box_xyxy_to_cxcywh(self, x):

        x0 = np.expand_dims(x[..., 0], axis=-1)
        y0 = np.expand_dims(x[..., 1], axis=-1)
        x1 = np.expand_dims(x[..., 2], axis=-1)
        y1 = np.expand_dims(x[..., 3], axis=-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return np.concatenate(b, axis=-1)


# class default_trans(trans.PyTensorOperation):
#     def __init__(self):
#         self.cast = ops.Cast()
#         self.video = []


#     def __call__(self, path, label):
#         for im in path:
#             im = Image.open(im).convert('RGB')


def make_coco_transforms():
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    # scales = [480]
    return Compose([
        RandomHorizontalFlip(),
        RandomResize(scales, max_size=800),
        PhotometricDistort(),
        Compose([
            RandomResize([400]),
            RandomSizeCrop(384, 600),
            RandomResize([300], max_size=540)  # for r50
            # T.RandomResize([280], max_size=504),#for r101
        ]),
        normalize,
    ])

# def make_coco_transforms():
#     normalize = Compose([
#         ToTensor(),
#         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     return Compose([
#         RandomResize([400]),
#         RandomSizeCrop(384, 600),
#         RandomResize([300], max_size=540),
#         normalize,
#     ])
