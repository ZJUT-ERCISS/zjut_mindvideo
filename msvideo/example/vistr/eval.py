import os
from pickletools import optimize
import cv2
import json
import math
from PIL import Image
import mindspore
import numpy as np
from mindspore import nn
from mindspore.dataset.transforms.c_transforms import Compose
from mindspore.dataset.vision import c_transforms as T
from mindspore.dataset.vision import py_transforms as T_p
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, ops
from torch import sigmoid
from mindvision.msvideo.models.vistr import vistr_r50
from mindvision.msvideo.utils import misc
from mindvision.msvideo.dataset.transforms import video_normalize

import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F

import pynvml

context.set_context(mode=context.PYNATIVE_MODE)
context.set_context(device_id=0)

cast = ops.Cast()
sub = ops.Sub()
div = ops.Div()

transform = T_p.ToTensor()
# transform2 = Compose([T.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224])])
# transforms_list = [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# mean = Tensor(mean, mindspore.float32).view(-1, 1, 1)
# std = Tensor(std, mindspore.float32).view(-1, 1, 1)
mean = np.array(mean)
std = np.array(std)




img_path = "/usr/dataset/VOS/valid/valid_all_frames/JPEGImages/"
ann_path = "/usr/dataset/VOS/annotations/valid.json"

num_frames = 36
num_ins = 10

folder = img_path
videos = json.load(open(ann_path, 'rb'))['videos']
# ann = json.load(open(ann_path, 'rb'))['annotations']


ms_model = vistr_r50(pretrained=False)
param_dict = load_checkpoint("/home/zgz/VisTR/vistr_r50.ckpt")
load_param_into_net(ms_model, param_dict)


vis_num = len(videos)
result = []

# resize_bilinear = ops.ResizeBilinear((720,1280))
ms_sigmoid = ops.Sigmoid()
concat = ops.Concat(axis=0)
softmax = nn.Softmax(axis=-1)
expand_dims = ops.ExpandDims()
input_perm = (1, 2, 0)
input_perm2 = (2, 0, 1)
transpose = ops.Transpose()


# i = 1
# tgt_category_id = 2
result = []
logits_list = []
masks_list = []
score_list = []
for i in range(vis_num):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("Process video: ",i)
    print(meminfo.used)
    id_ = videos[i]['id']
    length = videos[i]['length']
    file_names = videos[i]['file_names']
    clip_num = math.ceil(length/num_frames)

    img_set = []
    if length<num_frames:
        clip_names = file_names*(math.ceil(num_frames/length))
        clip_names = clip_names[:num_frames]
    else:
        clip_names = file_names[:num_frames]
    if len(clip_names)==0:
        continue
    if len(clip_names)<num_frames:
        clip_names.extend(file_names[:num_frames-len(clip_names)])
    for k in range(num_frames):
        im = Image.open(os.path.join(folder,clip_names[k]))
        h = im.size[1]
        w = im.size[0]
        width = int((im.size[0]*300) / im.size[1])
        height = 300
        im = im.resize((width, height), resample=Image.Resampling.BILINEAR)
        im = transform(im)
        im = (im - mean[:, None, None]) / std[:, None, None]
        im = Tensor(im, mindspore.float32)
        im = expand_dims(im, 0)
        img_set.append(im)
    img = concat(img_set)
    # images = Tensor(img, mindspore.float32)
    images = Tensor(img, mindspore.float32)
    images, masks = misc.nested_tensor_from_tensor_list(images)
    masks = cast(masks, mindspore.float32)
    pred, pred_mask = ms_model((images, masks))
    
    pred_logits = pred[-1, ..., :42]
    pred_boxes = pred[-1, ..., 42:]
    # np.save("/home/zgz/VisTR/.vscode/data/result.npy", pred_logits.asnumpy())
    pred_logits = softmax(pred_logits)[0, :, :-1]
    pred_boxes = pred_boxes[0]
    pred_masks = pred_mask[0]


    pred_masks = pred_masks.reshape(36, 10, pred_masks.shape[-2], pred_masks.shape[-1])
    resize_bilinear = ops.ResizeBilinear((h, w))
    pred_masks = resize_bilinear(pred_masks)
    pred_masks = ms_sigmoid(pred_masks).asnumpy()>0.5

    pred_masks = pred_masks[:length]
    pred_logits = pred_logits.reshape(num_frames, num_ins, pred_logits.shape[-1]).asnumpy()
    pred_logits = pred_logits[:length]
    pred_scores = np.max(pred_logits, axis=-1)
    pred_logits = np.argmax(pred_logits, axis=-1)
    for m in range(num_ins):
        if pred_masks[:,m].max()==0:
            continue
        score = pred_scores[:, m].mean()
        category_id = np.argmax(np.bincount(pred_logits[:,m]))
        instance = {'video_id':id_, 'score':float(score), 'category_id':int(category_id)}
        segmentation = []
        for n in range(length):
            if pred_scores[n,m]<0.001:
                segmentation.append(None)
            else:
                mask = (pred_masks[n,m]).astype('uint8')
                rle = mask_util.encode(np.array(mask[:,:,np.newaxis], order='F'))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                segmentation.append(rle)
        instance['segmentations'] = segmentation
        result.append(instance)
with open("/home/zgz/vision/.vscode/result/result1.json", 'w', encoding='utf-8') as f:
        json.dump(result,f)