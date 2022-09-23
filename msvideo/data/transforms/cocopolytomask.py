import mindspore
import numpy as np
from mindspore import ops, Tensor
from pycocotools import mask as coco_mask


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    length = len(segmentations)
    for polygons in segmentations:
        if not polygons:
            mask = np.zeros((height, width), dtype="uint8")
            mask = np.expand_dims(mask, axis=0)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = np.any(mask, axis=2)
            mask = np.array(mask[None, ...], dtype="uint8")
        masks.append(mask)
    if masks:
        masks = np.concatenate(masks, axis=0)
    else:
        masks = np.zeros((length, height, width), dtype="uint8")
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, inds, num_frames):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        # image_id = torch.tensor([image_id])
        image_id = Tensor([image_id], mindspore.int64)
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = []
        classes = []
        segmentations = []
        valid = []
        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            for j in range(num_frames):
                bbox = ann['bboxes'][frame_id-inds[j]]
                segm = ann['segmentations'][frame_id-inds[j]]
                clas = ann["category_id"]
                # for empty boxes
                if bbox is None:
                    bbox = [0, 0, 0, 0]
                    valid.append(0)
                    clas = 0
                else:
                    valid.append(1)
                boxes.append(bbox)
                segmentations.append(segm)
                classes.append(clas)
        boxes = np.array(boxes, dtype="float32").reshape(-1, 4)
        boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]
        boxes1 = np.expand_dims(np.clip(boxes[:, 0], 0, w), axis=1)
        boxes2 = np.expand_dims(np.clip(boxes[:, 1], 0, h), axis=1)
        boxes3 = np.expand_dims(np.clip(boxes[:, 2], 0, w), axis=1)
        boxes4 = np.expand_dims(np.clip(boxes[:, 3], 0, h), axis=1)
        boxes = np.concatenate([boxes1, boxes2, boxes3, boxes4], axis=1)
        classes = Tensor(classes, mindspore.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        classes = classes.asnumpy()
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["valid"] = np.array(valid, dtype="int32")
        if self.return_masks:
            target["masks"] = masks
        return target
