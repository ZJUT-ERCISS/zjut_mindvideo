## mindvideo.loss

### GatherFeature

> class mindvideo.loss.GatherFeature()

Gather feature at specified position.

- base: nn.Cell

**Parameters:**

None

**Return:**

Tensor, feature at spectified position


### TransposeGatherFeature

> class mindvideo.loss.TransposeGatherFeature()

Transpose and gather feature at specified position

- base: nn.Cell

**Parameters:**

None

**Return:**

Tensor, feature at spectified position


### RegLoss

> class mindvideo.loss.RegLoss(mode='l1')

Warpper for regression loss.

- base: nn.Cell

**Parameters:**

- mode(str): L1 or Smoothed L1 loss. Default: "l1"

**Return:**

Tensor, regression loss.


### CenterNetMultiPoseLoss

> class mindvideo.loss.CenterNetMultiPoseLoss(reg_loss, hm_weight, wh_weight, off_weight, reg_offset, reid_dim, nid, batch_size)

Warpper for regression loss.

- base: nn.Cell

**Parameters:**

- reg_loss (str): Regression loss, it can be L1 loss or Smooth L1 loss: (['l1', 'sl1']). Default='l1'.
- hm_weight (int): Loss weight for keypoint heatmaps. Default=1.
- wh_weight (int): Loss weight for bounding box size. Default=0.1.
- off_weight (int): Loss weight for keypoint local offsets. Default=1.
- reg_offset (bool): Whether to use regress local offset. Default=True.
- reid_dim (int): Feature embed dim. Default=128.
- nID (int): Totoal number of identities in dataset. Default=14455.
- batch_size (int): Number of imgs.

**Return:**

Tensor, total loss.


### FocalLoss

> class mindvideo.loss.FocalLoss(alpha=2, beta=4)

nn.Cell warpper for focal loss.

- base: nn.Cell

**Parameters:**

- alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
- beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples. Default: 4.

**Return:**

Tensor, focal loss.


### DiceLoss

> class mindvideo.loss.DiceLoss()

Compute the DICE loss, similar to generalized IOU for masks

- base: nn.Cell

**Parameters:**

None

**Return:**

Tensor, DICE loss


### SetCriterion

> class mindvideo.loss.SetCriterion(num_classes, matcher, weight_dict, eos_coef, aux_loss)

vistr loss contains loss_labels, loss_masks and loss_boxes.

- base: nn.LossBase

**Parameters:**

- num_classes(int): Types of segmented objects.
- matcher(cell): Match predictions to GT.
- weight_dict(dict): Weights for different losses.
- eos_coef(float): Background class weights.
- aux_loss(bool): wether or not to computer aux loss.

**Return:**

Tensor, vistr loss


### SigmoidFocalLoss

> class mindvideo.loss.SigmoidFocalLoss()

Compute the sigmoid focal loss.

- base: nn.Cell

**Parameters:**

- alpha(float):Default: 0.25.
- gamma(float):Default: 2.

**Return:**

Tensor, sigmoid focal loss