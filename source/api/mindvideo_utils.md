## mindvideo.utils

### EvalLossMonitor

> class mindvideo.utils.EvalLossMonitor(model)

Monitor for loss in validation.

- base: Callback

**Parameters:**

- model(str): The model to monitor.

**Return:**

None


> def mindvideo.utils.EvalLossMonitor.epoch_begin(run_context)

Record time at the beginning of epoch.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


> def mindvideo.utils.EvalLossMonitor.epoch_end(run_context)

Print training info at the end of epoch.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


> def mindvideo.utils.EvalLossMonitor.step_begin(run_context)

Record time at the beginning of step.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


> def mindvideo.utils.EvalLossMonitor.step_end(run_context)

Print training info at the end of step.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


### ValAccMonitor

> class mindvideo.utils.ValAccMonitor(model: ms.Model,
                 dataset_val: ms.dataset,
                 num_epochs: int,
                 interval: int = 1,
                 eval_start_epoch: int = 1,
                 save_best_ckpt: bool = True,
                 ckpt_directory: str = "./",
                 best_ckpt_name: str = "best.ckpt",
                 metric_name: str = "Accuracy",
                 dataset_sink_mode: bool = True)

Monitors the train loss and the validation accuracy, after each epoch saves the best checkpoint file with highest validation accuracy.

- base: Callback

**Parameters:**

- model (ms.Model): The model to monitor.
- dataset_val (ms.dataset): The dataset that the model needs.
- num_epochs (int): The number of epochs.
- interval (int): Every how many epochs to validate and print information. Default: 1.
- eval_start_epoch (int): From which time to validate. Default: 1.
- save_best_ckpt (bool): Whether to save the checkpoint file which performs best. Default: True.
- ckpt_directory (str): The path to save checkpoint files. Default: './'.
- best_ckpt_name (str): The file name of the checkpoint file which performs best. Default: 'best.ckpt'.
- metric_name (str): The name of metric for model evaluation. Default: 'Accuracy'.
- dataset_sink_mode (bool): Whether to use the dataset sinking mode. Default: True.

**Raises:**

ValueError: If `interval` is not more than 1.

**Return:**

None


> def mindvideo.utils.ValAccMonitor.apply_eval()

Model evaluation, return validation accuracy.

**Parameters:**

None

**Return:**

Validation accuracy.


> def mindvideo.utils.ValAccMonitor.epoch_end(run_context)

After epoch, print train loss and val accuracy, save the best ckpt file with highest validation accuracy.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


> def mindvideo.utils.ValAccMonitor.end(run_context)

Print the best validation accuracy after network training.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


### SaveCallback

> class mindvideo.utils.SaveCallback(eval_model, ds_eval)

Callback for checkpoint saving.

- base: Callback

**Parameters:**

- model (ms.Model): The model to monitor.
- dataset_val (ms.dataset): The dataset that the model needs.
- num_epochs (int): The number of epochs.
- interval (int): Every how many epochs to validate and print information. Default: 1.
- eval_start_epoch (int): From which time to validate. Default: 1.
- save_best_ckpt (bool): Whether to save the checkpoint file which performs best. Default: True.
- ckpt_directory (str): The path to save checkpoint files. Default: './'.
- best_ckpt_name (str): The file name of the checkpoint file which performs best. Default: 'best.ckpt'.
- metric_name (str): The name of metric for model evaluation. Default: 'Accuracy'.
- dataset_sink_mode (bool): Whether to use the dataset sinking mode. Default: True.

**Return:**

None


> def mindvideo.utils.SaveCallback.step_end(run_context)

At the end of each step, save the maximum accuracy checkpoint.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


### LossMonitor

> class mindvideo.utils.LossMonitor(lr_init: Optional[Union[float, Iterable]] = None,
                 per_print_times: int = 1)

Loss Monitor for classification.

- base: Callback

**Parameters:**

- lr_init (Union[float, Iterable], optional): The learning rate schedule. Default: None.
- per_print_times (int): Every how many steps to print the log information. Default: 1.

**Return:**

None


> def mindvideo.utils.LossMonitor.epoch_begin(run_context)

Record time at the beginning of epoch.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


> def mindvideo.utils.LossMonitor.epoch_end(run_context)

Print training info at the end of epoch.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


> def mindvideo.utils.LossMonitor.step_begin(run_context)

Record time at the beginning of step.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


> def mindvideo.utils.LossMonitor.step_end(run_context)

Print training info at the end of step.

**Parameters:**

- run_context (RunContext): Context of the process running.

**Return:**

None


### ClassFactory

> class mindvideo.utils.ClassFactory()

Module class factory for builder.

**Parameters:**

None

**Return:**

None

> def mindvideo.utils.ClassFactory.register(cls, module_type=ModuleType.GENERAL, alias=None)

Register class into registry.

**Parameters:**

- module_type (ModuleType): Module type name, default: ModuleType.GENERAL.
- alias (str) : class alias, default: None.

**Returns:**

Wrapper.


> def mindvideo.utils.ClassFactory.wrapper(register_class)

Register class with wrapper function.

**Parameters:**

- register_class: Class which need to be register.

**Returns:**

Wrapper of register_class.


> def mindvideo.utils.ClassFactory.register_cls(cls, register_class, module_type=ModuleType.GENERAL, alias=None)

Register class with type name into registry.

**Parameters:**

- register_class: Class which need to be register.
- module_type(ModuleType): Module type name, default: ModuleType.GENERAL.
- alias(String): class name.

**Returns:**

register_class.


> def mindvideo.utils.ClassFactory.is_exist(cls, module_type, class_name=None)

Determine whether class name is in the current type group.

**Parameters:**

- module_type(ModuleType): Module type.
- class_name(string): Class name.

**Returns:**

Bool.


> def mindvideo.utils.ClassFactory.get_cls(cls, module_type, class_name=None)

Get class.

**Parameters:**

- module_type(ModuleType): Module type.
- class_name(String): class name.

**Returns:**

register_class.


> def mindvideo.utils.ClassFactory.get_instance_from_cfg(cls, cfg, module_type=ModuleType.GENERAL, default_args=None)

Get instance from configure.

**Parameters:**

- cfg(dict): Config dict which should at least contain the key "type".
- module_type(ModuleType): module type.
- default_args(dict, optional) : Default initialization arguments.

**Returns:**

obj: The constructed object.


> def mindvideo.utils.ClassFactory.get_instance(cls, module_type=ModuleType.GENERAL, obj_type=None, args=None)

Get instance by ModuleType with object type.

**Parameters:**

- module_type(ModuleType): Module type. Default: ModuleType.GENERAL.
- obj_type(String): Class type.
- args(dict): Object arguments.

**Returns:**

obj: The constructed object.


### recur_list2tuple

> def mindvideo.utils.recur_list2tuple(d)

Transform list data in dict into tuple recursively.

**Parameters:**

d(list).

**Returns:**

Tuple.


### Config

> class mindvideo.utils.Config(*args, **kwargs)

A Config class is inherit from dict. Config class can parse arguments from a config file of yaml or a dict.

- base: dict

**Parameters:**

- args (list) : config file_names
- kwargs (dict) : config dictionary list

**Returns:**

None


> `def mindvideo.utils.Config.__getattr__(key)`

Get a object attr by `key`.

**Parameters:**

- key(str): the name of object attr.

**Returns:**

Attr of object that name is `key`.


> `def mindvideo.utils.Config.__setattr__(key, value)`

Set a object value `key` with `value`.

**Parameters:**

- key(str): The name of object attr.
- value: the `value` need to set to the target object attr.

**Returns:**

None


> `def mindvideo.utils.Config.__delattr__(key)`

Delete a object attr by its `key`.

**Parameters:**

- key(str): The name of object attr.

**Returns:**

None


> `def mindvideo.utils.Config.merge_from_dict(options)`

Merge options into config file.

**Parameters:**

- options(dict): dict of configs to merge from.

**Returns:**

None


> `def mindvideo.utils.Config._merge_into(a, b)`

Merge dict ``a`` into dict ``b``, values in ``a`` will overwrite ``b``.

**Parameters:**

- a(dict): The source dict to be merged into b.
- b(dict): The origin dict to be fetch keys from ``a``.

**Returns:**

dict: The modified dict of ``b`` using ``a``.


> `def mindvideo.utils.Config._file2dict(file_name=None)`

Convert config file to dictionary.

**Parameters:**

- file_name(str): Config file.

**Returns:**

dict


> `def mindvideo.utils.Config._dict2config(config, dic)`

Convert dictionary to config.

**Parameters:**

- config: Config object.
- dic(dict): dictionary.

**Returns:**

None


### ActionDict

> class mindvideo.utils.ActionDict() 

Argparse action to split an option into `KEY=VALUE` from on the first `=` and append to dictionary. List options can be passed as comma separated values.

i.e. 'KEY=Val1,Val2,Val3' or with explicit brackets 'KEY=[Val1,Val2,Val3]'.

- base: Action

**Parameters:**

None

**Returns:**

None


> `def mindvideo.utils.ActionDict._parse_int_float_bool(val)`

Convert string val to int or float or bool or do nothing.

**Parameters:**

- val (str) : Value String

**Returns:**

Int or float or bool or str.


> `def mindvideo.utils.ActionDict.find_next_comma(val_str)`

Find the position of next comma in the string.

**Note:**

'(' and ')' or '[' and']' must appear in pairs or not exist.

**Parameters:**

- val (str) : Value String

**Returns:**

Int.


> `def mindvideo.utils.ActionDict._parse_value_iter(val)`

Convert string format as list or tuple to python list object or tuple object.

**Parameters:**

- val (str) : Value String

**Returns:**

List or tuple.

**Examples:**

```
>>> ActionDict._parse_value_iter('1,2,3')
[1,2,3]
>>> ActionDict._parse_value_iter('[1,2,3]')
[1,2,3]
>>> ActionDict._parse_value_iter('(1,2,3)')
(1,2,3)
>>> ActionDict._parse_value_iter('[1,[1,2],(1,2,3)')
[1, [1, 2], (1, 2, 3)]
```

### parse_args

> def mindvideo.utils.parse_args()

Parse arguments from `yaml` config file.

**Parameters:**

None

**Returns:**

object: arg parse object.


### gaussian_radius

> def mindvideo.utils.gaussian_radius(det_size, min_overlap=0.7)

Set label value of gt bbox within gaussian radius. Details of why using `gaussian_radius` can be found in paper: https://arxiv.org/abs/1808.01244.

**Parameters:**

- det_size (tuple[int]): Size of ground truth bounding box.
- min_overlap (float): Threshold of iou which is calculated by gt bbox and bbox that is within radius. Default: 0.7.

**Returns:**

Minimum radius that meet the overlap condition.


### gaussian2d

> def mindvideo.utils.gaussian2d(shape, sigma=1)

Gaussian2d heatmap.

**Parameters:**

- shape (tuple[int]): x, y radius of gaussian dustribution.
- sigma (int, float): Standard deviation of gaussian dustribution. Default: 1.

**Returns:**

Gaussian heatmap mask.


### draw_umich_gaussian

> def mindvideo.utils.draw_umich_gaussian(heatmap, center, radius, k=1)

Draw umich gaussian, apply gaussian distribution to heatmap.

**Parameters:**

- heatmap (numpy.ndarray): Heatmap.
- center (sequence[int]): Center of gaussian mask.
- radius (int, float): Radius of gaussian mask.
- k (int, float): Multiplier for gaussian mask values.

**Returns:**

Heatmap.


### draw_msra_gaussian

> def mindvideo.utils.draw_msra_gaussian(heatmap, center, sigma)

Draw msra gaussian, apply gaussian distribution to heatmap.

**Parameters:**

- heatmap (numpy.ndarray): Heatmap.
- center (sequence[int]): Center of gaussian mask.
- sigma (int, float): Standard deviation of gaussian dustribution.

**Returns:**

Heatmap.


### compute_mask

> def mindvideo.utils.compute_mask(depth, height, width, window_size, shift_size)

Calculate attention mask for SW-MSA.

**Parameters:**

- depth, height, width (int): Numbers of depth, height, width dimensions.
- window_size (Tuple(int)): Input window size.
- shift_size (Tuple(int)): Input shift_size.

**Returns:**

Tensor, attention mask.


### get_mask

> def mindvideo.utils.get_mask(tensor)

Get img masks.

**Parameters:**

Tensor.

**Returns:**

Tensor.


### _max_by_axis

> def mindvideo.utils._max_by_axis(the_list)

**Parameters:**

List[List[int]].

**Returns:**

List[int].


### nested_tensor_from_tensor_list

> def mindvideo.utils.nested_tensor_from_tensor_list(tensor_list, split=True)

Normalize the input image data.

**Parameters:**

- tensor_list (Tensor)
- split (bool)

**Returns:**

Two tensors.


### cal_for_frames

> def mindvideo.utils.cal_for_frames(video_path)

Calculate optical flow using a list of frames.

**Parameters:**

video_path (string): Path to video.

**Returns:**

List.


### cal_for_video

> def mindvideo.utils.cal_for_frames(video_path)

Calculate optical flow of a video.

**Parameters:**

video_path (string): Path to video.

**Returns:**

List.


### compute_tvl1

> def mindvideo.utils.compute_tvl1(prev, curr, bound=20)

Compute the TV-L1 optical flow.

**Parameters:**

- prev: previous frame.
- curr: current frame.

**Returns:**

array

### save_flow

> def mindvideo.utils.save_flow(video_flows, flow_path, save_format='jpg')

Save video flows in specified format.

**Parameters:**

- video_flows (obj): object of video flow
- flow_path (str): The path where saves the optical flow.
- save_format (str): Optical flow save format, can be 'npy' or 'jpg'. Default: 'jpg'.

**Returns:**

None


### extract_flow

> def mindvideo.utils.extract_flow(video_path, flow_path, save_format='jpg')

Extract flow from video frames.

**Parameters:**

- video_path (str): The path of video. If `video_path` is a file directory, the function will extract optical flow from jpeg images in the directory. Else if `video_path` is a video, then extract optical flow frame by frame.
- flow_path (str): The path where saves the optical flow.
- save_format (str): Optical flow save format, can be 'npy' or 'jpg'. Default: 'jpg'.

**Returns:**

None

**Example:**

```
>>> vpath = "./path_to_video"
>>> save_path = "./path_to_saved_flow"
>>> extract_flow(vpath, save_path)
```


### round_width

> def mindvideo.utils.round_width(width, multiplier, min_width=8, divisor=8)

Round width of filters based on width multiplier.

**Parameters:**

- width (int): the channel dimensions of the input.
- multiplier (float): the multiplication factor.
- min_width (int): the minimum width after multiplication.
- divisor (int): the new width should be dividable by divisor.

**Returns:**

Round width of filters: Int


### drop_path

> def mindvideo.utils.drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False)

Stochastic Depth per sample.

**Parameters:**

- x (Tensor): Input feature.
- drop_prob(float): The probabilit of dropping.
- training(bool): Determine whether the model is under training.

**Returns:**

Tensor


### reisze_mean

> def mindvideo.utils.reisze_mean(data_dir, save_dir=None, height=240, width=320, interpolation='bilinear', norm=True)

Calculate mean of resized video frames.

**Parameters:**

- data_dir (str): The directory of videos, the file structure should be like this:

```
|-- data_dir
    |-- class1
        |-- video1-1
        |-- video1-2
        ...
    |-- class2
        |-- video2-1
        |-- video2-2
```

- save_dir (Union[str, None]): The directory where saves the resized mean. If None, this function will not save it to disk.
- height (int): Height of resized video frames.
- width (int): Width of reiszed video frames.
- interpolation (str): Method of resize the frames, it can be 'bilinear', 'nearest', 'linear', 'bicubic'. Default: 'bilinear'.
- norm (bool): Whether to normalize resized frames, if True, the resize mean will divided by 255.

**Returns:**

resized mean (numpy.ndarray): Resized mean of video frames in shape of (height, width, 3).

**Example:**

```
>>> vmean = reisze_mean(data_dir="/home/publicfile/UCF101/train",
>>>                     save_dir="./",
>>>                     height=128,
>>>                     width=128)
>>> print(vmean.shape)
```


### six_padding

> def mindvideo.utils.six_padding(padding)

Convert padding list into a tuple of 6 integer.
If padding is an int, returns `(padding, padding, padding, padding, padding, padding)`,
If padding's length is 3, returns `(padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])`,
If padding's length is 6, returns `(padding[0], padding[1], padding[2], padding[3], padding[4], padding[5])`,

**Parameters:**

- padding(Union[int, tuple, list]): Padding list that has the length of 1, 3 or 6.

**Returns:**

Tuple of shape (6,).


### TaskAccuracy

> class mindvideo.utils.TaskAccuracy(label_format='one_hot')

Calculates the accuracy for classification and multilabel data.
The accuracy class has two local variables, the correct number and the total number of samples, that are used to compute the frequency with which `y_pred` matches `y`. This frequency is ultimately returned as the accuracy: an idempotent operation that simply divides the correct number by the total number.

**Parameters:**

- eval_type (str): The metric to calculate the accuracy over a dataset. Supports 'classification' and 'multilabel'. 'classification' means the dataset label is single. 'multilabel' means the dataset has multiple labels. Default: 'classification'.
- label_format (str): The format of output label.

**Return:**

None

**Examples:**

```
>>> import numpy as np
>>> import mindspore
>>> from mindspore import nn, Tensor
>>>
>>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
>>> y = Tensor(np.array([[1, 0], [1, 0], [0, 1]]), mindspore.float32)
>>> metric = nn.Accuracy('one_hot')
>>> metric.clear()
>>> metric.update(x, y)
>>> accuracy = metric.eval()
```

> def mindvideo.utils.TaskAccuracy.update(*inputs)

Updates the local variables. For 'classification', if the index of the maximum of the predict value matches the label, the predict result is correct. For 'multilabel', the predict value match the label, the predict result is correct.

**Parameters:**

- inputs: Logits and labels. `y_pred` stands for logits, `y` stands for labels. `y_pred` and `y` must be a `Tensor`, a list or an array. 
For the 'one_hot' evaluation type, `y_pred` is a list of floating numbers in range :math:`[0, 1]` and the shape is :math:`(1, N, C)` in most cases (not strictly), where :math:`N` is the number of cases and :math:`C` is the number of categories. `y` must be in one-hot format that shape is :math:`(1, N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`. 
For 'single' evaluation type, `y` is not one-hot format :match:'(1, N)`.

**Raises:**

ValueError: If the number of the inputs is not 2.


### limit_window_size

> def mindvideo.utils.limit_window_size(input_size, window_size, shift_size)

Limit the window size and shift size for window W-MSA and SW-MSA. If window size is larger than input size, we don't partition or shift windows.

**Parameters:**

- input_size (tuple[int]): Input size of features. E.g. (16, 56, 56).
- window_size (tuple[int]): Target window size. E.g. (8, 7, 7).
- shift_size (int): depth of video. E.g. (4, 3, 3).

**Returns:**

Tuple[int], limited window size and shift size.


### window_partition

> def mindvideo.utils.window_partition(features, window_size)

Window partition function for Swin Transformer.

**Parameters:**

- features: Original features of shape (B, D, H, W, C).
- window_size (tuple[int]): Window size.

**Returns:**

Tensor of shape (B * num_windows, window_size * window_size, C).


### window_reverse

> def mindvideo.utils.window_reverse(windows, window_size, batch_size, depth, height, width)

Window reverse function for Swin Transformer.

**Parameters:**

- windows: Partitioned features of shape (B*num_windows, window_size, window_size, C).
- window_size (tuple[int]): Window size.
- batch_size (int): Batch size of video.
- depth (int): depth of video.
- height (int): Height of video.
- width (int): Width of video.

**Returns:**

Tensor of shape (B, D, H, W, C).
