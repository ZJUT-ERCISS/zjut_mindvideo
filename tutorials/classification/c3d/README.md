# Contents

- [Contents](#contents)
  - [C3D Description](#c3d-description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [Quick Start](#quick-start)
  - [Script Parameters](#script-parameters)
  - [Training Process](#training-process)
    - [Training](#training)
  - [Evaluation Process](#evaluation-process)
    - [Evaluation](#evaluation)
  - [visulization](#visulization)
  - [Model Description](#model-description)
    - [Performance](#performance)
      - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
  - [ModelZoo Homepage](#modelzoo-homepage)

## [C3D Description](#contents)

C3D model is widely used for 3D vision task. The construct of C3D network is similar to the common 2D ConvNets, the main difference is that C3D use 3D operations like Conv3D while 2D ConvNets are anentirely 2D architecture. To know more information about C3D network, you can read the original paper Learning Spatiotemporal Features with 3D Convolutional Networks.

## [Model Architecture](#contents)

C3D net has 8 convolution, 5 max-pooling, and 2 fully connected layers, followed by a softmax output layer. All 3D convolution kernels are 3 × 3 × 3 with stride 1 in both spatial and temporal dimensions. The 3D pooling layers are denoted from pool1 to pool5. All pooling kernels are 2 × 2 × 2, except for pool1 is 1 × 2 × 2. Each fully connected layer has 4096 output units.

## [Dataset](#contents)

Dataset used: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

- Description: UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 data set which has 50 action categories.

- Dataset size：13320 videos
    - Note：Use the official Train/Test Splits([UCF101TrainTestSplits](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)).
- Data format：rar
    - Note：Data will be processed in dataset_preprocess.py
- Data Content Structure


```text
.
└─ucf101                                    // contains 101 file folder
  ├── ApplyEyeMakeup                        // contains 145 videos
  │   ├── v_ApplyEyeMakeup_g01_c01.avi      // video file
  │   ├── v_ApplyEyeMakeup_g01_c02.avi      // video file
  │    ...
  ├── ApplyLipstick                         // contains 114 image files
  │   ├── v_ApplyLipstick_g01_c01.avi       // video file
  │   ├── v_ApplyLipstick_g01_c02.avi       // video file
  │    ...
  ├── ucfTrainTestlist                      // contains category files
  │   ├── classInd.txt                      // Category file.
  │   ├── testlist01.txt                    // split file
  │   ├── trainlist01.txt                   // split file
  ...
```

## [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with Ascend or GPU(Nvidia).
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Processing raw data files

```text
# Convert video into image.
bash run_dataset_preprocess.sh UCF101 [RAR_FILE_PATH] 1

# for example: bash run_dataset_preprocess.sh UCF101 /Data/UCF101/UCF101.rar 1
```

- Download pretrained model from [c3d.ckpt](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EbVF6SuKthpGj046abA37jkBkfkhzLm36F8NJmH2Do3jhg?e=xh32kW)


Refer to `c3d.yaml`. We support some parameter configurations for quick start.

```bash
cd tools/classification

# run the following command for trainning
python train.py -c ../../mindvideo/config/c3d/c3d.yaml

# run the following command for evaluation
python eval.py -c ../../mindvideo/config/c3d/c3d.yaml

# run the following command for inference
python infer.py -c ../../mindvideo/config/c3d/c3d.yaml
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in c3d.yaml

- config for C3D, UCF101 dataset

```text
# ==============================================================================
# model architecture
model_name: "c3d"   # model name 

#global config
device_target: "GPU"
dataset_sink_mode: False
context:            # runtime
    mode: 0         #0--Graph Mode; 1--Pynative Mode
    device_target: "GPU"
    save_graphs: False
    device_id: 3

# model settings of every parts
model:
    type: C3D
    in_d: 16
    in_h: 112
    in_w: 112
    in_channel: 3
    kernel_size: [3, 3, 3]
    head_channel: [4096, 4096]
    num_classes: 101
    keep_prob: [0.5, 0.5, 1.0]

# learning rate for training process
learning_rate:
    lr_scheduler: "exponential"
    lr: 0.003
    lr_epochs: [15, 30, 75]
    steps_per_epoch: 596
    warmup_epochs: 1
    max_epoch: 150
    lr_gamma: 0.1

# optimizer for training process
optimizer:
    type: 'SGD'
    momentum: 0.9
    weight_decay: 0.0005
    loss_scale: 1.0

# train loss
loss:       
    type: SoftmaxCrossEntropyWithLogits
    sparse: True
    reduction: "mean"

# trainning setups, including pretrain model
train:       
    pre_trained: False
    pretrained_model: ""
    ckpt_path: "./output/"
    epochs: 150
    save_checkpoint_epochs: 5
    save_checkpoint_steps: 1875
    keep_checkpoint_max: 30
    run_distribute: False

# evaluation setups
eval:
    pretrained_model: ".vscode/ms_ckpts/c3d_20220912.ckpt"

# infer setups
infer:
    pretrained_model: ".vscode/ms_ckpts/c3d_20220912.ckpt"
    batch_size: 1
    image_path: ""
    normalize: True
    output_dir: "./infer_output"

# export model into ckpt in other format  
export:       
    pretrained_model: ""
    batch_size: 64
    image_height: 112
    image_width: 112
    input_channel: 3
    file_name: "c3d"
    file_formate: "MINDIR"

# dataloader and data augmentation setups
data_loader:
    train:
        dataset:
            type: UCF101
            path: "/home/publicfile/UCF101_splits"  # Path to data root dir
            split: "train"
            batch_size: 16
            seq: 16
            seq_mode: "average"
            num_parallel_workers: 6
            shuffle: True
        map:        # data augmentation
            operations:
                - type: VideoResize
                  size: [128, 171]
                - type: VideoRescale
                  shift: "src/example/c3d/resized_mean_sports1m.npy" # mean file
                - type: VideoRandomCrop
                  size: [112, 112]
                - type: VideoRandomHorizontalFlip
                  prob: 0.5
                - type: VideoReOrder
                  order: [3, 0, 1, 2]
            input_columns: ["video"]

    eval:
        dataset:
            type: UCF101
            path: "/home/publicfile/UCF101_splits"  # Path to data root dir
            split: "test"
            batch_size: 16
            seq: 16
            seq_mode: "average"
            num_parallel_workers: 1
            shuffle: False
        map:
            operations:
                - type: VideoResize
                  size: [128, 171]
                - type: VideoRescale
                  shift: "src/example/c3d/resized_mean_sports1m.npy"  # mean file
                - type: VideoCenterCrop
                  size: [112, 112]
                - type: VideoReOrder
                  order: [3, 0, 1, 2]
            input_columns: ["video"]
    group_size: 1
# ==============================================================================

```

## [Training Process](#contents)

### Training

- train.log for UCF101

```shell
epoch: 1 step: 1192, loss is 0.8381556
epoch time: 593197.024 ms, per step time: 301.297 ms
epoch: 2 step: 1192, loss is 0.5701107
epoch time: 576058.976 ms, per step time: 260.542 ms
epoch: 3 step: 1192, loss is 0.1724325
epoch time: 578041.281 ms, per step time: 235.868 ms
...
epoch: 99 step: 1192, loss is 6.3519354e-05
epoch time: 573493.252 ms, per step time: 225.237 ms
epoch: 100 step: 1192, loss is 4.852382e-05
epoch time: 575237.743 ms, per step time: 229.164 ms
```
## [Evaluation Process](#contents)

### Evaluation

- eval.log for UCF101

```text
start create network
pre_trained model: ./results/2021-11-02_time_07_30_42/ckpt_0/0-85_223.ckpt
setep: 1/237, acc: 0.75
setep: 21/237, acc: 1.0
setep: 41/237, acc: 0.625
setep: 61/237, acc: 1.0
setep: 81/237, acc: 0.875
setep: 101/237, acc: 1.0
setep: 121/237, acc: 0.9375
setep: 141/237, acc: 0.5625
setep: 161/237, acc: 1.0
setep: 181/237, acc: 1.0
setep: 201/237, acc: 0.5625
setep: 221/237, acc: 1.0
eval result: top_1 80.412%
```
## visulization
![result.gif](./pics/result.gif)


## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

- C3D for UCF101

| Parameters          | Ascend                                                      | GPU                               |
| ------------------- | ----------------------------------------------------------- | --------------------------------- |
| Model Version       | C3D                                                         | C3D                               |
| Resource            | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | V100                              |
| uploaded Date       | 09/22/2021 (month/day/year)                                 | 11/06/2021 (month/day/year)       |
| MindSpore Version   | 1.8.1                                                       | 1.8.1                             |
| Dataset             | UCF101                                                      | UCF101                            |
| Training Parameters | epoch = 30, batch_size = 16                                 | epoch = 150,  batch_size = 8      |
| Optimizer           | SGD                                                         | SGD                               |
| Loss Function       | Max_SoftmaxCrossEntropyWithLogits                           | Max_SoftmaxCrossEntropyWithLogits |
| Speed               | 1pc: 253.372ms/step                                         | 1pc:237.128ms/step                |
| Top_1               | 1pc: 80.33%                                                 | 1pc:80.138%                       |
| Total time          | 1pc: 1.31hours                                              | 1pc:4hours                        |
| Parameters (M)      | 78                                                          | 78                                |


- Benchmark

| Models                     | C3D  |
|----------------------------|------|
| Top-1 Acc(Origin,%)        | 61.1 |
| Top-1 Acc(Mindspore,%)     | 81.4 |
| Top-5 Acc(Origin,%)        | 85.2 |
| Top-5 Acc(Mindspore,%)     | 92.6 |

The Top-1 and Top-5 accuracy comparison between the original paper and Mindspore 

# [Description of Random Situation](#contents)

We set random seed to 666 in default_config.yaml and default_config_gpu.yaml

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
