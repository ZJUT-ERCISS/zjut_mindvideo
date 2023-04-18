# vist_mindspore/swin3d_mindspore

# Contents
- [Contents](#contents)
  - [Description](#description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [Quick Start](#quick-start)
    - [Requirements Installation](#requirements-installation)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Checkpoints](#model-checkpoints)
    - [Running](#running)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
  - [Performance](#performance)
    - [Evaluation Performance](#evaluation-performance)
  - [Benchmark](#benchmark)
  - [Visualization Result](#visualization-result)
  - [Citation](#citation)



## [Description](#contents)

This repository contains a Mindspore implementation of Video Swin Transformer based upon original Pytorch implementation (<https://github.com/SwinTransformer/Video-Swin-Transformer>). The training and validating scripts are also included, and the evaluation results are shown in the [Performance](#performance) section.

## [Model Architecture](#contents)
The major component of Swin3D is the Video Swin Transformer block, which contains a 3D shifted window based MSA (Multi-head Self Attention).
<div align=center>
<img src=https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/classification/vist/pics/swin3d_block.jpg> 

Figure 1 An illustration of two successive Video Swin Transformer blocks </div>
Figure 1 illustrates two successive Video Swin Transformer blocks. 3D W-MSA and 3D SW-MSA denote 3D window based multi-head self-attention using regular and shifted window partitioning configurations, respectively. Similar to [Swin2D](https://arxiv.org/abs/2103.14030), this 3D shifted window design introduces connections between neighboring non-overlapping 3D windows in the previous layer.
<div align=center>
<img src=https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/classification/vist/pics/swin3d_tiny.png> 

Figure 2 The architecture of swin3d tiny</div>
Figure 2 shows the overall architecture of the tiny version (Swin3D-T) of Video Swin Transformer. For more information, please read the original [paper](https://arxiv.org/abs/2106.13230).
## [Dataset](#contents)

Dataset used: [Kinetics400](https://www.deepmind.com/open-source/kinetics)

- Description: Kinetics-400 is a commonly used dataset for benchmarks in the video field. For details, please refer to its official website [Kinetics](https://www.deepmind.com/open-source/kinetics). For the download method, please refer to the official address [ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics), and use the download script provided by it to download the dataset.

- Dataset size：
    |category | Number of data  | 
    | :------: | :----------: | 
    |Training set | 238797 |  
    |Validation set | 19877 | 
Because of the expirations of some YouTube links, the sizes of kinetics dataset copies may be different.
```text
The directory structure of Kinetic-400 dataset looks like:

    .
    |-kinetic-400
        |-- train
        |   |-- ___qijXy2f0_000011_000021.mp4       // video file
        |   |-- ___dTOdxzXY_000022_000032.mp4       // video file
        |    ...
        |-- test
        |   |-- __Zh0xijkrw_000042_000052.mp4       // video file
        |   |-- __zVSUyXzd8_000070_000080.mp4       // video file
        |-- val
        |   |-- __wsytoYy3Q_000055_000065.mp4       // video file
        |   |-- __vzEs2wzdQ_000026_000036.mp4       // video file
        |    ...
        |-- kinetics-400_train.csv                  // training dataset label file.
        |-- kinetics-400_test.csv                   // testing dataset label file.
        |-- kinetics-400_val.csv                    // validation dataset label file.

        ...
```
## [Environment Requirements](#contents)
To run the python scripts in the repository, you need to prepare the environment as follow:

- Python and dependencies
    - python 3.7.5
    - decord 0.6.0
    - mindspore-gpu 1.6.1
    - ml-collections 0.1.1
    - numpy 1.21.5
    - Pillow 9.0.1
    - PyYAML 6.0
    - imageio 2.21.1
    - imageio-ffmpeg 0.4.7
- Hardware
    - Prepare hardware environment with GPU(Nvidia).
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

### [Requirements Installation](#contents)

Some packages in `requirements.txt` need Cython package to be installed first. For this reason, you should use the following commands to install dependencies:

```shell
pip install -r requirements.txt
```

### [Dataset Preparation](#contents)

Swin3D model uses kinetics400 dataset to train and validate in this repository. 

### [Model Checkpoints](#contents)

The pretrain model is trained on the the kinetics400 dataset. It can be downloaded here: 
- [Swin3D-Tiny](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EX1foDC63eNNgnxbfD2oEDYB9C5JoLUfEgqlJ_4QymoJqQ?e=ayseUu
)
- [Swin3D-Small](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EZKHu92j3SVLlAfvC-gv1pcBUvXcexXo7H5Kv8QymqHpZQ?e=B3FOkI
)
- [Swin3D-Base](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EXrE7hbSqCtJoSourHbcUIABmnskD5qO0o9c_hpJ-x86PA?e=zdQ02f)

### [Running](#contents)

To train or finetune the model, you can run the following script:

```shell

cd scripts/

# run training example
bash train_standalone.sh [PROJECT_PATH] [DATA_PATH]

# run distributed training example
bash train_distribute.sh [PROJECT_PATH] [DATA_PATH]


```
To validate the model, you can run the following script:
```shell
cd scripts/

# run evaluation example
bash eval_standalone.sh [PROJECT_PATH] [DATA_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
│  infer.py                                     // infer script
│  README.md                                    // descriptions about Swin3D
│  train.py                                     // training scrip
└─src
    ├─config
    │  └─swin3d                                 // swin3d parameter configuration
    |     |  swin3d_t.yaml
    |     |  swin3d_s.yaml
    |     |  swin3d_b.yaml
    ├─data
    │  │  builder.py                            // build data
    │  │  download.py                           // download dataset
    │  │  generator.py                          // generate video dataset
    │  │  images.py                             // process image
    │  │  kinetics400.py                        // kinetics400 dataset
    |  |  kinetics600.py                        // kinetics600 dataset
    │  │  meta.py                               // public API for dataset
    │  │  path.py                               // IO path
    │  │  video_dataset.py                      // video dataset
    │  │
    │  └─transforms
    │     |  builder.py                         // build transforms
    │     |  video_center_crop.py               // center crop
    │     |  video_normalize.py                 // normalize
    │     |  video_random_crop.py               // random crop
    │     |  video_random_horizontal_flip.py    // random horizontal flip
    │     |  video_reorder.py                   // reorder
    │     |  video_rescale.py                   // rescale
    │     |  video_short_edge_resize.py         // short edge resize
    │     |  video_three_crop.py                // three crop
    |
    ├─example
    │  |  swin3d_kinetics400_eval.py            // eval swin3d model
    │  |  swin3d_kinetics400_train.py           // train swin3d model
    │
    ├─loss
    │  |  builder.py                            // build loss
    │
    ├─models
    |  |  base.py                               // base
    │  │  builder.py                            // build model
    │  │  swin3d.py                             // swin3d model
    │  │
    │  └─layers
    │     | avgpoll3d.py                        // average pooling 3D.
    │     | dropout_dense.py                    // dense head
    │     | drop_path.py                        // drop path
    |     | feed_forward.py                     // feed forward layer
    |     | identity.py                         // identity block
    │     | roll3d.py                           // 3d roll operation
    │
    ├─optim
    │  |  builder.py                           // build optimizer
    │
    ├─schedule
    │  |  builder.py                           // build learning rate shcedule
    │  |  lr_schedule.py                       // learning rate shcedule
    │
    └─utils
       |  callbacks.py                        // eval loss monitor
       |  check_param.py                      // check parameters
       |  class_factory.py                    // class register
       |  config.py                           // parameter configuration
       |  mask.py                             // mask module for shifted windows based MSA
       |  windows.py                          // window operations


```

### [Script Parameters](#contents)

Here shows the parameters configuration for both training and evaluation of Swin3D-Tiny model.
- config for swin3d_t, Kinetics400 dataset

```text
# ==============================================================================
# model architecture
model_name: Swin3D-T

# The dataset sink mode.
dataset_sink_mode: False

# Context settings.
context:
    mode: 0 #0--Graph Mode; 1--Pynative Mode
    device_target: "GPU"
    device_id: 2

# Model settings.
model:
    type: swin3d_t
    num_classes: 400
    patch_size: [2, 4, 4]
    window_size: [8, 7, 7]
    embed_dim: 96
    depths: [2, 2, 6, 2]
    num_heads: [3, 6, 12, 24]
    representation_size: 768
    droppath_keep_prob: 0.9

learning_rate:
    lr_scheduler: "cosine_annealing_V2"
    lr: 0.001
    warmup_epochs: 2.5
    max_epoch: 30
    t_max: 30
    eta_min: 0

optimizer:
    type: 'AdamWeightDecay'
    beta1: 0.9
    beta2: 0.99
    # L2 regularization.
    weight_decay: 0.02

loss:
    type: SoftmaxCrossEntropyWithLogits
    sparse: True
    reduction: "mean"

train:
    pre_trained: True
    pretrained_model: "pretrained_models/ms_swin_tiny_patch244_window877_kinetics400_1k.ckpt"
    ckpt_path: "./swin3d_t/"
    epochs: 30
    save_checkpoint_epochs: 1
    save_checkpoint_steps: 1875
    keep_checkpoint_max: 10
    run_distribute: False

infer:
    pretrained_model: "pretrained_models/ms_swin_tiny_patch244_window877_kinetics400_1k.ckpt"

# kinetic dataset config
data_loader:
    train:
        dataset:
              type: Kinetic400
              path: "/usr/publicfile/kinetics-400"
              split: 'train'
              seq: 32
              seq_mode: 'interval'
              num_parallel_workers: 1
              shuffle: True
              batch_size: 1
              frame_interval: 2
              num_clips: 1

        map:
            operations:
                - type: VideoShortEdgeResize
                  size: 256
                  interpolation: 'linear'
                - type: VideoRandomCrop
                  size: [224, 224]
                - type: VideoRandomHorizontalFlip
                  prob: 0.5
                - type: VideoRescale
                  shift: 0
                - type: VideoReOrder
                  order: [3, 0, 1, 2]
                - type: VideoNormalize
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.225]

    eval:
        dataset:
            type: Kinetic400
            path: "/usr/publicfile/kinetics-400"
            split: 'test'
            seq: 32
            seq_mode: 'interval'
            num_parallel_workers: 1
            shuffle: False
            batch_size: 1
            frame_interval: 2
            num_clips: 4
        map:
            operations:
                - type: VideoShortEdgeResize
                  size: 224
                - type: VideoThreeCrop
                  size: [224, 224]
                - type: VideoRescale
                  shift: 0
                - type: VideoReOrder
                  order: [3, 0, 1, 2]
                - type: VideoNormalize
                  mean: [0.485, 0.456, 0.406]
                  std: [0.229, 0.224, 0.225]
    group_size: 1
==============================================================================

```
## [Performance](#contents)
### [Evaluation Performance](#contents)

- Mindspore version: 1.6.1

| Models              | Swin3D-Tiny                   | Swin3D-Small                  | Swin3D-Base                   |
| -------------       |-------------------------      |--------------                 |------------                   |
| Dataset             | kinetic400                    | kinetic400                    | kinetic400                    |
| Training Parameters | epoch = 30,  batch_size = 16  | epoch = 30,  batch_size = 16  | epoch = 30,  batch_size = 16  |
| Optimizer           | AdamWeightDecay               | AdamWeightDecay               | AdamWeightDecay               |
| Loss Function       | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |
| acc@1               | 77.27                         | 78.89                         | 81.16                         |
| acc@5               | 93.29                         | 93.88                         | 95.16                         |

## [Benchmark](#contents)
The Top-1 and Top-5 accuracy comparison between the original paper and Mindspore is shown below:

| Models | Swin3D-Tiny | Swin3D-Small | Swin3D-Base |
| ------ | ----------- | ------------ | ----------- |
| Top-1 Acc(Origin, %)     |  78.8 |  80.6 |  82.7 |
| Top-1 Acc(Mindspore, %)  | 77.27 | 78.89 | 81.16 |
| Top-5 Acc(Origin, %)     |  93.6 |  94.5 |  95.5 |
| Top-5 Acc(Mindspore, %)  | 93.29 | 93.88 | 95.16 |

## [Visualization Result](#contents)

<div align=center>
<img src=https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/classification/vist/pics/vis_result.gif>

Figure 3 Visualization Result of Swin3D-T</div>

## [Citation](#contents)
```BibTeX
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
@misc{swin3d_mindspore,
    title={Mindspore Video Models},
    author={ZJUT-ERCISS},
    year={2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ZJUT-ERCISS/swin3d_mindspore}}
}
```