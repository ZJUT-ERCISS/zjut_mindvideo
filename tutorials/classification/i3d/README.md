# I3D_mindspore
- [Description](https://github.com/ZJUT-ERCISS/fairmot_mindspore#description)
- [Model Architecture](https://github.com/ZJUT-ERCISS/fairmot_mindspore#model-architecture)
- [Dataset](https://github.com/ZJUT-ERCISS/fairmot_mindspore#dataset)
- [Environment Requirements](https://github.com/ZJUT-ERCISS/fairmot_mindspore#environment-requirements)
- Quick Start
  - [Requirements Installation](https://github.com/ZJUT-ERCISS/fairmot_mindspore#requirements-installation)
  - [Dataset Preparation](https://github.com/ZJUT-ERCISS/fairmot_mindspore#dataset-preparation)
  - [Model Checkpoints](https://github.com/ZJUT-ERCISS/fairmot_mindspore#model-checkpoints)
  - [Running](https://github.com/ZJUT-ERCISS/fairmot_mindspore#running)
- Script Description
  - Training Process
    - [Training](https://github.com/ZJUT-ERCISS/fairmot_mindspore#training)
    - [Distributed Training](https://github.com/ZJUT-ERCISS/fairmot_mindspore#distributed-training)
  - [Evaluation Process](https://github.com/ZJUT-ERCISS/fairmot_mindspore#evaluation-process)
- Model Description
  - [Performance](https://github.com/ZJUT-ERCISS/fairmot_mindspore#performance)
- [Citation](#Citation)

# Description

Inflated 3D ConvNet (I3D) that is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to leI3D seamless spatio-temporal feature extractors from video while leveraging successful ImageNet architecture designs and even their parameters. We show that, after pre-training on Kinetics, I3D models considerably improve upon the state-of-the-art in action classification, reaching 80.9% on HMDB-51 and 98.0% on UCF-101


# Model Architecture

The overall network architecture of I3D is shown below:

[link]([1705.07750.pdf (arxiv.org)](https://arxiv.org/pdf/1705.07750.pdf))

# Dataset

Dataset used: [Kinetics400](https://www.deepmind.com/open-source/kinetics)

- Description: Kinetics-400 is a commonly used dataset for benchmarks in the video field. For details, please refer to its official website [Kinetics](https://www.deepmind.com/open-source/kinetics). For the download method, please refer to the official address [ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics), and use the download script provided by it to download the dataset.

- Dataset size：

  | category       | Number of data |
  | -------------- | -------------- |
  | Training set   | 234619         |
  | Validation set | 19761          |

```
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

# Environment Requirements

- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)

- Requirements

```text
Python and dependencies
    - python 3.7.5
    - decord 0.6.0
    - imageio 2.21.1
    - imageio-ffmpeg 0.4.7
    - mindspore-gpu 1.6.1
    - ml-collections 0.1.1
    - matplotlib 3.4.1
    - numpy 1.21.5
    - Pillow 9.0.1
    - PyYAML 6.0
    - scikit-leI3D 1.0.2
    - scipy 1.7.3
    - pycocotools 2.0
```

- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick Start

## Requirements Installation

```text
pip install -r requirements.txt
```

## Dataset Preparation

I3D model uses [Kinetics400](https://www.deepmind.com/open-source/kinetics) dataset to train and validate in this repository.

**Configure path to dataset root** in `data/data.json` file.

## Model Checkpoints

The pretrain model is trained on the the Kinetics400 dataset. It can be downloaded here:[i3d_rgb_kinetics400.ckpt](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EeqkpDHObpBNj5ibeawTY0gBWd84YvFrhmbdGeu8qm5SDw?e=E3j8vM)

## Running

- Run on GPU

```text
cd scripts/

# run training example
bash run_standalone_train.sh [PROJECT_PATH] [DATA_PATH]

# run distributed training example
bash run_distribute_train.sh [PROJECT_PATH] [DATA_PATH]

# run evaluation example
bash run_standalone_eval.sh [PROJECT_PATH] [DATA_PATH]
```

# Script Description

## Training Process

### Training Alone

Run `scripts/run_standalone_train.sh` to train the model standalone. The usage of the script is:

#### Running on GPU

```
bash scripts/run_standalone_train.sh [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the training procedure:

```
bash scripts/run_standalone_train.sh ./config/i3d_rgb.yaml ./i3d_rgb_kinetics400.ckpt
```

The model checkpoint will be saved into `./output`.

### Distributed Training

Run `scripts/run_distribute_train.sh` to train the model distributed. The usage of the script is:

#### Running on GPU

```
bash scripts/run_distribute_train.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the distributed training procedure:

```
bash scripts/run_distribute_train.sh 8 0,1,2,3,4,5,6,7 ./config/i3d_rgb.yaml ./i3d_rgb_kinetics400.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file `train/tran.log`.

The model checkpoint will be saved into `train/ckpt`.

## Evaluation Process

The evaluation data set was [Kinetics400](https://www.deepmind.com/open-source/kinetics) 

Run `scripts/run_eval.sh` to evaluate the model. The usage of the script is:

```
bash scripts/run_standalone_eval.sh [device] [config] [load_ckpt] [dataset_dir]
```

For example, you can run the shell command below to launch the validation procedure.

```
bash scripts/run_standalone_eval.sh GPU ./config/i3d_rgb.yaml ./i3d_rgb_kinetics400.ckpt data_path
```

The eval results can be viewed in `eval/eval.log`.

# [Model Description](https://github.com/ZJUT-ERCISS/fairmot_mindspore#contents)

## [Performance](https://github.com/ZJUT-ERCISS/fairmot_mindspore#contents)

### I3D on Kinetics400 dataset with detector

#### Benchmark with paper

|               | MindSpore | original paper |
| ------------- | --------- | ----- |
| Top1 Accuracy | 67%       | 68%   |

#### Performance parameters

| Parameters          | GPU Standalone                | GPU Distributed               |
| ------------------- | ----------------------------- | ----------------------------- |
| Model Version       | I3D                           | I3D                           |
| Resource            | RTX 3090 24GB                 | 8x RTX 3090 24GB              |
| Uploaded Date       | 25/06/2021 (day/month/year)   | 21/02/2021 (day/month/year)   |
| MindSpore Version   | 1.2.0                         | 1.5.0                         |
| Training Dataset    | Kinetics400                   | Kinetics400                   |
| Evaluation Dataset  | Kinetics400                   | Kinetics400                   |
| Training Parameters | epoch=30, batch_size=4        | epoch=30, batch_size=12       |
| Optimizer           | SGD                           | SGD                           |
| Loss Function       | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |

#### Visual Result

![i3d_vis](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/classification/i3d/pics/result.gif)


# Citation

If you find this project useful in your research, please consider citing:

```text
@INPROCEEDINGS{8099985,
  author={Carreira, João and Zisserman, Andrew},
  booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset}, 
  year={2017},
  volume={},
  number={},
  pages={4724-4733},
  doi={10.1109/CVPR.2017.502}}
```



```latex
@misc{nonlocal_misdspore, author = {MindSpore Vision Contributors}, title = {MindVideo Models}, year = {2022}, publisher = {GitHub}, journal = {GitHub repository}, doi = {10.1109/CVPR.2017.502}, howpublished = {\url{https://github.com/ZJUT-ERCISS/i3d_mindspore}} } 
```