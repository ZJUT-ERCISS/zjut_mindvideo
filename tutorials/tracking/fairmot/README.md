# fairmot_mindspore

# Description
There has been remarkable progress on object detection and re-identification in recent years which are the core components for multi-object tracking. However, little attention has been focused on accomplishing the two tasks in a single network to improve the inference speed. The initial attempts along this path ended up with degraded results mainly because the re-identification branch is not appropriately learned. In this work, we study the essential reasons behind the failure, and accordingly present a simple baseline to addresses the problems. It remarkably outperforms the state-of-the-arts on the MOT challenge datasets at 30 FPS. This baseline could inspire and help evaluate new ideas in this field. More detail about this model can be found in:

Paper: Zhang Y, Wang C, Wang X, et al. FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking. 2020.
# Model Architecture

The overall network architecture of FairMOT is shown below:

![FairMOT_architecture](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/tracking/fairmot/pics/fairmot.png)


#Dataset
Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: 
[ETH](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) or (https://data.vision.ee.ethz.ch/cvl/aess/dataset/)
[CalTech](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) or (http://www.vision.caltech.edu/datasets/)
[MOT17](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) or (https://motchallenge.net/data/MOT17/)
[CUHK-SYSU](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) or (http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html)
[PRW](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) or (http://www.liangzheng.com.cn/Project/project_prw.html)
[CityPerson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)


# Environment Requirements

- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)

- Requirements

```text
Python and dependencies
    - python 3.7.5
	- Cython 0.29.30
	- cython-bbox 0.1.3
    - decord 0.6.0
    - mindspore-gpu 1.6.1
    - ml-collections 0.1.1
    - matplotlib 3.4.1
	- motmetrics 1.2.5
    - numpy 1.21.5
    - Pillow 9.0.1
    - PyYAML 6.0
    - scikit-learn 1.0.2
    - scipy 1.7.3
    - pycocotools 2.0
```

- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
  
# Quick Start

## Requirements Installation

Some packages in requirements.txt need Cython package to be installed first. For this reason, you should use the following commands to install dependencies:
```text
pip install -r requirements.txt
```

## Dataset Preparation

FairMot model uses mix dataset to train and validate in this repository. We use the training data as JDE in this part and we call it "MIX".

Please refer to their DATA ZOO to download and prepare all the training data including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16.

Then put all training and evaluation data into one directory and then change "data_root" to that directory in data.json , like this:

```text
"data_root": "/home/publicfile/dataset/tracking"
```

#Model Checkpoints
The pretrain model (DLA-34 backbone) is trained on the the MIX dataset for 30 epochs. It can be downloaded here: 
[fairmot_dla34-30_886.ckpt] (https://gitee.com/yanlq46462828/zjut_mindvideo/blob/master/tutorials/tracking/fairmot/README.md)


Model Checkpoints
The pretrain model (DLA-34 backbone) is trained on the the MIX dataset for 30 epochs. It can be downloaded here: fairmot_dla34-30_886.ckpt

#Running
Please run one of the following command in root directory fairmot_mindspore.

```
# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [DATA_JSON] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
# distributed training on GPU
bash scripts/run_standalone_train_gpu.sh [DATA_JSON] [NUM_DEVICES] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
```

To validate the model, run the shell `script scripts/run_eval.sh` with the format below:
```
bash scripts/run_eval.sh [CKPT_PATH] [DATA_ROOT] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
```
To infer using the model, run the shell script `scripts/run_infer.sh` with the format below:
```
bash scripts/run_eval.sh [CKPT_PATH] [DATA_ROOT] [SAVE_VIDEO] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)]
```
The validate and infer programme will generate pictures with predict bbox, and infer programme can generate video if SAVE_VIDEO is True.

If you want to train or evaluate the model in other parameter settings, please refer to `fairmot_mix_train.py` or `fairmot_mix_eval.py` to change your input parameters in script.

#Examples
##Training
```
bash scripts/run_standalone_train_gpu.sh datas/data.json 0
```
##Distributed Training
```
bash scripts/run_distribute_train_gpu.sh datas/data.json 0,1,2,3
```
The above shell script will run distribute training in the background.

You can view the results through the file `./output/train.log`.

The model checkpoint will be saved into `./output`.

##Evaluation Process
The evaluation data set was `MOT17`
```
bash scripts/run_eval.sh fairmot_dla34-30_886.ckpt /home/tracking/MOT17/images/train 0
```
The evaluation results can be viewed in `/output`.

##Infer Process
The infer data set was `MOT17`

```
bash scripts/run_eval.sh fairmot_dla34-30_886.ckpt /home/tracking/MOT17/images/test  True 0
```
The infer results can be viewed in `./output`.

# [Model Description](https://github.com/ZJUT-ERCISS/fairmot_mindspore#contents)

## [Performance](https://github.com/ZJUT-ERCISS/fairmot_mindspore#contents)

#### Performance and parameters

| Parameters          | GPU Distributed             |
| ------------------- | --------------------------- |
| Model Version       | FairMotNet                  |
| Resource            | 4x RTX 3090 24GB            |
| Uploaded Date       | 21/09/2022 (day/month/year) |
| MindSpore Version   | 1.6.1                       |
| Training Dataset    | MIX                         |
| Evaluation Dataset  | **MOT17**                       |
| Training Parameters | epoch=30, batch_size=2      |
| Optimizer           | Adam                        |
| Loss Function       | FocalLoss,RegLoss           |
| Train Performance   | **MOTA:71.3%** Prcn:91.2%       |

#### Benchmark

results from original paper：

| Dataset | MOTA | IDF1 | MT(%) | ML(%) | IDs  | FPS  |
| --  | --| -- | ---  | ---  | --- | --- |
| MOT15   | 60.6 | 64.7 | 47.6  | 11.0  | 591  | 30.5 |
| MOT16   | 74.9 | 72.8 | 44.7  | 15.9  | 1074 | 25.9 |
| **MOT17**   | **73.7** | 72.3 | 43.2  | 17.3  | 3303 | 25.9 |
| MOT20   | 61.8 | 67.3 | 68.8  | 7.6   | 5243 | 13.2 |
#### Visualization

Examples given below are the predictions this FairMOT model:

![1](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/tracking/fairmot/pics/MOT17_04_SDP.gif)
![2](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/tracking/fairmot/pics/MOT17_05_SDP.gif)
![3](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/tracking/fairmot/pics/MOT17_09_SDP.gif)
![4](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/tracking/fairmot/pics/MOT17_10_SDP.gif)
![5](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/tutorials/tracking/fairmot/pics/MOT17_13_SDP.gif)

# Citation

If you find this project useful in your research, please consider citing:

```text
@article{zhang2021fairmot,
  title={Fairmot: On the fairness of detection and re-identification in multiple object tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={International Journal of Computer Vision},
  volume={129},
  pages={3069--3087},
  year={2021},
  publisher={Springer}
}
```



```latex
@misc{MindSpore Vision 2022,
    title={{MindSpore Vision}:MindSpore Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://gitee.com/mindspore/vision}},
    year={2022}
}
```