## C3D Description

C3D model is widely used for 3D vision task. The construct of C3D network is similar to the common 2D ConvNets, the main difference is that C3D use 3D operations like Conv3D while 2D ConvNets are anentirely 2D architecture. To know more information about C3D network, you can read the original paper Learning Spatiotemporal Features with 3D Convolutional Networks.

## Model Architecture

C3D net has 8 convolution, 5 max-pooling, and 2 fully connected layers, followed by a softmax output layer. All 3D convolution kernels are 3 × 3 × 3 with stride 1 in both spatial and temporal dimensions. The 3D pooling layers are denoted from pool1 to pool5. All pooling kernels are 2 × 2 × 2, except for pool1 is 1 × 2 × 2. Each fully connected layer has 4096 output units.

## Dataset

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


## Quick Start

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
```

## Visulization
![result.gif](./pics/result.gif)

## Performance

Fine tune c3d model on ucf101 with sport1m pretrained weight.
we get 75.34 for clip-accuracy
third party implmentation : please refer to https://github.com/hx173149/C3D-tensorflow

|  Ours | third party 
:-: | :-: |
75.34 | 74.65|

## References
 -  C3D-TF : https://github.com/hx173149/C3D-tensorflow
 -  C3D-caffe: https://github.com/facebook/C3D
