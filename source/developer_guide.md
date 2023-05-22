# Developer Guide

### Requirements Installation
Use the following commands to install dependencies for each model, taking the non-local model as an example:

```text
pip install -r mindvideo/example/nonlocal/requirements.txt
```
### Configuration Files
The configuration files of each supported model are presented in ./mindvideo/config. Each .yaml file contains information about the supported model training, evaluation and inference, for example, model name, model, learning rate, loss, optimizer, etc.

### Load Model Checkpoints
All links to download the pre-train models are presented in https://gitee.com/yanlq46462828/zjut_mindvideo/tree/master

### Dataset Preparation
The links of MindVideo supported dataset are presented in: https://gitee.com/yanlq46462828/zjut_mindvideo/tree/master, including activitynet, Kinetics400, Kinetics600, UCF101, Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17, MOT16, charades, Collective Activity, columbia Consumer Video, davis, hmdb51, fbms, msvd, Sports-1M, THUMOS, UBI-Fights, tyvos.

Then put all training and evaluation data into one directory and then change **data_root** to that directory in data.json, like this:
```text
"data_root": "/home/publicfile/dataset/tracking"
```
Within mindvideo, all data processing methods according to each dataset used can be found under the data folder.

### Customize a Model
Here, we present how to use a model, and apply it to the MindSpore.
MindSpore supports C3D, I3D, X3D, R(2+1)D, NonLocal, ViST, fairMOT, VisTR and ARN models. 

- Create a Model

To begin with, we should create a model implementing from one of C3D, I3D, X3D, R(2+1)D, NonLocal, ViST, fairMOT, VisTR and ARN models. For example, we would like to develop a model named as I3D and write the code to builder.py.
```text
def build_model(cfg):
    """build model"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.MODEL)


def build_layer(cfg):
    """build layer"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.LAYER)
```
- Pass Parameters

Then, we need to indicate .yaml files to define the parameters of the model. Taking I3D model as example:
```text
model_name: i3d_rgb
dataset_sink_mode: False
```

### Context settings
```text
context:
    mode: 0 #0--Graph Mode; 1--Pynative Mode
    device_target: "GPU"
```

### Model settings
```text
model:
    type: i3d_rgb 
    num_classes: 400


learning_rate:
    lr_scheduler: "cosine_annealing"
    lr: 0.0012
    lr_epochs: [2, 4]
    lr_gamma: 0.1
    eta_min: 0.0
    t_max: 100
    max_epoch: 5
    warmup_epochs: 4

optimizer:
    type: 'SGD'
    momentum: 0.9
    weight_decay: 0.0004
    loss_scale: 1024

loss:
    type: SoftmaxCrossEntropyWithLogits
    sparse: True
    reduction: "mean"

train:
    pre_trained: False
    pretrained_model: ""
    ckpt_path: "./output/"
    epochs: 100
    save_checkpoint_epochs: 5
    save_checkpoint_steps: 1875
    keep_checkpoint_max: 10
    run_distribute: False

eval:
    pretrained_model: ""

infer:
    pretrained_model: ""
    batch_size: 16
    image_path: ""
    normalize: True
    output_dir: "./infer_output"
```

### Kinetics Dataset Config
```text
data_loader:
    train:
        dataset:
              type: Kinetic400
              path: "/home/publicfile/kinetics-400"
              shuffle: True
              split: 'train'
              seq: 64
              num_parallel_workers: 8
              shuffle: True
              batch_size: 16
              
        map:
            operations:
                - type: VideoResize
                  size: [256, 256]
                - type: VideoRandomCrop
                  size: [224, 224]
                - type: VideoRandomHorizontalFlip
                  prob: 0.5
                - type: VideoToTensor
            input_columns: ["video"]

    eval:
        dataset:
            type: Kinetic400
            path: "/home/publicfile/kinetics-dataset"
            split: 'val'
            seq: 64
            shuffle: Ture
            num_parallel_workers: 8
            seq_mode: 'discrete'
            
        map:
            operations:
                - type: VideoShortEdgeResize
                  size: 256
                - type: VideoCenterCrop
                  size: [224, 224]
                - type: VideoToTensor
            input_columns: ["video"]
group_size: 1
```

### Customize DataLoaders
Here, we present how to develop a new DataLoader, and apply it into our tool. If we have a model, and there is special requirement for loading the data, then we need to design a new DataLoader.

In this project, here is a abstract dataloaders: builder.py file in ./mindvideo/data.

In general, the new dataloader include four function: build_dataset_sampler, builder_dataset, build_transforms, register_builtin_dataset. The build_dataset_sampler function is used to build sampler, the build_dataset function is used to build dataset, the build_transforms function is used to build data transform pipeline, the register_builtin_dataset function is used to register MindSpore builtin dataset class.

### Customize Trainers
There are two approaches provided for training, evaluation and inference within mindvideo for each supported model. After installing MindSpore via the official website, one is to run the training or evaluation files under the example folder, which is a independent module for training and evaluation specifically designed for starters, according to each model's name. And the other is to use the train and inference interfaces for all models under the root folder of the repository when working with the YAML file containing the parameters needed for each model as we also support some parameter configurations for quick start. For this method, take I3D for example, just run following commands for training:
```text
python train.py -c zjut_mindvideo/mindvideo/config/i3d/i3d_rgb.yaml
```
and run following commands for inference and evaluation:
```text
python infer.py -c zjut_mindvideo/mindvideo/config/i3d/i3d_rgb.yaml
python eval.py -c zjut_mindvideo/mindvideo/config/i3d/i3d_rgb.yaml
```
