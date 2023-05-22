## Run on GPU

There are two approaches provided for training, evaluation and inference within `mindvideo` for each supported model. After installing MindSpore via the official website, one is to run the training or evaluation files under the `example` folder, which is a independent module for training and evaluation specifically designed for starters, according to each model's name. And the other is to use the train and inference interfaces for all models under the root folder of the repository when working with the `YAML` file containing the parameters needed for each model as we also support some parameter configurations for quick start.

### run with example files

```text
# run training example
!python src/example/[TRAIN_FILE] --data_url [DATASET_PATH] --epoch_size [NUM_OF_EPOCH] --batch_size [BATCH_SIZE]

# run evaluation example
!python src/example/[EVAL_FILE] --data_url [DATASET_PATH] --pretrained_path [CKPT_PATH] --batch_size [BATCH_SIZE]
```

For example when training and evaluating the C3D model, please run the following command in root directory.

```text
# run training example
!python src/example/c3d_ucf101_train.py --data_url /usr/dataset/ucf101 --pretrained True --pretrained_path ./c3d_pretrained.ckpt --batch_size 8

# run evaluation example
!python src/example/c3d_ucf101_eval.py --data_url /usr/dataset/ucf101 --pretrained_path ./c3d_pretrained.ckpt --batch_size 16
```

### run with scripts and config file

```text
!cd scripts
# run training example
!bash run_standalone_train_gpu.sh [CONFIG_PATH]
# run evaluation example
!bash run_standalone_eval_gpu.sh [CONFIG_PATH]
```

For example when training and evaluating the C3D model, please run the following command in root directory.

```text
# run training example
!bash run_standalone_train_gpu.sh src/config/c3d_ucf101.yaml
# run evaluation example
!bash run_standalone_eval_gpu.sh src/config/c3d_ucf101.yaml
```

You can choose how to train and test model by simply setting the config. For example, if you want to modify the training or evaluating configuration of c3d, you can enter the `/src/config` folder and modify the c3d_ucf101.yaml file. The following shows some configurations available for modification in the yaml file.

```yaml
# learning rate for training process
learning_rate:     # learning_rate scheduler
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

loss:       
    type: SoftmaxCrossEntropyWithLogits
    sparse: True
    reduction: "mean"

train:       # ckpt related parameters
    pre_trained: False
    pretrained_model: ""
    ckpt_path: "./output/"
    epochs: 150
    save_checkpoint_epochs: 5
    save_checkpoint_steps: 1875
    keep_checkpoint_max: 30
    run_distribute: False

eval:       # infer process
    pretrained_model: ".vscode/c3d_20220912.ckpt"
    batch_size: 1
    image_path: ""
    normalize: True
    output_dir: "./eval_output"
```
