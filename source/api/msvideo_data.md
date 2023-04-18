## msvideo.data

### Acvivitynet

> class msvideo.data.Activitynet(path, split="train", transform=None, target_transform=None, seq=16, seq_mode="part", align=True, batch_size=16, repeat_num=1, shuffle=None, num_parallel_workers=1, num_shards=None, shard_id=None, download=False)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: True.
- batch_size (int): Batch size of dataset. Default:16.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.

**Returns:**

None

> def msvideo.data.Activitynet.index2label()

Get the mapping of indexes and labels.

**Returns:**

The mapping of indexes and labels

### ParseActivitynet

> class ParseActivitynet()

Parse Activitynet dataset.

-Base: ParseDataset

> def msvideo.data.ParseActivitynet.loadjson()

Parse json file.

**Returns:**

- cls2index: dictionary
- index2cls: list

> def msvideo.data.ParseActivitynet.parse_dataset()

Traverse the Activitynet dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list

### Charades

> class msvideo.data.Charades(path,
                                split="train",
                                transform=None,
                                target_transform=None,
                                seq=16,
                                seq_mode="part",
                                align=True,
                                batch_size=16,
                                repeat_num=1,
                                shuffle=None,
                                num_parallel_workers=1,
                                num_shards=None,
                                shard_id=None,
                                download=False)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: True.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.

**Returns:**

None

> def msvideo.data.Charades.index2label()

Get the mapping of indexes and labels.

**Returns:**

The mapping of indexes and labels

### ParseCharades

> class ParseCharades()

Parse Charades dataset.

-Base: ParseDataset

> def msvideo.data.ParseCharades.parse_dataset()

Traverse the Charades dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list

### CollectiveActivity

> class msvideo.data.CollectiveActivity(path,
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=False,
                 batch_size=1,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False,
                 suffix="picture")

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: True.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.
- suffix(str): Storage format of video. Default: "picture".

**Returns:**

None

### ParseCollectiveActivity

> class ParseCollectiveActivity()

Parse CollectiveActivity dataset.

-Base: ParseDataset

> def msvideo.data.ParseCollectiveActivity.parse_dataset()

Traverse the CollectiveActivity dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list

### CCV

> class msvideo.data.CCV(path,
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=False,
                 batch_size=1,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False,
                 suffix="picture")

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: True.
- batch_size (int): Batch size of dataset. Default:1.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.
- suffix(str): Storage format of video. Default: "picture".

**Returns:**

None

> def msvideo.data.CCV.index2label()

Get the mapping of indexes and labels.

**Returns:**

The mapping of indexes and labels

### ParseCCV

> class msvideo.data.ParseCCV()

Parse columbia consumer video dataset.

-Base: ParseDataset

> def msvideo.data.ParseCCV.load_cls_file()

Parse the category file.

**Returns:**

a list of category name

> def msvideo.data.ParseCCV.parse_dataset()

Traverse the columbia consumer video dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list

### Davis

> class msvideo.data.Davis(path,
                 split="train",
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=True,
                 quality="480p",
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: True.
- quality(str):The Picture quality,"1080p" or "480p".Default:"480p".
- batch_size (int): Batch size of dataset. Default:1.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.

**Returns:**

None

> def msvideo.data.Davis.index2label()

Get the mapping of indexes and labels.

**Returns:**

The mapping of indexes and labels

### ParseDavis

> class msvideo.data.ParseDavis()

Parse Davis dataset.

-Base: ParseDataset

> def msvideo.data.ParseDavis.parse_dataset()

Traverse the Davis dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### DatasetGenerator

> class msvideo.data.DatasetGenerator(path,
                        label,
                        seq=16,
                        mode="part",
                        suffix="video",
                        align=False,
                        frame_interval=1,
                        num_clips=1)

Dataset generator for getting video path and its corresponding label.

**Parameters:**

- path(list): Video file path list.
- label(list): The label of each video,
- seq(int): The number of frames of the intercepted video.
- mode(str): Frame fetching method, options:["part", "discrete", "average", "interval"].
- suffix(str): Format of video file. options:["picture", "video"].
- align(boolean): The video contains multiple actions.
- frame_interval(int): Interval between sampling frames.
- num_clips(int): The number of samples of a video.

**Returns:**

None

### MixJDE

> class msvideo.data.MixJDE(data_json,
                            split="train",
                            batch_size=1,
                            repeat_num=1,
                            transform=None,
                            shuffle=None,
                            num_parallel_workers=1,
                            num_shards=None,
                            shard_id=None)

Multi-dataset based on jde datasets.

**Parameters:**

- data_json (str): Path to a json file that have the path to files that have the path to video frames.
- split (str): The dataset split supports "train", or "test". Default: "train".
- batch_size (int): Batch size of dataset. Default:1.
- repeat_num (int): The repeat num of dataset. Default:1.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.

**Returns:**

None

> def msvideo.data.MixJDE.run()

Dataset pipeline.

### JDE

> class msvideo.data.JDE(seq_path,
                        data_root,
                        split="train",
                        transform=None,
                        batch_size=1,
                        repeat_num=1,
                        resize=224,
                        shuffle=None,
                        num_parallel_workers=1,
                        num_shards=None,
                        shard_id=None,
                        schema_json=None,
                        trans_record=None)

- Base: `VideoDataset`

**Parameters:**

- seq_path (str): Path to a file that have the path to video frames.
- data_root (str): Path to
- split (str): The dataset split supports "train", or "test". Default: "train".
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq (int): The number of frames of captured video. Default: 16.
- seq_mode (str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- align (boolean): The video contains multiple actions.Default: False.
- batch_size (int): Batch size of dataset. Default:1.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.

**Returns:**

None

> class msvideo.data.JDE.read_dataset(*args)

**Returns:**

the path and the label of image: str

### ParseJDE

> class msvideo.data.ParseJDE()

Parse JDE dataset.

- Base: ParseDataset

> def msvideo.data.ParseJDE.parse_dataset()

Traverse the JDE dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### Kinetic400

> class msvideo.data.Kinetic400(path=path,
                         split=split,
                         load_data=load_data,
                         transform=transform,
                         target_transform=target_transform,
                         seq=seq,
                         seq_mode=seq_mode,
                         align=align,
                         batch_size=batch_size,
                         repeat_num=repeat_num,
                         shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         download=download,
                         frame_interval=frame_interval,
                         num_clips=num_clips)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part" or "discrete" fetch. Default: "part".
- align(boolean): The video contains multiple actions. Default: False.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel. Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool): Whether to download the dataset. Default: False.
- frame_interval (int): Frame interval of the sample strategy. Default: 1.
- num_clips (int): Number of clips sampled in one video. Default: 1.

**Returns:**

None

> def msvideo.data.Kinetic400.index2label()

Get the mapping of indexes and labels.

**Returns:**

The mapping of indexes and labels

> def msvideo.data.Kinetic400.default_transform()

Set the default transform for Kinetics400 dataset.

**Returns:**

transform: list


### ParseKinetic400

> class msvideo.data.ParseKinetic400()

Parse Kinetics400 dataset.

- Base: ParseDataset

> def msvideo.data.ParseKinetics400.load_cls_file()

Parse the category file.

**Returns:**

id2cls: list
cls2id: dict

> def msvideo.data.ParseKinetic400.parse_dataset()

Traverse the Kinetics400 dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### Kinetic600

> class msvideo.data.Kinetic600(path=path,
                         split=split,
                         load_data=load_data,
                         transform=transform,
                         target_transform=target_transform,
                         seq=seq,
                         seq_mode=seq_mode,
                         align=align,
                         batch_size=batch_size,
                         repeat_num=repeat_num,
                         shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         download=download)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" , "val" or "infer". Default: train.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
- align(boolean): The video contains multiple actions. Default: False.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.

**Returns:**

None

> def msvideo.data.Kinetic600.index2label()

Get the mapping of indexes and labels.

**Returns:**

The mapping of indexes and labels

> def msvideo.data.Kinetic600.default_transform()

Set the default transform for Kinetics400 dataset.

**Returns:**

transform: list


### ParseKinetic600

> class msvideo.data.ParseKinetic600()

Parse Kinetics600 dataset.

- Base: ParseDataset

> def msvideo.data.ParseKinetics600.load_cls_file()

Parse the category file.

**Returns:**

id2cls: list
cls2id: dict

> def msvideo.data.ParseKinetic600.parse_dataset()

Traverse the Kinetics600 dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### DatasetToMR

> class msvideo.data.DatasetToMR(load_data, destination, split, partition_number, schema_json, shard_id)

Transform dataset to MindRecord.

> def msvideo.data.DatasetToMR.trans_to_mr()

Execute transformation from dataset to MindRecord.

**Return:**

filename: str


### Dataset

> class msvideo.data.Dataset(path: str,
                            split: str,
                            load_data: Callable,
                            batch_size: int,
                            repeat_num: int,
                            shuffle: bool,
                            num_parallel_workers: Optional[int],
                            num_shards: int,
                            shard_id: int,
                            resize: Optional[int] = None,
                            transform: Optional[Callable] = None,
                            target_transform: Optional[Callable] = None,
                            mode: Optional[str] = None,
                            columns_list: Optional[list] = None,
                            schema_json: Optional[dict] = None,
                            trans_record: Optional[bool] = None)

Dataset is the base class for making dataset which are compatible with MindSpore Vision.

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- load_data(callable): The corresponding video data. 
- batch_size (int): Batch size of dataset. Default:16.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- mode(str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- columns_list(str): Column names of dataset.
- schema_json(dict): Mapping of category and id.
- trans_record(bool): Have the transform record or not. Defalt: None.

**Return:**

None

> def msvideo.data.Dataset.index2label()

Get the mapping of indexes and labels.

> def msvideo.data.Dataset.default_transform()

Default data augmentation.

> def msvideo.data.Dataset.pipelines()

Data augmentation.

> def msvideo.data.Dataset.run()

Dataset pipeline.


### ParseDataset

> class msvideo.data.ParseDataset(path: str, shard_id: Optional[int] = None)

Parse dataset.

> def msvideo.data.ParseDatasetparse_dataset(*args)

parse dataset from internet or compression file.


### Mot16

> class msvideo.data.Mot16(path,
                 split="train",
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=False,
                 batch_size=1,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False,
                 suffix="picture")

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", or "test". Default: "train".
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part"), "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: False.
- batch_size (int): Batch size of dataset. Default:1.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.
- suffix(str): Storage format of video. Default: "picture".

**Returns:**

None

> def msvideo.data.Mot16.index2label()

Get the mapping of indexes and labels.

> def msvideo.data.Mot16.download_dataset()

Download the Mot16 data if it doesn't exist already.


### ParseMot16

> class msvideo.data.ParseMot16()

Parse Mot16 dataset.

- Base: ParseDataset

> def msvideo.data.ParseMot16.parse_dataset()

Traverse the Mot16 dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### MSVD

> class msvideo.data.MSVD(path,
                 split="",
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=True,
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part"), "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: True.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.

**Returns:**

None

> def msvideo.data.MSVD.index2label()

Get the mapping of indexes and labels.

> def msvideo.data.MSVD.download_dataset()

Download the MSVD data if it doesn't exist already.


### ParseMSVD

> class msvideo.data.ParseMSVD()

Parse MSVD dataset.

- Base: ParseDataset

> def msvideo.data.ParseMSVD.load_label()

Parse annotation file.

**Returns:**

label2index: dict
index2label: list

> def msvideo.data.ParseMSVD.parse_dataset()

Traverse the MSVD dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### TaskGenerator

> class msvideo.data.TaskGenerator(path, cls, n, k, q)

N-way K-Shot Tasks generator for getting video path and its corresponding label. There are N categories in each task, including K labeled samples in each category.

**Parameters:**

- path(str): video file path.
- cls(list): the ending index of the video for each category, the index is start from 1.
- n(int): the number of categories per task.
- k(int): the number of label samples in each category
- q(int): the number of unlabeled samples in each category.


### Thumos

> class msvideo.data.Thumos(path,
                 split="train",
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=False,
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test", "val" or "background". Default: "infer".
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
- align(boolean): The video contains multiple actions. Default: False.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.

**Returns:**

None

> def msvideo.data.Thumos.index2label()

Get the mapping of indexes and labels.


### ParseThumos

> class msvideo.data.ParseThumos()

Parse Thumos dataset.

- Base: ParseDataset

> def msvideo.data.ParseThumos.parse_dataset()

Traverse the Thumos dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### UbiFights

> class msvideo.data.UbiFights(path,
                 split="test",
                 transform=None,
                 target_transform=None,
                 seq=16,
                 seq_mode="part",
                 align=True,
                 batch_size=16,
                 repeat_num=1,
                 shuffle=None,
                 num_parallel_workers=1,
                 num_shards=None,
                 shard_id=None,
                 download=False)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: None.
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part", "discrete", or "align" fetch. Default: "align".
- align(boolean): The video contains multiple actions.Default: True.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.

**Returns:**

None

> def msvideo.data.UbiFights.index2label()

Get the mapping of indexes and labels.

> def msvideo.data.UbiFights.download_dataset()

Download the UBI-fights data if it doesn't exist already.


### ParseUbiFights

> class msvideo.data.ParseUbiFights()

Parse UbiFights dataset.

- Base: ParseDataset

> def msvideo.data.ParseUbiFights.parse_dataset()

Traverse the UbiFights dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list


### UCF101

> class msvideo.data.UCF101(path: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 seq: int = 16,
                 seq_mode: str = "part",
                 align: bool = False,
                 batch_size: int = 16,
                 repeat_num: int = 1,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[bool] = None,
                 shard_id: Optional[bool] = None,
                 download: bool = False,
                 suffix: str = "video",
                 task_num: int = 0,
                 task_n: int = 0,
                 task_k: int = 0,
                 task_q: int = 0)

- Base: `VideoDataset`

**Parameters:**

- path (string): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: "train".
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part") or "discrete" fetch. Default: "part".
- align(boolean): The video contains multiple actions. Default: False.
- batch_size (int): Batch size of dataset. Default:16.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel.Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.
- suffix(str): Video format to be processed. Optional:("video", "picture", "task"). Default:"video".
- task_num(int): Number of tasks in few shot learning. Default: 0.
- task_n(int): Number of categories per task in few shot learning. Default:0.
- task_k(int): Number of support sets per task in few shot learning. Default:0.
- task_q(int): Number of query sets per task in few shot learning. Default:0.

**Returns:**

None

> def msvideo.data.UCF101.index2label()

Get the mapping of indexes and labels.

> def msvideo.data.UCF101.download_dataset()

Download the UCF101 data if it doesn't exist already.

> def msvideo.data.UCF101.default_transform()

Set the default transform for UCF101 dataset.

> def msvideo.data.UCF101.create_task()

Create task list in few shot learning.


### ParseUCF101

> class msvideo.data.ParseUCF101()

Parse UCF101 dataset.

- Base: ParseDataset

> def msvideo.data.ParseUCF101.parse_dataset()

Traverse the UCF101 dataset file to get the path and label.

**Returns:**

- video_path: list
- video_label: list

> def msvideo.data.ParseUCF101.load_cls()

Parse category file.

> def msvideo.data.ParseUCF101.modify_struct()

If there is no category subdirectory in the folder, modify the file structure.


### VideoDataset

> class msvideo.data.VideoDataset(path: str,
                                split: str,
                                load_data: Union[Callable, Tuple],
                                transform: Optional[Callable],
                                target_transform: Optional[Callable],
                                seq: int,
                                seq_mode: str,
                                align: bool,
                                batch_size: int,
                                repeat_num: int,
                                shuffle: bool,
                                num_parallel_workers: Optional[int],
                                num_shards: int,
                                shard_id: int,
                                download: bool,
                                columns_list: List = ['video', 'label'],
                                suffix: str = "video",
                                frame_interval: int = 1,
                                num_clips: int = 1)

VideoDataset is the base class for making video dataset which are compatible with MindSpore Vision.

- Base: `Dataset`

**Parameters:**

- path (str): Root directory of the Mnist dataset or inference image.
- split (str): The dataset split supports "train", "test" or "infer". Default: "infer".
- transform (callable, optional): A function transform that takes in a video. Default:None.
- target_transform (callable, optional): A function transform that takes in a label. Default: None.
- seq(int): The number of frames of captured video. Default: 16.
- seq_mode(str): The way of capture video frames,"part" or "discrete" fetch. Default: "part".
- align(bool): The video contains multiple actions. Default: False.
- batch_size (int): Batch size of dataset. Default:32.
- repeat_num (int): The repeat num of dataset. Default:1.
- shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default:None.
- num_parallel_workers (int): Number of subprocess used to fetch the dataset in parallel. Default: 1.
- num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
- shard_id (int, optional): The shard ID within num_shards. Default: None.
- download (bool) : Whether to download the dataset. Default: False.
- frame_interval(int):The number of frame interval when reading video. Default: 1.
- num_clips(int):The number of video clips read per video.

> def msvideo.data.VideoDataset.index2label()

Get the mapping of indexes and labels.

> def msvideo.data.VideoDataset.download_dataset()

Download the VideoDataset data if it doesn't exist already.

> def msvideo.data.VideoDataset.default_transform()

Set the default transform for VideoDataset dataset.


### check_file_exist

> def msvideo.data.check_file_exist(file_name: str)

Check the input filename is exist or not.

**Parameters:**

- file_name (str): File name.

**Returns:**

None

**Raises:**

FileNotFoundError: If file is not exist, print "File `{file_name}` does not exist."

### check_file_valid

> def msvideo.data.check_file_valid(filename: str, extension: Tuple[str, ...])

Check image file is valid through the extension.

**Parameters:**

- filename (str): File name.
- extension (Tuple[str, ...]): Extension of files.

**Returns:**

Str




### check_dir_exist

> def msvideo.data.check_dir_exist(dir_name: str)

Check the input directory is exist or not.

**Parameters:**

- dir_name (str): Name of directory.

**Returns:**

None

**Raises:**

FileNotFoundError: If the directory is not exist, print "Directory `{dir_name}` does not exist."

### save_json_file

> def msvideo.data.save_json_file(filename: str, data: Dicts)

Save json file.

**Parameters:**

- filename (str): File to be saved.
- data (dict): Data of json file.

**Returns:**

None


### load_json_file

> def msvideo.data.load_json_file(filename: str)

Load json file.

**Parameters:**

- filename (str): File to be loaded.

**Returns:**

None

### detect_file_type

> def msvideo.data.detect_file_type(filename: str)

Detect file type by suffixes and return tuple(suffix, archive_type, compression).

**Parameters:**

- filename (str): File to be detected.

**Returns:**

Tuple(suffix, archive_type, compression)