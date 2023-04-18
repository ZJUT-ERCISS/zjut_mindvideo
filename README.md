# Welcome

![LOGO.png](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/msvideo/resource/ms_video_logo.png)

[![docs](https://camo.githubusercontent.com/d5d535f53f2cb047c2b4382b8fd3c2913519abad35badcd4f22bd45d174f450a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f63732d6c61746573742d626c7565)](https://msvideo-guidebook.readthedocs.io/en/latest/) [![license](https://camo.githubusercontent.com/d4dc5ba23f0f26ac45a8419e6669afe324f992b413b2006d5f59ac548b1da725/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f6d696e6473706f72652d6c61622f6d696e6463762e737667)](https://gitee.com/yanlq46462828/zjut_mindvideo/blob/master/LICENSE) [![open issues](https://camo.githubusercontent.com/746aed3806dcfd86e6ada45e8f0be5e79c349bcaa5f44317b1feef8dc3498abb/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6973737565732f6d696e6473706f72652d6c61622f6d696e646376)](https://gitee.com/yanlq46462828/zjut_mindvideo/issues)[![PRs](https://camo.githubusercontent.com/64b454ccdf96dc519c389355e075c9d752f717216743d7cb3270643e27f49d1b/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5052732d77656c636f6d652d70696e6b2e737667)](https://
.com/yanlq46462828/zjut_mindvideo/pulls)

English|[中文](https://gitee.com/yanlq46462828/zjut_mindvideo/blob/master/README_CN.md)

## Introduction

MindSpore Video(msvideo) is an open source Video toolbox  for computer vision research and development based on MindSpore. It collects a series of classic and SoTA vision models, such as C3D and ARN, along with their pre-trained weights and training strategies.. With the decoupled module design, it is easy to apply or adapt msvideo to your own CV tasks.

### Major Features

- Modular Design

We decompose the video framework into different components and one can easily construct a customized video framework by combining different modules.

![ModularDesign.png](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/msvideo/resource/modular_design.png)

Currently, MindVideo supports the Action Recognition , Video Tracking, Video segmentation.

![result.gif](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/msvideo/resource/result.gif)

![result.png](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/msvideo/resource/result.png)

![MOT17_09_SDP.gif](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/msvideo/resource/MOT17_09_SDP.gif)

### Benchmark Results

The performance of the models trained with MindVideo is summarized in [benchmark.md](https://gitee.com/yanlq46462828/zjut_mindvideo/blob/master/source/introduction/benchmark.md), where the training recipes and weights are both available.

## Installation

### Dependency

Use the following commands to install dependencies:

```text
git clone https://gitee.com/yanlq46462828/zjut_mindvideo.git
cd zjut_mindvideo

# Please first install mindspore according to instructions on the official website: https://www.mindspore.cn/install

pip install -r requirements.txt
pip install -e .
```

### Dataset Preparation

MindSpore Video(msvideo) supported dataset can be downloaded from:

- [activitynet](http://activity-net.org/index.html)
- [Kinetics400](https://www.deepmind.com/open-source/kinetics) 
- [Kinetics600](https://www.deepmind.com/open-source/kinetics) 
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [Caltech Pedestrian](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [CityPersons](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [CUHK-SYSU](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [PRW](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [MOT17](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [MOT16](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)
- [charades](https://prior.allenai.org/projects/charades)
- [Collective Activity](https://cvgl.stanford.edu/projects/collective/collectiveActivity.html)
- [columbia Consumer Video](https://www.ee.columbia.edu/ln/dvmm/CCV/)
- [davis](https://davischallenge.org/davis2016/code.html)
- [hmdb51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
- [fbms](https://paperswithcode.com/dataset/fbms-59)
- [msvd](https://paperswithcode.com/dataset/msvd)
- [Sports-1M](https://paperswithcode.com/dataset/sports-1m)
- [THUMOS](https://www.crcv.ucf.edu/THUMOS14/download.html)
- [UBI-Fights](https://paperswithcode.com/dataset/ubi-fights)
- [tyvos](https://youtube-vos.org/dataset)

Then put all training and evaluation data into one directory and then change `"data_root"` to that directory in [data.json](https://github.com/ZJUT-ERCISS/fairmot_mindspore/blob/main/datas/data.json), like this:

```
"data_root": "/home/publicfile/dataset/tracking"
```

Within `msvideo`, all data processing methods according to each dataset used can be found under the `data` folder.

## Quick Start

### Running

There are two approaches provided for training, evaluation and inference within `msvideo` for each supported model. After installing MindSpore via the official website, one is to run the training or evaluation files under the `example` folder, which is a independent module for training and evaluation specifically designed for starters, according to each model's name. And the other is to use the train and inference interfaces for all models under the root folder of the repository when working with the `YAML` file containing the parameters needed for each model as we also support some parameter configurations for quick start. For this method, take I3D for example, just run following commands for training:

```
python train.py -c zjut_mindvideo/msvideo/config/i3d/i3d_rgb.yaml
```

and run following commands for evaluation:

```
python infer.py -c zjut_mindvideo/msvideo/config/i3d/i3d_rgb.yaml
```

Also, [paperswithcode](https://paperswithcode.com) is a good resource for browsing the models within `msvideo`, each can be found at:

| Model    | Link                                                         |
| :------- | :----------------------------------------------------------- |
| ARN      | https://paperswithcode.com/paper/few-shot-action-recognition-via-improved#code |
| C3D      | https://paperswithcode.com/paper/learning-spatiotemporal-features-with-3d#code |
| Fairmot  | https://paperswithcode.com/paper/a-simple-baseline-for-multi-object-tracking#code |
| I3D      | https://paperswithcode.com/paper/quo-vadis-action-recognition-a-new-model-and#code |
| Nonlocal | https://paperswithcode.com/paper/non-local-neural-networks   |
| R(2+1)D  | https://paperswithcode.com/paper/a-closer-look-at-spatiotemporal-convolutions |
| Vist     | https://paperswithcode.com/paper/video-swin-transformer#code |
| X3D      | https://paperswithcode.com/paper/x3d-expanding-architectures-for-efficient |
| Vistr    | https://paperswithcode.com/paper/end-to-end-video-instance-segmentation-with#code |

### Model Checkpoints

The links to download the pre-train models are as follows:

<table>
	<tr>
	    <td>Model</td>
	    <td>Link</td>
	</tr >
        <tr>
	    <td>ARN</td>
	    <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/ER55hujI22BOkyjL5UrBVt0BfKx8lmeW5DRctx46tfZRkA?e=hdIZIu">arn.ckpt</a></td>
	</tr>
	<tr>
	    <td>C3D</td>
	    <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EbVF6SuKthpGj046abA37jkBkfkhzLm36F8NJmH2Do3jhg?e=xh32kW">c3d.ckpt</a></td>
	</tr>
	<tr>
       <td>Fairmot</td>
	   <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EdU2TA3NrqVFpj-Adkh2RiEB_UZoxLHiNFj6tcuMDylQVA?e=YIWTiH">fairmot_dla34-30_886.ckpt</a></td>
	</tr>
    <tr>
       <td>I3D</td>
	   <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EeqkpDHObpBNj5ibeawTY0gBWd84YvFrhmbdGeu8qm5SDw?e=E3j8vM">i3d_rgb_kinetics400.ckpt</a></td>
	</tr>
    <tr>
       <td>Nonlocal</td>
	   <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/Ec-B_Hr00QRAs49Vd7Qg4PkBslya1SjAola4hg64tpI6Vg?e=YNm0Ig">nonlocal_mindspore.ckpt</a></td>
	</tr>
    <tr>
       <td>R(2+1)D</td>
	   <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EXT6cCmxV59Gp4U9VChcmuUB2Fmuhfg7SRkfuxGsOiyBUA?e=qJ9Wc1">r2plus1d18_kinetic400.ckpt</a></td>
	</tr>
    <tr>
       <td rowspan="4">X3D</td>
	   <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EaVbGiHvrf5Nl6TooLlq340B4LMrLF8Cqm9PH0w9Mlqx9Q?e=a2XEoh">x3d_l_kinetics400.ckpt</a></td>
	</tr>
    <tr>
        <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EVqLWmg7v4JBkLJPY3vP-1kBeq7uI5sE2Tin7kM5PcxQMw?e=S1wCy0">x3d_m_kinetics400.ckpt</a></td>
    </tr>
    <tr>
        <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EUH1YqWCkLlLlEMA9A8MuwQBSPQ0yjyUJVUIlsuWbP3YeQ?e=WK955U">x3d_s_kinetics400.ckpt</a></td>
    </tr>
    <tr>
        <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EYwLbhrIcCdIor3J_Dxj3foBMx2bFb7zcw9QRVBkamZE_A?e=p4tDBt">x3d_xs_kinetics400.ckpt</a></td>
    </tr>
    <tr>
       <td rowspan="3">Swin3D</td>
	   <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EXrE7hbSqCtJoSourHbcUIABmnskD5qO0o9c_hpJ-x86PA?e=zdQ02f">ms_swin_base_patch244_window877_kinetics400_22k.ckpt
       </a></td>
	</tr>
    <tr>
        <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EZKHu92j3SVLlAfvC-gv1pcBUvXcexXo7H5Kv8QymqHpZQ?e=B3FOkI">ms_swin_small_patch244_window877_kinetics400_1k.ckpt
        </a></td>
    </tr>
    <tr>
        <td><a href="https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EX1foDC63eNNgnxbfD2oEDYB9C5JoLUfEgqlJ_4QymoJqQ?e=ayseUu">ms_swin_tiny_patch244_window877_kinetics400_1k.ckpt
        </a></td>
    </tr>
</table>
## Model List

- [x] [C3D](msvideo/example/arn/README.md) for Action Recognition.

- [x] [I3D](msvideo/example/i3d/README.md) for Action Recognition.

- [x] [X3D](msvideo/example/x3d/README.md) for Action Recognition.

- [x] [R(2+1)d](msvideo/example/r(2+1)d/README.md) for Action Recognition.

- [x] [NonLocal](msvideo/example/nonlocal/README.md) for Action Recognition.

- [x] [ViST](msvideo/example/vist/README.md) for Action Recognition.

- [x] [fairMOT](msvideo/example/fairmot/README.md) for One-shot Tracking.

- [x] [VisTR](msvideo/example/vistr/README.md) for Instance Segmentation. 

- [x] [ARN](msvideo/example/arn/README.md) for Few-shot Action Recognition.

  The master branch works with **MindSpore 1.5+**.

## Build Documentation

1. Clone msvideo

```bash
git clone https://github.com/ZJUT-ERCISS/zjut_mindvideo.git
cd zjut_mindvideo
```

2. Install the building dependencies of documentation

```bash
pip install -r requirements.txt
```

3. Build documentation

```bash
make html
```

4. Open `_build/html/index.html` with browser

## License

This project is released under the [Apache 2.0 license](https://gitee.com/moeno_ss/zjut_mindvideo/blob/st2/LICENSE).

## Supported Algorithms

Supported algorithms:

- Action Recognition 
- Video Tracking
- Video segmentation

## Base Structure

MindSpore Video(msvideo) is a MindSpore-based Python package that provides high-level features:

- Base backbone of models like c3d and resnet series.
- Domain oriented rich dataset interface.
- Rich visualization and IO(Input/Output) interfaces.

![BaseArchitecture.png](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/msvideo/resource/base_architecture.png)

### Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [Gitee Issues](https://gitee.com/yanlq46462828/zjut_mindvideo/issues).

### Contributing

We appreciate all contributions to improve MindSpore Video. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

### License

This project is released under the [Apache 2.0 license](LICENSE).

### Acknowledgement

MindSpore is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.The contributors are listed in  [CONTRIBUTERS.md](source\introduction\CONTRIBUTERS.md)

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Video 2022,
    title={{MindSpore Video}:MindSpore Video Toolbox and Benchmark},
    author={MindSpore Video Contributors},
    howpublished = {\url{https://gitee.com/yanlq46462828/zjut_mindvideo}},
    year={2022}
}
```
