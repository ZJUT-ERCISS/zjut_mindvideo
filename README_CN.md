<div align="center">

# MindVideo

[![docs](https://camo.githubusercontent.com/d5d535f53f2cb047c2b4382b8fd3c2913519abad35badcd4f22bd45d174f450a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f63732d6c61746573742d626c7565)](https://mindvideo-guidebook.readthedocs.io/en/latest/) [![license](https://camo.githubusercontent.com/d4dc5ba23f0f26ac45a8419e6669afe324f992b413b2006d5f59ac548b1da725/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f6d696e6473706f72652d6c61622f6d696e6463762e737667)](https://gitee.com/yanlq46462828/zjut_mindvideo/blob/master/LICENSE) [![open issues](https://camo.githubusercontent.com/746aed3806dcfd86e6ada45e8f0be5e79c349bcaa5f44317b1feef8dc3498abb/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6973737565732f6d696e6473706f72652d6c61622f6d696e646376)](https://gitee.com/yanlq46462828/zjut_mindvideo/issues)[![PRs](https://camo.githubusercontent.com/64b454ccdf96dc519c389355e075c9d752f717216743d7cb3270643e27f49d1b/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5052732d77656c636f6d652d70696e6b2e737667)](https://gitee.com/yanlq46462828/zjut_mindvideo/pulls)

[English](https://gitee.com/yanlq46462828/zjut_mindvideo/blob/master/README.md)|中文

</div>

## 简介

MindVideo 是一个基于MindSpore的计算机视觉研究和开发的开源视频工具箱。它收集了一系列经典的和SoTA的视觉模型，如C3D和ARN，以及它们的预训练权重和训练策略。通过解耦模块的设计，很容易将MindVideo 应用于或适应你自己的CV任务。

### 主要特点

- 模块化设计

我们将视频框架分解成不同的组件，人们可以通过结合不同的模块轻松构建一个定制的视频框架。

![ModularDesign.png](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/mindvideo/resource/modular_design.png)

目前，MindVideo支持动作识别、视频跟踪、视频分割。

![result.gif](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/mindvideo/resource/result.gif)

![result.png](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/mindvideo/resource/result.png)

![MOT17_09_SDP.gif](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/mindvideo/resource/MOT17_09_SDP.gif)

### 性能结果

用MindVideo训练的模型的性能总结在[benchmark.md](https://gitee.com/yanlq46462828/zjut_mindvideo/blob/master/source/introduction/benchmark.md)中，其中训练配方和权重都是可用的。

## 安装	

### 依赖

使用以下指令来安装依赖:

```text
git clone https://gitee.com/yanlq46462828/zjut_mindvideo.git
cd zjut_mindvideo

# If you use vistr, the version of Python should be 3.7
# Please first install mindspore according to instructions on the official website: https://www.mindspore.cn/install

pip install -r requirements.txt
pip install -e .
```

### 数据集准备

MindVideo 支持的数据集可以从以下链接下载：

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

然后将所有的训练和评估数据放入一个目录中，然后将"data_root"改为[data.json](https://github.com/ZJUT-ERCISS/fairmot_mindspore/blob/main/datas/data.json)中的该目录。

```
"data_root": "/home/publicfile/dataset/tracking"
```

在`mindvideo`中, 根据所使用的每个数据集的所有数据处理方法都可以在`data` 文件夹中找到。 

## 快速开始

### 运行

每个`mindvideo`支持的模型都有两种方法用于训练、评估和推理。通过官方网站安装MindSpore后，可以在 `example` 文件夹下运行训练或评估文件，根据每个模型的名称，示例 "文件夹是一个专门为初学者设计的训练和评估的独立模块。另一种是在处理包含每个模型所需参数的`YAML`文件时，使用版本库根文件夹下所有模型的训练和推理接口，因为我们也支持一些参数配置以快速启动。对于这种方法，以I3D为例，只需运行以下命令进行训练。

```
python train.py -c zjut_mindvideo/mindvideo/config/i3d/i3d_rgb.yaml
```

使用以下指令来评估:

```
python infer.py -c zjut_mindvideo/mindvideo/config/i3d/i3d_rgb.yaml
```

同时, [paperswithcode](https://paperswithcode.com) 是浏览`mindvideo`模型的良好资源，不同的模型都可以在以下链接找到:

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

### 模型检查点

以下使预训练模型的下载链接:

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

- [x] [C3D](mindvideo/example/arn/README.md) 动作识别

- [x] [I3D](mindvideo/example/i3d/README.md) 动作识别

- [x] [X3D](mindvideo/example/x3d/README.md) 动作识别

- [x] [R(2+1)d](mindvideo/example/r(2+1)d/README.md) 动作识别

- [x] [NonLocal](mindvideo/example/nonlocal/README.md) 动作识别

- [x] [ViST](mindvideo/example/vist/README.md) 动作识别

- [x] [fairMOT](mindvideo/example/fairmot/README.md) 单次学习跟踪

- [x] [VisTR](mindvideo/example/vistr/README.md)实例分割

- [x] [ARN](mindvideo/example/arn/README.md) 少样本动作识别

  主干分支的工作原理是 **MindSpore 1.5+**.

## 文档建立

[API文档教程](https://msvideo-guidebook.readthedocs.io/en/latest/)

1. 克隆mindvideo

```bash
git clone https://gitee.com/yanlq46462828/zjut_mindvideo.git
cd zjut_mindvideo
```

2. 安装文档的依赖

```bash
pip install -r requirements.txt
```

3. 建立文档

```bash
make html
```

4. 通过浏览器打开 `build/html/index.html` 

## 支持算法

支持算法:

- 动作识别 
- 视频跟踪
- 视频实例分割

## 基本结构

MindVideo是一个基于MindSpore的Python包，提供以下高级功能:

- c3d和resnet系列等模型的基础骨干。.
- 面向领域的丰富数据集界面
- 丰富的可视化IO接口.

![BaseArchitecture.png](https://gitee.com/yanlq46462828/zjut_mindvideo/raw/master/mindvideo/resource/base_architecture.png)

### 反馈与联系

动态版本仍在开发中，如果您发现任何问题或对新功能有想法，请不要犹豫，通过 [Gitee Issues](https://gitee.com/yanlq46462828/zjut_mindvideo/issues)与我们联系

### 贡献

我们感谢所有对改善 MindVideo的贡献。请参考[CONTRIBUTING.md](CONTRIBUTING.md)的贡献指南

### 许可证

这个项目是在[Apache 2.0 license](https://gitee.com/moeno_ss/zjut_mindvideo/blob/st2/LICENSE)下发布的。

### 致谢

MindSpore 是一个开源项目，欢迎任何贡献和反馈。我们希望这个工具箱和基准能够通过提供一个灵活和标准化的工具箱，重新实施现有的方法和开发他们自己的新的计算机视觉方法，为日益增长的研究界服务。贡献人在 [CONTRIBUTERS.md](source\introduction\CONTRIBUTERS.md)中列出

### 引用

如果你觉得mindvideo对你的项目有帮助，请考虑引用：

```latex
@misc{MindVideo 2022,
    title={{MindVideo}:MindVideo Toolbox and Benchmark},
    author={MindVideo Contributors},
    howpublished = {\url{https://gitee.com/yanlq46462828/zjut_mindvideo}},
    year={2022}
}
```