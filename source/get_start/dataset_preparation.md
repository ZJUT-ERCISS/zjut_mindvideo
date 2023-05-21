## Dataset Preparation

MindVideo supported dataset can be downloaded from:

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

Within `mindvideo`, all data processing methods according to each dataset used can be found under the `data` folder.
