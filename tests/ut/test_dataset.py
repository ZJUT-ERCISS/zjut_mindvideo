import sys
sys.path.append('.')

from mindvideo.data import builder
import mindspore as ms
import pytest
import subprocess
from mindvideo.utils.config import Config



tests_dir = "./tests"


# test ytvos
@pytest.mark.parametrize('mode', [0, 1])
def test_create_dataset_ytvos(mode):
    ms.set_context(mode=mode)

    # download pass
    config_path = './mindvideo/config/vistr/vistr.yaml'
    temp_config_path = f'{tests_dir}/temp_config.yaml'

    type = 'Ytvos'
    path = "/usr/dataset/VOS"
    split = "train"
    seq = 36
    batch_size = 1
    repeat_num = 1
    shuffle = False
    num_parallel_workers = 1
    data_len = 'all'

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    type: .*#    type: {type}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    path: .*#    path: {path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    split: .*#    split: {split}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    seq: .*#    seq: {seq}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    batch_size: .*#    batch_size: {batch_size}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    repeat_num: .*#    repeat_num: {repeat_num}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    shuffle: .*#    shuffle: {shuffle}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    num_parallel_workers: .*#    num_parallel_workers: {num_parallel_workers}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    data_len: .*#    data_len: {data_len}#", temp_config_path])


    print(f'Preparing config file:')
    for cmd in prepare_cmds:
        print(cmd)
        ret = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        assert ret==0, 'Prepare config file fails.'

    # with open(temp_config_path) as file:
    #     args = yaml.load(file.read(), Loader=yaml.FullLoader)
    # config = Config(args.config)
    config = Config(temp_config_path)
    dataset = builder.build_dataset(config.data_loader.train.dataset)
    # assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None, 'Creating dataset fails'


# test UCF101
@pytest.mark.parametrize('mode', [0, 1])
def test_create_dataset_UCF101(mode):
    ms.set_context(mode=mode)

    # download dataset pass

    config_path = "./mindvideo/config/c3d/c3d.yaml"
    temp_config_path = f'{tests_dir}/temp_config.yaml'

    type = 'UCF101'
    path = '/home/publicfile/UCF101-dataset/data'
    split = 'train'
    batch_size = 16
    seq = 16
    seq_mode = "average"
    num_parallel_workers = 6
    shuffle = True

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    type: .*#    type: {type}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    path: .*#    path: {path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    split: .*#    split: {split}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    seq: .*#    seq: {seq}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    seq_mode: .*#    seq_mode: {seq_mode}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    batch_size: .*#    batch_size: {batch_size}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    shuffle: .*#    shuffle: {shuffle}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    num_parallel_workers: .*#    num_parallel_workers: {num_parallel_workers}#", temp_config_path])
      
    print(f'Preparing config file:')
    for cmd in prepare_cmds:
        print(cmd)
        ret = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        assert ret==0, 'Prepare config file fails.'

    config = Config(temp_config_path)
    dataset = builder.build_dataset(config.data_loader.train.dataset)
    # assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None, 'Creating dataset fails'


# test Kinetic400
@pytest.mark.parametrize('mode', [0, 1])
def test_create_dataset_Kinetic400(mode):

    ms.set_context(mode=mode)

    # download dataset pass

    config_path = "./mindvideo/config/i3d/i3d_rgb.yaml"
    temp_config_path = f'{tests_dir}/temp_config.yaml'

    type = 'Kinetic400'
    path = "/home/publicfile/kinetics-400"
    shuffle = True
    split = 'train'
    seq = 64
    num_parallel_workers = 8
    batch_size = 16

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    type: .*#    type: {type}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    path: .*#    path: {path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    split: .*#    split: {split}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    seq: .*#    seq: {seq}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    batch_size: .*#    batch_size: {batch_size}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    shuffle: .*#    shuffle: {shuffle}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    num_parallel_workers: .*#    num_parallel_workers: {num_parallel_workers}#", temp_config_path])

    print(f'Preparing config file:')
    for cmd in prepare_cmds:
        print(cmd)
        ret = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        assert ret==0, 'Prepare config file fails.'

    config = Config(temp_config_path)
    dataset = builder.build_dataset(config.data_loader.train.dataset)
    # assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None, 'Creating dataset fails'

# test
@pytest.mark.parametrize('mode', [0, 1])
def test_create_dataset_MixJDE(mode):
    ms.set_context(mode=mode)

    # download dataset pass

    config_path = "./mindvideo/config/fairmot/fairmot_dla34.yaml"
    temp_config_path = f'{tests_dir}/temp_config.yaml'

    type = 'MixJDE'
    data_json = "/home/publicfile/dataset/tracking/datas/data.json"
    split = 'train'
    batch_size = 2
    num_parallel_workers = 1
    shuffle = True

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    type: .*#    type: {type}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    data_json: .*#    data_json: {data_json}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    split: .*#    split: {split}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    batch_size: .*#    batch_size: {batch_size}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    shuffle: .*#    shuffle: {shuffle}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    num_parallel_workers: .*#    num_parallel_workers: {num_parallel_workers}#", temp_config_path])

    print(f'Preparing config file:')
    for cmd in prepare_cmds:
        print(cmd)
        ret = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        assert ret==0, 'Prepare config file fails.'

    config = Config(temp_config_path)
    dataset = builder.build_dataset(config.train.data_loader.dataset)
    # assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None, 'Creating dataset fails'
