
''' 
Test train and validate pipelines.
For training, both graph mode and pynative mode with ms_function will be tested.
'''
import sys
sys.path.append('.')

import subprocess
from subprocess import Popen, PIPE
import os
import pytest
import mindvideo
# from mindcv.utils.download import DownLoad

check_acc = True
tests_dir = "./tests"

@pytest.mark.parametrize('mode', ['GRAPH', 'PYNATIVE_FUNC'])
def test_train(mode, device_id=0, model='c3d'):
    ''' train on a UCF101 subset dataset '''
    # original c3d config file
    config_path = 'mindvideo/config/c3d/c3d.yaml'

    # dataset directory
    data_dir = '/home/publicfile/UCF101-dataset/data'
    if mode=='GRAPH':
        mode_num = 0
    else:
        mode_num = 1
    # dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    # if not os.path.exists(data_dir):
    #     DownLoad().download_and_extract_archive(dataset_url, './')

    # ---------------- test running train.py using the toy data ---------

    # path to pretrained ckpt
    ckpt_path = '.vscode/c3d.ckpt'
    # batch size 
    batch_size = 16
    # path of temporary config file
    temp_config_path = f'{tests_dir}/temp_config.yaml'
    # infer.py for classification
    infer_file = 'mindvideo/tools/classification/infer.py'

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    mode: .*#    mode: {mode_num}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    device_id: .*#    device_id: {device_id}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#            path: .*#            path: {data_dir}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    pretrained_model: .*#    pretrained_model: {ckpt_path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    batch_size: .*#    batch_size: {batch_size}#", temp_config_path])

    print(f'Preparing config file:')
    for cmd in prepare_cmds:
        print(cmd)
        ret = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        assert ret==0, 'Prepare config file fails.'


    cmd = f"python {infer_file} -c {temp_config_path}"
    print(f'Running command: \n{cmd}')
    # ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    print(out)
    # assert out==0, 'Infer fails.'