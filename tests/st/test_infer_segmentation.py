
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

@pytest.mark.parametrize('mode', ['PYNATIVE_FUNC'])
def test_train(mode, device_id=0, model='vistr_r50'):
    ''' infer on a ytvos subset dataset '''
    # original vistr config file
    config_path = './mindvideo/config/vistr/vistr.yaml'

    # dataset directory
    data_dir = '/usr/dataset/VOS/'
    if mode=='GRAPH':
        mode_num = 0
    else:
        mode_num = 1
    # dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    # if not os.path.exists(data_dir):
    #     DownLoad().download_and_extract_archive(dataset_url, './')

    # ---------------- test running infer.py using the toy data ---------
    # number of videos 
    video_num = 1
    # path to pretrained ckpt
    ckpt_path = '/home/publicfile/checkpoint/vistr/ckpt/vistr_r50_all.ckpt'
    # path to dcn.conv weights
    weight_path = '/home/publicfile/checkpoint/vistr/ckpt/weights_r50.npy'
    # path of temporary config file
    temp_config_path = f'{tests_dir}/temp_config.yaml'

    # infer.py for segmentation
    infer_file = 'mindvideo/tools/segmentation/infer.py'

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    mode: .*#    mode: {mode_num}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    device_id: .*#    device_id: {device_id}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    pretrained_model: .*#    pretrained_model: {ckpt_path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    weights: .*#    weights: {weight_path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    video_num: .*#    video_num: {video_num}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#            path: .*#            path: {data_dir}#", temp_config_path])
    
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