
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
def test_train(mode, device_id=0, model='fairmot'):
    ''' train on a UCF101 subset dataset '''
    # original fairmot_dla34 config file
    config_path = 'mindvideo/config/fairmot/fairmot_dla34.yaml'

    # dataset directory
    data_dir = '/home/publicfile/dataset/tracking'
    if mode=='GRAPH':
        mode_num = 0
    else:
        mode_num = 1
    # dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    # if not os.path.exists(data_dir):
    #     DownLoad().download_and_extract_archive(dataset_url, './')

    # ---------------- test running train.py using the toy data ---------
    # path to pretrained ckpt
    ckpt_path = '/home/yanlq/mindvideo_github/fairmot_mindspore/.vscode/fairmot_dla34-17_886_conv2d.ckpt'
    # path of temporary config file
    temp_config_path = f'{tests_dir}/temp_config.yaml'
    # infer.py for tracking
    infer_file = 'mindvideo/tools/tracking/infer.py'

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    mode: .*#    mode: {mode_num}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    device_id: .*#    device_id: {device_id}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#            path: .*#            path: {data_dir}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    ckpt_path: .*#    ckpt_path: {ckpt_path}#", temp_config_path])

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