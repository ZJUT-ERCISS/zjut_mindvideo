''' 
Test train and infer pipelines.
For training, pynative mode with ms_function will be tested.
'''
import sys
sys.path.append('.')

import subprocess
from subprocess import Popen, PIPE
import os
import pytest


check_acc = True
tests_dir = "./tests"


@pytest.mark.parametrize('mode', ['PYNATIVE_FUNC'])
def test_train(mode, device_id=2, model='vistr_r50'):
    ''' train on a ytvos subset dataset '''
    config_path = './msvideo/config/vistr/vistr.yaml'
    # prepare data
    data_dir = '/usr/dataset/VOS'
    mode_num = 1
    # dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    # if not os.path.exists(data_dir):
    #     DownLoad().download_and_extract_archive(dataset_url, './')

    # ---------------- test running train.py using the toy data ---------
    ckpt_path = './tests/ckpt_tmp'
    num_samples = 1
    num_epochs = 1
    batch_size = 1
    data_len = "1"
    temp_config_path = f'{tests_dir}/temp_config.yaml'

    train_file = './msvideo/engine/segmentation/train.py'

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    mode: .*#    mode: {mode_num}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    device_id: .*#    device_id: {device_id}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    ckpt_path: .*#    ckpt_path: {ckpt_path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    epoch_size: .*#    epoch_size: {num_epochs}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    data_len: .*#    data_len: {data_len}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    save_checkpoint_epochs: .*#    save_checkpoint_epochs: {num_epochs}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#            path: .*#            path: {data_dir}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#            batch_size: .*#            batch_size: {batch_size}#", temp_config_path])


    print(f'Preparing config file:')
    for cmd in prepare_cmds:
        print(cmd)
        ret = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        assert ret==0, 'Prepare config file fails.'

    # run train.py
    cmd = f"python {train_file} -c {temp_config_path}"

    print(f'Running command: \n{cmd}')
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret==0, 'Training fails.'
    # end training
    
    
    # --------- Test running infer.py using the trained model ------------- #
    infer_file = './msvideo/engine/segmentation/infer.py'

    #begin_ckpt = os.path.join(ckpt_dir, f'{model}-1_1.ckpt')
    prepare_cmds = []
    end_ckpt = os.path.join(ckpt_path, f'{model}-{num_epochs}_{num_samples//batch_size}.ckpt')
    save_path = 'tests/results.json'
    video_num = 1

    prepare_cmds.append(["sed", "-i", f"s#    pretrained_model: .*#    pretrained_model: {end_ckpt}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    save_path: .*#    save_path: {save_path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    video_num: .*#    video_num: {video_num}#", temp_config_path])
    for cmd in prepare_cmds:
        print(cmd)
        ret = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        assert ret==0, 'Prepare config file fails.'
    # "sed -i 's#    pretrained_model: .*#    pretrained_model: {end_ckpt}#' tests/temp_config.yaml"
    
    cmd = f"python {infer_file} -c {temp_config_path}"
    print(f'Running command: \n{cmd}')
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)

    assert ret==0, 'Infering fails'