
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
import msvideo
# from mindcv.utils.download import DownLoad

check_acc = True
tests_dir = "./tests"

@pytest.mark.parametrize('mode', ['GRAPH', 'PYNATIVE_FUNC'])
def test_train(mode, device_id=3, model='c3d'):
    ''' train on a UCF101 subset dataset '''
    config_path = 'msvideo/config/c3d/c3d.yaml'
    # prepare data
    data_dir = '/home/publicfile/UCF101_splits/data'
    # data_dir = './UCF101_splits'
    num_classes = 101
    if mode=='GRAPH':
        mode_num = 0
    else:
        mode_num = 1
    # dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    # if not os.path.exists(data_dir):
    #     DownLoad().download_and_extract_archive(dataset_url, './')

    # ---------------- test running train.py using the toy data ---------
    dataset = 'ucf101'
    ckpt_path = './tests/ckpt_tmp'
    num_samples = 9536
    num_epochs = 2
    batch_size = 16
    temp_config_path = f'{tests_dir}/temp_config.yaml'

    train_file = 'msvideo/engine/classification/train.py'

    prepare_cmds = []
    prepare_cmds.append(["cp", config_path, temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    mode: .*#    mode: {mode_num}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    device_id: .*#    device_id: {device_id}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    ckpt_path: .*#    ckpt_path: {ckpt_path}#", temp_config_path])
    prepare_cmds.append(["sed", "-i", f"s#    epochs: .*#    epochs: {num_epochs}#", temp_config_path])
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
    
    # --------- Test running validate.py using the trained model ------------- #
    valid_file = 'msvideo/engine/classification/eval.py'

    #begin_ckpt = os.path.join(ckpt_dir, f'{model}-1_1.ckpt')
    end_ckpt = os.path.join(ckpt_path, f'{model}-{num_epochs}_{num_samples//batch_size}.ckpt')
    print(num_samples//batch_size)
    ret = subprocess.call(["sed", "-i", f"s#            batch_size: .*#            batch_size: {batch_size}#", temp_config_path],
        stdout=sys.stdout, stderr=sys.stderr)
    ret = subprocess.call(["sed", "-i", f"s#    pretrained_model: .*#    pretrained_model: {end_ckpt}#", temp_config_path],
        stdout=sys.stdout, stderr=sys.stderr)
    # "sed -i 's#    pretrained_model: .*#    pretrained_model: {end_ckpt}#' tests/temp_config.yaml"
    assert ret==0, 'Prepare config file fails.'
    
    cmd = f"python {valid_file} -c {temp_config_path}"
    print(f'Running command: \n{cmd}')
    # ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    assert out==0, 'Validation fails'

    if check_acc: 
        res = out.decode()
        idx = res.find('Accuracy')
        acc = res[idx:].split(',')[0].split(':')[1]
        print('Val acc: ', acc)
        if float(acc) > 0.5:
            print('Acc is too low.')
        # assert float(acc) > 0.5, 'Acc is too low'
    
    # subprocess.call(["rm", temp_config_path], stdout=sys.stdout, stderr=sys.stderr)
    # subprocess.call(["rm", "-r", ckpt_path], stdout=sys.stdout, stderr=sys.stderr)