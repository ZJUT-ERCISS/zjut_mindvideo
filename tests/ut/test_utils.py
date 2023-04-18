import numpy as np

import mindspore
from mindspore import context, load_checkpoint, load_param_into_net, Tensor
from msvideo.data.builder import build_dataset, build_transforms
from msvideo.models import build_model

import pytest
from msvideo.utils.check_param import Validator, Rel
from msvideo.utils.config import Config


def test_config():

    config_path = '/home/shr/mindspore/dabao/zjut_mindvideo/msvideo/config/x3d/x3d_m.yaml'
    config = Config(config_path)
    context.set_context(**config.context)
    # perpare dataset
    transforms = build_transforms(config.data_loader.eval.map.operations)
    data_set = build_dataset(config.data_loader.eval.dataset)
    data_set.transform = transforms
    dataset_infer = data_set.run()
    Validator.check_int(dataset_infer.get_dataset_size(), 0, Rel.GT)
    # set network
    network = build_model(config.model)
    # load pretrain model
    param_dict = load_checkpoint(config.infer.pretrained_model)
    load_param_into_net(network, param_dict)


def test_six_padding():
    from msvideo.utils.six_padding import six_padding
    res = six_padding([1,2,3])
    assert len(res)==6, 'shape not match'


def test_lossMonitor():
    from msvideo.utils.callbacks import LossMonitor
    lr = [0.01, 0.008, 0.006, 0.005, 0.002]
    monitor = LossMonitor(lr_init=lr, per_print_times=100)
    assert monitor


# @pytest.mark.parametrize('number',123)
def test_check_param():
    from msvideo.utils.check_param import check_is_number
    number = 3
    res = check_is_number(number, int, "bias", "bias_class")
    assert res


def test_gaussian():
    from msvideo.utils.gaussian import gaussian_radius, gaussian2d
    det_size = (3,5)
    res = gaussian_radius(det_size=det_size, min_overlap=0.7)
    ans = gaussian2d(det_size)
    assert ans.shape==det_size, 'shape not match'
    assert res


# def test_resized_mean():
#     from msvideo.utils.resized_mean import reisze_mean
#     vmean = reisze_mean(data_dir="/home/publicfile/UCF-101/YoYo",
#                          save_dir="/home/shr/",
#                          height=128,
#                          width=128)
#     print(vmean.shape)


def test_windows():
    from msvideo.utils.windows import limit_window_size, window_partition
    res = limit_window_size((16, 56, 56),(8,7,7),(4,3,3))
    data_input = Tensor(np.random.rand(1, 16, 48, 48, 3), dtype=mindspore.float32)
    ans = window_partition(data_input, (4,3,3))
    assert res
    assert ans.shape==(1024,36,3), 'shape not match'

