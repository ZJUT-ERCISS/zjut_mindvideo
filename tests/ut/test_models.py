import sys
sys.path.append('.')

import pytest
import numpy as np

import mindspore
import mindspore as ms
from mindspore import Tensor

from msvideo.models.nonlocal3d import nonlocal3d
from msvideo.models.x3d import x3d, BlockX3D
from msvideo.models.c3d import C3D
from msvideo.models.i3d import I3D
from msvideo.models.r2plus1d import R2Plus1d18
from msvideo.models.arn import ARN
from msvideo.models.swin3d import swin3d_b
from msvideo.models.vistr import VistrCom


from msvideo.models.fairmot import FairmotDla34
def test_fairmot():
    model = FairmotDla34()


def test_x3d():
    batch = 1
    frames = 16
    image_size = 224
    class_nums = 400

    model = x3d(block=BlockX3D,
                depth_factor=2.2,
                num_frames=16,
                train_crop_size=224,
                num_classes=400,
                dropout_rate=0.5,
                eval_with_clips=False)

    dummy_input = Tensor(np.random.rand(batch, 3, frames, image_size, image_size), dtype=mindspore.float32)
    y = model(dummy_input)
    assert y.shape==(batch, class_nums), 'output shape not match'


def test_nonlocal():
    model = nonlocal3d(in_d=32,
                       in_h=224,
                       in_w=224,
                       num_classes=400,
                       keep_prob=1.0)
    batch = 1
    frames = 16
    image_size = 224
    class_nums = 400
    dummy_input = Tensor(np.random.rand(batch, 3, frames, image_size, image_size), dtype=mindspore.float32)
    y = model(dummy_input)
    assert y.shape==(batch, class_nums), 'output shape not match'


def test_C3D():
    model = C3D(in_d=16,
                in_h=112,
                in_w=112,
                in_channel=3,
                kernel_size=[3,3,3],
                head_channel=[4096, 4096],
                num_classes=101,
                keep_prob=[0.5, 0.5, 1.0])
    batch = 1
    frames = 16
    image_size = 128
    class_nums = 101     
    dummy_input = Tensor(np.random.rand(batch, 3, frames, image_size, image_size), dtype=mindspore.float32)
    y = model(dummy_input)
    assert y.shape==(batch, class_nums), 'output shape not match'


def test_I3D():
    data_input = Tensor(np.random.rand(1, 3, 16, 224, 224), dtype=mindspore.float32)
    model = I3D()
    assert model, 'model error'


def test_r2plus1d():
    model = R2Plus1d18(num_classes=400)
    data_input = Tensor(np.random.rand(1, 3, 16, 112, 112), dtype=mindspore.float32)
    out = model(data_input)
    assert out.shape==(1, 400), 'output shape not match'


def test_ARN():
    model = ARN(support_num_per_class=1,
                query_num_per_class=1, 
                class_num=5, 
                is_c3d=False, 
                in_channels=3, 
                out_channels=64,
                jigsaw=10,
                sigma=10)
    assert model, 'model error'
    # data_input = Tensor(np.random.rand(1, 3, 10, 16, 128, 128), dtype=mindspore.float32)
    # out = model(data_input)
    # assert out.shape==(1, 5, 5), 'output shape not match'


def test_swin_3d():
    model = swin3d_b()
    batch = 2
    frames = 16
    image_size = 224
    class_nums = 400
    dummy_input = Tensor(np.random.rand(batch, 3, frames, image_size, image_size), dtype=mindspore.float32)
    y = model(dummy_input)
    assert y.shape==(batch, class_nums), 'output shape not match'


def test_vistr():
    model = VistrCom()
    assert model, 'model error'


# if __name__== '__main__':

#     test_fairmot()    

#     test_x3d()

#     test_nonlocal()
    
#     test_C3D()

#     test_I3D()

#     test_r2plus1d()

#     test_ARN()

#     test_swin_3d()

#     test_vistr()
