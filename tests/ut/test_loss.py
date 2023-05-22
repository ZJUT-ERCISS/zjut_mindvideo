import sys
sys.path.append('.')
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.initializer import Normal
from mindspore.nn import TrainOneStepCell, WithLossCell

from mindvideo.data.transforms.jde_load import JDELoad
from mindvideo.data import MixJDE
from mindvideo.loss.centernet_multipose_loss import FocalLoss, RegLoss
from mindvideo.loss.vistr_loss import DiceLoss, SigmoidFocalLoss

ms.set_seed(1)
np.random.seed(1)

class SimpleNet(nn.Cell):
    def __init__(self, in_channels=72):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 36, 1, pad_mode='valid')
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

class LossCell_Seg(nn.Cell):
    def __init__(self, net, loss):
        super(LossCell_Seg, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, x, y, num_boxes):
        output = self._net(x)
        loss = self._loss(output[0], y[0], num_boxes)
        return loss

class SimpleNet2(nn.Cell):
    def __init__(self, in_channels=1):
        super(SimpleNet2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 1, 3, pad_mode='valid')
        self.softmax = nn.Softmax()

    def construct(self, x):
        x = self.conv1(x)
        x = self.softmax(x)
        return x

class SimpleNet3(nn.Cell):
    def __init__(self, in_channels=4):
        super(SimpleNet3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 4, 3, pad_mode='valid')
        self.softmax = nn.Softmax()

    def construct(self, x):
        x = self.conv1(x)
        x = self.softmax(x)
        return x

class LossCell_Tra_F(nn.Cell):
    def __init__(self, net, loss):
        super(LossCell_Tra_F, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, x, y):
        output = self._net(x)
        loss = self._loss(output, y)
        return loss

class LossCell_Tra_R(nn.Cell):
    def __init__(self, net, loss):
        super(LossCell_Tra_R, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, x, reg_mask, ind, wh):
        output = self._net(x)
        loss = self._loss(output, reg_mask, ind, wh)
        return loss


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['DiceLoss', 'SigmoidFocalLoss', 'FocalLoss', 'RegLoss'])
def test_loss(mode, name):
    print(f'mode={mode}; loss_name={name}')
    ms.set_context(mode=mode)

    bs = 1
    # set loss and model
    if name == 'DiceLoss':
        x = ms.Tensor(np.random.randn(bs, 72, 255), ms.float32)
        y = ms.Tensor(np.ones((bs, 36, 255)), ms.float32)
        num_boxes = 36
        net_loss = DiceLoss()
        network = SimpleNet(in_channels=72)

        net_with_loss = LossCell_Seg(network, net_loss)

        net_opt = nn.AdamWeightDecay(params=network.trainable_params())

        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(x, y, num_boxes)

        for _ in range(10):
            cur_loss = train_network(x, y, num_boxes)

        print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

        assert cur_loss < begin_loss, 'Loss does NOT decrease'

    elif name == 'SigmoidFocalLoss':
        x = ms.Tensor(np.random.randn(bs, 72, 255), ms.float32)
        y = ms.Tensor(np.ones((bs, 36, 255)), ms.float32)
        num_boxes = 36
        net_loss = SigmoidFocalLoss()
        network = SimpleNet(in_channels=72)

        net_with_loss = LossCell_Seg(network, net_loss)

        net_opt = nn.AdamWeightDecay(params=network.trainable_params())

        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(x, y, num_boxes)

        for _ in range(10):
            cur_loss = train_network(x, y, num_boxes)

        print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

        assert cur_loss < begin_loss, 'Loss does NOT decrease'
    
    elif name == 'FocalLoss':
        x = ms.Tensor(np.random.rand(bs, 1, 154, 274), ms.float32)
        y = ms.Tensor(np.ones((bs, 1, 152, 272)), ms.float32)
        net_loss = FocalLoss()
        network = SimpleNet2(in_channels=1)

        net_with_loss = LossCell_Tra_F(network, net_loss)

        net_opt = nn.Adam(network.trainable_params())

        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(x, y)

        for _ in range(10):
            cur_loss = train_network(x, y)

        print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

        assert cur_loss < begin_loss, 'Loss does NOT decrease'
    
    elif name == 'RegLoss':
        net_loss = RegLoss()
        network = SimpleNet3(in_channels=4)

        x = ms.Tensor(np.random.rand(bs, 4, 154, 274), ms.float32)
        reg_mask = ms.Tensor(np.ones((bs, 500)), ms.float32)
        ind = ms.Tensor(np.ones((bs, 500)), ms.int32)
        wh = ms.Tensor(np.ones((bs, 500, 4)), ms.float32)

        net_with_loss = LossCell_Tra_R(network, net_loss)

        net_opt = nn.Adam(network.trainable_params())

        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(x, reg_mask, ind, wh)

        for _ in range(10):
            cur_loss = train_network(x, reg_mask, ind, wh)

        print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

        assert cur_loss < begin_loss, 'Loss does NOT decrease'
