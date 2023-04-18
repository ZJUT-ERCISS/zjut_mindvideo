''' 
Test optimizer
'''
import numpy as np
from mindspore.common.initializer import Normal
from mindspore.nn import TrainOneStepCell, WithLossCell
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
import pytest
import sys
sys.path.append('.')


# optimizer:Adam SGD Momentum AdamweightDecay
def init_group_params(params, weight_decay):
    decay_params = []
    no_decay_params = []

    for param in params:
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
        {'order_params': params}
    ]

class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=10, in_channels=1, include_top=True):
        super(SimpleCNN, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.fc(x)
        return x


@pytest.mark.parametrize('opt', ['sgd', 'momentum'])
# @pytest.mark.parametrize('nesterov', [True, False])
@pytest.mark.parametrize('filter_bias_and_bn', [True, False])
def test_sgd_optimizer(opt, filter_bias_and_bn):

    network = SimpleCNN(in_channels=1, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    params = network.trainable_params()
    weight_decay = 1e-5
    if weight_decay and filter_bias_and_bn:
        params = init_group_params(network.trainable_params(), weight_decay)

    if opt == 'sgd':
        # note: nn.Momentum may perform better if momentum > 0.
        net_opt = nn.SGD(params=params,
                         learning_rate=0.01,
                         momentum=0.9,
                         weight_decay=1e-5,
                         loss_scale=1.0
                         )
    elif opt == 'momentum':
        net_opt = nn.Momentum(params=network.trainable_params(),
                              learning_rate=0.01,
                              momentum=0.9,
                              weight_decay=1e-5,
                              loss_scale=1.0
                              )

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f"{opt}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_adamweightdecay_optimizer():
    network = SimpleCNN(in_channels=1, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = nn.AdamWeightDecay(params=network.trainable_params(),
                                 learning_rate=0.001,
                                 beta1=0.9,
                                 beta2=0.999,
                                 weight_decay=0.001)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f"{'adamweightdecay'}, begin loss: {begin_loss}, end loss:  {cur_loss}")
    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('bs', [1, 2, 4, 8, 16])
# @pytest.mark.parametrize('opt', ['adam', 'adamW', 'rmsprop', 'adagrad'])
def test_bs_adam_optimizer(bs):

    network = SimpleCNN(num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = nn.Adam(params=network.trainable_params(),
                      learning_rate=0.01,
                      weight_decay=1e-5,
                      loss_scale=1.0,
                      use_nesterov=False
                      )

    bs = bs
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)

    print(f"{'adam'}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('momentum', [0.1, 0.2, 0.5, 0.9, 0.99])
def test_momentum_optimizer(momentum):
    network = SimpleCNN(in_channels=1, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = nn.Momentum(params=network.trainable_params(),
                          learning_rate=0.01,
                          momentum=momentum,
                          weight_decay=1e-5,
                          use_nesterov=False,
                          loss_scale=1.0,
                          )

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f"{momentum}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('momentum', [-0.1, -1.0, -2])
def test_wrong_momentum_optimizer(momentum):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

        net_opt = nn.Momentum(params=network.trainable_params(),
                              learning_rate=0.01,
                              momentum=momentum,
                              weight_decay=0.0001,
                              use_nesterov=False,
                              loss_scale=1.0,
                              )

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f"{momentum}, begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('loss_scale', [-0.1, -1.0])
def test_wrong_loss_scale_optimizer(loss_scale):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = nn.Momentum(params=network.trainable_params(),
                              learning_rate=0.01,
                              momentum=0.9,
                              weight_decay=0.0001,
                              use_nesterov=False,
                              loss_scale=loss_scale,
                              )

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f"{loss_scale}, begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        if cur_loss < begin_loss:
            raise ValueError


def test_param_lr_01_filter_bias_and_bn_optimizer():
    network = SimpleCNN(in_channels=1, num_classes=10)
    conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                    {'params': no_conv_params, 'lr': 0.1},
                    {'order_params': network.trainable_params()}]
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = create_optimizer(group_params, 'momentum', lr=0.01, weight_decay=1e-5, momentum=0.9,
                               nesterov=False, filter_bias_and_bn=False)
    net_opt = nn.Momentum(params=group_params,
                          learning_rate=0.01,
                          momentum=0.9,
                          weight_decay=1e-5,
                          use_nesterov=False
                          )

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_wrong_params_more_optimizer():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
        conv_params.append('test')
        no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
        group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                        {'params': no_conv_params, 'lr': 0.0},
                        {'order_params': network.trainable_params()}]
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

        net_opt = nn.Momentum(params=group_params,
                              learning_rate=0.01,
                              momentum=0.9,
                              weight_decay=1e-5,
                              use_nesterov=False
                              )

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_wrong_params_input_optimizer():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        conv_params = [1, 2, 3, 4]
        no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
        group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                        {'params': no_conv_params, 'lr': 0.0},
                        {'order_params': network.trainable_params()}]
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = nn.Momentum(params=group_params,
                              learning_rate=0.01,
                              momentum=0.9,
                              weight_decay=1e-5,
                              use_nesterov=False
                              )

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE, ])
def test_mode_mult_single_optimizer(mode):
    ms.set_context(mode=mode)
    network = SimpleCNN(in_channels=1, num_classes=10)
    conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                    {'params': no_conv_params, 'lr': 0.1},
                    {'order_params': network.trainable_params()}]
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = nn.Momentum(params=group_params,
                          learning_rate=0.01,
                          momentum=0.9,
                          weight_decay=1e-5,
                          use_nesterov=False
                          )

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'
