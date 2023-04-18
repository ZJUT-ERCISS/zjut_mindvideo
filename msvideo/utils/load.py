import os
import mindspore as ms
from mindspore import context,  load_checkpoint, load_param_into_net, nn, ops, Tensor


def load_model(path, net=None):
    ''' Load saved model.

    Args:
        path (str): Path of saved model to be loaded.
        net (nn.Cell):  The network object should be passed into this 
            implementation if the suffix of file is `.ckpt`.

    Returns:
        Loaded model.

    '''
    
    if os.path.splitext(path)[-1] == '.ckpt':
        assert isinstance(net, nn.Cell), '''Please create your model in advance 
            then load the checkpoint.'''
        param_dict = load_checkpoint(path)
        load_param_into_net(net, param_dict)

    else:
        graph = ms.load(path)
        net = nn.GraphCell(graph)

    
    return net