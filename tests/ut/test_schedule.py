import sys
sys.path.append('.')

import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor
from lr_schedule import warmup_step_lr, warmup_cosine_annealing_lr_v1, warmup_cosine_annealing_lr_v2
from lr_schedule import dynamic_lr, piecewise_constant_lr


def test_scheduler():
    # warmup_step_lr
    lrs_manually = [0.025, 0.05, 0.05, 0.05, 0.005, 0.005,
                    0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
                    0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
    lrs_ms = warmup_step_lr(0.05, [2,3], steps_per_epoch=2, warmup_epochs=1, max_epoch=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # warmup_cosine_annealing_lr_v1
    lrs_manually = [0.025, 0.05, 0., 0., 0.05, 0.05, 0., 0., 0.05,
                    0.05, 0., 0., 0.05, 0.05, 0., 0., 0.05, 0.05,
                    0., 0.]
    lrs_ms = warmup_cosine_annealing_lr_v1(0.05, steps_per_epoch=2, warmup_epochs=1, max_epoch=10, t_max=1.0)
    assert np.allclose(lrs_ms, lrs_manually)

    # warmup_cosine_annealing_lr_v2
    lrs_manually = [0.025, 0.05, 0., 0., 0.05, 0.05, 0., 0., 0.05,
                    0.05, 0., 0., 0.05, 0.05, 0.0375, 0.0375, 0.0125, 0.0125,
                    0., 0.]
    lrs_ms = warmup_cosine_annealing_lr_v2(0.05, steps_per_epoch=2, warmup_epochs=1, max_epoch=10, t_max=1.0)
    assert np.allclose(lrs_ms, lrs_manually)

    # dynamic_lr
    lrs_manually = [0.1, 0.55, 1.0, 0.9949107209404664, 0.9797464868072487, 0.9548159976772592,
                    0.9206267664155906, 0.8778747871771292, 0.8274303669726426, 0.7703204087277988,
                    0.7077075065009433, 0.6408662784207149, 0.5711574191366426, 0.5, 0.42884258086335747, 0.35913372157928514,
                    0.29229249349905684, 0.2296795912722014, 0.1725696330273575, 0.12212521282287092, 0.07937323358440934, 0.04518400232274078]
    lrs_ms = dynamic_lr(1.0, steps_per_epoch=2, warmup_steps=2, warmup_ratio=0.1, epoch_size=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # piecewise_constant_lr
    lrs_manually = [0.05, 0.05, 0.0375, 0.0375, 0.0375, 0.025, 0.025, 0.025, 0.025, 0.025]
    lrs_ms = piecewise_constant_lr([2,5,10], [0.05, 0.0375, 0.025])
    assert np.allclose(lrs_ms, lrs_manually)

if __name__ == "__main__":
    test_scheduler()