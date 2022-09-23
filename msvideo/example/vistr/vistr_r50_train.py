import argparse
import os
from mindspore import ParallelMode, context, load_checkpoint, load_param_into_net
from mindspore import nn
import mindspore

from mindspore.common import set_seed
from mindspore.communication import init, get_rank, get_group_size
from mindspore.profiler import Profiler
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindvision.engine.callback import LossMonitor
from mindvision.engine.lr_schedule.lr_schedule import warmup_cosine_annealing_lr_v1
from mindvision.msvideo.dataset import ytvos2
from mindvision.msvideo.models.vistr import vistr_r50
from mindvision.msvideo.models.blocks.vistr_loss import SetCriterion
from mindvision.msvideo.models.neck.instance_sequence_match import HungarianMatcher


set_seed(10)


class LossCell(nn.Cell):
    """Cell with loss function.."""

    def __init__(self, net, loss):
        super(LossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, image, labels, boxes, valid, masks):
        """Cell with loss function."""
        feature = self._net(image)
        return self._loss(feature, (labels, boxes, valid, masks))

    @property
    def backbone_network(self):
        """Return net."""
        return self._net


def vistr_r50_train(args_opt):
    # init("nccl")
    context.set_context(mode=context.GRAPH_MODE)
    # context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(device_id=0)
    # mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=True)
    # mindspore.set_auto_parallel_context(pipeline_stages=2, full_batch=True)
    if args_opt.profiler:
        profiler = Profiler(output_path='./profiler_data')

    dataset_train = ytvos2.ytvos(path="/usr/dataset/VOS/",
                                 split='train',
                                 seq=36,
                                 batch_size=1,
                                 repeat_num=1,
                                 shuffle=False)
    dataset_train = dataset_train.run()
    step_size = dataset_train.get_dataset_size()
    ckpt_save_dir = args_opt.ckpt_save_dir

    network = vistr_r50(pretrained=False)
    # param_dict = load_checkpoint(
    #     "/home/zgz/VisTR/vistr_r50.ckpt")
    # load_param_into_net(network, param_dict)

    # Set learning rate scheduler.
    # lr = warmup_cosine_annealing_lr_v1(lr=0.0001, steps_per_epoch=step_size,
    #                                    warmup_epochs=2.5, max_epoch=30, t_max=30, eta_min=0)
    lr = nn.piecewise_constant_lr(
        [step_size * args_opt.lr_drop, step_size * args_opt.epochs],
        [args_opt.lr, args_opt.lr * 0.1]
    )
    lr_embed = nn.piecewise_constant_lr(
        [step_size * args_opt.lr_drop, step_size * args_opt.epochs],
        [args_opt.lr_embed, args_opt.lr_embed * 0.1]
    )

    #  Define optimizer.
    # network_opt = nn.AdamWeightDecay(
    #     network.trainable_params(), lr, beta1=0.9, beta2=0.999, weight_decay=0.0001)
    param_dicts = [
        {
            'params': [par for par in network.trainable_params() if 'embed' not in par.name],
            'lr': lr,
            'weight_decay': args_opt.weight_decay
        },
        {
            'params': [par for par in network.trainable_params() if 'embed' in par.name],
            'lr': lr_embed,
            'weight_decay': args_opt.weight_decay
        }
    ]
    network_opt = nn.AdamWeightDecay(param_dicts)

    # Define losgs function.
    matcher = HungarianMatcher(num_frames=args_opt.num_frames,
                               cost_class=args_opt.cost_class,
                               cost_bbox=args_opt.cost_bbox,
                               cost_giou=args_opt.cost_giou)
    network_loss = SetCriterion(num_classes=args_opt.num_classes,
                                matcher=matcher,
                                weight_dict=args_opt.weight_dict,
                                eos_coef=args_opt.eos_coef,
                                losses=args_opt.losses,
                                aux_loss=args_opt.aux_loss)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size, keep_checkpoint_max=args_opt.ckpt_max)
    ckpt_callback = ModelCheckpoint(
        prefix=args_opt.model_name, directory=ckpt_save_dir, config=ckpt_config)

    # Init the model.
    # model = Model(network, loss_fn=network_loss,
    #               optimizer=network_opt, metrics={"class": vistr_metric()})
    net_with_loss = LossCell(network, network_loss)
    vistr_net = nn.TrainOneStepCell(net_with_loss, network_opt)

    model = Model(vistr_net, optimizer=network_opt)

    # Begin to train.
    print('[Start training `{}`]'.format(args_opt.model_name))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=False)
    print('[End of training `{}`]'.format(args_opt.model_name))
    if args_opt.profiler:
        profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisTR_r50 Train.")
    parser.add_argument("--dataset_path", type=str,
                        default="/data0/VOS")
    parser.add_argument("--model_name", type=str, default="vistr_r50")
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=2)
    parser.add_argument("--epoch_size", type=int,
                        default=18, help="Train epoch size.")
    parser.add_argument('--epochs', default=18, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_embed', default=0.00001, type=float)
    parser.add_argument('--lr_drop', default=12, type=int)
    parser.add_argument("--pretrained", type=bool,
                        default=True, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of batch size.")
    parser.add_argument("--ckpt_save_dir", type=str,
                        default="./vistr_r50", help="Location of training outputs.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default="/home/spicyww/zgz/vision_2/data/vistr_r50.ckpt",
                        help="Location of Pretrained Model.")
    parser.add_argument("--ckpt_max", type=int, default=100,
                        help="Max number of checkpoint files.")
    parser.add_argument("--dataset_sink_mode", default=False,
                        help="The dataset sink mode.")
    parser.add_argument("--run_distribute", type=bool,
                        default=False, help="Run distribute.")
    parser.add_argument("--num_parallel_workers", type=int,
                        default=8, help="Number of parallel workers.")
    parser.add_argument("--profiler", type=bool,
                        default=False, help="Use Profiler.")
    # matcher
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--cost_class", type=int, default=1)
    parser.add_argument("--cost_bbox", type=int, default=1)
    parser.add_argument("--cost_giou", type=int, default=1)
    # loss
    parser.add_argument("--num_classes", type=int, default=41)
    parser.add_argument("--weight_dict", type=dict)
    parser.add_argument("--eos_coef", type=float, default=0.1)
    parser.add_argument("--losses", type=list,
                        default=['labels', 'boxes', 'masks'])
    parser.add_argument("--aux_loss", type=bool, default=True)

    args = parser.parse_known_args()[0]

    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    weight_dict["loss_mask"] = 1
    weight_dict["loss_dice"] = 1

    aux_weight_dict = {}
    for i in range(6 - 1):
        aux_weight_dict.update(
            {k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    args.weight_dict = weight_dict
    vistr_r50_train(args)
