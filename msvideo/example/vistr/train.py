from mindspore import Parameter, string
import numpy as np
import argparse
from mindspore import dtype as mstype
from mindspore import nn, ops, context, Tensor, load_checkpoint, load_param_into_net
from mindspore.common import initializer as init
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindvision.msvideo.models.vistr import vistr_r50
from mindvision.msvideo.dataset import ytvos2
from mindvision.msvideo.example.vistr.callbacks import get_callbacks
from mindvision.msvideo.models.loss.vistr_loss import build_criterion
from mindspore.communication.management import get_rank
from mindspore.communication.management import get_group_size
from mindspore.communication.management import init

class TrainOneStepCellWithSense(nn.Cell):
    def __init__(self, network, optimizer, initial_scale_sense1, initial_scale_sense2, max_grad_norm=0.1):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense1 = Parameter(Tensor(initial_scale_sense1, dtype=mstype.float32), name="scale_sense1")
        self.scale_sense2 = Parameter(Tensor(initial_scale_sense2, dtype=mstype.float32), name="scale_sense2")
        self.reducer_flag = False
        self.grad_reducer = None
        self.max_grad_norm = max_grad_norm
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def set_sense(self, data):
        """set sense"""
        pred, masks = data
        self.scale_sense1.set_data(Tensor(pred, dtype=mstype.float32))
        self.scale_sense2.set_data(Tensor(masks, dtype=mstype.float32))

    def construct(self, *inputs):
        """construct"""
        pred = self.network(*inputs)
        
        grads = self.grad(self.network, self.weights)(*inputs, (self.scale_sense1 * 1., self.scale_sense2 * 1.))
        grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        pred = ops.depend(pred, self.optimizer(grads))
        return pred

class TrainOneStepWrapper:
    """train one step wrapper"""
    def __init__(self, one_step_cell, criterion, aux_loss=True, n_dec=6):
        self.one_step_cell = one_step_cell
        self.criterion = criterion
        self.aux_loss = aux_loss
        self.n_dec = n_dec

    def _get_sens_np(self, loss_np):
        """get sens np"""
        ce_w = self.criterion.weight_dict['loss_ce']
        bbox_w = self.criterion.weight_dict['loss_bbox']
        giou_w = self.criterion.weight_dict['loss_giou']
        mask_w = self.criterion.weight_dict['loss_mask']
        dice_w = self.criterion.weight_dict['loss_dice']
        sens_np = np.concatenate(
            [ce_w * loss_np['loss_ce_grad_src'],
             bbox_w * loss_np['loss_bbox_grad_src'] + giou_w * loss_np['loss_giou_grad_src']],
            axis=-1)

        sens_np2 = np.concatenate(
            [mask_w * np.expand_dims(loss_np['loss_mask_grad_src'], 0), 
             dice_w * np.expand_dims(loss_np['loss_dice_grad_src'], 0)],
             axis=0
        )

        if self.aux_loss:
            sens_np_aux = np.stack([
                np.concatenate([
                    ce_w * loss_np[f'loss_ce_grad_src_{i}'],
                    bbox_w * loss_np[f'loss_bbox_grad_src_{i}'] +
                    giou_w * loss_np[f'loss_giou_grad_src_{i}']
                ], axis=-1)
                for i in range(self.n_dec - 1)
            ])
            sens_np = np.concatenate([sens_np_aux, np.expand_dims(sens_np, 0)])
        return sens_np, sens_np2


    def __call__(self, inputs, gt):
        """call"""
        # first pass data through the network for calculating the loss and its gradients
        network_output = self.one_step_cell.network(*inputs)

        loss_np = self.criterion(network_output, gt)
        loss_value = sum(loss_np[k] * self.criterion.weight_dict[k]
                         for k in loss_np.keys() if k in self.criterion.weight_dict)[0]

        # update sensitivity parameter
        sens_np = self._get_sens_np(loss_np)

        self.one_step_cell.set_sense(Tensor(sens_np))
        # second pass data through the network for backpropagation step
        pred = self.one_step_cell(*inputs)
        return loss_value, pred, network_output

def get_optimizer(model, args_opt):
    """get optimizer"""
    # epochs 18 steps_per_epoch = int(length_dataset / config.batch_size / config.device_num)
    # lr = 0.0001 lr_embed = 0.00001 lr_drop = 12 weight_decay = 0.0001 clip_max_norm = 0.1
    lr = nn.piecewise_constant_lr(
        [args_opt.steps_per_epoch * args_opt.lr_drop, args_opt.steps_per_epoch * args_opt.epochs],
        [args_opt.lr, args_opt.lr * 0.1]
    )
    lr_embed = nn.piecewise_constant_lr(
        [args_opt.steps_per_epoch * args_opt.lr_drop, args_opt.steps_per_epoch * args_opt.epochs],
        [args_opt.lr_embed, args_opt.lr_embed * 0.1]
    )
    param_dicts = [
        {
            'params': [par for par in model.trainable_params() if 'embed' not in par.name],
            'lr': lr,
            'weight_decay': args_opt.weight_decay
        },
        {
            'params': [par for par in model.trainable_params() if 'embed' in par.name],
            'lr': lr_embed,
            'weight_decay': args_opt.weight_decay
        }
    ]
    optimizer = nn.AdamWeightDecay(param_dicts)
    return optimizer

def prepare_train(args_opt):
    """prepare train"""
    network = vistr_r50(pretrained=False)
    param_dict = load_checkpoint(
        "/home/zgz/VisTR/vistr_r50.ckpt")
    load_param_into_net(network, param_dict)
    criterion = build_criterion()

    dataset_train = ytvos2.ytvos(path="/usr/dataset/VOS/",
                                split='train',
                                seq=36,
                                batch_size=1,
                                repeat_num=1,
                                shuffle=False)
    dataset_train = dataset_train.run()
    step_size = dataset_train.get_dataset_size()


    args_opt.steps_per_epoch = int(step_size / 1 / 1)


    optimizer = get_optimizer(network, args_opt)
    return network, optimizer, criterion, dataset_train

def run_train(args_opt):
    """run training process"""
    # set_default()
    init('nccl')
    rank = get_rank()
    # args.device_num = get_group_size()
    # context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE)
    # context.set_context(device_target=GPU)

    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL, search_mode="dynamic_programming")

    detr, optimizer, criterion, dataset_train = prepare_train(args_opt)
    data_loader = dataset_train.create_dict_iterator()
    detr.set_train()

    if args_opt.aux_loss:
        sens_param1 = np.ones([args_opt.dec_layers, args_opt.batch_size, args_opt.num_queries, 46])
    else:
        sens_param1 = np.ones([args_opt.batch_size, args_opt.num_queries, 46])
    sens_param2 = np.ones([2, args_opt.batch_size, args_opt.num_queries, 300, 500])
    step_cell = TrainOneStepCellWithSense(detr, optimizer, sens_param1, sens_param2, args_opt.clip_max_norm)
    train_wrapper = TrainOneStepWrapper(step_cell, criterion,
                                        args_opt.aux_loss, args_opt.dec_layers)

    if args_opt.save_ckpt_logs:
        callbacks = get_callbacks(args_opt, detr)

    for sample in data_loader:
        image = sample['video']
        labels = sample['labels']
        boxes = sample['boxes']
        valid = sample['valid']
        masks = sample['masks']
        gt = (labels, boxes, valid, masks)
        out = train_wrapper(image, gt)
        loss_value, _, _ = out
        if args_opt.save_ckpt_logs:
            callbacks(loss_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VisTR_r50 Train.")
    parser.add_argument("--model_name", type=str, default="vistr_r50")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=6)
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_queries', default=360, type=int,
                        help="Number of query slots")
    parser.add_argument('--save_ckpt_logs', default=True, type=bool)
    parser.add_argument('--aux_loss', default=False, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_embed', default=0.00001, type=float)
    parser.add_argument('--lr_drop', default=12, type=int)
    parser.add_argument('--epochs', default=18, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--clip_max_norm', default=0.1 , type=float)
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--keep_checkpoint_max", default=10, type=int)
    parser.add_argument("--save_path", default="/home/zgz/vision/.vscode/ckpt")
    parser.add_argument("--log_frequency_step", default=100, type=int)
    parser.add_argument("--device_num", default=2, type=int)

    args = parser.parse_known_args()[0]



    run_train(args)

