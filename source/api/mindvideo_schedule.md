## mindvideo.schedule

### linear_warmup_learning_rate

> def mindvideo.schedule.linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr)

Linear warmup learning rate.


### warmup_step_lr

> def mindvideo.schedule.warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1)

Warmup step learning rate.


### warmup_cosine_annealing_lr_v1

> def mindvideo.schedule.warmup_cosine_annealing_lr_v1(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0)

Cosine annealing learning rate.


### warmup_cosine_annealing_lr_v2

> def mindvideo.schedule.warmup_cosine_annealing_lr_v2(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0)

Cosine annealing learning rate V2.


### cosine_learning_rate

> def mindvideo.schedule.cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps)

Cosine learning rate.


### dynamic_lr

> def mindvideo.schedule.dynamic_lr(base_lr, steps_per_epoch, warmup_steps, warmup_ratio, epoch_size)

Dynamic learning rate generator.


### piecewise_constant_lr

> def mindvideo.schedule.piecewise_constant_lr(milestone, learning_rates)

Piecewise constant learning rate.