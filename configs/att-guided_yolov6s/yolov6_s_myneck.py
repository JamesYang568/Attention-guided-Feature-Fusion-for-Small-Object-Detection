# James Yang
_base_ = '../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# ======================= Frequently modified parameters =====================
# -----train val related-----
# Base learning rate for optim_wrapper
max_epochs = 100  # Maximum training epochs
num_last_epochs = 10  # Last epoch number to switch training pipeline
save_epoch_intervals = 1  # 5
train_batch_size_per_gpu = 20
train_num_workers = 2
# ============================== Unmodified in most cases ===================

deepen_factor = _base_.deepen_factor
widen_factor = 0.5

model = dict(
    backbone=dict(
        out_indices=[1, 2, 3, 4]
    ),
    neck=dict(
        type='YOLOv6RepPAFPN_AFFM',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],
        num_csp_blocks=12,
        affm_cfg=dict(
            type='AttentionAlignedFeatureFusion',
            parallel=False, aggregate=True, kernel=5,
            ratio=4,
        ),
        finer_cfg=dict(
            type='FeatureSupplementModule',
            shallow_channels=int(128 * widen_factor), deep_channels=int(128 * widen_factor),  # caution
            ratio=4, dilations=[1],
            share=True, fix=False,
        ),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers
)

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu // 2,
    num_workers=train_num_workers // 2
)
test_dataloader = val_dataloader

base_lr = _base_.base_lr / 8
optim_wrapper = dict(optimizer=dict(lr=base_lr))
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs,
        warmup_epochs=3),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=2,
        save_best='auto')
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_last_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
