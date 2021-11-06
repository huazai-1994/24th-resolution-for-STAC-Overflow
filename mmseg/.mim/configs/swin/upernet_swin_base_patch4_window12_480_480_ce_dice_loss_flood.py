_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/flood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    pretrained='/data0/projects/pre_trained_model/swin_base_patch4_window12_384_22k.pth', # noqa
    backbone=dict(
        pretrain_img_size=384,
        in_channels=9,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
    decode_head=dict(in_channels=[128, 256, 512, 1024],
                     num_classes=2,
                     loss_decode=dict(
                         type='DC_and_CE_loss',
                         soft_dice_kwargs={},
                         ce_kwargs={'use_sigmoid':False, 'loss_weight':1.0})
                     ),
    auxiliary_head=dict(in_channels=512,
                        num_classes=2,
                        loss_decode=dict(
                            type='DC_and_CE_loss',
                            soft_dice_kwargs={},
                            ce_kwargs={'use_sigmoid':False, 'loss_weight':1.0})
                        ))


img_scale = (520, 520)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile_flood'),
    dict(type='LoadAnnotations_flood'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=180, pad_val=0,seg_pad_val=0),

    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect_flood', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=2,
    train=dict(
        pipeline=train_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)