_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/flood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

img_scale = (640, 640)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile_flood'),
    dict(type='LoadAnnotations_flood'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.5)),
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

test_pipeline = [
    dict(type='LoadImageFromFile_flood'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(480, 480), (512, 512), (640, 640)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect_flood_test', keys=['img']),
        ])
]
data = dict(
    test=dict(
        pipeline=test_pipeline))

model = dict(
    backbone=dict(
        in_channels=9),
    test_cfg=dict(crop_size=(480, 480), stride=(320, 320)))
# evaluation = dict(metric='mDice')
