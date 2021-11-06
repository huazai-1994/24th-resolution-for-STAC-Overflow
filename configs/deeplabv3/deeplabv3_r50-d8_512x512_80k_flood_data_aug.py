_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/flood.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
img_scale = (520, 520)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile_flood'),
    dict(type='LoadAnnotations_flood'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 1.5)),
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
    train=dict(
        pipeline=train_pipeline))
model = dict(
    pretrained = None,
    backbone=dict(in_channels=9),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
evaluation = dict(interval=4000, metric='mIoU')

