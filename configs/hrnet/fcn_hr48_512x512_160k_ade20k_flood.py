_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/flood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
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
    samples_per_gpu=6,
    train=dict(
        pipeline=train_pipeline))

model = dict(
    pretrained=None,
    backbone=dict(
        in_channels=9,
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        num_classes=2))

load_from = "/data/projects/pre_trained/best.pth"