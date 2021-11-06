_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/flood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
img_scale = (520, 520)
crop_size = (480, 480)


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

test_pipeline = [
    dict(type='LoadImageFromFile_flood'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(480, 480),(512, 512), (640, 640)],
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

load_from = "/data/projects/pre_trained/best.pth"
