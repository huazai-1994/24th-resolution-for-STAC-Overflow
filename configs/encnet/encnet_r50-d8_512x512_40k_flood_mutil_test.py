_base_ = [
    '../_base_/models/encnet_r50-d8.py', '../_base_/datasets/flood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

img_scale = (640, 640)
crop_size = (480, 480)


model = dict(
    pretrained = None,
    backbone=dict(in_channels=9),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

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
