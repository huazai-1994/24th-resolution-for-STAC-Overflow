_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/flood.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

albu_train_transforms = [
    dict(
        type='RandomCrop',
        height=256,
        width=256,
        p=0.5),
    dict(
        type='RandomRotate90',
        p=0.5),
    dict(
        type='HorizontalFlip',
        p=0.5),
    dict(
        type='VerticalFlip',
        p=0.5),

    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.2),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=10,
    #             g_shift_limit=10,
    #             b_shift_limit=10,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0)
    #     ],
    #     p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile_flood'),
    dict(type='LoadAnnotations_flood'),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'masks'
        },
        update_pad_shape=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect_flood', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline))

model = dict(
    pretrained = None,
    backbone=dict(in_channels=2),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
evaluation = dict(interval=4000, metric='mIoU')

