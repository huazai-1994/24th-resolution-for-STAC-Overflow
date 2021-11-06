# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
# recommand use this config for BEiT models which are self-supervised pretrained and then intermediate fine-tuned on imagenet
_base_ = [
    '../../_base_/models/upernet_beit.py', '../../_base_/datasets/flood.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_40k.py'
]
img_scale = (520, 520)
crop_size = (480, 480)

model = dict(
    pretrained='/data/projects/pre_trained/beit_base_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        type='BEiT',
        img_size=480,
        patch_size=16,
        in_chans=9,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=0.1,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11]
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=2,
        channels=768,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=2
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(426, 426))
)


train_pipeline = [
    dict(type='LoadImageFromFile_flood'),
    dict(type='LoadAnnotations_flood'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(1, 1.5), keep_ratio=True),
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

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=2,
    train=dict(
        pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


# runner = dict(type='IterBasedRunnerAmp')
#
# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
