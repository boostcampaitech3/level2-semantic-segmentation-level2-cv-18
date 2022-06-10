# dataset settings
# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset
# import os.path as osp
classes = ('Background','General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
palette = [[0,0,0],[192,0,128],[0,128,192],[0,128,64],[128,0,0],[64,0,128],[64,0,192],[192,128,64],[192,192,128],[64,64,128],[128,0,192]]

from mmseg.datasets.builder import PIPELINES
from mmcls.datasets.pipelines.transforms import Albu
PIPELINES.register_module(module=Albu)
data_root = '/opt/ml/input/last/'
img_dir = 'images'
ann_dir = 'annotations'
dataset_type='CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
albu_train_transforms =[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0,
                rotate_limit=30,
                p=0.5,
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='ElasticTransform', p=1.0),
                    dict(type='Perspective', p=1.0),
                    dict(type='PiecewiseAffine', p=1.0),
                ],
                p=0.3),
            dict(
                type='Affine',
              p=0.3  
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RGBShift', r_shift_limit=20, g_shift_limit=20,b_shift_limit=20,always_apply=False,p=1.0),
                    dict(type='ChannelShuffle', p=1.0)
                ],
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.1,
                contrast_limit=0.15,
                p=0.5),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=10,
                p=0.5),
            dict(type='GaussNoise', p=0.3),
            dict(type='CLAHE', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='MedianBlur', blur_limit=5, p=1.0)
                ],
                p=0.3),
        ]
    
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False,
        ),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5,1.0,1.5],
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type = 'MultiImageMixDataset',
        dataset = dict(
            type=dataset_type,
            classes=classes,
            palette=palette,
            
            reduce_zero_label=False,
            data_root=data_root,
            img_dir=data_root+'images/train',
            ann_dir=data_root+'annotations/train',
            pipeline=[dict(type='LoadImageFromFile'),
                      dict(type='LoadAnnotations')
            ]),
            pipeline=train_pipeline),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root+'images/valid',
        ann_dir=data_root+'annotations/valid',
        pipeline=valid_pipeline),
    test=dict(
        classes=classes,
        palette=palette,
        
        reduce_zero_label=False,
        type=dataset_type,
        data_root=data_root,
        img_dir=data_root+'images/test',
        pipeline=test_pipeline))
