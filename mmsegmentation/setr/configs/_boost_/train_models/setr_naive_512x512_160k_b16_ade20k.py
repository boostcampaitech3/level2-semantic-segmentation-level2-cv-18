_base_ = [
    '../models/setr_naive.py', 
    '../datasets/dataset.py',
    '../default_runtime.py', 
    '../schedules/schedule.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        img_size=(512, 512),
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='/opt/ml/input/code/mmsegmentation/configs/my_base/models/pretrain/jx_vit_large_p16.pth')
        ),
    decode_head=dict(num_classes=11),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=11,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

# optimizer = dict(
#     lr=0.01,
#     weight_decay=0.0,
#     paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

# num_gpus: 8 -> batch_size: 16
data = dict(samples_per_gpu=2)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=7e-6)

lr = 1e-4
optimizer = dict(
    type='AdamW', 
    lr=lr, 
    weight_decay=0.00
)
