_base_ = [
    '../_base_/datasets/custom.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BEiT',
        pretrained = '../pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
        img_size=512,
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=(7, 11, 15, 23),
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_cfg=dict(type='LN', eps=1e-06),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        final_norm=False,
        norm_eval=False,
        init_values=0.1,
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=1024,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,avg_non_ignore=True)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(512,512), stride=(341, 341)))
    # test_cfg=dict(mode='whole'))


optimizer = dict(_delete_=True, type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)