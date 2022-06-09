_base_ = ['./segformer_mit-b0_512x512_160k_ade20k.py']

# model settings
model = dict(
    # pretrained='pretrain/mit_b5.pth',
    # pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth',
    pretrained='/opt/ml/input/code/mmsegmentation/configs/my_base/models/pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

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
    weight_decay=0.01
)